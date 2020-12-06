#- Imports
from time import time
import os, re, hy
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from beacon.bert.bert_model import read_bert
from beacon.bert.semantic_relations import build_relator, compile_DG, compile_prolog

#- For Mapping Lexs to Unique Characters (LIMIT 58 unique Lexs)
DEFAULTLEX_PATH = os.path.dirname(os.path.abspath(__file__))+"/lexicon"
VP_PATTERN = [{'POS': 'VERB', 'OP': '?'},
              {'POS': 'ADV', 'OP': '*'},
              {'POS': 'AUX', 'OP': '*'},
              {'POS': 'VERB', 'OP': '+'}]
NTAGS = ["NSUBJ", "DOBJ", "POBJ", "ATTR", "CONJ", "NSUBJPASS", "CSUBJ", "COMPOUND"]
VTAGS =["APPOS", "CCOMP", "PCOMP", "ADVCL", "ROOT", "AUX",  "XCOMP", "PARATAXIS", "ADVMOD", "AMOD", "ACL", "RELCL", "PREP", "DATIVE", "CSUBJPASS", "OPRD", "AUXPASS", "QUANTMOD"]
TAGS = NTAGS + VTAGS

#-
class Beacon(object):
  """ Class defines a refactored implementation of Clever which now
  uses re, spacy and pandas packages to locate text snippets according to some
  predifined lexicon. This implemenation was design to serve as a single callable
  object which extracts snippets over a single text string at a time and returns
  a dataframe of tagged results. After initializing Beacon with desired lexicon
  files, the object's call function is defined by the following monad:

  beacon(text::str) ->  df::pd.DataFrame

  With dataframe columns:
    ["index", "rels_layers", "rels_threshold", "rels_index", "rels_lex",
     "lex","startx","endx","text", "snippet_rule",
     "snippet_matchx","snippet_startx","snippet_endx","snippet_text"])

  For every passed text string, this Clever implementation outputs a dataframe
  with the above specification. If no matches are found, an empty dataframe that
  conforms to the above specifications will be returned instead. This will be
  useful for applying Clever functionally over a collection of texts. For example,
  an anonymous function can be built using beacon to process a corpus of notes:

  beacon = Beacon(return_snippet_text=False)
  beacon_lambda = lambda x: beacon(x).to_json()
  corpus["beacon_output"] = corpus["ReportTex†"].apply(beacon_lambda)

  In the above example, we build an anonymous function which takes an argument x
  and passes it to beacon and then converts the dataframe to a json to be stored
  in the corpus dataframe. This will let us naively parallelize the procedure over
  a collection of texts.

  Notes:

  Previous implementions of Clever use n-grams along snippet tags to logically
  infer postive or negative conformations of target concepts by checking the
  presence of modifier tags in the n-grams. The results are prone to noise as it
  does not check to see if the modfier is being applied to the concept, or if a
  concept is tied to the patient and not another entity in the snippet.

  We explore the use of pretrained BERT language models to infer the relationships
  between lexicon terms in order to confirm dependence between modifiers and concepts.
  This is done by:
    1) extracting the token correlation matricies (attention) from BERT at the last N layers of the network,
    2) mapping lexicon substrings to BERT input tokens,
    3) aggregating attention for lexicon terms,
    4) selecting high values token relations using a dynamically programmed gini coefficeint search,
    5) and mapping those selected tokens back to lexicon terms.
  """

  def __init__(self,
               lexicon_path=DEFAULTLEX_PATH+"/lexicon_default.psv",
               target_lex="UMLS-HOMELESS",
               bert_model_path=None):

    #- Bert Model For Knowledge Graph Annotating
    self.bert_relate = build_relator(read_bert(bert_model_path))

    #- Load Lexicon
    self.lexicon = self._read_lexicon(lexicon_path)
    self.lexicon = self.lexicon[self.lexicon["lex"]!="UMLS-SUICIDE"]
    self._spacy_nlp = spacy.load("en_core_web_sm")
    self._verb_matcher = Matcher(self._spacy_nlp.vocab)
    self._verb_matcher.add("Verb phrase", None, VP_PATTERN)

    #- Compile search regexes at object intialization to improve performance
    self.target_lex = target_lex
    self._lexicon_regexs = dict([(grpn, self._build_regex(x["string"])) for (grpn, x) in self.lexicon.groupby("lex")])
    self._snippet_regex1 = re.compile(r"((^|([\r\n][\r\n].*?[\-\:]\s))([^\r^\n](?![\r\n]))*?)(" + self._lexicon_regexs[self.target_lex].pattern + ")(([^\.]*?\.){1,3})")
    self._snippet_regex2 = re.compile(r"((\.[^\.]*?){1,3})(" + self._lexicon_regexs[self.target_lex].pattern + ").*$")
    self._snippet_regex3 = re.compile(r"((\.[^\.]*?){1,3})(" + self._lexicon_regexs[self.target_lex].pattern + ")(([^\.]*?\.){1,3})")
    self._snippet_regex4 = re.compile(r".*(" + self._lexicon_regexs[self.target_lex].pattern + ").*")
    lexs = list(self._lexicon_regexs.keys()) + TAGS

  def __call__(self, text, bert_layers=[4,8], bert_threshold=None, return_snippet_text=True):
    preprocessed_text = self._preprocess(text) # First preprocess text
    t = time()
    snippets = self._fetch_snippets(preprocessed_text)
    tags = []
    for (i, row) in snippets[["snippet_text","snippet_startx"]].iterrows():
        text, startx = row
        tagged_text = self._tag_nounchunks(text) # Then tag noun phrases
        tagged_text = self._tag_verbchunks(text, tagged_text) # Then tag verb phrases
        tagged_text = self._tag_by_lexicon(text, tagged_text) # Then tag lexs over text
        tagged_text = tagged_text.sort_values("endx",ascending=True).sort_values(["startx"]).reset_index(drop=True)#.reset_index()
        tagged_text["snippet_index"] = i
        tagged_text["startx"] += startx
        tagged_text["endx"] += startx
        tags.append(tagged_text)
    snippets = snippets.merge(pd.concat(tags), on="snippet_index")
    snippets = snippets.groupby(["startx","endx","text","snippet_index","snippet_startx","snippet_endx","snippet_text"])["lex"].apply(lambda x: "|".join(x)).reset_index().reset_index()

    kg_annotated = self.bert_relate(snippets, bert_layers, bert_threshold) if snippets is not None else None # Annonate tag relations using BERT
    out = kg_annotated if kg_annotated is not None else pd.DataFrame([], columns=["index", "rels_layers", "rels_threshold", "rels_index", "rels_lex", "rels_tokens",
                                                                                   "lex","startx","endx","text", "snippet_index","snippet_startx","snippet_endx","snippet_text"])
    out = out if return_snippet_text else out.drop(columns=["snippet_text"])
    return out

  #- FUTURE: Standardize Lexicon and Headers file type. (i.e csv, parquet, etc).
  def _read_lexicon(self, path):
    return pd.read_csv(path, sep="|")

  def _preprocess(self, text):
    text = text.lower()
    text = text.replace("...", " ")
    return text.lower()

  def _build_regex(self, series):
    """ Method is used to build a compiled regular expression which will be used to split and
    search text. The regex is built from a pandas Series of substrings for a subset of lexicon
    terms grouped by particular lex type.
    """
    series = [re.sub(r'([^0-9a-zA-Z\s])', r'\\\1', x) for x in list(series)] # Add forward slash to non-alphanumeric characters
    series = [x.replace(" ", "\s") for x in series] # Change spaces to arbitrary whitespace character.
    series = sorted(series, key=len, reverse=True)
    regex = r"|".join(["({})".format(s) if s.startswith("\\") else "({})".format(s) for s in series]) # Join terms and encapsulate each with grouping parenthesis to keep term on splits
    return re.compile(regex)

  def _fetch_snippets(self, text):
    """ """
    matches = list(re.finditer(self._snippet_regex1, text))
    if len(matches) < 1:
        matches = list(re.finditer(self._snippet_regex3, text))
        df = [(span[0], span[1], match) for (span, match) in [(x.span(), x.group(0)) for x in matches]]
        df = pd.DataFrame(df).reset_index()
        df.columns = ["snippet_index","snippet_startx","snippet_endx","snippet_text"]
        return df
    snippets = [(match, span[0]) for (span, match) in [(x.span(), x.group(0)) for x in matches]]
    df = []
    for (text, startx) in snippets:
      out = list(re.finditer(self._snippet_regex2, text))
      if len(out) < 1:  out = list(re.finditer(self._snippet_regex4, text))
      df += [(startx+span[0], startx+span[1], match) for (span, match) in [(x.span(), x.group(0)) for x in out]]
    df = pd.DataFrame(df).reset_index()
    df.columns = ["snippet_index","snippet_startx","snippet_endx","snippet_text"]
    return df

  def _match_on_regex(self, text, key):
    """ Methods searches for a given lex type and returns a Dataframe of the matched indexes and string."""
    out = re.finditer(self._lexicon_regexs[key], text)
    out = [(key, span[0], span[1], match) for (span, match) in [(x.span(), x.group(0)) for x in out]]
    return pd.DataFrame(out, columns=["lex","startx","endx","text"])

  def _tag_nounchunks(self, text):
    """ """
    doc =  self._spacy_nlp(text)
    df = [[chunk.root.dep_.upper(), chunk.start_char, chunk.end_char, chunk.text] for chunk in doc.noun_chunks]
    return pd.DataFrame(df, columns=["lex", "startx", "endx", "text"])

  def _tag_verbchunks(self, text, tags=None):
    """ """
    doc = self._spacy_nlp(text)
    matches = self._verb_matcher(doc)
    spans = filter_spans([doc[start:end] for _, start, end in matches])
    df = [[chunk.root.dep_.upper(), chunk.start_char, chunk.end_char, chunk.text] for chunk in spans]
    df = pd.DataFrame(df, columns=["lex", "startx", "endx", "text"])
    df = df.append(tags) if tags is not None else df
    return df

  def _tag_by_lexicon(self, text, tags=None):
    """ Method performs regex matching over all lex types and builds a pandas DataFrame of the results."""
    dfs = [self._match_on_regex(text, key) for key in self._lexicon_regexs]
    dfs = pd.concat(dfs)
    dfs = dfs.append(tags) if tags is not None else dfs
    dfs["text"] = dfs["text"].apply(lambda x: x.strip())
    return dfs if len(dfs) > 0 else None
