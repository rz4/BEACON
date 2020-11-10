#- Imports
import os, re, hy
import pandas as pd
import spacy
from beacon.bert.model import read_bert
from beacon.bert.semantic_relations import build_relator, compile_network

#- For Mapping Lexs to Unique Characters (LIMIT 58 unique Lexs)
BASE58 = '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
DEFAULTLEX_PATH = os.path.dirname(os.path.abspath(__file__))+"/lexicon"

#-
class Beacon(object):
  """ Beacon class defines a refactored implementation of Clever which now
  uses re, spacy and pandas packages to locate text snippets according to some
  predifined lexicon. This implemenation was design to serve as a single callable
  object which extracts snippets over a single text string at a time and returns
  a dataframe of tagged results. After initializing Beacon with desired lexicon
  files, the object's call function is definded by the following monad:

  beacon(text::str) ->  df::pd.DataFrame

  With dataframe columns:
    ["index", "rels_depth", "rels_threshold", "rels_index", "rels_lex",
     "lex","startx","endx","text", "snippet_rule",
     "snippet_matchx","snippet_startx","snippet_endx","snippet_text"])

  For every passed text string, this Clever implementation outputs a dataframe
  with the above specification. If no matches are found, an empty dataframe that
  conforms to the above specifications will be returned instead. This will be
  useful for applying Clever functionally over a collection of texts. For example,
  an anonymous function can be built using beacon to process a corpus of notes:

  beacon = Beacon(return_snippet_text=False)
  beacon_lambda = lambda x: beacon(x).to_json()
  corpus["beacon_ouput"] = corpus["ReportTex†"].apply(beacon_lambda)

  In the above example, we build an anonymous function which takes an argument x
  and passes it to beacon and then converts the dataframe to a json to be stored
  in the corpus dataframe. This will let us naively parallelize the procedure over
  a collection of texts.

  Notes:

  Previous implemenations of Clever use n-grams along snippet tags to logically
  infer postive or negative conformations of target concepts by checking the
  presence of modifier tags in the n-grams. The results are prone to noise as it
  does not check to see if the modier is being applied to the concept, or if a
  concept is tied to a patient and not another entity in the snippet.

  We explore the use of pretrained BERT language models to infer the relationships
  between lexicon terms in order to confirm dependence between modifiers and concepts.
  This is done by:
    1) extracting the token correlation matricies (attention) from BERT at the last N layers of the network,
    2) mapping lexicon substrings to BERT input tokens,
    3) aggregating attention for lexicon terms,
    4) selecting high values token relations using a dynamically programmed gini coefficeint search,
    5) and mapping those selected tokens back to lexicon terms.
  From prelimary results, we find that BERT's attention output is very useful at
  measuring grammatical dependencies between lexicon terms. We tested snippets which
  contain both a patient and family entities, and the method returned the concept
  term relating to patient and not the family memember.
  """

  def __init__(self, lexicon_path=DEFAULTLEX_PATH+"/lexicon_default.psv",
                     snippet_rules_path=DEFAULTLEX_PATH+"/snippet_rules_default.psv",
                     bert_model_path=None):
    """
    """

    #- Bert Model For Knowledge Graph Annotating
    self.bert_relate = build_relator(read_bert(bert_model_path))

    #- Load Lexicon
    self.lexicon = self._read_lexicon(lexicon_path)
    self.spacy_nlp = nlp = spacy.load("en_core_web_sm")

    #- Load Snippet pattern rules
    self.rules = self._read_snippet_rules(snippet_rules_path)

    #- Compile search regexes at object intialization to improve performance
    self._lexicon_regexs = dict([(grpn, self._build_regex(x["string"])) for (grpn, x) in self.lexicon.groupby("lex")])
    lexs = list(self._lexicon_regexs.keys()) + ["NSUBJ", "DOBJ", "POBJ", "ATTR", "ROOT", "CONJ", "NSUBJPASS", "APPOS"]
    self._lex_mapping = dict([(key, BASE58[i]) for (i, key) in enumerate(lexs)]) # Build lex to symbol mapping
    self._snippet_rules = dict([(key, self._build_rule(self.rules[key])) for key in self.rules])

  def __call__(self, text, bert_depth=2, bert_threshold=None, return_snippet_text=True):
    preprocessed_text = self._preprocess(text) # First preprocess text
    tagged_text = self._tag_nounchunks(preprocessed_text)
    tagged_text = self._tag_by_lexicon(preprocessed_text, tagged_text) # Then tag lexs over text
    snippets = self._snip_by_rules(preprocessed_text, tagged_text) if tagged_text is not None else None # Finally match on snippet patterns
    kg_annotated = self.bert_relate(snippets, bert_depth, bert_threshold) if snippets is not None else None # Annonate tag relations using BERT
    out = kg_annotated if kg_annotated is not None else pd.DataFrame([], columns=["index", "rels_depth", "rels_threshold", "rels_index", "rels_lex",
                                                                                  "lex","startx","endx","text", "snippet_rule",
                                                                                  "snippet_matchx","snippet_startx","snippet_endx","snippet_text"])
    out = out if return_snippet_text else out.drop(columns=["snippet_text"])
    return out

  def entity_graph(self, annotations):
    """
    """
    return compile_network(annotations)

  #- FUTURE: Standardize Lexicon and Headers file type. (i.e csv, parquet, etc).
  def _read_lexicon(self, path):
    return pd.read_csv(path, sep="|")

  # FUTURE: Load snippet rules from file.
  #- FUTURE: Standardize file type. (i.e csv, parquet, etc).
  def _read_snippet_rules(self, path):
    df = pd.read_csv(path, sep="|", header=None)
    return dict([r for (i, r) in df.iterrows()])

  def _preprocess(self, text):
    out = text.lower() # 1. Lowercase
    return " ".join(out.split())

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

  def _build_rule(self, rule):
    """ Method is used to build a compiled regular expression which will be used to match on snippet patterns.
    To allow definitions of patterns using the lex type names, a mapping of the names to individual characters is
    done using a BASE58 mapping. This limits the number of possible lexs to 58, but that is way above what is needed
    for Clever. The mapping also speeds up the pattern matching since the regex search is done over single characters.
    """
    regex = rule
    for key in self._lex_mapping: regex = regex.replace(key, self._lex_mapping[key]) # Substitute lex type names with encodings
    return re.compile(r""+regex)

  def _match_on_regex(self, text, key):
    """ Methods searches for a given lex type and returns a Dataframe of the matched indexes and string."""
    out = re.finditer(self._lexicon_regexs[key], text)
    out = [(key, span[0], span[1], match) for (span, match) in [(x.span(), x.group(0)) for x in out]]
    return pd.DataFrame(out, columns=["lex","startx","endx","text"])

  def _tag_nounchunks(self, text):
    """ """
    doc =  self.spacy_nlp(text)
    df = [[chunk.root.dep_.upper(), chunk.start_char, chunk.end_char, chunk.text] for chunk in doc.noun_chunks]
    return pd.DataFrame(df, columns=["lex", "startx", "endx", "text"])

  def _tag_by_lexicon(self, text, tags=None):
    """ Method performs regex matching over all lex types and builds a pandas DataFrame of the results."""
    dfs = [self._match_on_regex(text, key) for key in self._lexicon_regexs]
    dfs = pd.concat(dfs)
    dfs = dfs.append(tags) if tags is not None else dfs
    dfs["text"] = dfs["text"].apply(lambda x: x.strip())
    return dfs.sort_values("endx",ascending=True).sort_values("startx").reset_index(drop=True).reset_index() if len(dfs) > 0 else None

  def _match_on_rule(self, text, tagged_df, key):
    """ Methods searches for a given snippet pattern and returns a Dataframe of the matches."""
    tag_seq = "".join([self._lex_mapping[i] for i in list(tagged_df["lex"])])
    matches = re.finditer(self._snippet_rules[key], tag_seq)
    matches = [(key, span[0], span[1]) for span in [x.span() for x in matches]]
    out = []
    for (i, match) in enumerate(matches):
      (r, si, ei) = match
      df = tagged_df.loc[si:ei-1].copy()
      df["snippet_rule"] = r
      df["snippet_index"] = i
      sx = df["startx"].min()
      ex = df["endx"].max()
      df["snippet_startx"] = sx
      df["snippet_endx"] = ex
      df["snippet_text"] = text[sx:ex]
      out.append(df)

    return pd.concat(out) if len(out) > 0 else None

  def _snip_by_rules(self, text, tagged_df):
    """ Method performs regex matching over all snippet rules and builds a pandas DataFrame of the results."""
    dfs = [#- ADD checks for infered interpretation
           self._match_on_rule(text, tagged_df, key)
           for key in self.rules]
    return pd.concat(dfs).sort_values("endx",ascending=True).sort_values(["snippet_startx", "startx"]).reset_index(drop=True) if len(dfs) > 0 else None
