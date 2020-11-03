#- Imports
import os, re, hy
import pandas as pd
from beacon.bert.model import read_bert
from beacon.bert.semantic_relations import build_relator

#- For Mapping Lexs to Unique Characters (LIMIT 58 unique Lexs)
BASE58 = '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
DEFAULTLEX_PATH = os.path.dirname(os.path.abspath(__file__))+"/lexicon"

#-
class Beacon(object):
  """ Beacon class defines a refactored implementation of legacy Clever which now
  uses just re andpandas packages to locate text snippets according to some predifined lexicon. This implemenation
  was design to serve as a single callable object which extracts snippets over a single text string at a time and returns a dataframe of tagged results.
  After initializing Clever with desired lexicon and header files, the object's call function is definded by the following monad:

  beacon(text::str) ->  df::pd.DataFrame

  With dataframe columns:
    ["index" "rels_index" "rels_lex","lex","startx","endx","text", "snippet_rule","snippet_regex","snippet_matchx","snippet_startx","snippet_endx","snippet_text"]

  For every text string, this Clever implementation outputs a dataframe with the above specification. If no matches are found,
  an empty dataframe that conforms to the above specifications will be returned instead. This will be useful for applying Clever functionally
  over a collection of texts. For example, an anonymous function can be built using beacon to process a corpus of notes:

  beacon_lambda = lambda x: beacon(x).drop("snippet_text").to_json()
  corpus["beacon_ouput"] = corpus["ReportTex†"].apply(beacon_lambda)

  In the above example, we build an anonymous function which takes an argument x passes it to beacon then drops the "snippet_text"
  (we only want the annotation data) and then converts the dataframe to a json to be stored in the corpus dataframe. This will
  let us naively parallelize Clever over texts in the future.

  Notes:

  Previous implemenations of Clever use n-grams along snippet tags to logically infer postive or negative conformations of target concepts
  by checking the presence of modifier tags in the n-grams. This results in very noise prone results as it does not check to see if the
  modier is being applied to the concept, or if a concept is tied to a patient and not another entity in the snippet.

  We explore the use of pretrained BERT language models to infer the relationships between lexicon terms in order to confirm the subject of
  a modifiers and concepts. This is done by:
    1) extracting the token correlation matricies (attention) from BERT at the last N layers of the network,
    2) mapping lexicon substrings to BERT input tokens,
    3) aggregating attention for lexicon terms,
    4) selecting high values token relations using a dynamically programmed gini coefficeint search,
    5) and mapping those term -> token relations back to term -> term relations
  From prelimary results, we find that BERT's attention output is very useful at measuring grammatical dependencies between lexicon terms.
  We tested snippets which contain both a patient and family entities, and the method returned the concept term relating to patient and not the family memember.

  TODO: Add the rules from legacy Clever to match on positive or negative confirmations and check for modifiers (negation, subject, etc)
  """

  def __init__(self, lexicon_path=DEFAULTLEX_PATH+"/lexicon_default.psv",
                     snippet_rules_path=DEFAULTLEX_PATH+"/snippet_rules_default.psv",
                     bert_model_path=None, bert_depth=2,
                     return_snippet_text=True):

    #- Bert Model For Knowledge Graph Annotating
    self.bert_relate = build_relator(read_bert(bert_model_path), depth=bert_depth)

    #- Load Lexicon
    self.lexicon = self._read_lexicon(lexicon_path)

    #- Load Snippet pattern rules
    self.rules = self._read_snippet_rules(snippet_rules_path)

    #- Compile search regexes at object intialization to improve performance
    self._lexicon_regexs = dict([(grpn, self._build_regex(x["string"])) for (grpn, x) in self.lexicon.groupby("lex")])
    self._lex_mapping = dict([(key, BASE58[i]) for (i, key) in enumerate(self._lexicon_regexs.keys())]) # Build lex to symbol mapping
    self._snippet_rules = dict([(key, self._build_rule(self.rules[key])) for key in self.rules])
    self.return_snippet_text = return_snippet_text

  def __repr__(self):
    repr = "beacon_object:\n\tlexicon_terms: {}".format(len(self.lexicon))
    repr += "\n\tunique_lexs: {}".format(len(self._lex_mapping.keys()))
    repr += "\n\tunqiue_snippet_rules: {}".format(len(self.rules.keys()))
    return repr

  def __call__(self, text):
    preprocessed_text = self._preprocess(text) # First preprocess text
    tagged_text = self._tag_by_lexicon(preprocessed_text) # Then tag lexs over text
    snippets = self._snip_by_rules(preprocessed_text, tagged_text) if tagged_text is not None else None # Finally match on snippet patterns
    kg_annotated = self.bert_relate(snippets) if snippets is not None else None # Annonate tag relations using BERT
    out = kg_annotated if kg_annotated is not None else pd.DataFrame([], columns=["index", "rels_index", "rels_lex",
                                                                                   "lex","startx","endx","text", "snippet_rule","snippet_regex",
                                                                                   "snippet_matchx","snippet_startx","snippet_endx","snippet_text"])
    out = out if self.return_snippet_text else out.drop("snippet_text")
    return out

  #- FUTURE: Standardize Lexicon and Headers file type. (i.e csv, parquet, etc).
  def _read_lexicon(self, path):
    df = pd.read_csv(path, sep="|")
    return df

  # FUTURE: Load snippet rules from file.
  #- FUTURE: Standardize file type. (i.e csv, parquet, etc).
  def _read_snippet_rules(self, path):
    df = pd.read_csv(path, sep="|", header=None)
    return dict([r for (i, r) in df.iterrows()])

  def _preprocess(self, text):
    out = text.lower() # 1. Lowercase
    out = " ".join(out.split())
    return out

  def _build_regex(self, series):
    """ Method is used to build a compiled regular expression which will be used to split and
    search text. The regex is built from a pandas Series of substrings for a subset of lexicon
    terms grouped by particular lex type.
    """
    series = [re.sub(r'([^0-9a-zA-Z\s])', r'\\\1', x) for x in list(series)] # Add forward slash to non-alphanumeric characters
    series = [x.replace(" ", "\s") for x in series] # Change spaces to arbitrary whitespace character.
    regex = r"|".join(["({})".format(s) if s.startswith("\\") else "({})".format(s) for s in series]) # Join terms and encapsulate each with grouping parenthesis to keep term on splits
    regex = re.compile(regex)
    return regex

  def _build_rule(self, rule):
    """ Method is used to build a compiled regular expression which will be used to match on snippet patterns.
    To allow definitions of patterns using the lex type names, a mapping of the names to individual characters is
    done using a BASE58 mapping. This limits the number of possible lexs to 58, but that is way above what is needed
    for Clever. The mapping also speeds up the pattern matching since the regex search is done over single characters.
    """
    regex = rule
    for key in self._lex_mapping: regex = regex.replace(key, self._lex_mapping[key]) # Substitute lex type names with encodings
    regex = re.compile(r""+regex)
    return regex

  def _match_on_regex(self, text, key):
    """ Methods searches for a given lex type and returns a Dataframe of the matched indexes and string."""
    out = re.finditer(self._lexicon_regexs[key], text)
    out = [(key, span[0], span[1], match) for (span, match) in [(x.span(), x.group(0)) for x in out]]
    df = pd.DataFrame(out, columns=["lex","startx","endx","text"])
    return df

  def _tag_by_lexicon(self, text):
    """ Method performs regex matching over all lex types and builds a pandas DataFrame of the results."""
    dfs = [self._match_on_regex(text, key) for key in self._lexicon_regexs]
    return pd.concat(dfs).sort_values("endx",ascending=True).sort_values("startx").reset_index(drop=True).reset_index() if len(dfs) > 0 else None

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
