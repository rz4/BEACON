;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import os re
        [pandas :as pd]
        [beacon.bert [read-bert preprocess]]
        [beacon.parse.parser [BertParse]])

;-
(setv DEFAULT-LEXICON (.format "{}/{}" (os.path.dirname (os.path.abspath __file__)) "lexicon/")
      DEFAULT-LEXICON (lfor file (os.listdir DEFAULT-LEXICON) (.format "{}{}" DEFAULT-LEXICON file)))

;--
(defclass Beacon []

  ;--
  (defn __init__ [self targets &optional [context-len 3]
                                         [lexicon []]
                                         [from-pretrained "bert-base-uncased"]
                                         [layers [4 5 6]]]
    (setv self.targets targets
          self.context-len context-len
          self.lexicon (pd.concat (lfor l (+ DEFAULT-LEXICON lexicon) (pd.read-csv l :sep "|")))
          self.parser (BertParse :bert (read-bert :from-pretrained from-pretrained)
                                 :priors (self._compile_priors self.lexicon)
                                 :layers layers)
          self.target-regex (self._compile_target_regex self.targets self.lexicon)
          self.phrase-regex (self._compile_phrase_regex self.lexicon)))

  ;--
  (defn __call__ [self text &optional [return-G False]]
    (let [text (preprocess text)
          snippets (self._pull_snippet text)
          asts []]
      (for [snippet snippets]
        (let [terms (self._match_lexicon_phrases (first snippet))]
          (setv (, ast G) (self.parser (first snippet) :terms terms))
          (.append asts (if return-G
                          (, (last snippet) (first snippet) ast G)
                          (, (last snippet) (first snippet) ast)))))
      asts))

  ;--
  (defn _compile_target_regex [self targets lexicon]
    (let [
          df (get lexicon (.contains (. (.lower (. (get lexicon "lex") str)) str) (.format "^({})$" (.join "|" targets))))
          regex (.join "|" (get df "string"))
          regex (+
                   r"(\S+\s+){1,"
                   ;"([\.\?\!\n\,\-\]][^\.\?\!\n\,\-\]]+){1,"
                   (str (+ 1 self.context-len)) "}"
                   (.format "[:punct:]?({})[:punct:]?" regex)
                   "(\s+\S+){1,"
                   ;"([^\.\?\!\n\,\-\[]*[\.\?\!\n\,\-\[]+){1,"
                   (str self.context-len) "}")]
      (re.compile regex)))

  ;--
  (defn _compile_phrase_regex [self lexicon]
    (let [df (get lexicon (.contains (. (get lexicon "string") str) "\s"))
          regex (.join "|" (get df "string"))
          regex (.format "({})" regex)]
      (re.compile regex)))

  ;--
  (defn _compile_priors [self df]
    (let [priors []]
      (.append priors "%-- Lexicon Priors\n")
      (for [(, i row) (.iterrows df)]
        (setv (, lex token) row)
        (.append priors (.format "{}(\"{}\")." (.lower lex) token)))
      (.join "\n" priors)))

  ;--
  (defn _pull_snippet [self text]
    (let [x (lfor x (re.finditer self.target-regex text) (, (.group x 0) (.group x 2)))]
      x))

  ;--
  (defn _match_lexicon_phrases [self text]
    (let [groups (lfor x (re.finditer self.phrase-regex text) (, (.group x 0) (.span x)))
          matches []]
      (for [(, grp span) groups]
        (when (> (len (.split grp)) 1)
              (.append matches span)))
      matches)))
