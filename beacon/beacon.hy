;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import os
        [pandas :as pd]
        [beacon.parse.bert [read-bert]]
        [beacon.parse.parser [BertParse]])

;-
(setv DEFAULT-LEXICON (.format "{}/{}" (os.path.dirname (os.path.abspath __file__)) "lexicon/lexicon.psv"))

;--
(defclass Beacon []

  ;--
  (defn __init__ [self &optional [from-pretrained "bert-base-uncased"]
                                 [lexicon DEFAULT-LEXICON]
                                 [layers [4 5 6]]]
    (setv self.lexicon (pd.read-csv lexicon :sep "|")
          self.parser (BertParse :bert (read-bert :from-pretrained from-pretrained)
                                 :priors (self._compile_priors self.lexicon)
                                 :layers layers)))

  ;--
  (defn __call__ [self text &optional [return-G False]]
    (setv (, ast G) (self.parser text))
    (if return-G (, ast G) ast))

  ;--
  (defn _compile_priors [self df]
    (let [priors []]
      (.append priors "%-- Lexicon Priors\n")
      (for [(, i row) (.iterrows df)]
        (setv (, lex token) row)
        (.append priors (.format "{}(\"{}\")." (.lower lex) token)))
      (.join "\n" priors))))
