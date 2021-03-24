;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import os
        [collections [OrderedDict]]
        [pyswip [Prolog]]
        [networkx :as nx]
        [pandas :as pd]
        [hy.contrib.pprint [pformat]])

;-
(setv FILEPATH (os.path.dirname (os.path.abspath __file__)))

;--
(defn export-prolog [facts]
  (with [f (open (+ FILEPATH "/rules.swi") "r")]
    (setv script (f.read)))
  (.format script facts))

;--
(let [PATH (+ FILEPATH "/temp.swi")
      PROLOG (Prolog)]
  (defn eval-prolog [queries script]

    (with [f (open PATH "w")] (f.write script)
      (.consult PROLOG PATH)
      (os.remove PATH)

      (let [out {}]
        (for [key queries]
          (setv query (get queries key)
                output (lfor x (.query PROLOG query) x)
                encode (fn [x] (if (isinstance x bytes) (.decode x) (if (isinstance x int) x)))
                output (lfor y (set (lfor x output (tuple (lfor (, key val) (.items x) (, key (encode val)))))) (dict y)))
          (assoc out key output))

        out))))

;--
(defclass BertAST []

  ;--
  (defn __init__ [self token-index]
    (setv self.token-index token-index
          self.tokens (self._format_tokens token-index)
          self.tree {}
          self.facts None))

  ;--
  (defn __repr__ [self]
    (pformat (self.to-list)))

  ;--
  (defn _format_tokens [self df]
    (dfor (, i row) (.iterrows (get df ["index" "token"]))
      [(get row "index") (get row "token")]))

  ;--
  (defn _intersects [self key1 key2]
    (let [key1 key1
          key2 (.split key2 ",")]
      (> (sum (lfor i key1 (in i key2))) 0)))

  ;--
  (defn _expand [self new-leaf &optional [root None]]
    (let [branch (if (none? root) self.tree root)
          keys (.keys branch)]
      (for [key keys]
        (when (& (self._intersects new-leaf key) (not (in (.join "," new-leaf) keys)))
              (self._expand new-leaf :root (get branch key))
              (return)))
      (unless (in (.join "," new-leaf) keys)
        (assoc branch (.join "," new-leaf) {}))))

  ;--
  (defn _to-prolog [self &optional [root None] [agg None]]
    (let [branch (if (none? root) self.tree root)
          kid (.format "\"{}\"" (str (gensym)))]
      (.append agg
        (.format "tree({}, {})."
          kid
          (.join ", "
            (lfor key branch
              (if (empty? (get branch key))
                  (str (int key))
                  (self._to-prolog (get branch key) :agg agg))))))
      (if (none? root)
          (.join "\n" agg)
          kid)))

  ;--
  (defn _compile [self &optional [priors None]]
    ;-
    (let [script []]
      (unless (none? priors) (.append script priors))
      (.append script "%-- BERT-Parse Derived Facts:")

      ;-
      (let [token-facts []]
        (.append token-facts "% token(index, token, startx, endx).\n")
        (for [(, i row) (.iterrows self.token-index)]
          (setv (, index token _ startx endx) row)
          (.append token-facts (.format "token({}, \"{}\", {}, {})." index (.replace token "\"" "\\\"") startx endx)))
        (.append script (.join "\n" token-facts)))

      ;-
      (let [branch-facts []]
        (.append branch-facts "% tree(index, node1, node2).")
        (.append branch-facts (.format "{}" (self._to-prolog :agg [])))
        (.append script branch-facts))

      (setv self.facts (.join "\n\n" (flatten script)))))

  ;--
  (defn to-list [self &optional [root None]]
    (let [branch (if (none? root) self.tree root)]
      (lfor key branch
        (if (empty? (get branch key))
            (get self.tokens (int key))
            (self.to-list (get branch key))))))

  ;--
  (defn query [self queries &optional [priors []]]
    (let [kb (export-prolog (.join "\n\n" (flatten [self.facts priors])))
          answers (eval_prolog queries kb)]
      answers)))
