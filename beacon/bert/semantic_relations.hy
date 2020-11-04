;-
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;-
(import copy
        [numpy :as np]
        [pandas :as pd]
        [beacon.bert.io [bert-input]]
        [beacon.bert.model [BertRepr]])

;--
(defmu Attention [x atten depth bertrepr]
  (setv (, hiddens attens) (bertrepr x atten depth))
  attens)

;--
(defn gini-score [x]
  (let  [array (-> x .flatten (+ 1e-6) np.sort)
         index (np.arange 1 (+ 1 (get array.shape 0)))
         n (get array.shape 0)]
    (/ (np.sum (* array (- (* 2 index) n 1))) (* n (np.sum array)))))

;--
(defn gini-select [values &optional [intervals 100]]

  ;- Remove padding tokens and max/min normalize values
  (setv values (-> values .detach .numpy)
        values (get values (> values 0.0))
        values (/ (- values (.min values)) (- (.max values) (.min values))))

  ;- Calculate Ginis at all valid percentiles
  (setv df (.reset-index (pd.DataFrame (lfor v values [v]) :columns ["value"]))
        quantiles (list (.quantile (get df "value") (lfor i (range 0 intervals) (* (/ 1 intervals) i))))
        gini-global (-> df (get "value") .to-numpy gini-score)
        ginis (lfor (, i q) (enumerate quantiles)
                    (do (setv split (-> df (get (< (get df "value") q)) (get "value") .to-numpy))
                        [i q (len split) (gini-score split)])))

  ;- Dynamic Programming Method: Search for split along percentiles where gini coeffient falls under 0.5 or rebounds
  (let [minimum-value 1.0
        minimum-gini gini-global]
    (for [(, i q size gini) (reversed ginis)]
      (when (and (> size 0)
                 (>= gini 0.55)
                 (< gini minimum-gini))
            (setv minimum-value q
                  minimum-gini gini)))

    ;- Return Token Indexes
    (get df (> (get df "value") minimum-value) "index")))

;-- Annotate relations between lexicon terms and return a DataFrame
(defn bert-relate [beacon-tagged bert depth]

  ;- Build dataframe of relations along each snippet
  (let [text-snippets (-> beacon-tagged (get ["snippet_index" "snippet_text" "snippet_startx"]) .drop-duplicates)
        df []]
    (for [(, i row) (.iterrows text-snippets)]
      (setv (, snippet-idx snippet-text snippet-startx) row
            snippet (get beacon-tagged (= (get beacon-tagged "snippet_index") snippet-idx))
            (, tensor atten tokens offsets) (bert-input snippet-text)

      ;- Compile mappings between Bert Tokens and Clever lexicon terms
            idx-to-lex {}
            lex-to-idx {}
            lex-labels {})

      (for [(, _ lex) (.iterrows (get snippet ["index" "lex" "startx" "endx"]))]
        (setv (, index label startx endx) lex)
        (when (in label ["DOT" "PUNCT"]) (continue))
        (for [(, i r) (enumerate offsets)]
          (setv (, ix ie) r)
          (when (and (>= (+ snippet-startx ix) startx) (<= (+ snippet-startx ie) endx) (!= (+ ix ie) 0))
            (assoc idx-to-lex i index)
            (if (in index lex-to-idx) (setv (get lex-to-idx index) (+ (get lex-to-idx index) [i])) (assoc lex-to-idx index [i]))
            (assoc lex-labels index label))))

      ;- Gather Attentions for lexicon terms and select token relation candidates using gini coeffient
      (setv attens (bert tensor atten 2))
      (for [key lex-to-idx]
        (setv values (.sum (get attens 0 (get lex-to-idx key)) :dim 0)
              values (* values (get atten 0))
              selections (gini-select values)

              ;- Of selected tokens only append if token is part of lexicon term

              rels-index (set (flatten (lfor k selections (if (and (in k idx-to-lex) (!= (get idx-to-lex k) key)) (str (get idx-to-lex k)) []))))
              rels-lex (set (flatten (lfor k selections (if (and (in k idx-to-lex) (!= (get idx-to-lex k) key)) (get lex-labels (get idx-to-lex k)) [])))))
        (.append df [key (.join "|" (sorted rels-index)) (.join "|" (sorted rels-lex))])))

    ;- Merge Bert Relation Annoations to Clever Snippet Dataframe
    (->  beacon-tagged
         (.merge (pd.DataFrame df :columns ["index" "rels_index" "rels_lex"]) :on "index" :how "outer")
         (.fillna "")
         (.sort-values "index"))))

;-  Make an interaface for the bert relations
(defn build-relator [bert-model]
  (spec/assert :BertModel bert-model)
  (let [bert (Attention :bertrepr (BertRepr :bert (copy.deepcopy bert-model.bert)))]
    (fn [df depth] (bert-relate df bert depth))))
