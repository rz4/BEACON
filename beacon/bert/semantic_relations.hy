;-
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;-
(import copy torch
        [networkx :as nx]
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
(defn gini-select [values &optional [intervals 100] [threshold 0.5]]

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
                 (>= gini threshold)
                 (< gini minimum-gini))
            (setv minimum-value q
                  minimum-gini gini)))

    ;- Return Token Indexes
    (get df (> (get df "value") minimum-value) "index")))

;-- Annotate relations between lexicon terms and return a DataFrame
(defn bert-relate [beacon-tagged bert depth threshold-]

  ;- Build dataframe of relations along each snippet
  (let [text-snippets (-> beacon-tagged (get ["snippet_index" "snippet_text" "snippet_startx"]) .drop-duplicates)
        df []]
    (for [(, i row) (.iterrows text-snippets)]
      (setv (, snippet-idx snippet-text snippet-startx) row
            snippet (get beacon-tagged (= (get beacon-tagged "snippet_index") snippet-idx))
            (, tensor atten tokens offsets) (bert-input snippet-text)

            ;- Adaptive threshold based on length of tokens.
            ;- Lower thresholds work best for shorter sequences while higher thresholds reduce
            ;- False postive noise in longer sequences.
            threshold (if threshold- threshold- (+ 0.4 (cond [(< (len tokens) 17) 0.05]
                                                             [(< (len tokens) 25) 0.1]
                                                             [True 0.15])))

      ;- Compile mappings between Bert Tokens and Clever lexicon terms
            idx-to-lex {}
            lex-to-idx {}
            lex-labels {})

      (for [(, _ lex) (.iterrows (get snippet ["index" "lex" "startx" "endx"]))]
        (setv (, index label startx endx) lex)
        (when (in label ["DOT" "PUNCT" "HEADER"]) (continue))
        (for [(, i r) (enumerate offsets)]
          (setv (, ix ie) r)
          (when (and (>= (+ snippet-startx ix) startx) (<= (+ snippet-startx ie) endx) (!= (+ ix ie) 0))
            (if (in i idx-to-lex) (setv (get idx-to-lex i) (+ (get idx-to-lex i) [index])) (assoc idx-to-lex i [index]))
            (if (in index lex-to-idx) (setv (get lex-to-idx index) (+ (get lex-to-idx index) [i])) (assoc lex-to-idx index [i]))
            (assoc lex-labels index label))))

      ;- Gather Attentions for lexicon terms and select token relation candidates using gini coeffient
      (setv attens (bert tensor atten depth))
      (for [key lex-to-idx]
        (setv values (.sum (get attens 0 (get lex-to-idx key)) :dim 0)
              values (* values (get atten 0))
              selections (gini-select values :threshold threshold)

              ;- Of selected tokens only append if token is part of lexicon term
              rels-index (lfor k selections (if (and (in k idx-to-lex))
                                                (lfor l (get idx-to-lex k) (if (= l key) [] l))
                                                []))
              rels-lex (lfor k selections (if (and (in k idx-to-lex))
                                              (lfor l (get idx-to-lex k) (if (= l key) [] (get lex-labels l)))
                                              [])))
        (.append df [key
                     depth
                     threshold
                     (.join "|" (lfor x (sorted (set (flatten rels-index))) (str x)))
                     (.join "|" (sorted (set (flatten rels-lex))))])))


    ;- Merge Bert Relation Annoations to Clever Snippet Dataframe
    (->  beacon-tagged
         (.merge (pd.DataFrame df :columns ["index" "rels_depth" "rels_threshold" "rels_index" "rels_lex"]) :on "index" :how "outer")
         (.fillna "")
         (.sort-values "index"))))

;-  Make an interaface for the bert relations
(defn build-relator [bert-model]
  (spec/assert :BertModel bert-model)
  (let [bert (Attention :bertrepr (BertRepr :bert (copy.deepcopy bert-model.bert)))]
    (fn [df depth threshold] (bert-relate df bert depth threshold))))

;-
(defn compile-network [df]
  ;- Split relations for entitys
  (let [df (get df (!= (get df "rels_depth") ""))

        G (nx.DiGraph)]
    (setv (get df "node") (get df "index")
          (get df "edge") (.split (. (get df "rels_index") str) "|")
          df (.explode df "edge")
          df (get df ["node" "edge" "lex" "text"]))
          ;df (get df (~ (.isin (get df "edge") (lfor e (.unique (get df (= (get df "lex") "PT") "node")) (str e))))))

    ;- Gather nodes and edges
    (let [nodes (lfor (, i row) (.iterrows (get df ["node" "lex" "text"])) (, (get row 0) {"concept" (.format "{}:{}:{}" (get row 0) (get row 1) (get row 2))}))
          pairs (lfor (, i row) (.iterrows (get (get df (!= (get df "edge") "")) ["node" "edge"])) (if (, (get row 0) (int (get row 1)))))]
      (.add-nodes-from G nodes)
      (.add-edges-from G pairs))

    ;- Reduce Patient Nodes
    (let [pts (= (get df "lex") "PT")
          patient-nodes (get df (.isin (get df "text") (get df pts "text")) "node")
          patient-nodes (list (.unique (.append patient-nodes (get df pts "node"))))
          start (first patient-nodes)]
      (for [i (rest patient-nodes)] (setv G (nx.contracted_nodes G start i :self_loops False))))

    G))
