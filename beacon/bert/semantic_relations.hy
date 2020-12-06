;-
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;-
(import copy torch os
        [networkx :as nx]
        [numpy :as np]
        [pandas :as pd]
        [beacon.bert.bert_io [bert-input]]
        [beacon.bert.bert-model [BertRepr]])

;- STATICS
(setv NTAGS ["NSUBJ" "DOBJ" "POBJ" "ATTR" "CONJ" "NSUBJPASS" "CSUBJ" "COMPOUND"]
      PROLOG-TEMPLATE (with [f (open (+ (os.path.dirname (os.path.abspath __file__)) "/beacon_rules.swi") "r")] (f.read)))

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
        minimum-gini 1.0]
    (for [(, i q size gini) (reversed ginis)]
      (if (and (> size 0)
               (> gini gini-global))
          (setv minimum-value q
                minimum-gini gini)
          (if (and (>= gini threshold)
                   (<= gini minimum-gini))
              (setv minimum-value q
                    minimum-gini gini)
              (break))))

    ;- Return Token Indexes
    (get df (> (get df "value") minimum-value) "index")))

;-- Annotate relations between lexicon terms and return a DataFrame
(defn bert-relate [beacon-tagged bert layers threshold-]

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
            threshold (if threshold- threshold- (+ 0.4 (cond
                                                             [(< (len tokens) 50) 0.025]
                                                             [(< (len tokens) 80) 0.05]
                                                             [(< (len tokens) 100) 0.1]
                                                             [(< (len tokens) 150) 0.15]
                                                             [True 0.25])))

      ;- Compile mappings between Bert Tokens and Clever lexicon terms
            idx-to-lex {}
            lex-to-idx {}
            lex-labels {})

      (for [(, _ lex) (.iterrows (get snippet ["index" "lex" "startx" "endx"]))]
        (setv (, index label startx endx) lex)
        (when (or (in "HEADER" label)) (continue))
        (for [(, i r) (enumerate offsets)]
          (setv (, ix ie) r)
          (when (and (>= (+ snippet-startx ix) startx) (<= (+ snippet-startx ie) endx) (!= (+ ix ie) 0))
            (if (in i idx-to-lex) (setv (get idx-to-lex i) (+ (get idx-to-lex i) [index])) (assoc idx-to-lex i [index]))
            (if (in index lex-to-idx) (setv (get lex-to-idx index) (+ (get lex-to-idx index) [i])) (assoc lex-to-idx index [i]))
            (assoc lex-labels index label))))

      ;- Gather Attentions for lexicon terms and select token relation candidates using gini coeffient
      (setv attens (last (bert tensor atten (get layers 0) (get layers 1))))
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
                     (.join "|" (lfor x layers (str x)))
                     threshold
                     (.join "|" (lfor x (sorted (set (flatten rels-index))) (str x)))
                     (.join "|" (sorted (set (flatten (lfor x (flatten rels-lex) (.split x "|"))))))
                     (len tokens)])))

    ;- Merge Bert Relation Annoations to Clever Snippet Dataframe
    (->  beacon-tagged
         (.merge (pd.DataFrame df :columns ["index" "rels_layers" "rels_threshold" "rels_index" "rels_lex" "rels_tokens"]) :on "index" :how "outer")
         (.fillna "")
         (.sort-values "index"))))

;-  Make an interaface for the bert relations
(defn build-relator [bert-model]
  (spec/assert :BertModel bert-model)
  (let [bert (BertRepr :bert (copy.deepcopy bert-model.bert))]
    (fn [df layers threshold] (bert-relate df bert layers threshold))))

;- Build a directed graph from annotation dataframe
(defn compile-DG [df &optional [merge-on (fn [x] (in "PT" (get x "lex")))]]

  ;- Split relations for entitys
  (let [df (get df (!= (get df "rels_layers") "")) ; Remove unconnected nodes
        G (nx.DiGraph)]
    (setv (get df "node") (get df "index")
          (get df "edge") (.split (. (get df "rels_index") str) "|")
          df (.explode df "edge")
          df (get df ["node" "edge" "lex" "text" "startx" "endx"]))

    ;- Gather nodes and edges
    (let [nodes (lfor (, i row) (.iterrows (get df ["node" "lex" "text" "startx" "endx"]))
                      (, (get row 0) {"index" (get row 0)
                                      "lex" (get row 1)
                                      "text" (get row 2)
                                      "startx" (get row 3)
                                      "endx" (get row 4)}))
          pairs (lfor (, i row) (.iterrows (get (get df (!= (get df "edge") "")) ["node" "edge"]))
                      (, (get row 0) (int (get row 1))))]
      (.add-nodes-from G nodes)
      (.add-edges-from G pairs))

    ;- Reduce Nodes
    (let [mask (.apply (get df ["lex" "text"]) merge-on 1)
          merged (get df (.isin (get df "text") (get df mask "text")) "node")
          merged (list (.unique (.append merged (get df mask "node"))))
          start (first merged)]
      (for [i (rest merged)] (setv G (nx.contracted_nodes G start i :self_loops False)))
      (let [cores (nx.core_number G)
            ranks (nx.pagerank G)]
        (nx.set_node_attributes G cores "core")
        (nx.set_node_attributes G ranks "pagerank"))

      (, G (int start)))))

;-
(defn has-shortest-path-of-length [G start end length]
  (and (in start G)
       (in end G)
       (nx.has-path G start end)
       (< (len (nx.shortest-path G start end)) length)))

;-
(defn compile-prolog [tags graph pt]
  ;-
  (let [objs (-> tags (get (.contains (. (get tags "lex") str) (.join "|" NTAGS)) "index")  .to-list)
        roots (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "ROOT"))) "index") .to-list)
        concepts (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "UMLS"))) "index") .to-list)
        fams (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "FAM"))) "index") .to-list)
        caretakers (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "CG"))) "index") .to-list)
        hxs (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "HX"))) "index") .to-list)
        rxs (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "RISK"))) "index") .to-list)
        negexs (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "NEGEX"))) "index") .to-list)
        dots (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "DOT"))) "index") .to-list)
        puncts (-> tags (get (.contains (. (get tags "lex") str) (.join "|" (, "PUNCT"))) "index") .to-list)
        objs (flatten (lfor x objs (if (in x (+ [pt] fams caretakers)) [] x)))
        impassables (+ [pt] objs dots puncts roots)
        graph1 (flatten (lfor x (.nodes graph) (if (in x objs) [] x)))
        graph1 (.subgraph graph graph1)
        graph2 (flatten (lfor x (.nodes graph) (if (in x impassables) [] x)))
        graph2 (.subgraph graph graph2)
        script ""
        jumps 5
        nodes []]

    ;-
    (for [node (+ [pt] concepts fams hxs negexs)])
    (+= script (.format "patient({}).\n" pt))

    ;-
    (for [concept concepts]
      (if (has-shortest-path-of-length graph1 pt concept jumps)
          (do (setv path (nx.shortest-path graph1 pt concept)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "dx({}, {}, [{}]).\n" pt concept path))
              (+= nodes (+ path- [concept pt])))
          (when (has-shortest-path-of-length graph1 concept pt jumps)
                (setv path (nx.shortest-path graph1 pt concept)
                      path- path path (.join ", " (lfor p path (str p))))
                (+= script (.format "dx({}, {}, [{}]).\n" pt concept path))
                (+= nodes (+ path- [concept pt]))))
      (for [fam fams]
        (when (has-shortest-path-of-length graph2 fam concept jumps)
              (setv path (nx.shortest-path graph2 fam concept)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "dx({}, {}, [{}]).\n" fam concept path))
              (+= nodes (+ path- [concept fam]))))
      (for [hx hxs]
        (when (has-shortest-path-of-length graph2 hx concept jumps)
              (setv path (nx.shortest-path graph2 hx concept)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "hx({}, {}, [{}]).\n" hx concept path))
              (+= nodes (+ path- [concept hx])))
        (for [negex negexs]
          (when (has-shortest-path-of-length graph2 negex hx jumps)
                (setv path (nx.shortest-path graph2 negex hx)
                      path- path path (.join ", " (lfor p path (str p))))
                (+= script (.format "negex({}, {}, [{}]).\n" negex hx path))
                (+= nodes (+ path- [negex hx])))))
      (for [rx rxs]
        (when (has-shortest-path-of-length graph2 rx concept jumps)
              (setv path (nx.shortest-path graph2 rx concept)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "risk({}, {}, [{}]).\n" rx concept path))
              (+= nodes (+ path- [concept rx])))
        (for [negex negexs]
          (when (has-shortest-path-of-length graph2 negex rx jumps)
                (setv path (nx.shortest-path graph2 negex rx)
                      path- path path (.join ", " (lfor p path (str p))))
                (+= script (.format "negex({}, {}, [{}]).\n" negex rx path))
                (+= nodes (+ path- [negex rx])))))
      (for [negex negexs]
        (when (has-shortest-path-of-length graph2 negex concept jumps)
              (setv path (nx.shortest-path graph2 negex concept)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "negex({}, {}, [{}]).\n" negex concept))
              (+= nodes (+ path- [negex concept])))))
    ;-
    (for [fam fams]
      (when (has-shortest-path-of-length graph1 pt fam jumps)
            (setv path (nx.shortest-path graph1 pt fam)
                  path- path path (.join ", " (lfor p path (str p))))
            (+= script (.format "family({}, {}, [{}]).\n" rx concept path))
            (+= nodes (+ path- [pt fam])))
      (when (has-shortest-path-of-length graph1 fam pt jumps)
            (setv path (nx.shortest-path graph1 fam pt)
                  path- path path (.join ", " (lfor p path (str p))))
            (+= script (.format "family({}, {}, [{}]).\n" fam pt path))
            (+= nodes (+ path- [pt fam])))
      (for [negex negexs]
        (when (has-shortest-path-of-length graph2 negex fam jumps)
              (setv path (nx.shortest-path graph2 negex fam)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "negex({}, {}, [{}]).\n" negex fam path))
              (+= nodes (+ path- [negex fam])))))

    ;-
    (for [caretaker caretakers]
      (when (has-shortest-path-of-length graph1 pt caretaker jumps)
            (setv path (nx.shortest-path graph1 pt caretaker)
                  path- path path (.join ", " (lfor p path (str p))))
            (+= script (.format "caregiver({}, {}, [{}]).\n" pt caretaker path))
            (+= nodes (+ path- [pt caretaker])))
      (when (has-shortest-path-of-length graph1 caretaker pt jumps)
            (setv path (nx.shortest-path graph1 caretaker pt)
                  path- path path (.join ", " (lfor p path (str p))))
            (+= script (.format "caregiver({}, {}, [{}]).\n" caretaker pt path))
            (+= nodes (+ path- [pt caretaker])))
      (for [negex negexs]
        (when (has-shortest-path-of-length graph2 negex caretaker jumps)
              (setv path (nx.shortest-path graph2 negex caretaker)
                    path- path path (.join ", " (lfor p path (str p))))
              (+= script (.format "negex({}, {}, [{}]).\n" negex caretaker path))
              (+= nodes (+ path- [negex caretaker])))))

    (for [node (list (set nodes))]
      (let [n (get graph.nodes node)]
        (setv script (+ (.format "lex({}, \"{}\", \"{}\", {}, {}).\n" node (get n "lex") (get n "text") (get n "startx") (get n "endx"))
                        script))))

    (.format PROLOG-TEMPLATE script)))
