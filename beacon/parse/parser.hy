;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import torch transformers
        [time [time]]
        [numpy :as np]
        [pandas :as pd]
        [networkx :as nx]
        [networkx.algorithms.community.centrality [girvan_newman]]
        [networkx.algorithms.centrality [edge_betweenness_centrality :as betweenness]]
        [torch.nn.functional :as F])
(transformers.logging.set_verbosity transformers.logging.ERROR)

;- Local Imports
(import [beacon.bert [bert-input tokenizer]]
        [beacon.parse.ast [BertAST]])

;--
(defclass BertParse []

  ;--
  (defn __init__ [self bert &optional [priors None] [device (torch.device "cpu")] [layers [4 5 6]]]
    (setv self.bert (.to bert device)
          self.device device
          self.layers layers ; Extract These Layers From Bert; Standard Bert has 12 layers.
          self.priors priors))

  ;--
  (defn __call__ [self text &optional [terms []]]

    ;- Main Parsing Routine
    (setv ;- Tokenize, Infer on Input and Extract Attention
          (, attention-states token-index collapse-on) (self._inference text :layers self.layers :terms terms)

          ;- Combine Attention Heads and Format For Gini Select
          attention (self._aggregate_attention attention-states)

          ;- Build DataFrame of HIgh Valued Token Relations
          token-relations (self._relate_tokens attention token-index)

          ;- Build Directed Graph Using Tokens as Nodes and Relations as Edges
          (, G token-index) (self._build_DG token-relations token-index collapse-on)

          ;- Compile Graph Into a Abstract Syntax Tree Using Divisive Graph Partitioning
          ast (self._compile_ast G token-index))
    (, ast G))

  ;--
  (defn _inference [self text &optional [layers (range 12)] [terms []]]

    ;- Tokenize Inputs
    (let [inputs (bert-input text)]
      (setv (, inputs attens tokens inds) inputs

            ;- Build DataFrame of Token Indexes and Collapsable Sub Tokens
            (, token-index collapse-on) (self._build_token_index inputs tokens inds text terms)

            ;- Run Inference on Input using Bert
            (, inputs attens) (lfor i (, inputs attens) (.to i self.device))
            output-dict (self.bert :input_ids inputs
                                   :attention_mask attens
                                   :output-hidden-states True
                                   :output-attentions True
                                   :return-dict True)
            token-logits (get output-dict "prediction_logits")
            hidden-states (torch.cat (lfor i layers (get output-dict "hidden_states" i)))
            attention-states (torch.cat (lfor i layers (get output-dict "attentions" i))))

      (, attention-states token-index collapse-on)))

  ;--
  (defn _build_token_index [self inputs tokens inds text terms]

    ;- Build DataFrame of index, Token, Token_id, and String Start and End Indexes
    (let [inds (-> inds .int .detach .numpy first)
          inputs (-> inputs .int .detach .numpy first)
          df (lfor i (range (len tokens)) (+ [i (get tokens i) (get inputs i)] (list (get inds i))))
          df (pd.DataFrame df :columns ["index" "token" "token_id" "startx" "endx"])

          ;- Remove Special Tokens
          df (.reset-index (get df (~ (.contains (. (get df "token") str) r"\[SEP\]|\[CLS\]"))) :drop True)

          ;- Using Start and End indexes, collect groups of dependent subtokens
          ; These will be contracted when building the Directed Graph
          intersects (fn [startx endx term] (and (>= startx (first term)) (<= endx (last term))))
          collapse-on [] accum1 [] accum2 [] n None i None]
      (for [(, _ row) (.iterrows (get df ["index" "token" "startx" "endx"]))]
          (setv (, index token startx endx) row)
          (if (none? n)
              (setv n index i endx accum1 [n] accum2 [token])

              ;- Contract on valid subtoken; Disregard punctuation.
              (if (and (= i startx)
                       (not (in token ["." "," ":" "!" "?"]))
                       (> 1 (sum (lfor term terms (intersects startx endx term)))))
                  (do (.append accum1 index)
                      (.append accum2 token)
                      (setv i endx))
                  (do (when (> (len accum1) 1)
                        (.append collapse-on (, (tokenizer.convert_tokens_to_string accum2) accum1)))
                      (setv n index i endx accum1 [n] accum2 [token])))))
      ;--
      (for [term terms]
        (setv intersects (fn [x] (and (>= (get x "startx") (first term)) (<= (get x "endx") (last term))))
              term-tokens (get df (.apply (get df ["startx" "endx"]) intersects :axis 1))
              accum1 (.to-list (get term-tokens "index"))
              accum2 (tokenizer.convert_tokens_to_string (.to-list (get term-tokens "token"))))
        (.append collapse-on (, accum2 accum1)))
      (, df collapse-on)))

  ;--
  (defn _aggregate_attention [self attention-states]
    ;- Combine Heads and Layers by Sum
    (-> attention-states (.sum :dim 1) (.sum :dim 0)))

  ;--
  (defn _relate_tokens [self pmat token-index]
    ;- Select HIgh Value Token Relations Using A Dynamic Programming Search.
    ; A percentile cutoff is found by measuring gini coeffient across percentile splits of the data.
    ; When the gini coeffient of data under the percentile cutoff falls under the selected threshold
    ; (0.6 of global gini), return dataframe of relations which are above the percentile cutoff.
    (let [selected (self._gini_select (.flatten pmat))]

      ;- Remap indexes to a NxN indices
      (setv (get selected "hit_index") (// (get selected "index") (len pmat))
            (get selected "index") (% (get selected "index") (len pmat))

            ;- Remove edges that are not indexed in token dataframe
            selected (.dropna (.merge selected token-index :on "index" :how "left"))
            selected (.dropna (.merge selected (.rename token-index :columns {"index" "hit_index"}) :on "hit_index" :how "left"))
            token-relations (get selected ["index" "hit_index" "value"]))
      token-relations))

  ;--
  (defn _gini_score [self vector]
    ;- Calculate Gini-Coefficient Over a Vector of Values
    (let [array (-> vector .flatten (+ 1e-6) np.sort)
          index (np.arange 1 (+ 1 (get array.shape 0)))
          n (get array.shape 0)]
      (/ (np.sum (* array (- (* 2 index) n 1)))
         (+ 1e-6 (* n (np.sum array))))))

  ;--
  (defn _gini_select [self tensor &optional [intervals 10] [threshold 0.6]]
    ;- Remove padding tokens and max/min normalize values
    (setv values (-> tensor .cpu .detach .numpy)
          values (/ (- values (.min values)) (- (.max values) (.min values))))

    ;- Reduce threshold on short sequences
    (when (< (len values) 1001) (setv threshold 0.5))
    (when (< (len values) 1001) (setv threshold 0.5))
    (when (< (len values) 901) (setv threshold 0.4))
    (when (< (len values) 100) (setv threshold 0.1))


    ;- Calculate Ginis at all valid percentiles
    (setv df (-> (pd.DataFrame (lfor (, i v) (enumerate values) [i v]) :columns ["index" "value"]))
          quantiles (list (.quantile (get df "value") (lfor i (range 0 intervals) (* (/ 1 intervals) i))))
          gini-global (-> df (get "value") .to-numpy self._gini-score)
          ginis (lfor (, i q) (enumerate quantiles)
                  (do (setv split (-> df (get (< (get df "value") q)) (get "value") .to-numpy))
                      [i q (len split) (self._gini-score split)])))

    ;- Dynamic Programming Method: Search for split along percentiles where gini coeffient falls under 0.5 or rebounds
    (let [minimum-value 1.0
          minimum-gini 1.0]
      (for [(, i q size gini) (reversed ginis)]
        (if (and (> size 0)
                 (> gini gini-global))
            (setv minimum-value q
                  minimum-gini gini)
            (if (and (>= gini (* gini-global threshold))
                     (<= gini minimum-gini))
                (setv minimum-value q
                      minimum-gini gini)
                (break))))

      ;- Return Token Indexes Sorted By Value
      (setv df (get df (> (get df "value") minimum-value))
            df (.sort-values df "value" :ascending False))
      df))

  ;--
  (defn _build_DG [self token-relations token-index collapse-on]

    ;- Create A New Directed Graph
    (let [G (nx.DiGraph)]

      ;- Gather nodes and edges
      (let [nodes (lfor (, i row) (.iterrows (get token-index ["index" "token" "token_id" "startx" "endx"]))
                        (, (int (get row 0)) {"index" (get row 0)
                                              "token" (get row 1)
                                              "token_id" (get row 2)
                                              "startx" (get row 3)
                                              "endx" (get row 4)}))
            pairs (lfor (, i row) (.iterrows (get token-relations ["index" "hit_index" "value"]))
                        (, (int (get row 0)) (int (get row 1)) {"weight" (get row 2)}))]

      ;- Add edges and nodes to graph; Remove self loops
        (.add-nodes-from G nodes)
        (.add-edges-from G pairs))
      (.remove_edges_from G (nx.selfloop_edges G))

      ;-- Collapse Nodes on Sub Token Groups
      (for [c collapse-on]
        (setv (, token c) c)

        (let [c1 (first c)]
          (for [cn (rest c)]
            (setv G (nx.contracted-nodes G c1 cn :self-loops False)))
          (nx.set_node_attributes G {c1 {"token" token}})))

      ;- Return Updated Token Index
      (let [df (lfor node (do (setv accum [])
                              (for [(, key node) (-> G .nodes .data)]
                                (when (in "token" node) (.append accum node)))
                              accum)
                 {"index" (get node "index")
                  "token" (get node "token")
                  "token_id" (get node "token_id")
                  "startx" (get node "startx")
                  "endx" (get node "endx")})
            df (pd.DataFrame df)]
        (, G df))))

  ;--
  (defn _compile_ast [self G token-index]

    ;-
    (let [scorer (fn [G] (let [centrality (betweenness G :weight "weight")] (max centrality :key centrality.get)))
          hierarchy (girvan_newman G :most_valuable_edge scorer)
          ast (BertAST token-index)]
      (for [(, i level)  (enumerate hierarchy)]
        (for [leaf level]
          (let [leaf (lfor n (sorted leaf) (str n))]
            (._expand ast leaf))))
      (._compile ast :priors self.priors)
      ast)))
