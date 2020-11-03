;-
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;-
(import torch copy
        [numpy :as np]
        [torch.nn :as nn]
        [torch.nn.functional :as F])

;--
(defmu PairwiseCosineDistance [x1 x2]
  (setv (, x1 x2) (|-> [x1 x2] (.unsqueeze 0))
        (, pdist cdist) [(*-> [x1 x2] F.pairwise_distance (/ 500) (** 2) (/ -2) torch.exp)
                         (*-> [x1 x2] F.cosine_similarity)])
  (* 2 (/ (* pdist cdist) (+ pdist cdist))))

;--
(defmu SemanticSimilarity [inputs1 attens1 inputs2 attens2 bertrepr similarity]
  """ Module takes in token and attention tensors for two sentences, a bert encoder
  module and a similarity measure. The token and attention tensors are passed to
  the bert encoder in parallel for inference; then the mean of the output representations
  is found for each sentence. Similarity is then measured between the two aggreagates.
  """
  (setv (, brepr1 brepr2) (|-> [[inputs1 attens1] [inputs2 attens2]]
                               (*-> bertrepr (get 0)))
    attention (get (bertrepr (torch.cat [inputs1 inputs2] 1) (torch.cat [attens1 attens2] 1)) 1)
    att1 (F.softmax (/ (geta attention [:] [(get (.size attens1) 1) :] [: (get (.size attens1) 1)])
                       (np.sqrt (get (.size attens1) 1))) -1)
    brepr1 (@ att1 brepr1))
  (*-> (|-> [brepr1 brepr2]
            [(get (.bool attens2)) (get (.bool attens2))] ; Remove padded values
            (.sum 0))
       similarity))

;--
(defn build-SEMSIM [model]
  """Function takes Bert model as input and returns lambda function which measures the
  similarity between two strings.

  'Network surgery' is performed on model by removing the subgraph formed between the input layer
  and the intermediate layer with the required data representation (output of the Pooling layer).
  Extracting the subgraph lets us do two things: 1) Remove unecessary computation instead of
  applying a forward hook to an intermediate opertation, and 2) build a new graph which performs
  further computation using the extracted subgraph as a component.

  Args:
      torch.Module : a model with :BertModel specification

  Returns:
      fn(str, str) -> float
  """
  (spec/assert :BertModel model)
  (let [measure (SemanticSimilarity :bertrepr (BertRepr :bert (copy.deepcopy model.bert))
                             :similarity (PairwiseCosineDistance))]
    (fn [str1 str2]
      (setv (, inputs1 attens1 tokens1) (bert-input str1)
            (, inputs2 attens2 tokens2) (bert-input str2))
      (.item (measure inputs1 attens1 inputs2 attens2)))))
