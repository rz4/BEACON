;- Macros
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;- Imports
(import torch copy transformers
        [tqdm [tqdm]]
        [numpy :as np]
        [pandas :as pd]
        [torch.nn :as nn]
        [torch.nn.functional :as F]
        [torch.utils.data [DataLoader]]
        [transformers [BertTokenizerFast BertForMaskedLM]])

;;------- DATA SPECS
(spec/def :Dataframe (fn [x] (instance? pd.DataFrame x))
          :HasTextCol (fn [x] (in "text" (. x columns)))
          :BertData (spec/and :Dataframe :HasTextCol)
          :BertModel (spec/modules bert (spec/modules embeddings (spec/modules)
                                                      encoder (spec/modules)
                                                      pooler (spec/modules)))
          :BertMLM (spec/and :BertModel (spec/modules cls (spec/modules))))

;-- Read a standard BERT architecture and model weights if provided.
(defn read-bert [&optional [file None] [from-pretrained "bert-base-uncased"]]
  (let [model (BertForMaskedLM.from_pretrained from-pretrained)]
    (when file
      (setv model2 (nn.DataParallel (BertForMaskedLM model.config)))
      (-> (torch.load file :map_location (torch.device "cpu"))
          (get "model_state_dict")
          (->> (.load-state-dict model2)))
      (setv model model2.module))
    (.eval model)))

;-- Module extracts BERTS internal representations (hidden layers and attention)
(defmu BertRepr [inputs attens layersi layersj bert]
  (setv (, _ __ hidden attention) (*-> [inputs attens]
                                       (bert :output_hidden_states True :output_attentions True))
        hiddens (torch.cat (cut hidden layersi layersj) -1)
        attentions (.sum (get attention layersi) 1))
  (for [a (cut attention (+ 1 layersi) layersj)] (+= attentions (.sum a 1))) ; combine attention layers using product
  [hiddens attentions])

;-- Return a language model inference model for single text strings
(defn build-LM [model]
  (spec/assert :BertModel model)
  (fn [text]
    (setv (, inputs1 attens1 tokens1) (bert-input text))
    (model_output (list (get (torch.argmax (get (model inputs1 attens1) 0) -1) (.bool attens1))))))

;-- Train standard BERT models
(defn train-bert-mlm
  [model data &optional [masked-percent 0.25] [batch-size 10] [epochs 100] [devices [(torch.device "cpu")]]]
  (spec/assert :BertMLM model)
  (spec/assert :BertData data)
  (let [dataset (mino/apply (.to-numpy (get data "text"))
                          (fn [x]
                            (setv (, inputs attens _) (model-input x)
                                  (, masked-inputs mask) (mask-input inputs (len _) masked-percent))
                            [(get inputs 0) (get attens 0) (get masked-inputs 0) (get mask 0)]))
        dataloader (DataLoader dataset :batch-size batch-size :shuffle True :num_workers 1 :pin_memory False)
        optimizer (transformers.AdamW (.parameters model) :lr 2e-5 :correct_bias False)
        scheduler (transformers.get_linear_schedule_with_warmup optimizer 1000 (* (len dataset) epochs))
        device (get devices 0)]

    (print "Training BertMLM...")
    (setv model (.to (.train model) device))
    (for [e (range epochs)]
      (for [(, i batch) (enumerate dataloader)]
        (setv (, labels attens inputs mask) (lfor b batch (.to b device))
              (, loss logits) (model :input_ids inputs :attention_mask attens :labels labels)
              vals (- (get labels mask) (get (torch.argmax logits -1) mask))
              vals (- 1 (torch.clamp (torch.abs vals) 0 1))
              acc (/ (.sum vals) (.sum (.float mask))))
        (.zero_grad model)
        (.zero_grad optimizer)
        (.backward loss)
        (torch.nn.utils.clip_grad_norm_ (.parameters model) 1)
        (.step optimizer)
        (.step scheduler)
        (print (.format "Epoch {}: Batch {} of {}: [Loss: {}; Acc: {}]"
                  e i (int (/ (len dataset) batch-size)) (-> loss .detach .cpu) (-> acc .detach .cpu)))))
    (.cpu (.eval model))))

;-- Evaluate standard BERT models
(defn eval-bert-mlm
  [model data &optional [masked-percent 0.25] [batch-size 10] [devices [(torch.device "cpu")]]]
  (spec/assert :BertMLM model)
  (spec/assert :BertData data)
  (let [dataset (mino/apply (.to-numpy (get data "text"))
                          (fn [x]
                            (setv (, inputs attens _) (model-input x)
                                  (, masked-inputs mask) (mask-input inputs (len _) masked-percent))
                            [(get inputs 0) (get attens 0) (get masked-inputs 0) (get mask 0)]))
        dataloader (DataLoader dataset :batch-size batch-size :shuffle True :num_workers 1 :pin_memory False)
        device (get devices 0)
        tp 0.0 N 0.0 losses []]

    (print "Evaluating BertMLM...")
    (setv model (.to (.eval model) device))
    (for [(, i batch) (enumerate (tqdm dataloader))]
      (setv (, labels attens inputs mask) (lfor b batch (.to b device))
            (, loss logits) (model :input_ids inputs :attention_mask attens :labels labels)
            vals (- (get labels mask) (get (torch.argmax logits -1) mask))
            vals (- 1 (torch.clamp (torch.abs vals) 0 1)))
      (+= tp (.cpu (.detach (.sum vals))))
      (+= N (.cpu (.detach (.sum mask))))
      (+= losses [(.cpu (.detach loss))]))
    (print (.format "[Loss: {}; Acc: {}]" (np.mean losses) (/ tp N)))
    (.cpu (.eval model))))
