;- Macros
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;- Imports
(import torch copy transformers json re
        [time [time]]
        [numpy :as np]
        [pandas :as pd]
        [torch.nn :as nn]
        [torch.nn.functional :as F]
        [torch.utils.data [DataLoader]]
        [transformers [BertTokenizerFast BertForPreTraining BertForMaskedLM BertForSequenceClassification pipeline]]
        [sklearn.metrics [accuracy_score precision_recall_fscore_support]]
        [nltk.tokenize.treebank [TreebankWordTokenizer]])

;-- BERT input/output interface
(setv tokenizer (BertTokenizerFast.from_pretrained "bert-base-uncased")) ;"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"))

;-- Read a standard BERT architecture and model weights if provided.
(defn read-bert [&optional [from-pretrained "bert-base-uncased"]] ;"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"]]
  (let [model (.from_pretrained BertForPreTraining from-pretrained)]
    (.eval model)))

;-- Read a standard BERT architecture and model weights if provided.
(defn read-bert-MLM [&optional [from-pretrained "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"]]
  (let [model (.from_pretrained BertForMaskedLM from-pretrained)]
    (.eval model)))

;-- Read a standard BERT architecture and model weights if provided.
(defn read-bert-classifier [&optional [from-pretrained "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"] [nb-classes None]]
  (let [model (if nb-classes
                (.from_pretrained BertForSequenceClassification from-pretrained :num_labels nb-classes)
                (.from_pretrained BertForSequenceClassification from-pretrained))]
    (.eval model)))

;--
(let [regex1 (re.compile r"[\r\n]+")
      regex2 (re.compile r"[^\x00-\x7F]+")
      ;regex3 (re.compile r"\s's\b")
      tokenizer (TreebankWordTokenizer)]
  (defn preprocess [text]
    (let [value (.lower text)
          value (.sub regex1 " " value)
          value (.sub regex2 " " value)
          sentence value]
          ;tokenized (.tokenize tokenizer value)
          ;sentence (.join " " tokenized)]
          ;sentence (.sub regex3 "'s" sentence)]
      sentence)))

;--
(defn bert-input [x &optional [nb-tokens None] [tokenizer tokenizer]]
  (let [tokenized-dict (tokenizer.encode-plus x
                                              :add-special-tokens True
                                              :max_length (if nb-tokens nb-tokens 512)
                                              :truncation True
                                              :padding (if nb-tokens "max_length" "longest")
                                              :return-attention-mask True
                                              :return-tensors "pt"
                                              :return_offsets_mapping True)
        tokens (tokenizer.convert_ids_to_tokens (geta (get tokenized-dict "input_ids") [0]))]
    (, (get tokenized-dict "input_ids") (get tokenized-dict "attention_mask") tokens (get tokenized-dict "offset_mapping"))))

;--
(defn bert-output [x &optional [tokenizer tokenizer]]
  (if (isinstance x torch.Tensor)
      (tokenizer.decode x :skip-special-tokens False :clean-up-tokenization-spaces True)
      (tokenizer.convert_tokens_to_string x)))

;-- BERT token masking for training
(defn mask-input [x &optional [nb-tokens None] [mask-percent 0.2] [indices None]]
  (let [size (.size x)
        masked-tensor (.clone x)
        boolean-mask (torch.zeros size)
        sep-mask (= masked-tensor tokenizer.sep_token_id)]
    (let [nb-tokens (if nb-tokens nb-tokens (get size 1))
          inds (if indices
                   indices
                   (np.random.choice nb-tokens (int (* nb-tokens mask-percent)) :replace False))]
      (setv (get masked-tensor (, 0 inds)) tokenizer.mask_token_id
            (get masked-tensor sep-mask) tokenizer.sep_token_id
            (get boolean-mask (, 0 inds)) 1)
      (, masked-tensor (.bool boolean-mask)))))

;--
(defn pipeline-MLM [model]
  (pipeline "fill-mask" :model model :tokenizer tokenizer))

;--
(defn compute_metrics [labels preds]
  (setv (, precision recall f1 _) (precision_recall_fscore_support labels preds :average "macro")
        acc (accuracy_score labels preds))
  {"accuracy" acc "f1" f1 "precision" precision "recall" recall})

;-- Train standard BERT models
(defn train-bert
  [model data &optional [masked-percent 0.2] [batch-size 16] [epochs (range 1 10)] [nb-tokens 128] [dataset-limit None] [device (torch.device "cpu")]]
  (let [dataset (mino/apply (.to-numpy (get data ["text" "label"]))
                  (fn [x]
                    (setv (, text y) x
                          (, inputs attens _ __) (bert-input text :nb-tokens nb-tokens)
                          (, masked-inputs mask) (mask-input inputs (len _) masked-percent))
                    [(get inputs 0) (get attens 0) (get masked-inputs 0) (get mask 0) y]))
        dataset-len (if dataset-limit (* batch-size dataset-limit) (len dataset))
        dataloader (DataLoader dataset :batch-size batch-size :shuffle True :num_workers 1 :pin_memory False)
        optimizer (transformers.AdamW (.parameters model) :lr 2e-5 :correct_bias False)
        scheduler (transformers.get_linear_schedule_with_warmup optimizer 1000 (* dataset-len (max epochs)))
        t (time)
        results []]

    (print "Training Bert...")
    (setv model (.to (.train model) device))
    (for [e epochs]
      (for [(, i batch) (enumerate dataloader)]
        (setv (, labels attens inputs mask y) (lfor b batch (.to b device))
              (, loss mlm-logits sp-logits) (model :input_ids inputs :attention_mask attens :labels labels :next-sentence-label y :return_dict False)
              labelss (get labels mask)
              predicts (get (torch.argmax (F.softmax mlm-logits -1) -1) mask)
              sp (torch.argmax (F.softmax sp-logits -1) -1)
              l (-> loss .detach .cpu .numpy float)
              mlm-metrics (compute_metrics (-> labelss .detach .cpu .numpy) (-> predicts .detach .cpu .numpy))
              sp-metrics (compute_metrics (-> y .detach .cpu .numpy) (-> sp .detach .cpu .numpy))
              tt (- (time) t)
              result {"epoch" e "batch" i "total_batches" (int (/ dataset-len batch-size))
                      "time_elasped" tt "average_time_per_batch" (/ tt (+ 1 i))
                      "estimated_time_left" (* (/ tt (+ 1 i)) (- (int (/ dataset-len batch-size)) i))
                      "loss" l "mlm-metrics" mlm-metrics "sp-metrics" sp-metrics})
        (.zero_grad model)
        (.zero_grad optimizer)
        (.backward loss)
        (torch.nn.utils.clip_grad_norm_ (.parameters model) 1)
        (.step optimizer)
        (.step scheduler)
        (.append results result)
        (print (json.dumps result :indent 4))
        (when (and dataset-limit (>= i dataset-limit)) (break))))

    (, (.cpu (.eval model)) results)))

;-- Evaluate standard BERT models
(defn eval-bert
  [model data &optional [masked-percent 0.2] [batch-size 16] [nb-tokens 128] [dataset-limit None] [device (torch.device "cpu")]]
  (let [dataset (mino/apply (.to-numpy (get data ["text" "label"]))
                  (fn [x]
                    (setv (, text y) x
                          (, inputs attens _ __) (bert-input text :nb-tokens nb-tokens)
                          (, masked-inputs mask) (mask-input inputs (len _) masked-percent))
                    [(get inputs 0) (get attens 0) (get masked-inputs 0) (get mask 0) y]))
        dataloader (DataLoader dataset :batch-size batch-size :shuffle True :num_workers 1 :pin_memory False)
        results None]

    (print "Evaluating Bert...")
    (setv model (.to (.eval model) device))
    (for [(, i batch) (enumerate dataloader)]
      (setv (, labels attens inputs mask y) (lfor b batch (.to b device))
            (, loss mlm-logits sp-logits) (model :input_ids inputs :attention_mask attens :labels labels :next-sentence-label y :return_dict False)
            labelss (get labels mask)
            predicts (get (torch.argmax (F.softmax mlm-logits -1) -1) mask)
            sp (torch.argmax (F.softmax sp-logits -1) -1)
            l (-> loss .detach .cpu .numpy float)
            mlm-metrics (compute_metrics (-> labelss .detach .cpu .numpy) (-> predicts .detach .cpu .numpy))
            sp-metrics (compute_metrics (-> y .detach .cpu .numpy) (-> sp .detach .cpu .numpy))
            result {"loss" [l] "mlm-metrics" (dfor (, key val) (.items mlm-metrics) [key [val]]) "sp-metrics" (dfor (, key val) (.items sp-metrics) [key [val]])})
      (if (none? results)
          (setv results result)
          (do (+= (get results "loss") (get result "loss"))
              (+= (get results "mlm-metrics" "accuracy") (get result "mlm-metrics" "accuracy"))
              (+= (get results "mlm-metrics" "precision") (get result "mlm-metrics" "precision"))
              (+= (get results "mlm-metrics" "recall") (get result "mlm-metrics" "recall"))
              (+= (get results "mlm-metrics" "f1") (get result "mlm-metrics" "f1"))
              (+= (get results "sp-metrics" "accuracy") (get result "sp-metrics" "accuracy"))
              (+= (get results "sp-metrics" "precision") (get result "sp-metrics" "precision"))
              (+= (get results "sp-metrics" "recall") (get result "sp-metrics" "recall"))
              (+= (get results "sp-metrics" "f1") (get result "sp-metrics" "f1"))))
      (when (and dataset-limit (>= i dataset-limit)) (break)))
    (setv results {"loss" (/ (sum (get results "loss")) (len (get results "loss")))
                   "mlm-metrics" (dfor (, key vals) (.items (get results "mlm-metrics")) [key (/ (sum vals) (len vals))])
                   "sp-metrics" (dfor (, key vals) (.items (get results "sp-metrics")) [key (/ (sum vals) (len vals))])})
    (print (json.dumps results :indent 4))
    (, (.cpu (.eval model)) results)))
