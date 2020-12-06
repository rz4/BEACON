;- Macros
(require [mino.mu [*]]
         [mino.thread [*]]
         [mino.spec [*]]
         [hy.contrib.walk [let]])

;- Imports
(import torch copy
        [numpy :as np]
        [transformers [BertTokenizerFast]])

;-- BERT input/output interface
(setv tokenizer (BertTokenizerFast.from_pretrained "bert-base-uncased"))

(defn bert-input [x &optional [nb-tokens None] [tokenizer tokenizer]]
  """
  """
  (let [tokenized-dict (tokenizer.encode-plus x
                                              :add-special-tokens True
                                              :max_length (if nb-tokens nb-tokens 512)
                                              :truncation True
                                              :pad-to-max-length (if nb-tokens True False)
                                              :return-attention-mask True
                                              :return-tensors "pt"
                                              :return_offsets_mapping True)
        tokens (tokenizer.convert_ids_to_tokens (geta (get tokenized-dict "input_ids") [0]))]
    (, (get tokenized-dict "input_ids") (get tokenized-dict "attention_mask") tokens (get tokenized-dict "offset_mapping"))))

(defn bert-output [x &optional [tokenizer tokenizer]]
  """
  """
  (tokenizer.decode x :skip-special-tokens False :clean-up-tokenization-spaces True))

;-- BERT token masking for training
(defn mask-input [x &optional [nb-tokens None] [mask-percent 0.25] [indices None]]
  """
  """
  (let [size (.size x)
        masked-tensor (.clone x)
        boolean-mask (torch.zeros size)]
    (let [nb-tokens (if nb-tokens nb-tokens (get size 1))
          inds (if indices
                   indices
                   (np.random.choice nb-tokens (int (* nb-tokens mask-percent)) :replace False))]
      (setv (get masked-tensor (, 0 inds)) 103
            (get boolean-mask (, 0 inds)) 1)
      (, masked-tensor (.bool boolean-mask)))))
