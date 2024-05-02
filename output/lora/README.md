---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /public10_data/wtl/model/WizardCoder-15B-V1.0/
model-index:
- name: customs_4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# customs_4

This model is a fine-tuned version of [/public10_data/wtl/model/WizardCoder-15B-V1.0/](https://huggingface.co//public10_data/wtl/model/WizardCoder-15B-V1.0/) on the customs_data dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0082

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 10.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0475        | 1.39  | 100  | 0.0514          |
| 0.0247        | 2.78  | 200  | 0.0082          |
| 0.0087        | 4.17  | 300  | 0.0106          |
| 0.0012        | 5.56  | 400  | 0.0393          |
| 0.0011        | 6.94  | 500  | 0.0213          |
| 0.0012        | 8.33  | 600  | 0.0168          |
| 0.0014        | 9.72  | 700  | 0.0162          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.0.1+cu117
- Datasets 2.18.0
- Tokenizers 0.15.2