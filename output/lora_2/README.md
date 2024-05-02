---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /root/autodl-tmp/model/WizardCoder-15B-V1
model-index:
- name: lora_2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lora_2

This model is a fine-tuned version of [/root/autodl-tmp/model/WizardCoder-15B-V1](https://huggingface.co//root/autodl-tmp/model/WizardCoder-15B-V1) on the customs_data dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3414

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
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 2
- total_train_batch_size: 4
- total_eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.3.0+cu121
- Datasets 2.18.0
- Tokenizers 0.19.1