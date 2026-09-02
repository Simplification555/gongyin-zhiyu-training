---
library_name: peft
license: other
base_model: /mnt/cpfs/epic-user/niutianle-20260612/models/Qwen3-8B-Base
tags:
- base_model:adapter:/mnt/cpfs/epic-user/niutianle-20260612/models/Qwen3-8B-Base
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: m3_trajectory_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# m3_trajectory_sft

This model is a fine-tuned version of [/mnt/cpfs/epic-user/niutianle-20260612/models/Qwen3-8B-Base](https://huggingface.co//mnt/cpfs/epic-user/niutianle-20260612/models/Qwen3-8B-Base) on the a100_reasoner_m3_trajectory dataset.
It achieves the following results on the evaluation set:
- Loss: 2.8789

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
- train_batch_size: 4
- eval_batch_size: 8
- seed: 20260831
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 4.55.4
- Pytorch 2.3.1+cu121
- Datasets 3.6.0
- Tokenizers 0.21.4