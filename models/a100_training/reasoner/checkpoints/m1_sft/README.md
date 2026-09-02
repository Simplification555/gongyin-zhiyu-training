---
library_name: peft
license: other
base_model: Qwen/Qwen3-8B
tags:
- base_model:adapter:Qwen/Qwen3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: m1_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# m1_sft

This model is a LoRA adapter fine-tuned from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on the `a100_reasoner` dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6063

## Model description

Load this adapter with PEFT or Transformers together with the base model `Qwen/Qwen3-8B`.

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 8e-05
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

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6055        | 0.3476 | 250  | 0.6103          |
| 0.5391        | 0.6952 | 500  | 0.6063          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.55.4
- Pytorch 2.3.1+cu121
- Datasets 3.6.0
- Tokenizers 0.21.4
