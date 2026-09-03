# M4 Chaos Tool Policy Delivery Correction

The statement that the M4 chaos dataset is empty is incorrect.

## Delivered data

- `models/a100_training/chaos_policy/train.jsonl`: 2,160 samples
- `models/a100_training/chaos_policy/dev.jsonl`: 240 samples
- `models/a100_training/chaos_policy/all.jsonl`: 2,400 samples
- `models/a100_training/chaos_policy/dataset_info.json`: LLaMA-Factory registration

Each training record explicitly contains a failure mode, observation, allowed actions, forbidden actions, and an expected bounded policy. The target policy must not silently pass, invent tool results, override hard rules, or expose PII.

## Training configuration

Use `config/a100_reasoner_m4_chaos_lora.yaml`. It starts from the M1 adapter and writes a separate `m4_chaos_policy` output directory, preserving M1.

## Important correction to M1 interpretation

The initial mixed Reasoner data builder did not serialize chaos samples into usable message pairs. Therefore the M1 checkpoint is a grounded/complex SFT artifact, not evidence that Chaos Tool Policy training has been completed or evaluated. M4 must be trained using the independent dataset above.

All M4 data is synthetic and remains `do_not_eval=true`. It is not Human Gold or an external benchmark.

## GPU 7 training result

M4 training completed on physical NVIDIA A100-SXM4-80GB GPU 7 using the local Qwen3-8B base and M1 adapter. The run used 2,160 train records, 240 independent dev records, 136 optimization steps, BF16, and effective batch size 32.

- train loss: `2.250633351943072`
- best dev loss: `2.115617513656616` at step 100
- runtime: `236.0339` seconds
- adapter SHA-256: `4d0d32f5e69101faed50602309e315ae9d7916ec1815a9bc71d6d736b6226550`

This is a synthetic/silver candidate checkpoint, not a production or external benchmark result. The release manifest is stored beside the adapter.
