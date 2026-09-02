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
