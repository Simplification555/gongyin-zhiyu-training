# Gongyin Zhiyu A100 Training Pack

Training scripts and prepared datasets for the Gongyin Zhiyu fiduciary-governance prototype.

## Scope

This repository contains the first single-A100 training run for:

- Guard-Lite multi-label compliance classification
- Regulation evidence reranking
- ProductNLI (`SUPPORTED`, `CONTRADICTED`, `UNKNOWN`)
- Reasoner-8B LoRA SFT data assembly

All included records are synthetic, silver, or programmatically assembled training assets. They are marked `do_not_eval=true` where applicable. They are not human Gold, not customer production data, and not evidence of model or commercial performance.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
pip install datasets accelerate scikit-learn sentencepiece llamafactory
.\scripts\run_a100_training.ps1 prepare
```

See [docs/A100_TRAINING_RUNBOOK.md](docs/A100_TRAINING_RUNBOOK.md) for the recommended order and release boundaries.

Reasoner configs use `QWEN3_8B_PATH` for the local Qwen3-8B base model. Before loading the published LoRA adapter, run `git lfs install` and `git lfs pull`. Use `scripts/run_reasoner_gpu6.sh` or the PowerShell wrapper; both hard-pin training to physical GPU 6 (`CUDA_VISIBLE_DEVICES=6`) and refuse other device values.

## Evaluation boundary

GYZ-Bench, Hidden, Regression, private answer keys, human Gold candidates, API keys, databases, audit logs, and model weights are intentionally excluded. Do not report training or silver validation metrics as external benchmark or production results.
