param([ValidateSet('prepare','guard','reranker','nli','reasoner','all')][string]$Task = 'prepare')
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root
$env:PYTHONPATH = "$root\src"
python scripts/prepare_a100_training.py
if ($Task -eq 'prepare') { exit 0 }
if ($Task -in @('guard','all')) { python scripts/train_multilabel_transformer_compliance.py --data models/a100_training/guard_lite/all.jsonl --out-dir models/a100_training/guard_lite/model --model-name hfl/chinese-macbert-base --epochs 4 --batch-size 32 --grad-accum 1 --lr 2e-5 --max-length 384 }
if ($Task -in @('reranker','all')) { python scripts/train_regulation_cross_encoder.py --data models/a100_training/reranker/all.jsonl --out-dir models/a100_training/reranker/model --model-name hfl/chinese-macbert-base --epochs 3 --batch-size 32 --grad-accum 1 --lr 2e-5 --max-length 512 }
if ($Task -in @('nli','all')) { python scripts/train_product_nli.py --data models/a100_training/product_nli/all.jsonl --out-dir models/a100_training/product_nli/model --model-name hfl/chinese-macbert-base --epochs 3 --batch-size 32 --max-length 384 }
if ($Task -in @('reasoner','all')) { llamafactory-cli train config/a100_reasoner_qwen3_8b_lora.yaml }
