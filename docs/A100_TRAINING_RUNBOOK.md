# A100 单卡训练执行手册

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
pip install datasets accelerate scikit-learn sentencepiece llamafactory
```

先确认 `nvidia-smi` 和 `torch.cuda.is_bf16_supported()`。数据准备：

```powershell
.\scripts\run_a100_training.ps1 prepare
```

输出在 `models/a100_training/`。只包含 V2/V3 训练资产，保留 `do_not_eval=true`；不会读取 GYZ-Bench、Hidden、Regression 或 Gold Candidate。

## 推荐顺序

1. `./scripts/run_a100_training.ps1 guard`
2. `./scripts/run_a100_training.ps1 reranker`
3. `./scripts/run_a100_training.ps1 nli`，三分类标签为 SUPPORTED/CONTRADICTED/UNKNOWN；`data/evaluation/gyz_product_nli_v1.jsonl` 是银标/评测边界，不能当正式 Gold。
4. `./scripts/run_a100_training.ps1 reasoner`，先保存 M1 SFT，再单独做 trajectory、chaos policy 和 hard DPO checkpoint。

Guard/Reranker 使用 BF16 batch 32。Reasoner-8B 使用 BF16 LoRA、batch 2、gradient accumulation 16、4096 context；显存不足时再切 QLoRA NF4。

每个 checkpoint 都必须记录 GYZ dev/public、专项安全子集、CFR、Evidence Recall、Temporal Leakage、HITL Recall、ECE 和 P95 延迟。Hidden 只在候选冻结后运行一次。
