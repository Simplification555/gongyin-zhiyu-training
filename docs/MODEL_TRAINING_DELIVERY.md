# V6 模型训练交付文档

更新日期：2026-09-02

仓库：`Simplification555/gongyin-zhiyu-training`  
分支：`master`

## 交付结论

本仓库已经具备单张 A100 继续训练所需的首批数据、脚本和配置，并已交付一个可复现的 Reasoner M1 LoRA checkpoint。当前不能表述为“所有模型均已训练完成”：Guard-Lite、Reranker、ProductNLI 尚未在仓库交付训练结果；Reasoner 的 M2/M3/M4/M5/M6 也尚未完成。

## 已交付资产

| 模块 | 数据 | 训练入口 | 当前状态 |
|---|---:|---|---|
| Guard-Lite 多标签合规分类 | train 5,396 / dev 604 | `scripts/train_multilabel_transformer_compliance.py` | 可训练，结果待跑 |
| Regulation Reranker | train 9,062 / dev 938 | `scripts/train_regulation_cross_encoder.py` | 可训练，结果待跑 |
| ProductNLI 三分类 | train 2,663 / dev 337 | `scripts/train_product_nli.py` | 可训练，结果待跑 |
| Reasoner-8B 混合 SFT | train 20,715 / dev 2,301 | LLaMA-Factory + `config/a100_reasoner_qwen3_8b_lora.yaml` | M1 已完成 |

所有数据来自 V2/V3 synthetic、silver 或程序组合训练资产，保留 `do_not_eval=true` 边界。GYZ-Bench、Hidden、Regression、Gold Candidate 和私有 answer key 不进入训练。

## M1 Reasoner checkpoint

- 基座：`Qwen/Qwen3-8B`
- 方式：LoRA/PEFT
- 训练设备：NVIDIA A100-SXM4-80GB，物理 GPU 6
- 训练步数：720
- epoch：1.0
- train loss：0.7357495493359036
- eval loss：0.6062836647033691
- 训练时长：1904.311 秒
- adapter SHA-256：`bb36a1820081774c03000e089167310aab9a7e3873beb22ad18e19347dd16b14`
- 交付目录：`models/a100_training/reasoner/checkpoints/m1_sft/`

上述结果来自 checkpoint 的 `RELEASE_MANIFEST.json`、`train_results.json` 和 `eval_results.json`。M1 是工程 checkpoint，不等于外部 benchmark、真人 Gold 或生产效果证明。

## 推荐执行顺序

```powershell
python scripts/prepare_a100_training.py
pwsh scripts/run_a100_training.ps1 guard
pwsh scripts/run_a100_training.ps1 reranker
pwsh scripts/run_a100_training.ps1 nli
pwsh scripts/run_a100_training.ps1 reasoner
```

Reasoner 后续应按独立 checkpoint 推进：

```text
M1 Grounded/Complex SFT
M2 Complex F-MAG SFT
M3 Deep Trajectory / Tool Policy
M4 Chaos Tool Policy
M5 Hard DPO
M6 Quantized release
```

每个阶段必须保留训练日志、评估结果、配置快照和 SHA-256。不要覆盖 M1。

## 发布 Gate

训练完成后至少记录：Macro/Micro F1、CONTRADICTED Recall、UNKNOWN Precision、Evidence Recall、Temporal Leakage、HITL Recall、ECE、P95 latency、显存和成本。Hidden Benchmark 只能在候选 checkpoint 冻结后运行一次。

## 安全与边界

- 不上传 API key、数据库、审计日志或私有会话。
- 不把 synthetic/silver 指标写成真人 Gold 或生产效果。
- Reasoner 不能绕过确定性的 Decision Kernel 决定最终动作。
- 如果模型服务失败，系统必须降级或转人工复核，而不是静默 PASS。
