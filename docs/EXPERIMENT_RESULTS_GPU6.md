# GPU 6 Experiment Results

更新日期：2026-09-02

本次运行只使用 NVIDIA A100-SXM4-80GB 物理 GPU 6（`CUDA_VISIBLE_DEVICES=6`）。GPU 7 未启动、未用于计算，实验结束后 GPU 6/7 均为 0 MiB 显存占用。

## 银行智能体训练结果

以下指标是本仓库 synthetic/silver/programmatically assembled 数据的开发集指标，不是 GYZ-Bench、Hidden、真人 Gold 或生产效果。

| 模块 | 训练集 | 开发集 | 训练配置 | 主要结果 |
|---|---:|---:|---|---|
| Guard-Lite 多标签合规分类 | 5,396 | 604 | MacBERT；4 epoch；batch 32；长度 384；BF16 | Macro/Micro F1 1.0000；高风险召回 1.0000；exact match 1.0000 |
| Regulation Reranker | 9,062 | 938 | MacBERT；3 epoch；batch 32；长度 512；BF16 | Accuracy/Precision/Recall/F1 1.0000 |
| ProductNLI | 2,663 | 337 | MacBERT；3 epoch；batch 32；长度 384；BF16 | Accuracy/Macro F1 1.0000；CONTRADICTED Recall 1.0000；UNKNOWN Precision 1.0000 |
| Reasoner-8B M1 | 20,715 | 2,301 | Qwen3-8B LoRA；1 epoch；BF16；GPU 6 | train loss 0.7357495；eval loss 0.6062837；720 steps |

完整训练指标位于 `models/a100_training/{guard_lite,reranker,product_nli}/model/metrics.json`。M1 发布信息位于 `models/a100_training/reasoner/checkpoints/m1_sft/RELEASE_MANIFEST.json`。

## MMLU-57x10 controlled subset 外部对比实验

这是 controlled external comparison，不是完整 MMLU，也不是 formal result。共 6 个 seed，每个 seed 使用 57 个 subject、每个 subject 10 道测试题（570 题）；搜索集为每 subject 5 题（285 题）。完整逐条输出、调用记录、结果 JSON、汇总 CSV 和运行 manifest 位于 `outputs/external_comparison/ec2_gpu6/`。

RPAS 在本次配置中仅执行 9 个预定义候选架构的 controlled candidate selection；`RPAS_MMLU_NEW_CANDIDATES=0`，没有新增 reflective mutation search。三个 RPAS seed 使用确定性解码和相同候选集合，因此结果相同属于预期现象。

| 方法 | seed 0 | seed 1 | seed 2 | 平均准确率 | 有效率 |
|---|---:|---:|---:|---:|---:|
| RPAS | 0.815789 | 0.815789 | 0.815789 | 0.815789 | 1.000000 |
| G-Designer | 0.763158 | 0.773684 | 0.764912 | 0.767251 | 0.997661 |

三 seed 的真实成本按 `result.json` 与 `calls.jsonl` 汇总，不将 G-Designer 的 `search_calls=0` 解读为没有额外推理：

| 方法 | 测试调用（均值） | 搜索调用（均值） | 搜索 token（均值） | 总调用（均值） | 测试 token（均值） |
|---|---:|---:|---:|---:|---:|
| RPAS | 570.0 | 5,806.3 | 1.925M | 6,376.3 | 141.8K |
| G-Designer | 1,739.3 | 0（未单独记录） | 0（未单独记录） | 1,739.3 | 856.6K |

在该 controlled subset 上，RPAS 平均准确率为 81.58%，G-Designer 为 76.73%，差值 4.85 个百分点；该比较同时存在搜索和通信成本差异，不能据此声称 RPAS 总成本更低。

RPAS 三个 seed 均为 0 模型错误、0 次长度触顶；G-Designer 三个 seed 均为 0 模型错误，长度触顶分别为 20、20、18 次。该对比使用 `Qwen/Qwen3.5-9B`，仅作为外部能力实验，不代表银行业务性能。

## 完整结果文件

每个 controlled seed 均提交以下文件：

- `summary.csv`
- `result.json`
- `run_manifest.json`
- `test_outputs.jsonl`
- `calls.jsonl`
- RPAS 额外的 `search_rows.jsonl`

不提交 smoke 运行、私有数据、API key、数据库、审计日志、Hidden/Gold answer key 或模型服务凭据。
