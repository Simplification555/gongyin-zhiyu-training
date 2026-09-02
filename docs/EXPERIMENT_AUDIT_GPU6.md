# GPU 6 Experiment Audit

Audit date: 2026-09-03

## Scope

This audit covers the published EC-2 comparison and the A100 training metadata. The retained comparison is a `MMLU-57x10 controlled subset`, not full MMLU and not a formal result.

## Main-table evidence

The table is regenerated from all three seeds for both methods by `scripts/aggregate_ec2_mmlu.py`. Each seed has 57 subjects x 10 test examples and 57 subjects x 5 search examples.

| Method | Accuracy | Valid answer rate | Test calls | Search calls | Total calls | Search tokens | Search accounting |
|---|---:|---:|---:|---:|---:|---:|---|
| RPAS | 81.5789% | 100.0000% | 570.0 | 5,806.3 | 6,376.3 | 1,924,959.3 | instrumented |
| G-Designer | 76.7251% | 99.7661% | 1,739.3 | 0 | 1,739.3 | 0 | not separately instrumented |

The G-Designer zero search counters mean that no separate search phase was instrumented. They do not prove zero search compute. The comparison does not support a claim that RPAS is cheaper overall.

## Protocol checks

- All six retained seed manifests have `formal_result: false`.
- All six retained artifacts are labeled `MMLU-57x10 controlled subset`.
- RPAS scope is nine predefined candidates with no new reflective mutation (`RPAS_MMLU_NEW_CANDIDATES=0`).
- No test labels or test scores are used for candidate generation or selection.
- M1 Git LFS pointer and object checks pass; adapter SHA-256 is recorded in `RELEASE_MANIFEST.json`.
- RPAS repository tests: `16 passed`.
- Static protocol validation: all configured files `OK` with `--require-native`.
- Rebuilt main-table CSVs in both repositories are byte-identical.

## Device audit

All retained training and evaluation runs are labeled physical GPU 6 and use `CUDA_VISIBLE_DEVICES=6`. During this audit an unrelated stale GPU 7 Reasoner process was discovered and stopped before any new experiment was run; its output is excluded from the published artifacts. At audit completion, both GPU 6 and GPU 7 reported 0 MiB, and no Reasoner, vLLM, or native MMLU process remained.

## Interpretation boundary

The result is suitable for a controlled-subset main table with the stated caveats. It must not be described as full MMLU, formal gated results, complete reflective mutation search, or production banking performance.
