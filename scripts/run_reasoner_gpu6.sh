#!/usr/bin/env bash
set -euo pipefail

# Hard device guard: all Reasoner training in this repository uses physical GPU 6.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "6" ]]; then
  echo "Reasoner training is restricted to physical GPU 6; refusing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ -z "${QWEN3_8B_PATH:-}" ]]; then
  echo "Set QWEN3_8B_PATH to the local Qwen3-8B base model." >&2
  exit 2
fi

CONFIG="${1:-config/a100_reasoner_qwen3_8b_lora.yaml}"
RENDERED="${TMPDIR:-/tmp}/gongyin_reasoner_gpu6.yaml"
python scripts/render_reasoner_config.py "${CONFIG}" "${RENDERED}"
python -m llamafactory.cli train "${RENDERED}"
