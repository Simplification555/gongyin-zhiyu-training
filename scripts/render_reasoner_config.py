"""Render public Reasoner YAML templates with a local base-model path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    base = os.environ.get("QWEN3_8B_PATH", "").strip()
    if not base:
        raise SystemExit("QWEN3_8B_PATH must point to the local Qwen3-8B base model")
    text = args.source.read_text(encoding="utf-8")
    text = text.replace("${QWEN3_8B_PATH}", base.replace("\\", "/"))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
