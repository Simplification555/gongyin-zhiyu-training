"""Prepare non-empty Reasoner continuation-stage datasets from the assembled pack."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "a100_training" / "reasoner"
STAGES = {
    "m2_complex": "FMAG-SFT-",
    "m3_trajectory": "FMAG-TRJ-",
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usable(row: dict) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    return any(str(message.get("content", "")).strip() for message in messages)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    train = read(SOURCE / "train.jsonl")
    dev = read(SOURCE / "dev.jsonl")
    manifest = {"schema": "gongyin_zhiyu.reasoner_stages.v1", "stages": {}}
    for stage, prefix in STAGES.items():
        stage_train = [row for row in train if str(row.get("id", "")).startswith(prefix) and usable(row)]
        stage_dev = [row for row in dev if str(row.get("id", "")).startswith(prefix) and usable(row)]
        out = SOURCE / stage
        write(out / "train.jsonl", stage_train)
        write(out / "dev.jsonl", stage_dev)
        write(out / "all.jsonl", stage_train + stage_dev)
        manifest["stages"][stage] = {"train": len(stage_train), "dev": len(stage_dev), "total": len(stage_train) + len(stage_dev)}
    (SOURCE / "stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
