"""Prepare V3 hard-boundary DPO data for LLaMA-Factory."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/training/gyz_trainpack_v3_fmag/preference"
OUT = ROOT / "models/a100_training/hard_dpo"

def convert(path: Path):
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        rows.append({"prompt":r.get("prompt",""),"chosen":r.get("chosen",""),"rejected":r.get("rejected",""),"id":r.get("id"),"source":r.get("provenance_type","hard_boundary_preference"),"do_not_eval":True})
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train=convert(SRC/"hard_dpo_train.jsonl"); dev=convert(SRC/"hard_dpo_validation.jsonl")
    for name,rows in (("train.jsonl",train),("dev.jsonl",dev),("all.jsonl",train+dev)):
        (OUT/name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    info={"a100_hard_dpo":{"file_name":"all.jsonl","formatting":"sharegpt","columns":{"prompt":"prompt","chosen":"chosen","rejected":"rejected"}}}
    (OUT/"dataset_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"train":len(train),"dev":len(dev),"do_not_eval":True},ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
