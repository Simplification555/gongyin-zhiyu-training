"""Prepare A100 training datasets without benchmark contamination."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "training"
OUT = ROOT / "models" / "a100_training"
SEED = 20260831

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def as_messages(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    elif row.get("input"):
        messages = [{"role":"system","content":"你是银行AI治理执行Agent，只输出动作、依据和下一步。"},{"role":"user","content":str(row["input"])},{"role":"assistant","content":str(row.get("expected_action","HUMAN_REVIEW"))}]
    else:
        messages = [{"role":"user","content":str(row.get("prompt",""))},{"role":"assistant","content":str(row.get("chosen",""))}]
    return {"messages": messages, "id": row.get("id"), "source": row.get("provenance_type","synthetic"), "do_not_eval": True}

def prepare_guard() -> dict[str,int]:
    p = SRC / "gyz_trainpack_v2" / "classification"; o = OUT / "guard_lite"
    tr, dv = read_jsonl(p/"compliance_multilabel_train.jsonl"), read_jsonl(p/"compliance_multilabel_validation.jsonl")
    write_jsonl(o/"train.jsonl", tr); write_jsonl(o/"dev.jsonl", dv); write_jsonl(o/"all.jsonl", tr + dv); return {"train":len(tr),"dev":len(dv)}

def prepare_reranker() -> dict[str,int]:
    p = SRC / "gyz_trainpack_v2" / "retrieval"; o = OUT / "reranker"
    def convert(rows):
        out=[]
        for r in rows:
            common={"id":r.get("id"),"case_id":r.get("case_id", r.get("id")),"query":r.get("query",""),"source_id":r.get("source_id"),"split":r.get("split","train"),"do_not_eval":True}
            out += [{**common,"passage":r.get("positive_text",""),"label":1},{**common,"passage":r.get("negative_text",""),"label":0}]
        return out
    tr, dv = convert(read_jsonl(p/"regulation_reranker_pairs_train.jsonl")), convert(read_jsonl(p/"regulation_reranker_pairs_validation.jsonl"))
    write_jsonl(o/"train.jsonl", tr); write_jsonl(o/"dev.jsonl", dv); write_jsonl(o/"all.jsonl", tr + dv); return {"train":len(tr),"dev":len(dv)}

def prepare_nli() -> dict[str,int]:
    p = SRC / "gyz_trainpack_v2" / "nli"; o = OUT / "product_nli"
    tr, dv = read_jsonl(p/"product_nli_train_train.jsonl"), read_jsonl(p/"product_nli_train_validation.jsonl")
    write_jsonl(o/"train.jsonl", tr); write_jsonl(o/"dev.jsonl", dv); write_jsonl(o/"all.jsonl", tr + dv); return {"train":len(tr),"dev":len(dv)}

def prepare_reasoner() -> dict[str,int]:
    v2, v3 = SRC/"gyz_trainpack_v2", SRC/"gyz_trainpack_v3_fmag"
    pools={"grounded":read_jsonl(v2/"sft/grounded_sft_train.jsonl"),"complex":read_jsonl(v3/"sft/complex_fmag_sft_train.jsonl"),"temporal":read_jsonl(v2/"sft/temporal_rule_training_train.jsonl"),"trajectory":read_jsonl(v3/"trajectories/deep_fmag_trajectories_train.jsonl"),"chaos":read_jsonl(v3/"tooluse/chaos_tool_policy_train.jsonl")}
    targets={"grounded":12000,"complex":12000,"temporal":3500,"trajectory":3500,"chaos":3500}; rng=random.Random(SEED); mixed=[]; counts={}
    for name,n in targets.items():
        pick=pools[name][:min(n,len(pools[name]))]; rng.shuffle(pick); mixed += [as_messages(x) for x in pick]; counts[name]=len(pick)
    rng.shuffle(mixed); cut=max(1,len(mixed)//10); o=OUT/"reasoner"; write_jsonl(o/"dev.jsonl",mixed[:cut]); write_jsonl(o/"train.jsonl",mixed[cut:]); write_jsonl(o/"all.jsonl",mixed); counts.update({"train":len(mixed)-cut,"dev":cut,"total":len(mixed)}); return counts

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    m={"schema":"gongyin_zhiyu.a100_training_manifest.v1","seed":SEED,"do_not_eval":True,"datasets":{"guard_lite":prepare_guard(),"reranker":prepare_reranker(),"product_nli":prepare_nli(),"reasoner":prepare_reasoner()}}
    (OUT/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
