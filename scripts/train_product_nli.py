"""Train a three-way ProductNLI classifier: SUPPORTED/CONTRADICTED/UNKNOWN."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

LABELS = ["SUPPORTED", "CONTRADICTED", "UNKNOWN"]

def read(path: Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=Path("models/a100_training/product_nli/all.jsonl")); ap.add_argument("--out-dir",type=Path,default=Path("models/a100_training/product_nli/model")); ap.add_argument("--model-name",default="hfl/chinese-macbert-base"); ap.add_argument("--epochs",type=float,default=3); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--max-length",type=int,default=384); args=ap.parse_args()
    try:
        import numpy as np, torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc: raise SystemExit("Install torch transformers datasets accelerate scikit-learn") from exc
    rows=read(args.data); train=[r for r in rows if r.get("split")=="train"]; dev=[r for r in rows if r.get("split") in {"dev","validation"}]
    if not dev: dev=read(args.data.parent/"dev.jsonl")
    label_to_id={x:i for i,x in enumerate(LABELS)}
    def text(r): return "claim="+str(r.get("claim",r.get("text","")))+"\nproduct_snapshot="+str(r.get("product_snapshot_id", ""))+"\nas_of_date="+str(r.get("as_of_date", ""))
    def tok(batch):
        out=tokenizer(batch["text"],truncation=True,max_length=args.max_length); out["labels"]=[label_to_id.get(str(x).upper(),2) for x in batch["label"]]; return out
    for r in train+dev: r["text"]=text(r)
    tokenizer=AutoTokenizer.from_pretrained(args.model_name); tr=Dataset.from_list(train).map(tok,batched=True,remove_columns=list(train[0].keys())); dv=Dataset.from_list(dev).map(tok,batched=True,remove_columns=list(dev[0].keys()))
    model=AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=3,id2label=dict(enumerate(LABELS)),label2id=label_to_id)
    def metrics(p: Any):
        pred=np.argmax(p.predictions,axis=-1); y=p.label_ids
        return {"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro",zero_division=0),"contradicted_recall":recall_score(y,pred,labels=[1],average="macro",zero_division=0),"unknown_precision":precision_score(y,pred,labels=[2],average="macro",zero_division=0)}
    ta=TrainingArguments(output_dir=str(args.out_dir),num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,learning_rate=2e-5,weight_decay=.01,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="macro_f1",bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),report_to=[])
    trainer=Trainer(model=model,args=ta,train_dataset=tr,eval_dataset=dv,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics,processing_class=tokenizer)
    trainer.train(); result=trainer.evaluate(); args.out_dir.mkdir(parents=True,exist_ok=True); trainer.save_model(str(args.out_dir/"best_model")); tokenizer.save_pretrained(str(args.out_dir/"best_model")); (args.out_dir/"metrics.json").write_text(json.dumps({"labels":LABELS,"train":len(train),"dev":len(dev),"metrics":result},ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
