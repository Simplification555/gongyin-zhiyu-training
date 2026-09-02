# -*- coding: utf-8 -*-
"""Fine-tune a Chinese transformer for multi-label compliance classification."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "models" / "compliance_classifier" / "data" / "multilabel_compliance.jsonl"
DEFAULT_LABELS = ROOT / "models" / "compliance_classifier" / "data" / "multilabel_label_names.json"
DEFAULT_OUT = ROOT / "models" / "compliance_classifier" / "transformer_multilabel"

HIGH_RISK = {
    "suitability_mismatch",
    "principal_guarantee",
    "return_guarantee",
    "senior_misleading",
    "prompt_injection",
    "privacy_leak",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score, precision_score, recall_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install: pip install torch transformers datasets accelerate scikit-learn"
        ) from exc

    parser = argparse.ArgumentParser(description="Train multi-label MacBERT/RoBERTa compliance classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    label_names = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = read_jsonl(args.data)
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    test_rows = [row for row in rows if row.get("split") == "blind_test"]
    if not train_rows or not dev_rows:
        raise SystemExit("Need train/dev splits. Run prepare_multilabel_compliance_dataset.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        encoded["labels"] = [[float(x) for x in labels] for labels in batch["labels"]]
        return encoded

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=list(train_rows[0].keys()))
    dev_ds = Dataset.from_list(dev_rows).map(tokenize, batched=True, remove_columns=list(dev_rows[0].keys()))
    test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True, remove_columns=list(test_rows[0].keys()))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
    )
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = base_collator(features)
        batch["labels"] = torch.as_tensor(batch["labels"], dtype=torch.float32)
        return batch

    high_indices = [i for i, label in enumerate(label_names) if label in HIGH_RISK]

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= args.threshold).astype(int)
        labels = labels.astype(int)
        high_truth = labels[:, high_indices].max(axis=1) if high_indices else labels.max(axis=1)
        high_pred = preds[:, high_indices].max(axis=1) if high_indices else preds.max(axis=1)
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
            "high_risk_recall": recall_score(high_truth, high_pred, zero_division=0),
            "exact_match": float((preds == labels).all(axis=1).mean()),
        }

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    dev_metrics = trainer.evaluate(dev_ds)
    test_metrics = trainer.evaluate(test_ds) if len(test_rows) else {}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (args.out_dir / "label_names.json").write_text(json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "model_name": args.model_name,
        "threshold": args.threshold,
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "blind_test_size": len(test_rows),
        "dev": dev_metrics,
        "blind_test": test_metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
