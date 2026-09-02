# -*- coding: utf-8 -*-
"""Train a cross-encoder reranker for case-to-regulation evidence retrieval."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "models" / "regulation_reranker" / "data" / "regulation_reranker_pairs.jsonl"
DEFAULT_OUT = ROOT / "models" / "regulation_reranker" / "cross_encoder"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recall_at_k(rows: list[dict[str, Any]], scores: list[float], k: int = 5) -> float:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["case_id"]].append((score, int(row["label"])))
    recalls = []
    for items in grouped.values():
        if not any(label for _, label in items):
            continue
        top = sorted(items, key=lambda item: item[0], reverse=True)[:k]
        recalls.append(float(any(label for _, label in top)))
    return sum(recalls) / len(recalls) if recalls else 0.0


def main() -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

    parser = argparse.ArgumentParser(description="Train regulation evidence cross-encoder.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    test_rows = [row for row in rows if row.get("split") == "blind_test"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(batch["query"], batch["passage"], truncation=True, max_length=args.max_length)
        encoded["labels"] = [int(label) for label in batch["label"]]
        return encoded

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=list(train_rows[0].keys()))
    dev_ds = Dataset.from_list(dev_rows).map(tokenize, batched=True, remove_columns=list(dev_rows[0].keys()))
    test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True, remove_columns=list(test_rows[0].keys()))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
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
        metric_for_best_model="f1",
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
    test_metrics = trainer.evaluate(test_ds) if test_rows else {}

    predictions = trainer.predict(test_ds) if test_rows else None
    if predictions is not None:
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1].tolist()
        test_metrics["recall_at_5"] = recall_at_k(test_rows, probs, k=5)
        pred_rows = []
        for row, score in zip(test_rows, probs):
            pred_rows.append({**row, "score": float(score)})
        (args.out_dir / "test_predictions.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "test_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in pred_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    metrics = {
        "model_name": args.model_name,
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
