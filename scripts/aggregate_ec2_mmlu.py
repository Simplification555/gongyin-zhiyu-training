"""Build the EC-2 MMLU-57x10 main-table artifact from seed results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def ci95(values: list[float]) -> tuple[float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, center
    margin = 1.96 * stdev(values) / math.sqrt(len(values))
    return center - margin, center + margin


def load(root: Path, method: str, seed: int) -> dict:
    run = root / method / f"seed_{seed}"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    summary = result.get("summary", result)
    if manifest.get("formal_result") is not False or summary.get("num_examples") != 570:
        raise ValueError(f"invalid controlled-subset artifact: {run}")
    test_calls = int(summary["inference_calls"])
    search_calls = int(summary.get("search_calls", manifest.get("search_calls", 0)))
    test_tokens = int(summary.get("inference_tokens", 0))
    search_tokens = int(summary.get("search_tokens", manifest.get("search_tokens", 0)))
    return {"method": method, "seed": seed, "accuracy": float(summary["score"]),
            "valid_answer_rate": float(summary["valid_answer_rate"]),
            "test_calls": test_calls, "search_calls": search_calls,
            "total_calls": test_calls + search_calls, "test_tokens": test_tokens,
            "search_tokens": search_tokens, "total_tokens": test_tokens + search_tokens}


def aggregate(root: Path, output: Path) -> None:
    rows = [load(root, method, seed) for method in ("rpas", "gdesigner") for seed in (0, 1, 2)]
    table = []
    for method in ("rpas", "gdesigner"):
        group = [row for row in rows if row["method"] == method]
        scores = [row["accuracy"] for row in group]
        row = {"method": method, "seeds": 3, "test_examples": 570,
               "accuracy_mean": mean(scores), "accuracy_ci95_low": ci95(scores)[0],
               "accuracy_ci95_high": ci95(scores)[1],
               "valid_answer_rate_mean": mean(row["valid_answer_rate"] for row in group)}
        for field in ("test_calls", "search_calls", "total_calls", "test_tokens", "search_tokens", "total_tokens"):
            row[f"{field}_mean"] = mean(item[field] for item in group)
        row["search_cost_note"] = "instrumented" if method == "rpas" else "not separately instrumented"
        table.append(row)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "MMLU-57x10 controlled subset", "formal_result": False, "rows": table, "seed_rows": rows}
    (output / "main_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/external_comparison/ec2_gpu6"))
    parser.add_argument("--output", type=Path, default=Path("results/external_comparison/ec2_gpu6/aggregate"))
    args = parser.parse_args()
    aggregate(args.root, args.output)
