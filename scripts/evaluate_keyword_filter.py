#!/usr/bin/env python3
"""
Evaluate a deterministic keyword-based prompt filter against a labeled CSV.

Example:
    python3 scripts/evaluate_keyword_filter.py

The script expects a CSV with at least:
    - prompt
    - label

Label values may be binary (0/1) or string-like (benign/malicious).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "data/processed/validation_dataset.csv"
DEFAULT_OUTPUT_DIR = "results/keyword_eval"

BENIGN_LABELS = {"0", "benign", "safe", "normal", "false"}
MALICIOUS_LABELS = {"1", "malicious", "unsafe", "attack", "inject", "true"}

KEYWORD_CATEGORIES: dict[str, dict[str, int]] = {
    "instruction_override": {
        "ignore previous": 3,
        "ignore all previous": 3,
        "disregard previous": 3,
        "override": 2,
        "forget previous": 2,
        "new instructions": 2,
        "system prompt": 2,
        "hidden policy": 2,
    },
    "exfiltration": {
        "reveal": 2,
        "expose": 2,
        "show prompt": 3,
        "hidden prompt": 3,
        "policy text": 2,
        "secret": 2,
        "confidential": 2,
        "api key": 3,
        "password": 3,
        "token": 2,
        "credentials": 3,
    },
    "jailbreak": {
        "jailbreak": 3,
        "dan": 3,
        "do anything now": 3,
        "godmode": 3,
        "unfiltered": 2,
        "uncensored": 2,
        "no restrictions": 2,
        "bypass": 2,
        "disable safety": 3,
        "disable guardrails": 3,
    },
    "payload": {
        "malware": 3,
        "ransomware": 3,
        "keylogger": 3,
        "exploit": 3,
        "payload": 2,
        "shellcode": 3,
        "reverse shell": 3,
        "phishing": 3,
        "steal credentials": 3,
        "hack": 2,
        "breach": 2,
        "exfiltrate": 3,
    },
    "obfuscation": {
        "base64": 2,
        "rot13": 2,
        "hex encoded": 2,
        "obfuscate": 2,
        "unicode escape": 2,
        "roleplay": 1,
        "pretend to be": 1,
        "act as": 1,
    },
}


@dataclass(frozen=True)
class DatasetRow:
    row_id: int
    prompt: str
    expected_label: str
    expected_is_malicious: bool


@dataclass(frozen=True)
class KeywordResult:
    predicted_label: str
    predicted_is_malicious: bool
    risk_score: int
    matched_keyword_count: int
    matched_categories: list[str]
    matched_keywords: list[str]
    reasons: list[str]
    latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic keyword-based malicious prompt detector against a labeled CSV.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to labeled CSV dataset. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where results should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to evaluate from the dataset.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Risk score threshold required to label a prompt as malicious. Default: 3",
    )
    return parser.parse_args()


def normalize_expected_label(raw_label: Any) -> tuple[str, bool]:
    value = str(raw_label).strip().lower()
    if value in MALICIOUS_LABELS:
        return "malicious", True
    if value in BENIGN_LABELS:
        return "benign", False
    raise ValueError(f"Unsupported label value: {raw_label!r}")


def load_dataset(dataset_path: Path, limit: int | None = None) -> list[DatasetRow]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"prompt", "label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Dataset must contain columns {sorted(required_columns)}. "
                f"Found: {reader.fieldnames!r}"
            )

        rows: list[DatasetRow] = []
        for index, record in enumerate(reader, start=1):
            expected_label, expected_is_malicious = normalize_expected_label(record["label"])
            rows.append(
                DatasetRow(
                    row_id=index,
                    prompt=record["prompt"],
                    expected_label=expected_label,
                    expected_is_malicious=expected_is_malicious,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def safe_join(values: list[str]) -> str:
    return " | ".join(values)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_malicious_prompt(prompt: str, threshold: int) -> KeywordResult:
    started_at = time.perf_counter()
    lowered = prompt.lower()
    matched_keywords: list[str] = []
    matched_categories: list[str] = []
    reasons: list[str] = []
    score = 0

    for category, keywords in KEYWORD_CATEGORIES.items():
        for keyword, weight in keywords.items():
            if contains_keyword(lowered, keyword):
                score += weight
                matched_keywords.append(keyword)
                matched_categories.append(category)
                reasons.append(f"{category}:{keyword} (+{weight})")

    unique_categories = sorted(set(matched_categories))
    if len(unique_categories) >= 2:
        score += 1
        reasons.append("multi_category_signal_bonus (+1)")

    predicted_is_malicious = score >= threshold
    latency_ms = (time.perf_counter() - started_at) * 1000

    return KeywordResult(
        predicted_label="malicious" if predicted_is_malicious else "benign",
        predicted_is_malicious=predicted_is_malicious,
        risk_score=score,
        matched_keyword_count=len(matched_keywords),
        matched_categories=unique_categories,
        matched_keywords=matched_keywords,
        reasons=reasons,
        latency_ms=latency_ms,
    )


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(
        1
        for record in records
        if record["expected_is_malicious"] and record["predicted_is_malicious"]
    )
    tn = sum(
        1
        for record in records
        if (not record["expected_is_malicious"]) and (not record["predicted_is_malicious"])
    )
    fp = sum(
        1
        for record in records
        if (not record["expected_is_malicious"]) and record["predicted_is_malicious"]
    )
    fn = sum(
        1
        for record in records
        if record["expected_is_malicious"] and (not record["predicted_is_malicious"])
    )

    accuracy = safe_divide(tp + tn, len(records))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    latency_values = [record["latency_ms"] for record in records]
    category_counter = Counter(
        category
        for record in records
        if record["predicted_is_malicious"]
        for category in record["matched_categories"]
    )

    false_positives = [
        {
            "row_id": record["row_id"],
            "prompt": record["prompt"],
            "predicted_label": record["predicted_label"],
            "risk_score": record["risk_score"],
            "reasons": record["reasons"],
        }
        for record in records
        if (not record["expected_is_malicious"]) and record["predicted_is_malicious"]
    ]
    false_negatives = [
        {
            "row_id": record["row_id"],
            "prompt": record["prompt"],
            "predicted_label": record["predicted_label"],
            "risk_score": record["risk_score"],
            "reasons": record["reasons"],
        }
        for record in records
        if record["expected_is_malicious"] and (not record["predicted_is_malicious"])
    ]

    return {
        "total_rows": len(records),
        "successful_requests": len(records),
        "failed_requests": 0,
        "accuracy": accuracy,
        "precision_malicious": precision,
        "recall_malicious": recall,
        "f1_malicious": f1,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "avg_latency_ms": safe_divide(sum(latency_values), len(latency_values)),
        "max_latency_ms": max(latency_values) if latency_values else 0.0,
        "min_latency_ms": min(latency_values) if latency_values else 0.0,
        "predicted_malicious_count": sum(1 for record in records if record["predicted_is_malicious"]),
        "matched_category_distribution": dict(category_counter),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def write_detailed_results(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_id",
        "expected_label",
        "expected_is_malicious",
        "predicted_label",
        "predicted_is_malicious",
        "correct",
        "prompt",
        "risk_score",
        "matched_keyword_count",
        "matched_categories",
        "matched_keywords",
        "latency_ms",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{
                        key: record.get(key)
                        for key in fieldnames
                        if key not in {"matched_categories", "matched_keywords", "reasons"}
                    },
                    "matched_categories": safe_join(record.get("matched_categories", [])),
                    "matched_keywords": safe_join(record.get("matched_keywords", [])),
                    "reasons": safe_join(record.get("reasons", [])),
                }
            )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def write_markdown_report(
    path: Path,
    dataset_path: Path,
    threshold: int,
    metrics: dict[str, Any],
) -> None:
    confusion = metrics["confusion_matrix"]
    false_positives = metrics["false_positives"][:10]
    false_negatives = metrics["false_negatives"][:10]
    matched_category_distribution = metrics["matched_category_distribution"]

    lines = [
        "# Keyword Filter Evaluation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Detector: `deterministic keyword filter`",
        f"- Malicious threshold: **{threshold}**",
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Summary",
        "",
        f"- Total rows: **{metrics['total_rows']}**",
        f"- Successful requests: **{metrics['successful_requests']}**",
        f"- Failed requests: **{metrics['failed_requests']}**",
        f"- Accuracy: **{format_metric(metrics['accuracy'])}**",
        f"- Precision (malicious): **{format_metric(metrics['precision_malicious'])}**",
        f"- Recall (malicious): **{format_metric(metrics['recall_malicious'])}**",
        f"- F1 (malicious): **{format_metric(metrics['f1_malicious'])}**",
        f"- Predicted malicious: **{metrics['predicted_malicious_count']}**",
        f"- Avg endpoint latency: **{metrics['avg_latency_ms']:.4f} ms**",
        "",
        "## Confusion Matrix",
        "",
        f"- True positives: **{confusion['true_positive']}**",
        f"- True negatives: **{confusion['true_negative']}**",
        f"- False positives: **{confusion['false_positive']}**",
        f"- False negatives: **{confusion['false_negative']}**",
        "",
        "## Matched Category Distribution",
        "",
    ]

    if matched_category_distribution:
        for category, count in sorted(matched_category_distribution.items()):
            lines.append(f"- {category}: **{count}**")
    else:
        lines.append("- No malicious prompts were predicted.")

    lines.extend(["", "## Sample False Positives", ""])
    if false_positives:
        for item in false_positives:
            lines.append(
                f"- Row {item['row_id']}: score **{item['risk_score']}**, predicted "
                f"`{item['predicted_label']}` for `{item['prompt']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Sample False Negatives", ""])
    if false_negatives:
        for item in false_negatives:
            lines.append(
                f"- Row {item['row_id']}: score **{item['risk_score']}**, predicted "
                f"`{item['predicted_label']}` for `{item['prompt']}`"
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_dataset(rows: list[DatasetRow], threshold: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(rows)

    for position, row in enumerate(rows, start=1):
        print(f"[{position}/{total}] Evaluating row {row.row_id}...", flush=True)
        result = detect_malicious_prompt(row.prompt, threshold=threshold)
        records.append(
            {
                "row_id": row.row_id,
                "prompt": row.prompt,
                "expected_label": row.expected_label,
                "expected_is_malicious": row.expected_is_malicious,
                "predicted_label": result.predicted_label,
                "predicted_is_malicious": result.predicted_is_malicious,
                "correct": result.predicted_is_malicious == row.expected_is_malicious,
                "risk_score": result.risk_score,
                "matched_keyword_count": result.matched_keyword_count,
                "matched_categories": result.matched_categories,
                "matched_keywords": result.matched_keywords,
                "latency_ms": result.latency_ms,
                "reasons": result.reasons,
            }
        )

    return records


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_root = Path(args.output_dir)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(dataset_path, limit=args.limit)
    print(f"Loaded {len(rows)} labeled prompts from {dataset_path}")
    print(f"Running deterministic keyword detection with threshold={args.threshold}")

    records = evaluate_dataset(rows, threshold=args.threshold)
    metrics = compute_metrics(records)

    detailed_csv_path = run_dir / "detailed_results.csv"
    raw_jsonl_path = run_dir / "raw_results.jsonl"
    summary_json_path = run_dir / "summary.json"
    report_md_path = run_dir / "report.md"

    write_detailed_results(detailed_csv_path, records)
    write_jsonl(raw_jsonl_path, records)
    summary_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown_report(
        report_md_path,
        dataset_path=dataset_path,
        threshold=args.threshold,
        metrics=metrics,
    )

    print("\nEvaluation complete.")
    print(f"Detailed results: {detailed_csv_path}")
    print(f"Raw results:      {raw_jsonl_path}")
    print(f"Summary JSON:     {summary_json_path}")
    print(f"Markdown report:  {report_md_path}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision:        {metrics['precision_malicious']:.4f}")
    print(f"Recall:           {metrics['recall_malicious']:.4f}")
    print(f"F1:               {metrics['f1_malicious']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
