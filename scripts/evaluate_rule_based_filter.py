#!/usr/bin/env python3
"""
Evaluate a deterministic rule-based prompt filter against a labeled CSV.

Example:
    python3 scripts/evaluate_rule_based_filter.py

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
DEFAULT_OUTPUT_DIR = "results/rule_base_eval"

BENIGN_LABELS = {"0", "benign", "safe", "normal", "false"}
MALICIOUS_LABELS = {"1", "malicious", "unsafe", "attack", "inject", "true"}


@dataclass(frozen=True)
class DatasetRow:
    row_id: int
    prompt: str
    expected_label: str
    expected_is_malicious: bool


@dataclass(frozen=True)
class RuleMatch:
    category: str
    name: str
    weight: int
    pattern: str


@dataclass(frozen=True)
class RuleBasedResult:
    predicted_label: str
    predicted_is_malicious: bool
    risk_score: int
    matched_rule_count: int
    matched_categories: list[str]
    matched_rules: list[str]
    reasons: list[str]
    latency_ms: float


RULES: tuple[RuleMatch, ...] = (
    RuleMatch(
        category="instruction_override",
        name="ignore_previous_instructions",
        weight=3,
        pattern=r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|system)\b.{0,40}\b(instruction|instructions|rules|rule|prompt|message|context)\b",
    ),
    RuleMatch(
        category="instruction_override",
        name="follow_new_instructions",
        weight=2,
        pattern=r"\b(new|updated)\b.{0,20}\b(instruction|instructions|rules|rule|task)\b",
    ),
    RuleMatch(
        category="instruction_override",
        name="you_are_now",
        weight=2,
        pattern=r"\byou are now\b",
    ),
    RuleMatch(
        category="instruction_override",
        name="developer_system_prompt_reference",
        weight=2,
        pattern=r"\b(system prompt|developer message|hidden prompt|hidden policy|internal policy)\b",
    ),
    RuleMatch(
        category="exfiltration",
        name="reveal_hidden_instructions",
        weight=3,
        pattern=r"\b(reveal|show|print|display|output|expose|repeat)\b.{0,50}\b(system prompt|hidden prompt|prompt|instructions|policy|policies|chain of thought|internal message|secret)\b",
    ),
    RuleMatch(
        category="exfiltration",
        name="extract_sensitive_data",
        weight=3,
        pattern=r"\b(extract|leak|dump|export|return)\b.{0,40}\b(api key|password|credentials|token|secret|private data|confidential|sensitive data)\b",
    ),
    RuleMatch(
        category="jailbreak",
        name="do_anything_now",
        weight=3,
        pattern=r"\b(do anything now|dan|godmode|unfiltered|uncensored|no restrictions|without restrictions)\b",
    ),
    RuleMatch(
        category="jailbreak",
        name="bypass_safety",
        weight=3,
        pattern=r"\b(bypass|disable|neutralize|evade)\b.{0,40}\b(safety|guardrail|filter|defense|policy|moderation|restriction|restrictions)\b",
    ),
    RuleMatch(
        category="payload",
        name="malware_or_exploit_request",
        weight=3,
        pattern=r"\b(malware|ransomware|keylogger|payload|exploit|shellcode|reverse shell|dropper)\b",
    ),
    RuleMatch(
        category="payload",
        name="credential_theft_or_phishing",
        weight=3,
        pattern=r"\b(phishing|steal credentials|credential harvesting|social engineer|impersonate support|fake login)\b",
    ),
    RuleMatch(
        category="payload",
        name="harmful_action_instructions",
        weight=2,
        pattern=r"\b(how to|step by step|walk me through|instructions for)\b.{0,50}\b(exploit|breach|hack|steal|exfiltrate|bypass|disable)\b",
    ),
    RuleMatch(
        category="obfuscation",
        name="encoding_or_obfuscation",
        weight=2,
        pattern=r"\b(base64|rot13|hex-encoded|hex encoded|obfuscat|encoded payload|unicode escape)\b",
    ),
    RuleMatch(
        category="obfuscation",
        name="roleplay_to_evade",
        weight=2,
        pattern=r"\b(pretend to be|roleplay as|act as)\b.{0,40}\b(unrestricted|unfiltered|evil|malicious|attacker|red team)\b",
    ),
)

COMPILED_RULES = tuple(
    (rule, re.compile(rule.pattern, flags=re.IGNORECASE | re.DOTALL)) for rule in RULES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic rule-based malicious prompt detector against a labeled CSV.",
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


def detect_malicious_prompt(prompt: str, threshold: int) -> RuleBasedResult:
    started_at = time.perf_counter()
    matched_rules: list[str] = []
    matched_categories: list[str] = []
    reasons: list[str] = []
    score = 0
    lowered = prompt.lower()

    for rule, compiled in COMPILED_RULES:
        if compiled.search(lowered):
            score += rule.weight
            matched_rules.append(rule.name)
            matched_categories.append(rule.category)
            reasons.append(f"{rule.category}:{rule.name} (+{rule.weight})")

    unique_categories = sorted(set(matched_categories))
    if len(unique_categories) >= 2:
        score += 1
        reasons.append("multi_category_signal_bonus (+1)")

    predicted_is_malicious = score >= threshold
    latency_ms = (time.perf_counter() - started_at) * 1000

    return RuleBasedResult(
        predicted_label="malicious" if predicted_is_malicious else "benign",
        predicted_is_malicious=predicted_is_malicious,
        risk_score=score,
        matched_rule_count=len(matched_rules),
        matched_categories=unique_categories,
        matched_rules=matched_rules,
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
        "matched_rule_count",
        "matched_categories",
        "matched_rules",
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
                        if key not in {"matched_categories", "matched_rules", "reasons"}
                    },
                    "matched_categories": safe_join(record.get("matched_categories", [])),
                    "matched_rules": safe_join(record.get("matched_rules", [])),
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
        "# Rule-Based Filter Evaluation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Detector: `deterministic rule-based filter`",
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
                "matched_rule_count": result.matched_rule_count,
                "matched_categories": result.matched_categories,
                "matched_rules": result.matched_rules,
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
    print(f"Running deterministic rule-based detection with threshold={args.threshold}")

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
