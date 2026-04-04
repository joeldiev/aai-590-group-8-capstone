#!/usr/bin/env python3
"""
Evaluate the prompt-security API's /api/v1/prompt endpoint against a labeled CSV.

Example:
    python3 scripts/evaluate_prompt_endpoint.py

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
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_DATASET = "data/processed/validation_dataset.csv"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/v1/prompt"
DEFAULT_OUTPUT_DIR = "results/prompt_endpoint_eval"
DEFAULT_TIMEOUT = 30.0

BENIGN_LABELS = {"0", "benign", "safe", "normal", "false"}
MALICIOUS_LABELS = {"1", "malicious", "unsafe", "attack", "inject", "true"}


@dataclass(frozen=True)
class DatasetRow:
    row_id: int
    prompt: str
    expected_label: str
    expected_is_malicious: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call /api/v1/prompt for each labeled prompt and generate an evaluation report.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to labeled CSV dataset. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Full prompt endpoint URL. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where results should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to evaluate from the dataset.",
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


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            status_code = response.getcode()
            data = json.loads(response.read().decode("utf-8"))
            return status_code, data
    except error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            data = {"detail": raw_body}
        return exc.code, data


def safe_join(values: list[str]) -> str:
    return " | ".join(values)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["request_ok"]]
    failed = [record for record in records if not record["request_ok"]]

    tp = sum(
        1
        for record in successful
        if record["expected_is_malicious"] and record["predicted_is_malicious"]
    )
    tn = sum(
        1
        for record in successful
        if (not record["expected_is_malicious"]) and (not record["predicted_is_malicious"])
    )
    fp = sum(
        1
        for record in successful
        if (not record["expected_is_malicious"]) and record["predicted_is_malicious"]
    )
    fn = sum(
        1
        for record in successful
        if record["expected_is_malicious"] and (not record["predicted_is_malicious"])
    )

    evaluated_count = len(successful)
    total_count = len(records)
    accuracy = safe_divide(tp + tn, evaluated_count)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    endpoint_latency_values = [
        record["endpoint_latency_ms"]
        for record in successful
        if record["endpoint_latency_ms"] is not None
    ]
    severity_latency_values = [
        record["severity_latency_ms"]
        for record in successful
        if record["severity_latency_ms"] is not None
    ]
    severity_counter = Counter(
        record["severity_tier"] for record in successful if record["severity_tier"]
    )

    false_positives = [
        {
            "row_id": record["row_id"],
            "prompt": record["prompt"],
            "predicted_label": record["predicted_label"],
            "reasons": record["reasons"],
        }
        for record in successful
        if (not record["expected_is_malicious"]) and record["predicted_is_malicious"]
    ]
    false_negatives = [
        {
            "row_id": record["row_id"],
            "prompt": record["prompt"],
            "predicted_label": record["predicted_label"],
            "reasons": record["reasons"],
        }
        for record in successful
        if record["expected_is_malicious"] and (not record["predicted_is_malicious"])
    ]

    return {
        "total_rows": total_count,
        "successful_requests": evaluated_count,
        "failed_requests": len(failed),
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
        "avg_endpoint_latency_ms": safe_divide(
            sum(endpoint_latency_values), len(endpoint_latency_values)
        ),
        "max_endpoint_latency_ms": max(endpoint_latency_values) if endpoint_latency_values else 0.0,
        "min_endpoint_latency_ms": min(endpoint_latency_values) if endpoint_latency_values else 0.0,
        "avg_severity_latency_ms": safe_divide(
            sum(severity_latency_values), len(severity_latency_values)
        ),
        "severity_distribution": dict(severity_counter),
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
        "request_ok",
        "http_status",
        "error",
        "prompt",
        "anomaly_score",
        "anomaly_threshold",
        "anomaly_is_anomalous",
        "classifier_label",
        "classifier_confidence",
        "classifier_is_uncertain",
        "severity_tier",
        "severity_score",
        "endpoint_latency_ms",
        "severity_latency_ms",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fieldnames if key not in {"reasons"}},
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
    endpoint: str,
    metrics: dict[str, Any],
) -> None:
    confusion = metrics["confusion_matrix"]
    false_positives = metrics["false_positives"][:10]
    false_negatives = metrics["false_negatives"][:10]
    severity_distribution = metrics["severity_distribution"]

    lines = [
        "# Prompt Endpoint Evaluation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Endpoint: `{endpoint}`",
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
        f"- Avg endpoint latency: **{metrics['avg_endpoint_latency_ms']:.2f} ms**",
        f"- Avg severity latency: **{metrics['avg_severity_latency_ms']:.2f} ms**",
        "",
        "## Confusion Matrix",
        "",
        f"- True positives: **{confusion['true_positive']}**",
        f"- True negatives: **{confusion['true_negative']}**",
        f"- False positives: **{confusion['false_positive']}**",
        f"- False negatives: **{confusion['false_negative']}**",
        "",
        "## Severity Distribution",
        "",
    ]

    if severity_distribution:
        for tier, count in sorted(severity_distribution.items()):
            lines.append(f"- {tier}: **{count}**")
    else:
        lines.append("- No severity results were returned.")

    lines.extend(["", "## Sample False Positives", ""])
    if false_positives:
        for item in false_positives:
            lines.append(
                f"- Row {item['row_id']}: predicted `{item['predicted_label']}` for `{item['prompt']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Sample False Negatives", ""])
    if false_negatives:
        for item in false_negatives:
            lines.append(
                f"- Row {item['row_id']}: predicted `{item['predicted_label']}` for `{item['prompt']}`"
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_dataset(
    rows: list[DatasetRow],
    endpoint: str,
    timeout: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(rows)

    for position, row in enumerate(rows, start=1):
        print(f"[{position}/{total}] Evaluating row {row.row_id}...", flush=True)
        try:
            started_at = time.perf_counter()
            status_code, response = post_json(endpoint, {"prompt": row.prompt}, timeout=timeout)
            endpoint_latency_ms = (time.perf_counter() - started_at) * 1000
            request_ok = 200 <= status_code < 300
            predicted_label = response.get("final_label") if request_ok else None
            predicted_is_malicious = bool(response.get("is_malicious")) if request_ok else None
            anomaly = response.get("anomaly") or {}
            classification = response.get("classification") or {}
            severity = response.get("severity") or {}
            error_message = None if request_ok else str(response.get("detail", response))

            records.append(
                {
                    "row_id": row.row_id,
                    "prompt": row.prompt,
                    "expected_label": row.expected_label,
                    "expected_is_malicious": row.expected_is_malicious,
                    "predicted_label": predicted_label,
                    "predicted_is_malicious": predicted_is_malicious,
                    "correct": (
                        predicted_is_malicious == row.expected_is_malicious if request_ok else False
                    ),
                    "request_ok": request_ok,
                    "http_status": status_code,
                    "error": error_message,
                    "anomaly_score": anomaly.get("anomaly_score"),
                    "anomaly_threshold": anomaly.get("threshold"),
                    "anomaly_is_anomalous": anomaly.get("is_anomalous"),
                    "classifier_label": classification.get("predicted_label"),
                    "classifier_confidence": classification.get("confidence"),
                    "classifier_is_uncertain": classification.get("is_uncertain"),
                    "severity_tier": severity.get("severity_tier"),
                    "severity_score": severity.get("severity_score"),
                    "endpoint_latency_ms": endpoint_latency_ms,
                    "severity_latency_ms": severity.get("latency_ms"),
                    "reasons": response.get("reasons", []),
                    "raw_response": response,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "row_id": row.row_id,
                    "prompt": row.prompt,
                    "expected_label": row.expected_label,
                    "expected_is_malicious": row.expected_is_malicious,
                    "predicted_label": None,
                    "predicted_is_malicious": None,
                    "correct": False,
                    "request_ok": False,
                    "http_status": None,
                    "error": str(exc),
                    "anomaly_score": None,
                    "anomaly_threshold": None,
                    "anomaly_is_anomalous": None,
                    "classifier_label": None,
                    "classifier_confidence": None,
                    "classifier_is_uncertain": None,
                    "severity_tier": None,
                    "severity_score": None,
                    "endpoint_latency_ms": None,
                    "severity_latency_ms": None,
                    "reasons": [],
                    "raw_response": None,
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
    print(f"Sending requests to {args.endpoint}")

    records = evaluate_dataset(rows, endpoint=args.endpoint, timeout=args.timeout)
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
        endpoint=args.endpoint,
        metrics=metrics,
    )

    print("\nEvaluation complete.")
    print(f"Detailed results: {detailed_csv_path}")
    print(f"Raw responses:    {raw_jsonl_path}")
    print(f"Summary JSON:     {summary_json_path}")
    print(f"Markdown report:  {report_md_path}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision:        {metrics['precision_malicious']:.4f}")
    print(f"Recall:           {metrics['recall_malicious']:.4f}")
    print(f"F1:               {metrics['f1_malicious']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
