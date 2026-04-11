#!/usr/bin/env python3
"""
Summarize the latest validation results across all evaluation approaches.

The script looks for the newest timestamped run under each known validation
directory, loads its summary metrics, and writes a comparison report.

Current approaches:
    - AGL (prompt endpoint validation)
    - Rule-based filter
    - Keyword filter
    - LLM refusal baseline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "results/validation_comparison"


@dataclass(frozen=True)
class ValidationSource:
    label: str
    directory_name: str
    summary_kind: str


SOURCES: tuple[ValidationSource, ...] = (
    ValidationSource(label="AGL", directory_name="prompt_endpoint_eval", summary_kind="endpoint"),
    ValidationSource(label="Rule-Based", directory_name="rule_base_eval", summary_kind="filter"),
    ValidationSource(label="Keyword", directory_name="keyword_eval", summary_kind="filter"),
    ValidationSource(label="LLM Detection", directory_name="llm_eval", summary_kind="llm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the latest validation results across AGL and baseline approaches.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Base results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the comparison report should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def find_latest_summary(results_dir: Path, source: ValidationSource) -> tuple[Path, dict[str, Any]] | None:
    base_dir = results_dir / source.directory_name
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    candidate_dirs = sorted(
        [
            child
            for child in base_dir.iterdir()
            if child.is_dir() and (child / "summary.json").exists()
        ],
        key=lambda path: path.name,
    )
    if not candidate_dirs:
        return None

    latest_dir = candidate_dirs[-1]
    summary_path = latest_dir / "summary.json"
    return latest_dir, json.loads(summary_path.read_text(encoding="utf-8"))


def metric_or_none(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return float(value)
    return None


def int_or_zero(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    return int(value)


def format_float(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def format_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} ms"


def build_normalized_record(source: ValidationSource, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    accuracy = metric_or_none(summary, "accuracy")
    precision = metric_or_none(summary, "precision_malicious")
    recall = metric_or_none(summary, "recall_malicious")
    f1 = metric_or_none(summary, "f1_malicious")
    avg_latency = metric_or_none(summary, "avg_endpoint_latency_ms", "avg_latency_ms")
    min_latency = metric_or_none(summary, "min_endpoint_latency_ms", "min_latency_ms")
    max_latency = metric_or_none(summary, "max_endpoint_latency_ms", "max_latency_ms")

    confusion = summary.get("confusion_matrix", {})

    return {
        "approach": source.label,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "report_path": str(run_dir / "report.md"),
        "total_rows": int_or_zero(summary, "total_rows"),
        "successful_requests": int_or_zero(summary, "successful_requests"),
        "failed_requests": int_or_zero(summary, "failed_requests"),
        "accuracy": accuracy,
        "precision_malicious": precision,
        "recall_malicious": recall,
        "f1_malicious": f1,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "true_positive": int(confusion.get("true_positive", 0)),
        "true_negative": int(confusion.get("true_negative", 0)),
        "false_positive": int(confusion.get("false_positive", 0)),
        "false_negative": int(confusion.get("false_negative", 0)),
    }


def compare_against_agl(records: list[dict[str, Any]]) -> list[str]:
    agl = next((record for record in records if record["approach"] == "AGL"), None)
    if agl is None:
        return ["- AGL results were not found, so no baseline comparison could be computed."]

    lines: list[str] = []
    for record in records:
        if record["approach"] == "AGL":
            continue

        delta_accuracy = (
            None
            if agl["accuracy"] is None or record["accuracy"] is None
            else record["accuracy"] - agl["accuracy"]
        )
        delta_f1 = (
            None
            if agl["f1_malicious"] is None or record["f1_malicious"] is None
            else record["f1_malicious"] - agl["f1_malicious"]
        )
        delta_recall = (
            None
            if agl["recall_malicious"] is None or record["recall_malicious"] is None
            else record["recall_malicious"] - agl["recall_malicious"]
        )
        latency_ratio = (
            None
            if agl["avg_latency_ms"] in (None, 0.0) or record["avg_latency_ms"] is None
            else record["avg_latency_ms"] / agl["avg_latency_ms"]
        )

        comparison = (
            f"- {record['approach']}: "
            f"accuracy delta vs AGL = {format_float(delta_accuracy)}, "
            f"F1 delta = {format_float(delta_f1)}, "
            f"recall delta = {format_float(delta_recall)}, "
            f"avg latency ratio vs AGL = {format_float(latency_ratio)}x"
        )
        lines.append(comparison)

    return lines


def build_markdown_report(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Latest Validation Comparison Report",
        "",
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "- Proposed solution: `AGL`",
        "",
        "## Latest Runs",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"### {record['approach']}",
                "",
                f"- Run directory: `{record['run_dir']}`",
                f"- Summary: `{record['summary_path']}`",
                f"- Report: `{record['report_path']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Comparison",
            "",
            "| Approach | Accuracy | Precision | Recall | F1 | Avg Latency | Min Latency | Max Latency | TP | TN | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["approach"],
                    format_float(record["accuracy"]),
                    format_float(record["precision_malicious"]),
                    format_float(record["recall_malicious"]),
                    format_float(record["f1_malicious"]),
                    format_latency(record["avg_latency_ms"]),
                    format_latency(record["min_latency_ms"]),
                    format_latency(record["max_latency_ms"]),
                    str(record["true_positive"]),
                    str(record["true_negative"]),
                    str(record["false_positive"]),
                    str(record["false_negative"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Comparison Vs AGL", ""])
    lines.extend(compare_against_agl(records))

    if records:
        best_accuracy = max(records, key=lambda record: record["accuracy"] if record["accuracy"] is not None else -1.0)
        best_f1 = max(records, key=lambda record: record["f1_malicious"] if record["f1_malicious"] is not None else -1.0)
        fastest = min(records, key=lambda record: record["avg_latency_ms"] if record["avg_latency_ms"] is not None else float("inf"))

        lines.extend(
            [
                "",
                "## Highlights",
                "",
                f"- Best accuracy: **{best_accuracy['approach']}** at **{format_float(best_accuracy['accuracy'])}**",
                f"- Best malicious F1: **{best_f1['approach']}** at **{format_float(best_f1['f1_malicious'])}**",
                f"- Lowest average latency: **{fastest['approach']}** at **{format_latency(fastest['avg_latency_ms'])}**",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source in SOURCES:
        latest = find_latest_summary(results_dir, source)
        if latest is None:
            print(f"Skipping {source.label}: no summary.json found in {results_dir / source.directory_name}")
            continue
        run_dir, summary = latest
        records.append(build_normalized_record(source, run_dir, summary))

    if not records:
        print("No validation summaries were found.")
        return 1

    generated_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"latest_validation_comparison_{generated_at}.json"
    md_path = output_dir / f"latest_validation_comparison_{generated_at}.md"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    build_markdown_report(records, md_path)

    print("Comparison complete.")
    print(f"JSON summary:    {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
