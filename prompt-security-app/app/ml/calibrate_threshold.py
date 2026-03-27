from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml.inference import InferenceService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate reconstruction_mse threshold for anomaly API path."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV/JSONL/Parquet with prompt data.",
    )
    parser.add_argument(
        "--text-column",
        default="prompt",
        help="Column containing prompt text (default: prompt).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional label column for supervised calibration.",
    )
    parser.add_argument(
        "--benign-values",
        default="0,benign,normal,ham,false",
        help="Comma-separated label values treated as benign when label-column is provided.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target false-positive rate on benign set (default: 0.05).",
    )
    parser.add_argument(
        "--unlabeled-quantile",
        type=float,
        default=0.95,
        help="Quantile used when labels are not provided (default: 0.95).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max rows to score for calibration (default: 5000).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output threshold JSON path. Defaults to THRESHOLD_PATH from settings.",
    )
    return parser.parse_args()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use CSV/JSONL/Parquet.")


def _to_bool_like(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    return None


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = _load_table(input_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column `{args.text_column}` in input data.")

    df = df.dropna(subset=[args.text_column]).copy()
    if df.empty:
        raise ValueError("No non-null prompts found after filtering.")

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)

    svc = InferenceService(settings=settings)
    svc.load()

    scores: list[float] = []
    for prompt in df[args.text_column].astype(str).tolist():
        score, _, _ = svc.score_prompt(prompt)
        scores.append(score)

    score_arr = np.asarray(scores, dtype=np.float64)
    threshold = None
    method = ""

    if args.label_column and args.label_column in df.columns:
        benign_values = {v.strip().lower() for v in args.benign_values.split(",") if v.strip()}
        benign_mask = []
        for raw in df[args.label_column].tolist():
            if raw is None:
                benign_mask.append(False)
                continue
            sval = str(raw).strip().lower()
            parsed_bool = _to_bool_like(sval)
            if parsed_bool is not None:
                benign_mask.append(parsed_bool is False)
            else:
                benign_mask.append(sval in benign_values)

        benign_mask_arr = np.asarray(benign_mask, dtype=bool)
        benign_scores = score_arr[benign_mask_arr]
        if benign_scores.size == 0:
            raise ValueError("No benign rows detected for supervised calibration.")

        q = float(np.clip(1.0 - args.target_fpr, 0.0, 1.0))
        threshold = float(np.quantile(benign_scores, q))
        method = f"benign_quantile@{q:.4f}"
    else:
        q = float(np.clip(args.unlabeled_quantile, 0.0, 1.0))
        threshold = float(np.quantile(score_arr, q))
        method = f"unlabeled_quantile@{q:.4f}"

    out_path = Path(args.output).resolve() if args.output else settings.threshold_path
    payload = {
        "threshold": threshold,
        "score_name": "reconstruction_mse",
        "calibration_method": method,
        "sample_count": int(score_arr.size),
        "mean_score": float(np.mean(score_arr)),
        "std_score": float(np.std(score_arr)),
        "min_score": float(np.min(score_arr)),
        "max_score": float(np.max(score_arr)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved calibrated threshold -> {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
