# AGL — Adaptive Guardrail Layer

## Commands

```bash
# Setup
pip install -r requirements.txt

# Unified CLI (MUST use python3 — `python` is Python 2.7 on this machine)
python3 -m src.run --stage data [--csv PATH]
python3 -m src.run --stage train --mode classifier|anomaly|both
python3 -m src.run --stage evaluate [--checkpoint PATH]
python3 -m src.run --stage demo --text "..."
```

Data pipeline CSV (`data/processed/dataset_cleaned.csv`) is produced by `notebooks/data_pipeline/` on Google Colab. The `--stage data` command reads this CSV, deduplicates, balances, and splits into train/val/test parquets.

## Architecture

**Binary classification:** Benign(0) vs Malicious(1) — defined in `src/config.py`

Two modeling approaches (for comparison in the final report):
1. **Fine-tuned RoBERTa** (roberta-base, 125M params) — transfer learning (Joel)
2. **From-scratch DL model** — built with PyTorch/Keras (Alex, required by syllabus)
3. **TF-IDF + SVM** — traditional ML baseline

Optional: Mahalanobis OOD detector on [CLS] embeddings (PCA to 100 dims).

**Data pipeline flow:** `load_dataset_csv` (read Alex's cleaned CSV) → `build_dataset` (dedup/balance/split) → `tokenize_dataset`

**Key modules:**
- `src/config.py` — single source of truth for paths, hyperparameters, label maps
- `src/data/build_dataset.py` — dataset construction pipeline (reads from CSV)
- `src/data/load_datasets.py` — CSV loader
- `src/training/train.py` — `WeightedTrainer` (custom HF Trainer subclass with class-weighted CE loss)
- `src/models/agl_pipeline.py` — `AGLPipeline`, end-to-end inference class
- `src/evaluation/` — metrics (F1, ROC-AUC, PR-AUC), baselines, visualizations
- `notebooks/data_pipeline/` — Alex's data fetching, cleaning, EDA, feature engineering

Notebooks are designed for Google Colab with T4 GPU.

## Conventions

- All hyperparameters are centralized in `src/config.py`, not scattered across files
- Data pipeline reads from `data/processed/dataset_cleaned.csv` (binary labels)
- Processed splits: `data/processed/{train,val,test}.parquet` (gitignored)
- Model checkpoints: `models/` (gitignored)
- Results/figures: `results/` and `results/figures/`
- `data/`, `models/`, `*.parquet`, `*.csv` are all gitignored
