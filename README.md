# Adaptive Guardrail Layer (AGL)

**A Hybrid Neural Security Filter for LLM Prompt Classification**

Defending Generative AI Systems Against Prompt Injection and Jailbreak Attacks

## Team — Group 8

- Alexander J Padin
- Joel Dievendorf
- Jack Baxter

*AAI-590 Capstone — University of San Diego, MS Applied Artificial Intelligence*

## Project Overview

AGL is a lightweight, high-performance security filter designed to intercept and classify malicious inputs before they reach a production LLM. It performs **binary classification** of user prompts:

- **Benign** — safe, normal user input
- **Malicious** — prompt injection, jailbreak attempts, or data exfiltration attempts

The system combines three components in a hybrid architecture:

1. **Fine-tuned RoBERTa classifier** (transfer learning) — captures contextual and semantic patterns in adversarial prompts
2. **Denoising deep autoencoder** (built from scratch in PyTorch) — flags out-of-distribution inputs by measuring reconstruction error against a learned benign distribution
3. **Rule-based decision module** — aggregates classifier confidence and anomaly scores into a final binary verdict

For prompts classified as malicious, a **severity evaluation pipeline** activates:

4. **Threat type classifier** (TF-IDF + Gradient Boosting) — categorizes attacks into subtypes: injection, jailbreak, exfiltration, or unknown
5. **Severity scoring engine** — computes a weighted severity score (critical/high/medium/low) from classifier confidence, anomaly margin, threat type risk, and text heuristics
6. **Threat intelligence service** — maps detected threats to MITRE ATLAS techniques and OWASP Top 10 for LLM Applications with recommended remediation actions

## System Diagram

![High-level AGL architecture](docs/diagram.png)

## Key Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.952 |
| Malicious Precision | 0.912 |
| Malicious Recall | 1.000 |
| F1 Score | 0.954 |
| False Negatives | 0 |
| Avg Latency | ~102 ms/prompt |
| Avg Severity Latency | ~0.06 ms |

Evaluated on a 2,000-prompt validation dataset. The hybrid architecture achieves **perfect recall** for malicious prompts (zero false negatives), which is critical for security applications where missed threats can lead to system compromise. Full evaluation report: `results/prompt_endpoint_eval/`.

## Validation Workflows

The repository includes multiple validation approaches so AGL can be compared against simpler baselines and an external LLM refusal baseline:

- `AGL` — end-to-end endpoint validation of the proposed solution
- `Rule-Based` — deterministic regex-style filter
- `Keyword` — deterministic keyword filter
- `LLM Refusal` — OpenAI API answer-vs-refusal baseline

Validation scripts live in `scripts/`, and the recommended step-by-step usage guide is:

- [scripts/VALIDATION_README.md](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/VALIDATION_README.md)

That guide covers:

- how to run each validation
- where each validation writes its results
- how to generate one comparison summary across the latest runs

## Repository Structure

```
aai-590-group-8-capstone/
├── artifacts/                # Local experiment outputs and training artifacts
├── data/
│   ├── raw/                  # Downloaded source datasets (gitignored)
│   └── processed/            # Cleaned, balanced, split data (gitignored)
├── src/                      # Training & evaluation pipeline
│   ├── run.py                # Unified CLI entry point
│   ├── config.py             # Paths, hyperparameters, label maps
│   ├── data/
│   │   ├── build_dataset.py          # Read CSV, dedup, balance, stratified split
│   │   └── tokenize_dataset.py       # RoBERTa tokenization
│   ├── models/
│   │   ├── classifier.py            # RoBERTa sequence classifier
│   │   ├── anomaly_detector.py      # Anomaly detection module
│   │   └── agl_pipeline.py          # Full hybrid inference pipeline
│   ├── training/
│   │   ├── train.py              # Training loops (classifier, anomaly, both)
│   │   └── callbacks.py          # Metrics logging callback
│   ├── evaluation/
│   │   ├── metrics.py            # P/R/F1, confusion matrix, latency
│   │   ├── baselines.py          # Keyword blocklist, TF-IDF/SVM baselines
│   │   └── visualizations.py     # Plots and figures
│   └── utils/
│       └── reproducibility.py    # Seed setting
├── prompt-security-app/      # Deployed web application (FastAPI + UI)
│   ├── app/
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── api/routes.py         # API endpoints
│   │   ├── ml/
│   │   │   ├── inference.py      # Anomaly detection inference service
│   │   │   ├── classification.py # RoBERTa classification service
│   │   │   ├── decision.py       # Rule-based decision aggregation
│   │   │   ├── severity.py       # Severity scoring orchestration
│   │   │   ├── threat_classifier.py  # TF-IDF + Gradient Boosting threat subtyping
│   │   │   ├── threat_intel.py   # MITRE ATLAS / OWASP threat intelligence
│   │   │   ├── feature_engineering.py  # Feature pipeline for anomaly detection
│   │   │   ├── model.py          # Denoising autoencoder architecture
│   │   │   └── artifacts.py      # Model artifact loading
│   │   ├── schemas/              # Pydantic request/response models
│   │   └── static/               # Frontend UI (HTML/CSS/JS)
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── data_pipeline/            # Data fetching, cleaning, EDA, feature engineering
│   ├── experimentation/          # Model prototyping and training experiments
│   ├── train_roberta_heavy.ipynb # Final RoBERTa training notebook
│   └── AGL_denoising_autoencoder.ipynb  # Autoencoder training notebook
├── models/                   # Local trained/inference artifacts (gitignored in a fresh clone)
│   ├── anomaly_detection/        # Autoencoder checkpoints and thresholds
│   ├── classifier/               # RoBERTa checkpoints and tokenizer files
│   ├── feature_engineering/      # PCA / feature pipeline exports
│   └── severity/threat_intel/    # Threat intelligence mapping
├── results/                  # Evaluation outputs and figures
│   └── prompt_endpoint_eval/     # Automated validation reports
├── docs/                     # Project documents and report drafts
├── scripts/
│   ├── VALIDATION_README.md        # How to run all validations and compare them
│   ├── evaluate_prompt_endpoint.py # AGL endpoint validation
│   ├── evaluate_rule_based_filter.py # Deterministic rule-based baseline
│   ├── evaluate_keyword_filter.py  # Deterministic keyword baseline
│   ├── evaluate_llm_refusal.py     # OpenAI refusal baseline
│   ├── summarize_latest_validation_results.py # Cross-validation comparison report
│   └── download_datasets.py        # Dataset download helper
└── requirements.txt          # Python dependencies
```

## Setup

### Training and Evaluation Pipeline

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Use `python3` throughout. The repository has been run locally with Python 3.11+ and the app Docker image uses Python 3.11.

### Web Application

```bash
cd prompt-security-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Model artifacts must be placed in the repository-level `models/` directory. See `prompt-security-app/README.md` for the expected artifact structure.

### Deployment Notes

- The supported development path is the local FastAPI startup flow above.
- A Dockerfile is included in `prompt-security-app/`, but it requires the `models/` artifacts to be available in the image build context and may need path adjustments depending on how you build or deploy the container.
- If you deploy with Docker, verify that the container includes both the application code and the expected `models/` directory layout before starting the API.

## Usage

### CLI Pipeline

The recommended workflow for the ML pipeline is to run the notebooks in `notebooks/data_pipeline/` in order:

1. `0_data_fetching.ipynb`
2. `1_data_cleaning.ipynb`
3. `2_eda.ipynb`
4. `3_feature_engineering.ipynb`

Dataset setup requirements:

- Download the required source datasets first and place any manual files in `data/raw/`.
- Use `notebooks/experimentation/datasets_exploration.ipynb` as the reference for dataset source links and endpoints.
- Some datasets are pulled automatically from Hugging Face inside the notebooks or supporting code.
- The following datasets are not pulled automatically and must be downloaded manually into `data/raw/`:
  - `MPDD.csv`
    - Dataset: `MPDD`
    - Source: `https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd`
  - `Prompt_INJECTION_And_Benign_DATASET.jsonl`
    - Dataset: `Prompt Injection & Benign Prompt Dataset`
    - Source: `https://www.kaggle.com/datasets/cyberprince/prompt-injection-and-benign-prompt-dataset`

Hugging Face API setup:

```bash
# Create a Hugging Face token:
# https://huggingface.co/docs/hub/en/security-tokens

echo "HF_TOKEN=<your-key>" > .env
```

The notebooks load the token from `.env` and authenticate with the Hugging Face API using:

```python
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
login(hf_token)
```

After the data pipeline notebooks have produced the processed datasets and features, you can use run the training notebooks:

Primary training notebooks:

- `notebooks/AGL_denoising_autoencoder.ipynb` trains the denoising autoencoder.
- `notebooks/train_roberta_heavy_with_api_exports.ipynb` trains the RoBERTa classifier and exports the API-ready artifacts.

These are the main notebooks to use for model training. The notebooks under `notebooks/data_pipeline/` are for dataset preparation and feature engineering, not final model training.

### Web Application API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health and model status |
| `/api/v1/predict` | POST | Anomaly detection (autoencoder) |
| `/api/v1/classify` | POST | RoBERTa classification |
| `/api/v1/decision` | POST | Combined decision from both models |
| `/api/v1/prompt` | POST | End-to-end analysis (anomaly + classification + decision + severity) |

The web UI is served at `http://127.0.0.1:8000/` and provides an interactive interface for submitting prompts and visualizing model outputs in real time.

## Web Application UI

![Prompt Security web UI screenshot](docs/UI.png)

## Datasets

The composite dataset contains approximately **104,000 labeled examples** aggregated from seven publicly available adversarial prompt corpora, structured as a binary classification task (benign vs malicious):

| Source | Description |
|--------|-------------|
| WildGuardMix (Allen AI, 2024) | Benign + adversarial prompts |
| Prompt Injection and Benign Prompt Dataset (CyberPrince, 2024) | Injection + benign |
| Malicious Prompt Detection Dataset (Jebbar, 2024) | Malicious prompt detection |
| Malicious LLM Prompts v1 & v4 (Codesagar, 2024) | Malicious prompt variants |
| deepset/prompt-injections (deepset, 2023) | Benign + injection (662 samples) |
| Jailbreak Classification (Hao, 2023) | Jailbreak classification |

The aggregated dataset exhibits a relatively balanced class distribution (~55K benign, ~50K malicious). Hash-based deduplication and fuzzy matching were applied to remove duplicates.

## Model Architecture

### RoBERTa Classifier (Transfer Learning)

Fine-tuned `roberta-base` (125M parameters) for binary sequence classification. Trained with AdamW optimizer, cross-entropy loss, learning rate 2e-5, batch size 32, max sequence length 128, for 3 epochs with 10% warmup and early stopping.

### Denoising Deep Autoencoder (Built From Scratch)

Fully connected encoder-decoder network built from scratch in PyTorch, trained exclusively on benign prompts. The autoencoder learns a compressed representation of normal prompt behavior and flags deviations via reconstruction error (MSE). Input features include:

- Lexical/structural indicators (token counts, punctuation ratios, whitespace, non-ASCII characters)
- Phrase-based indicators (instruction override, roleplay, payload requests, social engineering, obfuscation)
- Dense semantic embeddings (sentence-transformer + PCA reduction)

### Decision Module

Rule-based aggregation that combines classifier confidence and anomaly scores into a cumulative maliciousness score, providing an interpretable mechanism for resolving agreement, disagreement, and uncertainty between models.

### Severity Pipeline (Malicious Prompts Only)

When both models agree a prompt is malicious, a post-hoc severity pipeline activates:

1. **Threat type classification** — TF-IDF + Gradient Boosting classifier assigns one of four subtypes (injection, jailbreak, exfiltration, unknown_malicious), with a heuristic regex fallback when model artifacts are unavailable
2. **Severity scoring** — Weighted composite score (0-10) from classifier confidence, anomaly margin, threat type risk weight, and text heuristics (URL presence, encoding patterns). Mapped to tiers: critical (8+), high (5-7), medium (3-4), low (0-2)
3. **Threat intelligence** — Structured lookup returning MITRE ATLAS technique IDs, OWASP LLM Top 10 mappings, and recommended remediation actions for each threat type

## Technologies

- **Python 3.11+**, PyTorch 2.x, Transformers 4.x
- **FastAPI** + Uvicorn for the API backend
- **scikit-learn** for baselines and preprocessing
- **sentence-transformers** for semantic embeddings
- **pandas**, **numpy**, **matplotlib**, **seaborn** for data analysis and visualization

## AI Use Disclosure

This project made use of AI-assisted tools during development:

- **Claude Code (Anthropic)** — Used for code scaffolding, debugging assistance, refactoring, documentation drafting, and iterative development of the training pipeline, evaluation harness, and web application. AI suggestions were reviewed, tested, and modified by team members before integration.
- **ChatGPT / Codex (OpenAI)** — Used for research support, brainstorming, code assistance, and drafting portions of the written report.
- **Gemini (Google)** — Used for select development and research tasks.

All AI-generated content was critically reviewed by team members. Model architecture decisions, experimental design, data analysis, and final interpretations are the original work of the team. No AI tool was used to generate evaluation results or fabricate data.

#### License
This project is licensed under the GNU License - see the [LICENSE.txt](LICENSE.txt) file for details.
