# Adaptive Guardrail Layer (AGL)

**A Multi-Stage Neural Classifier for LLM Security**

Defending Generative AI Systems Against Prompt Injection and Jailbreak Attacks

## Team — Group 8

- Alexander J Padin
- Joel Dievendorf
- Jack Baxter

*AAI-590 Capstone — University of San Diego, MS Applied Artificial Intelligence*

## Project Overview

AGL is a lightweight, high-performance security filter designed to intercept and classify malicious inputs before they reach a production LLM. It categorizes user prompts into four risk levels:

- **Benign** — safe, normal user input
- **Injection** — attempts to override system instructions
- **Jailbreak** — attempts to bypass safety/content filters
- **Exfiltration** — attempts to extract proprietary data or system prompts

## Repository Structure

```
├── data/
│   ├── raw/            # Original downloaded datasets
│   └── processed/      # Cleaned, merged, and labeled data
├── notebooks/          # Jupyter notebooks (EDA, training, analysis)
├── src/                # Python source code
├── models/             # Trained model artifacts
├── results/            # Evaluation results, figures, metrics
├── docs/               # Project documents, report drafts
└── TODO.md             # Project task tracking
```

## Setup

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn jupyter
```

## Datasets

See [TODO.md](TODO.md) for the full list of candidate datasets and download status.
