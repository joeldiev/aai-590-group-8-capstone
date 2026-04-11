# Validation Guide

This guide shows how to run each validation workflow in the repository and how to generate a single comparison report across all of them.

The validation approaches currently supported are:

- `AGL` — the proposed solution, evaluated through the deployed `/api/v1/prompt` endpoint
- `Rule-Based` — deterministic regex-style malicious prompt detection
- `Keyword` — deterministic keyword-matching malicious prompt detection
- `LLM Detection` — OpenAI API baseline based on answer-vs-refusal behavior

All validation scripts assume the labeled dataset is available at:

```text
data/processed/validation_dataset.csv
```

## Validation Outputs

Each evaluator writes timestamped outputs under its own folder:

- `results/prompt_endpoint_eval/<timestamp>/`
- `results/rule_base_eval/<timestamp>/`
- `results/keyword_eval/<timestamp>/`
- `results/llm_eval/<timestamp>/`

Typical files produced by each validation run:

- `detailed_results.csv`
- `raw_results.jsonl`
- `summary.json`
- `report.md`

The comparison script writes to:

- `results/validation_comparison/`

## 1. Run The AGL Validation

This is the evaluation of the proposed AGL solution.

Start the Prompt Security API first:

```bash
cd prompt-security-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then, from the repository root in a separate terminal:

```bash
python3 scripts/evaluate_prompt_endpoint.py \
  --dataset data/processed/validation_dataset.csv \
  --endpoint http://127.0.0.1:8000/api/v1/prompt
```

Quick smoke test:

```bash
python3 scripts/evaluate_prompt_endpoint.py --limit 10
```

## 2. Run The Rule-Based Validation

```bash
python3 scripts/evaluate_rule_based_filter.py \
  --dataset data/processed/validation_dataset.csv
```

Optional threshold override:

```bash
python3 scripts/evaluate_rule_based_filter.py \
  --dataset data/processed/validation_dataset.csv \
  --threshold 4
```

Quick smoke test:

```bash
python3 scripts/evaluate_rule_based_filter.py --limit 10
```

## 3. Run The Keyword Validation

```bash
python3 scripts/evaluate_keyword_filter.py \
  --dataset data/processed/validation_dataset.csv
```

Optional threshold override:

```bash
python3 scripts/evaluate_keyword_filter.py \
  --dataset data/processed/validation_dataset.csv \
  --threshold 4
```

Quick smoke test:

```bash
python3 scripts/evaluate_keyword_filter.py --limit 10
```

## 4. Run The LLM Refusal Validation

This baseline calls the OpenAI Responses API and classifies outcomes as:

- `refused => malicious`
- `answered => benign`

By default, it loads `OPENAI_API_KEY` from:

```text
notebooks/.env
```

Run it like this:

```bash
python3 scripts/evaluate_llm_refusal.py \
  --dataset data/processed/validation_dataset.csv
```

Example with explicit model selection:

```bash
python3 scripts/evaluate_llm_refusal.py \
  --dataset data/processed/validation_dataset.csv \
  --model gpt-4.1-nano
```

Example with a custom guardrail system prompt:

```bash
python3 scripts/evaluate_llm_refusal.py \
  --dataset data/processed/validation_dataset.csv \
  --system-prompt "You are a helpful assistant. Refuse unsafe or out-of-scope requests."
```

Quick smoke test:

```bash
python3 scripts/evaluate_llm_refusal.py --limit 10
```

## 5. Summarize All Validation Results Into One Comparison Report

After at least one run exists for each approach, generate the latest cross-approach summary:

```bash
python3 scripts/summarize_latest_validation_results.py
```

This script:

- finds the newest run in each validation folder
- treats `prompt_endpoint_eval` as `AGL`
- compares metrics and latency across all approaches
- writes:
  - `results/validation_comparison/latest_validation_comparison_<timestamp>.json`
  - `results/validation_comparison/latest_validation_comparison_<timestamp>.md`

## Recommended End-To-End Workflow

1. Run the AGL endpoint validation.
2. Run the rule-based validation.
3. Run the keyword validation.
4. Run the LLM refusal validation.
5. Run the comparison summarizer.

## Related Files

- [scripts/README.md](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/README.md)
- [scripts/evaluate_prompt_endpoint.py](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/evaluate_prompt_endpoint.py)
- [scripts/evaluate_rule_based_filter.py](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/evaluate_rule_based_filter.py)
- [scripts/evaluate_keyword_filter.py](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/evaluate_keyword_filter.py)
- [scripts/evaluate_llm_refusal.py](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/evaluate_llm_refusal.py)
- [scripts/summarize_latest_validation_results.py](https://github.com/joeldiev/aai-590-group-8-capstone/blob/main/scripts/summarize_latest_validation_results.py)
