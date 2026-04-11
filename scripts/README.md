# Scripts

This folder contains utility scripts for the project.

For the full validation workflow, including how to run each evaluator and how to combine them into one comparison report, see [VALIDATION_README.md](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/VALIDATION_README.md).

## Evaluate `/api/v1/prompt`

Use [`evaluate_prompt_endpoint.py`](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/evaluate_prompt_endpoint.py) to test the FastAPI prompt-security endpoint against a labeled dataset and generate an accuracy report.

### What it does

- Reads labeled prompts from `data/processed/validation_dataset.csv`
- Sends each prompt to `POST /api/v1/prompt`
- Stores per-prompt results and raw API responses
- Computes summary metrics such as:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix
  - severity distribution

### Prerequisites

Before running the script:

1. Start the Prompt Security API.
2. Make sure the server is available at `http://127.0.0.1:8000`.
3. Make sure the model artifacts are loaded correctly.

Example:

```bash
cd prompt-security-app
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the evaluator

From the repository root:

```bash
python3 scripts/evaluate_prompt_endpoint.py \
  --dataset data/processed/validation_dataset.csv \
  --endpoint http://127.0.0.1:8000/api/v1/prompt
```

### Optional arguments

```bash
python3 scripts/evaluate_prompt_endpoint.py --help
```

Available options:

- `--dataset`: path to the labeled CSV file
- `--endpoint`: full URL for the API endpoint
- `--output-dir`: directory where evaluation outputs are written
- `--timeout`: request timeout in seconds
- `--limit`: evaluate only the first N rows

### Output files

Each run creates a timestamped folder under:

```text
results/prompt_endpoint_eval/<timestamp>/
```

Generated files:

- `detailed_results.csv`: flattened row-by-row evaluation results
- `raw_results.jsonl`: raw endpoint responses for each prompt
- `summary.json`: aggregate metrics and confusion matrix
- `report.md`: human-readable evaluation report

### Quick test

To test the script on a smaller sample first:

```bash
python3 scripts/evaluate_prompt_endpoint.py --limit 10
```

## Evaluate A Deterministic Rule-Based Filter

Use [`evaluate_rule_based_filter.py`](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/evaluate_rule_based_filter.py) to score the same labeled validation prompts with a conservative, deterministic filter instead of the FastAPI model stack.

### What it does

- Reads labeled prompts from `data/processed/validation_dataset.csv`
- Applies a weighted set of explicit regex rules for:
  - instruction override attempts
  - hidden prompt / policy exfiltration
  - jailbreak phrasing
  - harmful payload requests
  - obfuscation / evasion signals
- Produces row-level results with matched rules, score, and latency
- Computes summary metrics such as:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix

### Run the evaluator

From the repository root:

```bash
python3 scripts/evaluate_rule_based_filter.py \
  --dataset data/processed/validation_dataset.csv
```

### Optional arguments

```bash
python3 scripts/evaluate_rule_based_filter.py --help
```

Available options:

- `--dataset`: path to the labeled CSV file
- `--output-dir`: directory where evaluation outputs are written
- `--limit`: evaluate only the first N rows
- `--threshold`: minimum rule score required to mark a prompt as malicious

### Output files

Each run creates a timestamped folder under:

```text
results/rule_base_eval/<timestamp>/
```

Generated files:

- `detailed_results.csv`: flattened row-by-row evaluation results
- `raw_results.jsonl`: raw per-prompt detection outputs
- `summary.json`: aggregate metrics and confusion matrix
- `report.md`: human-readable evaluation report

## Evaluate LLM Refusal Behavior

Use [`evaluate_llm_refusal.py`](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/evaluate_llm_refusal.py) to send the validation prompts to the OpenAI API and evaluate whether the model answered or refused each prompt.

### What it does

- Reads labeled prompts from `data/processed/validation_dataset.csv`
- Loads `OPENAI_API_KEY` from `notebooks/.env` by default
- Calls the OpenAI Responses API for each prompt
- Prepends a neutral system guardrail prompt before each validation prompt
- Detects refusal behavior from the typed API response content
- Interprets:
  - refused => malicious
  - answered => benign
- Produces row-level results with the answer text, refusal text, latency, and raw API response
- Computes summary metrics such as:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix

### Run the evaluator

From the repository root:

```bash
python3 scripts/evaluate_llm_refusal.py \
  --dataset data/processed/validation_dataset.csv
```

### Optional arguments

```bash
python3 scripts/evaluate_llm_refusal.py --help
```

Available options:

- `--dataset`: path to the labeled CSV file
- `--output-dir`: directory where evaluation outputs are written
- `--env-file`: env file containing `OPENAI_API_KEY`
- `--model`: OpenAI model to call
- `--timeout`: request timeout in seconds
- `--max-output-tokens`: response token cap per request
- `--system-prompt`: system guardrail prompt prepended before each validation prompt
- `--limit`: evaluate only the first N rows

### Output files

Each run creates a timestamped folder under:

```text
results/llm_eval/<timestamp>/
```

Generated files:

- `detailed_results.csv`: flattened row-by-row evaluation results
- `raw_results.jsonl`: raw API responses for each prompt
- `summary.json`: aggregate metrics and confusion matrix
- `report.md`: human-readable evaluation report

## Summarize The Latest Validation Results

Use [`summarize_latest_validation_results.py`](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/summarize_latest_validation_results.py) to compare the latest available run for each validation approach.

### What it does

- Finds the newest timestamped run under:
  - `results/prompt_endpoint_eval/`
  - `results/rule_base_eval/`
  - `results/keyword_eval/`
  - `results/llm_eval/`
- Treats the prompt endpoint evaluation as **AGL**
- Normalizes the main comparison metrics:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix
  - average / min / max latency
- Writes a combined comparison report

### Run the summarizer

From the repository root:

```bash
python3 scripts/summarize_latest_validation_results.py
```

### Output files

Each run writes files under:

```text
results/validation_comparison/
```

Generated files:

- `latest_validation_comparison_<timestamp>.json`
- `latest_validation_comparison_<timestamp>.md`

## Evaluate A Deterministic Keyword Filter

Use [`evaluate_keyword_filter.py`](/Users/apadin/Desktop/Capstone/aai-590-group-8-capstone/scripts/evaluate_keyword_filter.py) to score the same labeled validation prompts with a simpler keyword-matching baseline.

### What it does

- Reads labeled prompts from `data/processed/validation_dataset.csv`
- Applies a weighted set of explicit keywords across categories such as:
  - instruction override
  - exfiltration
  - jailbreak
  - payload
  - obfuscation
- Produces row-level results with matched keywords, score, and latency
- Computes summary metrics such as:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix

### Run the evaluator

From the repository root:

```bash
python3 scripts/evaluate_keyword_filter.py \
  --dataset data/processed/validation_dataset.csv
```

### Optional arguments

```bash
python3 scripts/evaluate_keyword_filter.py --help
```

Available options:

- `--dataset`: path to the labeled CSV file
- `--output-dir`: directory where evaluation outputs are written
- `--limit`: evaluate only the first N rows
- `--threshold`: minimum keyword score required to mark a prompt as malicious

### Output files

Each run creates a timestamped folder under:

```text
results/keyword_eval/<timestamp>/
```

Generated files:

- `detailed_results.csv`: flattened row-by-row evaluation results
- `raw_results.jsonl`: raw per-prompt detection outputs
- `summary.json`: aggregate metrics and confusion matrix
- `report.md`: human-readable evaluation report
