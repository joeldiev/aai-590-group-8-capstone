# Scripts

This folder contains utility scripts for the project.

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
