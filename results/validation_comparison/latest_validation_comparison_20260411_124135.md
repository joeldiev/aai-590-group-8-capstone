# Latest Validation Comparison Report

- Generated at (UTC): `2026-04-11T19:41:35.744524+00:00`
- Proposed solution: `AGL`

## Latest Runs

### AGL

- Run directory: `results/prompt_endpoint_eval/20260329_153350`
- Summary: `results/prompt_endpoint_eval/20260329_153350/summary.json`
- Report: `results/prompt_endpoint_eval/20260329_153350/report.md`

### Rule-Based

- Run directory: `results/rule_base_eval/20260411_105350`
- Summary: `results/rule_base_eval/20260411_105350/summary.json`
- Report: `results/rule_base_eval/20260411_105350/report.md`

### Keyword

- Run directory: `results/keyword_eval/20260411_105952`
- Summary: `results/keyword_eval/20260411_105952/summary.json`
- Report: `results/keyword_eval/20260411_105952/report.md`

### LLM Detection

- Run directory: `results/llm_eval/20260411_120327`
- Summary: `results/llm_eval/20260411_120327/summary.json`
- Report: `results/llm_eval/20260411_120327/report.md`

## Metric Comparison

| Approach | Accuracy | Precision | Recall | F1 | Avg Latency | Min Latency | Max Latency | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AGL | 0.9520 | 0.9124 | 1.0000 | 0.9542 | 101.81 ms | 72.49 ms | 496.39 ms | 1000 | 904 | 96 | 0 |
| Rule-Based | 0.7665 | 1.0000 | 0.5330 | 0.6954 | 0.07 ms | 0.01 ms | 22.23 ms | 533 | 1000 | 0 | 467 |
| Keyword | 0.7800 | 1.0000 | 0.5600 | 0.7179 | 0.17 ms | 0.08 ms | 11.01 ms | 560 | 1000 | 0 | 440 |
| LLM Detection | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1066.54 ms | 433.24 ms | 9255.10 ms | 0 | 1000 | 0 | 1000 |

## Comparison Vs AGL

- Rule-Based: accuracy delta vs AGL = -0.1855, F1 delta = -0.2588, recall delta = -0.4670, avg latency ratio vs AGL = 0.0006x
- Keyword: accuracy delta vs AGL = -0.1720, F1 delta = -0.2362, recall delta = -0.4400, avg latency ratio vs AGL = 0.0017x
- LLM Detection: accuracy delta vs AGL = -0.4520, F1 delta = -0.9542, recall delta = -1.0000, avg latency ratio vs AGL = 10.4756x

## Highlights

- Best accuracy: **AGL** at **0.9520**
- Best malicious F1: **AGL** at **0.9542**
- Lowest average latency: **Rule-Based** at **0.07 ms**
