# LLM-as-a-Judge Evaluation Report

**Judge model(s):** gpt-4o  
**Total evaluations:** 95

## Scores by Model

| Model | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 40 | 8.93 ± 1.47 | 5.80 ± 1.36 | 8.05 ± 1.68 | 5.33 ± 1.89 | 6.42 ± 1.52 |
| meta-llama_Llama-2-70b-chat-hf | 15 | 10.00 ± 0.00 | 4.67 ± 0.82 | 7.27 ± 1.62 | 4.13 ± 1.41 | 5.40 ± 0.91 |
| mistralai_Mistral-7B-Instruct-v0.3 | 40 | 9.15 ± 1.61 | 5.15 ± 1.03 | 7.55 ± 1.47 | 4.47 ± 1.26 | 5.80 ± 1.11 |

## Scores by Prompt Configuration

| Config | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| cq-only | 23 | 9.52 ± 0.99 | 5.61 ± 0.84 | 8.17 ± 1.07 | 5.30 ± 1.18 | 6.39 ± 0.78 |
| scenario-cq | 16 | 8.25 ± 2.05 | 5.25 ± 1.53 | 6.88 ± 2.03 | 4.69 ± 1.78 | 5.75 ± 1.61 |
| scenario-cq-constraints | 16 | 9.44 ± 1.26 | 5.88 ± 1.09 | 8.06 ± 1.61 | 5.50 ± 1.55 | 6.56 ± 1.36 |
| scenario-cq-reasoning | 16 | 8.69 ± 1.62 | 5.62 ± 1.20 | 7.69 ± 1.49 | 5.06 ± 1.69 | 6.12 ± 1.50 |
| scenario-only | 24 | 9.67 ± 1.01 | 4.62 ± 1.13 | 7.62 ± 1.64 | 3.67 ± 1.43 | 5.33 ± 1.13 |

## Scores by Scenario

| Scenario | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-w01-01 | 12 | 9.33 ± 0.98 | 5.25 ± 1.22 | 7.67 ± 1.23 | 4.83 ± 1.85 | 5.92 ± 1.24 |
| 2026-w04-01 | 12 | 9.50 ± 1.00 | 5.00 ± 0.85 | 8.25 ± 1.36 | 4.83 ± 1.40 | 6.00 ± 0.95 |
| 2026-w05-01 | 12 | 10.00 ± 0.00 | 6.50 ± 1.00 | 8.92 ± 1.00 | 6.33 ± 1.67 | 7.17 ± 1.11 |
| 2026-w06-01 | 12 | 9.17 ± 1.27 | 4.83 ± 1.11 | 6.83 ± 1.53 | 3.83 ± 1.11 | 5.42 ± 1.24 |
| 2026-w07-01 | 12 | 8.75 ± 1.48 | 4.83 ± 1.03 | 7.00 ± 1.21 | 4.00 ± 1.13 | 5.50 ± 0.80 |
| 2026-w09-01 | 12 | 9.25 ± 1.36 | 5.58 ± 1.31 | 7.92 ± 1.31 | 4.83 ± 1.64 | 6.17 ± 1.40 |
| 2026-w10-01 | 12 | 10.00 ± 0.00 | 5.75 ± 1.22 | 8.83 ± 0.72 | 5.50 ± 1.51 | 6.67 ± 1.23 |
| 2026-w11-01 | 11 | 7.36 ± 2.42 | 5.00 ± 1.26 | 6.18 ± 2.18 | 4.00 ± 1.34 | 5.09 ± 1.51 |

## Overall Score: Model × Config

| Model | cq-only | scenario-cq | scenario-cq-constraints | scenario-cq-reasoning | scenario-only |
| --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 6.8 (n=8) | 5.8 (n=8) | 6.6 (n=8) | 6.9 (n=8) | 6.1 (n=8) |
| meta-llama_Llama-2-70b-chat-hf | 6.0 (n=7) | — | — | — | 4.9 (n=8) |
| mistralai_Mistral-7B-Instruct-v0.3 | 6.4 (n=8) | 5.8 (n=8) | 6.5 (n=8) | 5.4 (n=8) | 5.0 (n=8) |
