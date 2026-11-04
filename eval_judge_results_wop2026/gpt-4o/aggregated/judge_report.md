# LLM-as-a-Judge Evaluation Report

**Judge model(s):** gpt-4o  
**Total evaluations:** 160

## Scores by Model

| Model | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 40 | 10.00 ± 0.00 | 6.00 ± 1.48 | 9.03 ± 0.70 | 5.58 ± 1.95 | 6.78 ± 1.40 |
| meta-llama_Llama-2-70b-chat-hf | 40 | 9.65 ± 1.17 | 4.88 ± 0.76 | 7.35 ± 1.46 | 4.47 ± 1.13 | 5.62 ± 0.87 |
| meta-llama_Llama-3.1-8B-Instruct | 40 | 9.85 ± 0.53 | 5.20 ± 0.79 | 8.00 ± 1.18 | 4.83 ± 1.03 | 6.08 ± 0.73 |
| mistralai_Mistral-7B-Instruct-v0.3 | 40 | 10.00 ± 0.00 | 5.33 ± 0.94 | 8.28 ± 1.04 | 4.85 ± 1.23 | 6.22 ± 0.97 |

## Scores by Prompt Configuration

| Config | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| cq-only | 32 | 9.94 ± 0.35 | 5.66 ± 0.90 | 8.50 ± 0.92 | 5.50 ± 1.08 | 6.50 ± 0.80 |
| scenario-cq | 32 | 9.84 ± 0.63 | 5.44 ± 1.22 | 8.28 ± 1.14 | 5.09 ± 1.38 | 6.28 ± 1.11 |
| scenario-cq-constraints | 32 | 9.75 ± 1.11 | 5.56 ± 1.08 | 8.06 ± 1.44 | 5.28 ± 1.35 | 6.34 ± 1.21 |
| scenario-cq-reasoning | 32 | 9.91 ± 0.53 | 5.44 ± 1.05 | 8.06 ± 1.29 | 5.06 ± 1.34 | 6.28 ± 1.05 |
| scenario-only | 32 | 9.94 ± 0.35 | 4.66 ± 1.04 | 7.91 ± 1.47 | 3.72 ± 1.33 | 5.47 ± 1.02 |

## Scores by Scenario

| Scenario | N | Syntactic | Semantic | Logical | Functional | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-w01-01 | 20 | 10.00 ± 0.00 | 5.30 ± 1.30 | 8.25 ± 1.25 | 5.10 ± 1.62 | 6.20 ± 1.15 |
| 2026-w04-01 | 20 | 9.70 ± 1.34 | 4.95 ± 0.76 | 8.65 ± 1.04 | 4.95 ± 1.47 | 6.05 ± 1.10 |
| 2026-w05-01 | 20 | 10.00 ± 0.00 | 6.25 ± 0.91 | 8.50 ± 1.36 | 5.80 ± 1.51 | 6.80 ± 1.06 |
| 2026-w06-01 | 20 | 9.70 ± 0.73 | 4.95 ± 1.00 | 7.40 ± 1.35 | 4.15 ± 1.04 | 5.75 ± 0.97 |
| 2026-w07-01 | 20 | 9.90 ± 0.45 | 5.00 ± 1.03 | 7.55 ± 1.15 | 4.25 ± 1.12 | 5.70 ± 0.80 |
| 2026-w09-01 | 20 | 9.85 ± 0.67 | 5.35 ± 1.14 | 8.35 ± 0.99 | 4.75 ± 1.37 | 6.20 ± 1.06 |
| 2026-w10-01 | 20 | 10.00 ± 0.00 | 5.50 ± 1.05 | 8.35 ± 1.04 | 5.40 ± 1.31 | 6.50 ± 1.05 |
| 2026-w11-01 | 20 | 9.85 ± 0.67 | 5.50 ± 1.15 | 8.25 ± 1.52 | 5.05 ± 1.36 | 6.20 ± 1.28 |

## Overall Score: Model × Config

| Model | cq-only | scenario-cq | scenario-cq-constraints | scenario-cq-reasoning | scenario-only |
| --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 7.2 (n=8) | 6.2 (n=8) | 7.1 (n=8) | 7.0 (n=8) | 6.2 (n=8) |
| meta-llama_Llama-2-70b-chat-hf | 6.0 (n=8) | 6.1 (n=8) | 5.5 (n=8) | 5.6 (n=8) | 4.9 (n=8) |
| meta-llama_Llama-3.1-8B-Instruct | 6.2 (n=8) | 6.4 (n=8) | 6.2 (n=8) | 6.0 (n=8) | 5.5 (n=8) |
| mistralai_Mistral-7B-Instruct-v0.3 | 6.5 (n=8) | 6.4 (n=8) | 6.5 (n=8) | 6.5 (n=8) | 5.2 (n=8) |
