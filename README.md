# Arabic Fake News Detection

Two fine-tuned BERT models for detecting fake news in Arabic — one for formal news (ARBERT) and one for informal social media text (MARBERTv2).

Supervised by Prof. Mohamed M. Abbassy.

## Overview

We fine-tuned two BERT-family models to classify Arabic text as `credible` or `not_credible`:

- **ARBERT** — trained on ~607,000 formal Modern Standard Arabic (MSA) news articles (AFND dataset).
- **MARBERTv2** — trained on ~128,000 informal, dialectal Arabic social media posts (ArabicFakeTweets dataset).

Each team member then took ownership of a separate evaluation domain, stress-testing the trained models from a different angle: calibration, error analysis, dialectal robustness, interpretability, and adversarial security.

## Team & Domains

| Domain | Owner | Folder |
|---|---|---|
| Model Training (ARBERT & MARBERTv2) | Team | [`/model-training`](./model-training) |
| Domain 1 — Advanced Statistical & Calibration Metrics | Mahmoud Hazem Ali Nadeem (2405477) | [`/domain-01-calibration-metrics`](./domain-01-calibration-metrics) |
| Domain 2 — Linguistic & Dialectal Robustness | Mohamed Shreif Abdelsattar (2405467) | [`/domain-02-dialectal-robustness`](./domain-02-dialectal-robustness) |
| Domain 3 — Error Surface & Failure Mode Analysis | Abdullah Abbas Ezzat (2405415) | [`/domain-03-error-surface`](./domain-03-error-surface) |
| Domain 4 — Interpretability & Black-Box Probing | Ali Alaa Salah (2405472) | [`/domain-04-interpretability`](./domain-04-interpretability) |
| Domain 5 — Adversarial & Security Testing | Mohammed Tamer (2405605) | [`/domain-05-adversarial-security`](./domain-05-adversarial-security) |

Domain numbers follow the team's presentation numbering. As of now, only Domain 5 has code and results uploaded — the rest will be filled in by each member as they push their work.

## Repository structure

```
arabic-fake-news-detection/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── model-training/                    (shared notebooks that train ARBERT & MARBERTv2)
│   ├── arbert_training.ipynb
│   ├── marbertv2_training.ipynb
│   └── README.md
│
├── domain-01-calibration-metrics/     (Mahmoud Hazem Ali Nadeem)
├── domain-02-dialectal-robustness/    (Mohamed Shreif Abdelsattar)
├── domain-03-error-surface/           (Abdullah Abbas Ezzat)
├── domain-04-interpretability/        (Ali Alaa Salah)
├── domain-05-adversarial-security/    (Mohammed Tamer)
│   each with: notebooks/  figures/  results/  README.md
│
└── docs/
    (project slide deck / final report go here)
```

## Models at a glance

| | ARBERT | MARBERTv2 |
|---|---|---|
| Text type | Formal MSA news | Informal / dialectal tweets |
| Dataset | AFND (~607K articles) | ArabicFakeTweets (~128K posts) |
| Pre-training | 61 GB MSA text | 29B tokens (tweets + news) |
| MAX_LEN / BATCH | 128 / 16 | 64 / 32 |
| Trainer | Standard Trainer | Weighted Trainer |

Shared training config: learning rate `2e-5`, weight decay `0.01`, warmup = first 10% of steps, 3 epochs, random seed `42`, 80/10/10 stratified split, macro-F1 for `compute_metrics`, LIME explainability integration.


Each domain folder has its own `README.md` with instructions for running that member's notebooks and reproducing their results.

## Acknowledgements

This project was carried out under the academic supervision of Prof. Mohamed M. Abbassy.


