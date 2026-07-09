# Domain 2 — Linguistic & Dialectal Robustness
Owner: Mohamed Shreif Abdelsattar (2405467)
Supervised by Prof. Mohamed M. Abbassy

## What this domain covers
ARBERT is trained on formal Modern Standard Arabic (MSA) news. This domain tests what happens when the same facts are rewritten in Egyptian/Levantine/Gulf/Maghrebi dialect — measuring whether the model's predictions change purely because of style, not substance (the "Dialect Gap").

## Files

```
notebooks/
├── cross_model_probing.py    # main pipeline: MSA -> Mild dialect -> Heavy dialect perturbation + evaluation
├── evaluate_arbert.py        # standalone ARBERT evaluation script (metrics, HTML report, misclassification review)
└── afnd_downloader.py        # small helper to pull the AFND parquet dataset from Kaggle Hub

figures/
├── VIZ1_performance_comparison.png   # Accuracy/F1 across Original / Mild / Heavy dialect versions
├── VIZ2_flip_rate.png                # % of predictions that flipped label after dialectal rewriting
├── VIZ3_confidence_drop.png          # mean model confidence across the three versions
├── VIZ4_confusion_matrices.png       # per-version confusion matrices
└── VIZ5_degradation.png              # accuracy degradation ("Dialect Gap") magnitude

results/
└── VIZ1-5 .txt                       # written findings for each chart, in the author's own words
```

## How to run

```bash
pip install pandas torch transformers scikit-learn matplotlib seaborn tqdm numpy pyarrow fastparquet scipy kagglehub
python afnd_downloader.py         # pulls the AFND dataset
python cross_model_probing.py     # runs the MSA -> Mild -> Heavy dialect experiment, produces the 5 VIZ charts
python evaluate_arbert.py         # optional standalone evaluation + HTML report
```

`afnd_downloader.py` uses `kagglehub` to fetch the dataset directly, so it needs a Kaggle account/API credentials configured locally if you're not running on Kaggle itself.

## Key results

| Metric | Original (MSA) | Mild Dialect | Heavy Dialect |
|---|---|---|---|
| Accuracy | 0.942 | 0.924 | 0.922 |
| Real-news True Positive Rate | 92.5% | 89.1% | 89.5% |
| Fake-news detection rate | ~96% (stable) | ~96% | ~96% |
| Mean confidence | 0.966 | 0.965 | 0.965 |
| Prediction flip rate | — | 3.0% | 3.2% |
| Dialect Gap (accuracy drop) | — | −1.8% | −2.0% |

**Findings:**
- ARBERT stays above the 90% accuracy threshold even under heavy dialectal rewriting — the overall Dialect Gap is small (~2%, classified as low severity).
- The degradation is not symmetric: **real news is disproportionately misclassified as fake** under dialect pressure (8 additional real-news samples flagged as fake, facts unchanged), while fake-news detection stays stable. This is a measurable style-veracity bias, not just random noise.
- Confidence stays flat (0.966 → 0.965) even as accuracy drops — the model doesn't "know" it's getting less reliable under dialect pressure. A production system relying on ARBERT's confidence score to flag uncertain predictions for human review would miss these dialect-induced errors.
- The gap between Mild and Heavy dialect (only 0.002) suggests diminishing returns: most of the sensitivity comes from the first wave of common substitutions (negation, pronouns, question words), not from piling on more dialectal swaps.

Full write-ups for each chart are in `results/VIZ1_performance_comparison.txt` through `VIZ5_degradation.txt`.
