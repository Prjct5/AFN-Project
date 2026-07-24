# Domain 3 — Error Surface & Failure Mode Analysis
Owner: Abdullah Abbas Ezzat (2405415)
Supervised by Prof. Mohamed M. Abbassy

## What this domain covers
A 99% accuracy number can hide a model that doesn't actually know when it's wrong. This domain performs a forensic analysis of MARBERTv2's failures specifically — mapping where errors cluster, whether the model's confidence can be trusted, and what structural or lexical patterns separate correct predictions from incorrect ones.

## Files

```
notebooks/
└── evaluate_marbertv2.py     # full evaluation pipeline: metrics, failure-mode classification, all 8 figures

figures/
├── 1_confidence_kde.png                      # confidence distribution: correct vs incorrect predictions
├── 2_correlation_heatmap.png                 # Spearman correlation between text/metadata features
├── 3_failure_mode_distribution.png           # error breakdown by failure type
├── 4_marbertv2_confidence_distribution.png   # confidence histogram, correct vs incorrect
├── 5_marbertv2_confusion_matrix.png          # TP / TN / FP / FN counts
├── 6_marbertv2_roc_curve.png                 # ROC curve and AUC
├── 7_structural_binning_accuracy.png         # accuracy across text-length quintiles
└── 8_trigger_words_tfidf.png                 # top TF-IDF words in the error set

results/
└── graphs_explanation.txt    # full written analysis of each figure, in the author's own words
```

## How to run

```bash
pip install pandas torch transformers scikit-learn matplotlib seaborn numpy scipy tqdm
python evaluate_marbertv2.py
```

The script is interactive — it prompts for the CSV path, column names, batch size, and output directory, then runs inference with MARBERTv2, computes all failure-mode metrics, and saves the 8 figures above plus a results CSV.

## Key results

| Metric | Value |
|---|---|
| Accuracy | ~99.1% |
| AUC (ROC) | 0.565 |
| True Positives (Fake correctly flagged) | 35,365 |
| True Negatives (Real correctly cleared) | 11,609 |
| False Negatives (Fake missed as Real) | 302 |
| False Positives (Real flagged as Fake) | 209 |
| High-confidence failures | ~246 cases |
| Calibration-instability failures | ~178 cases |
| Trigger-word-dependency failures | ~131 cases |
| Short-context failures | ~9 cases |

**Findings:**
- The headline accuracy (~99%) is misleading on its own — the AUC of 0.565 is barely above random (0.5), meaning the model's confidence score does not reliably separate correct predictions from incorrect ones, despite the model being right most of the time in absolute terms.
- Most failures are **confidence-related, not comprehension-related** — the model is frequently wrong while being highly confident (246 high-confidence failures), which is the hardest failure mode to catch with a simple confidence threshold.
- Text length is **not** a meaningful driver of errors — accuracy stayed stable (99.3%–99.8%) across all text-length quintiles.
- TF-IDF analysis of the error set surfaced a cluster of errors around **Saudi financial/business Arabic content** (words like "company," "riyal," "Riyadh," "communication/social media"), pointing to a concrete, fixable domain gap rather than a vague generalization problem.
- Of the two error types, false negatives (fake news missed as real, 302 cases) are the more costly mistake in a misinformation-detection context than false positives.

Full write-up for every figure is in `results/graphs_explanation.txt`.
