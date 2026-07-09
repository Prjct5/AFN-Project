# Domain 1 — Advanced Statistical & Calibration Metrics
Owner: Mahmoud Hazem Ali Nadeem (2405477)
Supervised by Prof. Mohamed M. Abbassy

## What this domain covers
Accuracy alone is a poor metric on an imbalanced fake-news dataset, so this domain evaluates whether the model's confidence can actually be trusted. It computes Matthews Correlation Coefficient (MCC), Expected Calibration Error (ECE), and Brier Score, then runs 1,000-iteration bootstrapping to produce 95% confidence intervals and confirm the results are stable rather than a fluke of one test split.

## Files

```
notebooks/
├── calibration_metrics_gpu.py     # full pipeline, GPU-optimized (FP16, TF32, cuDNN autotune)
└── calibration_metrics_cpu.py     # same pipeline, CPU-only fallback

figures/
├── reliability_diagram.png              # predicted confidence vs actual accuracy
├── calibration_by_bin.png               # per-bin accuracy/confidence bar chart
├── bootstrap_confidence_intervals.png   # 95% CI per metric across 1000 runs
├── confidence_histogram.png             # distribution of model confidence
└── calibration_dashboard.png            # 9-panel combined view

results/
├── advanced_metrics_20260508_232220.json                # all metrics, bootstrap CIs, run config
└── advanced_calibration_report_20260508_232220.html      # rendered HTML report
```

## How to run

Both scripts are interactive — they prompt for the input file path and run parameters (sample size, batch size, threshold, number of bootstrap iterations, number of calibration bins) instead of hardcoding them.

```bash
pip install pandas torch transformers scikit-learn matplotlib seaborn tqdm numpy pyarrow fastparquet scipy
python calibration_metrics_gpu.py   # or calibration_metrics_cpu.py on machines without a GPU
```

The script will ask for a CSV or Parquet file containing text + label columns, then run inference with the fine-tuned ARBERT model (`zhafyz/arabic-fake-news-arbert-afnd`), compute all calibration metrics, generate the 5 figures above, and save a results CSV, a metrics JSON, and an HTML report.

## Key results

Run on `afnd_clean.parquet`, batch size 24, FP16 enabled, 1000 bootstrap iterations, 15 calibration bins:

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.9263 | [0.9255, 0.9271] |
| Balanced Accuracy | 0.9287 | — |
| F1 | 0.9201 | [0.9192, 0.9211] |
| Matthews Correlation Coefficient (MCC) | 0.8534 | [0.8518, 0.8551] |
| Expected Calibration Error (ECE) | 0.0446 | [0.0438, 0.0453] |
| Brier Score | 0.0583 | [0.0577, 0.0589] |

The very tight confidence intervals (e.g. MCC varying by only ~0.003 across 1000 bootstrap resamples) confirm these results are stable, not dependent on a lucky test split. The model is reasonably well-calibrated overall (ECE ≈ 0.045), meaning its stated confidence is a fair approximation of its real-world accuracy — useful for setting a production rule such as "route predictions below 70% confidence to a human reviewer."

Full metric details and bin-by-bin breakdowns are in `results/advanced_metrics_20260508_232220.json`.
