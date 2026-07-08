# Domain 5 — Adversarial & Security Testing
Owner: Mohammed Tamer (2405605)
Supervised by Prof. Mohamed M. Abbassy

## What this domain covers
A black-box adversarial security audit of the Arabic fake-news classifiers. Ten attack types (character-level, word-level, and structural) are run against the models to measure how easily a prediction can be flipped, and whether the model is dangerously confident while wrong.

## Files

```
notebooks/
├── arbert_afnd_training.ipynb          # ARBERT fine-tuning on AFND (formal MSA news)
├── marbertv2_tweets_training.ipynb     # MARBERTv2 fine-tuning on ARABICFAKETWEETS (dialectal tweets)
└── domain5_adversarial_security.py     # Adversarial attack + evaluation suite (this domain's core deliverable)

figures/
├── domain5_flip_rates_20260507_194837.png       # Flip rate per attack type
├── domain5_conf_delta_20260507_194838.png       # Confidence delta distribution per attack
├── domain5_flip_heatmap_20260507_194839.png     # Per-sample flip heatmap across attacks
└── domain5_class_asymmetry_20260507_194840.png  # Fake->Credible vs Credible->Fake flip asymmetry

results/
├── domain5_scorecard_20260507_194840.csv                # Summary metrics per attack
├── domain5_per_sample_20260507_194840.csv               # Per-sample, per-attack flip/confidence data
└── domain5_perturbed_Homoglyph_Swap_20260507_194840.csv # Original vs perturbed text for the worst attack
```

## How to run

1. Run `arbert_afnd_training.ipynb` and/or `marbertv2_tweets_training.ipynb` first, or point `domain5_adversarial_security.py` at an already fine-tuned model on the Hugging Face Hub.
2. Set `HF_MODEL_ID` and `CSV_PATH` at the top of `domain5_adversarial_security.py`.
3. Run:
   ```bash
   pip install pyarabic lime textattack transformers torch scikit-learn pandas numpy matplotlib seaborn tqdm
   python domain5_adversarial_security.py
   ```
4. Outputs (scorecard CSV, per-sample CSV, perturbed-text CSV, and 4 charts) are written to the working directory with a timestamp.

**Note:** both notebooks were originally run on Kaggle. Kaggle-specific paths (`/kaggle/input`, `/kaggle/working`) and the `kaggle_secrets` import for the Hugging Face token are guarded with fallbacks — see the runtime note at the top of each notebook for how to adapt them to a different environment.

## Key results (this run — MARBERTv2 tested cross-domain on AFND formal-news text)

| Attack | Flip Rate | High-Conf Flips | Avg Confidence Delta |
|---|---|---|---|
| Homoglyph Swap | 49.7% | 187 | -0.038 |
| Compound Attack | 49.3% | 185 | -0.039 |
| Whitespace Flood | 23.8% | 56 | -0.034 |
| Zero-Width Chars | 14.7% | 22 | -0.024 |
| Numeral Injection | 10.7% | 6 | -0.020 |
| Hashtag Padding | 1.2% | 0 | -0.002 |
| Diacritic Inject / Tatweel Stretch / Synonym Substitution / URL Injection | 0.0% | 0 | ~0.000 |

This particular run tests MARBERTv2 (trained on dialectal tweets) against out-of-domain **formal MSA news** text, which is why flip rates are much higher here than MARBERTv2's near-zero flip rate on in-domain tweet data reported elsewhere in the project. Read together, the two results tell the same story: MARBERTv2 is highly robust *within* its training domain but destabilizes on out-of-domain input, which is itself a security-relevant weakness — an attacker doesn't need a clever adversarial attack if they can just shift the input domain.

Full attack-by-attack numbers are in `results/domain5_scorecard_20260507_194840.csv`.
