"""
===============================================================
  DOMAIN 5 — ADVERSARIAL SECURITY AUDIT
  Arabic Fake News Detection · AI Security Evaluation Suite
  Models: arabic-fake-news-arbert-afnd
          arabic-fake-news-marbertv2-tweets
===============================================================
  Simulates real-world adversarial attacks an attacker would
  use to bypass automated Arabic fake-news filters:

  Attack Surface:
  · Character-level  : Unicode homoglyph swaps, diacritic
                       injection, tatweel stretching, zero-
                       width character insertion
  · Word-level       : Synonym substitution of trigger words,
                       deliberate misspellings, leet-style
                       numeral injection
  · Structural-level : Sentence reordering, hashtag padding,
                       URL injection, whitespace flooding

  Metrics per attack:
  · Flip Rate        — % of predictions changed by the attack
  · Confidence Delta — avg drop in prediction confidence
  · High-Conf Flip   — flips where model was ≥90% confident
  · Class Asymmetry  — does the attack target Fake or Credible?

  Output:
  · Per-attack vulnerability scorecard
  · Cumulative attack success heatmap
  · Confidence delta distribution
  · Most vulnerable example deep-dive
  · Sanity Gate verdict + recommended mitigations
===============================================================
"""

# Runtime notes:
# - Developed and run on Kaggle (GPU T4/P100). Runs on any machine with
#   a CUDA GPU or CPU; CPU will just be much slower for 3000 attack runs.
# - CSV_PATH below expects a local file. On Kaggle this was read from
#   /kaggle/input/<dataset>/afnd_clean.csv; set it to your own dataset
#   path when running locally or in another environment.
# - The model is pulled directly from the Hugging Face Hub (HF_MODEL_ID),
#   so no local model files are required.

import re
import unicodedata
import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')
random.seed(42)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
HF_MODEL_ID  = "arabic-fake-news-marbertv2-tweets"
CSV_PATH     = "afnd_clean.csv"
SAMPLE_SIZE  = 300       # per class — kept lower (attacks are 1-sample-at-a-time)
BATCH_SIZE   = 16
MAX_LENGTH   = 128
HIGH_CONF    = 0.90      # confidence threshold for "dangerous" flips
# ─────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "credible", 1: "not_credible"}


# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = text.replace('#', ' ')
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'\u0640+', '', text)
    for ch in 'أإآ':
        text = text.replace(ch, 'ا')
    text = re.sub(r'[^\u0600-\u06FF0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════
def load_model(model_id: str):
    print(f"\nDownloading model from HuggingFace: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE)
    model.eval()
    print(f"Model loaded on: {DEVICE}")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════
#  SINGLE-SAMPLE INFERENCE
# ══════════════════════════════════════════════════════════════
def predict_single(text: str, tokenizer, model):
    clean = preprocess(text) or "."
    inputs = tokenizer(clean, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def predict_batch(texts, tokenizer, model):
    all_probs, all_preds = [], []
    clean = [preprocess(t) for t in texts]
    for i in tqdm(range(0, len(clean), BATCH_SIZE), desc="  Baseline inference"):
        batch  = [t if t else "." for t in clean[i: i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        all_probs.extend(probs); all_preds.extend(probs.argmax(axis=1))
    return np.array(all_probs), np.array(all_preds)


# ══════════════════════════════════════════════════════════════
#  ATTACK LIBRARY
#  Each attack is a function: str → str
#  The attacker does NOT know the model — pure black-box.
# ══════════════════════════════════════════════════════════════

# ── 1. Character-level homoglyph swap ─────────────────────────
HOMOGLYPH_MAP = {
    'ة': 'ه', 'ه': 'ة',
    'ى': 'ي', 'ي': 'ى',
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
    'ؤ': 'و', 'ئ': 'ي',
}

def attack_homoglyph(text: str) -> str:
    """Swaps visually similar Arabic characters (Unicode homoglyphs)."""
    result = []
    for ch in text:
        result.append(HOMOGLYPH_MAP.get(ch, ch))
    return ''.join(result)


# ── 2. Diacritic injection ─────────────────────────────────────
DIACRITICS = ['\u064E', '\u064F', '\u0650', '\u0651', '\u0652']  # fatha/damma/kasra/shadda/sukun

def attack_diacritic_inject(text: str) -> str:
    """Inserts random Arabic diacritics between characters to confuse tokenizer."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < 0.4:
            pos  = random.randint(1, len(word) - 1)
            word = word[:pos] + random.choice(DIACRITICS) + word[pos:]
        result.append(word)
    return ' '.join(result)


# ── 3. Tatweel stretching ──────────────────────────────────────
def attack_tatweel(text: str) -> str:
    """Inserts Arabic tatweel (ـ) into random words — common in social media."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < 0.35:
            pos  = random.randint(1, len(word) - 1)
            word = word[:pos] + 'ـ' + word[pos:]
        result.append(word)
    return ' '.join(result)


# ── 4. Zero-width character injection ─────────────────────────
ZWC = ['\u200B', '\u200C', '\u200D', '\uFEFF']  # zero-width space / joiner / non-joiner / BOM

def attack_zero_width(text: str) -> str:
    """Inserts invisible zero-width Unicode characters to break tokenization."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < 0.3:
            zw = random.choice(ZWC)
            pos  = random.randint(0, len(word))
            word = word[:pos] + zw + word[pos:]
        result.append(word)
    return ' '.join(result)


# ── 5. Synonym substitution (trigger-word swap) ────────────────
SYNONYM_MAP = {
    "عاجل":    "خبر جديد",
    "كارثة":   "حدث كبير",
    "هام":     "تنبيه",
    "فضيحة":   "أمر مستغرب",
    "خطير":    "مقلق",
    "مريب":    "غريب",
    "تحذير":   "تنبيه",
    "صادم":    "مفاجئ",
    "كشف":     "أوضح",
    "وثائق":   "معلومات",
    "سري":     "خاص",
    "مصدر":    "جهة",
}

def attack_synonym(text: str) -> str:
    """Replaces known fake-news trigger words with neutral synonyms."""
    for word, replacement in SYNONYM_MAP.items():
        text = text.replace(word, replacement)
    return text


# ── 6. Deliberate misspelling (numeral injection) ─────────────
ARABIC_NUMERAL_SUBS = {
    'ا': '4', 'و': '0', 'ه': '8', 'ة': '8'
}

def attack_numeral_inject(text: str) -> str:
    """Replaces some Arabic letters with visually similar numerals (leet-style)."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 4 and random.random() < 0.25:
            chars = list(word)
            for idx, ch in enumerate(chars):
                if ch in ARABIC_NUMERAL_SUBS and random.random() < 0.4:
                    chars[idx] = ARABIC_NUMERAL_SUBS[ch]
            word = ''.join(chars)
        result.append(word)
    return ' '.join(result)


# ── 7. Whitespace flooding ─────────────────────────────────────
def attack_whitespace(text: str) -> str:
    """Inserts extra spaces inside words to disrupt word-piece tokenization."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 4 and random.random() < 0.3:
            mid  = len(word) // 2
            word = word[:mid] + '  ' + word[mid:]
        result.append(word)
    return ' '.join(result)


# ── 8. Structural: hashtag padding ────────────────────────────
FAKE_HASHTAGS = [
    '#عاجل', '#خبر_عاجل', '#تحذير', '#كشف_حصري',
    '#مصدر_موثوق', '#وثيقة_مسربة', '#الحقيقة'
]

def attack_hashtag_pad(text: str) -> str:
    """Appends sensationalist hashtags to the text."""
    tags = random.sample(FAKE_HASHTAGS, k=min(2, len(FAKE_HASHTAGS)))
    return text + ' ' + ' '.join(tags)


# ── 9. Structural: URL injection ──────────────────────────────
FAKE_URLS = [
    'http://breaking-news-ar.com/exclusive',
    'https://t.me/urgent_leaks_ar',
    'http://scoops24.net/secret',
]

def attack_url_inject(text: str) -> str:
    """Injects a suspicious URL to test if the model ignores or reacts to links."""
    url = random.choice(FAKE_URLS)
    return text + ' ' + url


# ── 10. Combined multi-attack (compound adversary) ─────────────
def attack_compound(text: str) -> str:
    """Chains homoglyph + synonym + diacritic attacks together."""
    text = attack_homoglyph(text)
    text = attack_synonym(text)
    text = attack_diacritic_inject(text)
    return text


ATTACKS = {
    "Homoglyph Swap":        attack_homoglyph,
    "Diacritic Inject":      attack_diacritic_inject,
    "Tatweel Stretch":       attack_tatweel,
    "Zero-Width Chars":      attack_zero_width,
    "Synonym Substitution":  attack_synonym,
    "Numeral Injection":     attack_numeral_inject,
    "Whitespace Flood":      attack_whitespace,
    "Hashtag Padding":       attack_hashtag_pad,
    "URL Injection":         attack_url_inject,
    "Compound Attack":       attack_compound,
}


# ══════════════════════════════════════════════════════════════
#  EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════
def evaluate_attacks(texts, y_true, baseline_preds, baseline_probs, tokenizer, model):
    """
    For each attack, runs every sample through the perturbed text
    and records flip rate, confidence delta, and class asymmetry.
    """
    results      = []
    per_sample   = []    # for heatmap: (attack_name, sample_idx, flipped)

    print(f"\nLaunching {len(ATTACKS)} adversarial attacks on {len(texts)} samples...")

    for attack_name, attack_fn in ATTACKS.items():
        flips         = 0
        conf_deltas   = []
        high_conf_flips = 0
        fake_flips    = 0
        cred_flips    = 0
        vulnerable_examples = []

        for i, (text, baseline_pred, baseline_prob) in enumerate(
                zip(texts, baseline_preds, baseline_probs)):
            perturbed = attack_fn(text)
            adv_pred, adv_probs = predict_single(perturbed, tokenizer, model)

            flipped         = (adv_pred != baseline_pred)
            baseline_conf   = float(baseline_prob.max())
            adv_conf        = float(adv_probs.max())
            conf_delta      = adv_conf - baseline_conf

            conf_deltas.append(conf_delta)
            per_sample.append({
                "attack": attack_name,
                "sample": i,
                "flipped": int(flipped),
                "conf_delta": conf_delta,
            })

            if flipped:
                flips += 1
                if baseline_conf >= HIGH_CONF:
                    high_conf_flips += 1
                if baseline_pred == 1:
                    fake_flips += 1
                else:
                    cred_flips += 1
                if len(vulnerable_examples) < 3:
                    vulnerable_examples.append({
                        "text": text[:120],
                        "perturbed": perturbed[:120],
                        "true": int(y_true[i]),
                        "original_pred": int(baseline_pred),
                        "adv_pred": adv_pred,
                        "original_conf": baseline_conf,
                        "adv_conf": adv_conf,
                    })

        n = len(texts)
        flip_rate  = flips / n
        avg_delta  = float(np.mean(conf_deltas))

        results.append({
            "Attack":              attack_name,
            "Flip Rate":           flip_rate,
            "Flips":               flips,
            "High-Conf Flips":     high_conf_flips,
            "Avg Conf Delta":      avg_delta,
            "Fake→Credible Flips": fake_flips,
            "Credible→Fake Flips": cred_flips,
            "Vulnerable Examples": vulnerable_examples,
        })

        severity = "CRITICAL" if flip_rate > 0.15 else \
                   "HIGH"     if flip_rate > 0.08 else \
                   "MEDIUM"   if flip_rate > 0.03 else "LOW"
        print(f"  {severity}  {attack_name:<25}  Flip: {flip_rate:.1%}  "
              f"ΔConf: {avg_delta:+.4f}  HiConf Flips: {high_conf_flips}")

    return pd.DataFrame(results), pd.DataFrame(per_sample)


# ══════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════
def print_security_report(results_df, n_samples):
    print("\n" + "=" * 70)
    print("  DOMAIN 5 - ADVERSARIAL SECURITY AUDIT REPORT")
    print("=" * 70)

    print(f"\n  {'Attack':<28} {'Flip%':>7} {'Flips':>7} {'HiConfFlip':>11} "
          f"{'ΔConf':>8} {'F→C':>6} {'C→F':>6}")
    print("  " + "─" * 75)

    for _, row in results_df.sort_values("Flip Rate", ascending=False).iterrows():
        severity = "CRIT" if row["Flip Rate"] > 0.15 else \
                   "HIGH" if row["Flip Rate"] > 0.08 else \
                   "MED"  if row["Flip Rate"] > 0.03 else "LOW"
        print(f"  {severity} {row['Attack']:<26} "
              f"{row['Flip Rate']:>7.1%} "
              f"{int(row['Flips']):>7} "
              f"{int(row['High-Conf Flips']):>11} "
              f"{row['Avg Conf Delta']:>+8.4f} "
              f"{int(row['Fake→Credible Flips']):>6} "
              f"{int(row['Credible→Fake Flips']):>6}")

    overall_flip = results_df["Flip Rate"].mean()
    worst        = results_df.loc[results_df["Flip Rate"].idxmax()]
    best         = results_df.loc[results_df["Flip Rate"].idxmin()]

    print(f"\n  Overall avg flip rate  : {overall_flip:.1%}")
    print(f"  Most dangerous attack  : {worst['Attack']}  ({worst['Flip Rate']:.1%})")
    print(f"  Most resilient to      : {best['Attack']}   ({best['Flip Rate']:.1%})")
    print(f"  Total high-conf flips  : {int(results_df['High-Conf Flips'].sum())}")

    print("\n  VULNERABLE EXAMPLE DEEP-DIVE (top 2 per worst attack):")
    print("  " + "═" * 68)
    for ex in worst["Vulnerable Examples"][:2]:
        print(f"\n  True label   : {LABEL_MAP[ex['true']]}")
        print(f"  Original pred: {LABEL_MAP[ex['original_pred']]} ({ex['original_conf']:.1%} conf)")
        print(f"  Adv pred     : {LABEL_MAP[ex['adv_pred']]} ({ex['adv_conf']:.1%} conf)")
        print(f"  Original : {ex['text']}")
        print(f"  Perturbed: {ex['perturbed']}")
        print("  " + "─" * 68)

    print("\n  Sanity Gate - Recommended Mitigations:")
    for _, row in results_df.iterrows():
        if row["Flip Rate"] > 0.05:
            mitigations = {
                "Homoglyph Swap":        "→ Add Unicode normalization (NFKC) in preprocessing pipeline",
                "Diacritic Inject":      "→ Strip all Arabic diacritics (\\u064B–\\u065F) before inference",
                "Tatweel Stretch":       "→ Strip tatweel (\\u0640) in preprocessing",
                "Zero-Width Chars":      "→ Filter zero-width Unicode block in input sanitization",
                "Synonym Substitution":  "→ Train with synonym-augmented data; use sense embeddings",
                "Numeral Injection":     "→ Normalize Arabic–Indic numerals; strip non-Arabic chars",
                "Whitespace Flood":      "→ Collapse multiple spaces; re-tokenize after normalization",
                "Hashtag Padding":       "→ Strip hashtag tokens before tokenizer; test robustness",
                "URL Injection":         "→ Remove URLs in preprocessing; already done but verify",
                "Compound Attack":       "→ All above mitigations required; compound is the real threat",
            }
            mit = mitigations.get(row['Attack'], "→ Manual review required")
            print(f"    [!] {row['Attack']:<28}  {mit}")


# ══════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════
def plot_flip_rate_bar(results_df):
    ordered = results_df.sort_values("Flip Rate", ascending=False)
    colors  = ["#e63946" if r > 0.15 else
               "#f4a261" if r > 0.08 else
               "#e9c46a" if r > 0.03 else "#2a9d8f"
               for r in ordered["Flip Rate"]]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(ordered["Attack"], ordered["Flip Rate"], color=colors, alpha=0.9, edgecolor='white')
    ax.axhline(0.05, color='gray', linestyle='--', lw=1.2, label="5% danger threshold")
    ax.axhline(0.15, color='red',  linestyle='--', lw=1.2, label="15% critical threshold")

    for bar, val in zip(bars, ordered["Flip Rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.1%}", ha='center', va='bottom', fontsize=8.5)

    ax.set_title(f"Domain 5 — Adversarial Flip Rate per Attack\n{HF_MODEL_ID}",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_ylim(0, min(ordered["Flip Rate"].max() + 0.08, 1.0))
    ax.tick_params(axis='x', rotation=35)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = f"domain5_flip_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Flip rate bar chart saved: {out}")


def plot_conf_delta_box(per_sample_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    attack_order = (per_sample_df.groupby("attack")["conf_delta"]
                    .mean().sort_values().index.tolist())

    per_sample_df['attack'] = pd.Categorical(per_sample_df['attack'],
                                              categories=attack_order, ordered=True)
    grouped = [per_sample_df[per_sample_df['attack'] == a]['conf_delta'].values
               for a in attack_order]

    bp = ax.boxplot(grouped, labels=attack_order, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#457b9d')
        patch.set_alpha(0.7)

    ax.axhline(0, color='red', linestyle='--', lw=1.2, label="No change baseline")
    ax.set_title(f"Domain 5 — Confidence Delta Distribution per Attack\n{HF_MODEL_ID}",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Confidence Delta (Adv − Baseline)")
    ax.tick_params(axis='x', rotation=35)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = f"domain5_conf_delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Confidence delta boxplot saved: {out}")


def plot_flip_heatmap(per_sample_df, n_display=80):
    pivot = per_sample_df[per_sample_df['sample'] < n_display].pivot(
        index='attack', columns='sample', values='flipped'
    )
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', cbar_kws={'label': 'Flipped?'},
                linewidths=0.3, linecolor='white', vmin=0, vmax=1)
    ax.set_title(f"Domain 5 — Flip Heatmap (first {n_display} samples × all attacks)\n{HF_MODEL_ID}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("")
    plt.tight_layout()
    out = f"domain5_flip_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Flip heatmap saved: {out}")


def plot_class_asymmetry(results_df):
    fig, ax = plt.subplots(figsize=(13, 6))
    x    = np.arange(len(results_df))
    w    = 0.38
    ax.bar(x - w/2, results_df["Fake→Credible Flips"], w,
           label="Fake → Credible (evasion)", color="#e63946", alpha=0.85)
    ax.bar(x + w/2, results_df["Credible→Fake Flips"], w,
           label="Credible → Fake (injection)", color="#457b9d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Attack"], rotation=35, ha='right')
    ax.set_title(f"Domain 5 — Class Asymmetry of Attacks\n{HF_MODEL_ID}",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Flips")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = f"domain5_class_asymmetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Class asymmetry chart saved: {out}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  DOMAIN 5 — ADVERSARIAL SECURITY AUDIT")
    print("  Arabic Fake News · AI Security Evaluation Suite")
    print("=" * 70)

    print(f"\nLoading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    text_col  = next((c for c in df.columns if c.lower() in ['text', 'tweet', 'content']), None)
    label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'target']), None)
    if not text_col or not label_col:
        raise ValueError(f"Cannot find text/label columns. Found: {df.columns.tolist()}")

    credible_df = df[df[label_col] == 'credible'].dropna(subset=[text_col])
    fake_df     = df[df[label_col] == 'not_credible'].dropna(subset=[text_col])
    n           = min(SAMPLE_SIZE, len(credible_df), len(fake_df))
    texts       = credible_df[text_col].sample(n=n, random_state=42).tolist() + \
                  fake_df[text_col].sample(n=n, random_state=42).tolist()
    y_true      = np.array([0]*n + [1]*n)
    print(f"  Sample: {n} credible + {n} fake = {2*n} total")

    tokenizer, model = load_model(HF_MODEL_ID)

    print("\nRunning baseline inference...")
    baseline_probs, baseline_preds = predict_batch(texts, tokenizer, model)
    baseline_acc = accuracy_score(y_true, baseline_preds)
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")

    results_df, per_sample_df = evaluate_attacks(
        texts, y_true, baseline_preds, baseline_probs, tokenizer, model)

    print_security_report(results_df, len(texts))

    print("\nGenerating adversarial security plots...")
    plot_flip_rate_bar(results_df)
    plot_conf_delta_box(per_sample_df)
    plot_flip_heatmap(per_sample_df)
    plot_class_asymmetry(results_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed per-sample results
    per_sample_df.to_csv(f"domain5_per_sample_{ts}.csv", index=False, encoding="utf-8-sig")

    # Save summary scorecard
    summary_cols = ["Attack", "Flip Rate", "Flips", "High-Conf Flips",
                    "Avg Conf Delta", "Fake→Credible Flips", "Credible→Fake Flips"]
    results_df[summary_cols].to_csv(f"domain5_scorecard_{ts}.csv", index=False)

    # Save perturbed texts for worst attack
    worst_attack = results_df.loc[results_df["Flip Rate"].idxmax(), "Attack"]
    attack_fn    = ATTACKS[worst_attack]
    perturbed_rows = []
    for i, text in enumerate(texts):
        perturbed_rows.append({
            "original":  text,
            "perturbed": attack_fn(text),
            "true_label": int(y_true[i]),
            "baseline_pred": int(baseline_preds[i]),
        })
    pd.DataFrame(perturbed_rows).to_csv(
        f"domain5_perturbed_{worst_attack.replace(' ', '_')}_{ts}.csv",
        index=False, encoding="utf-8-sig")

    print(f"\nAll results saved with timestamp {ts}")
    print("Domain 5 complete.\n")


if __name__ == "__main__":
    main()
