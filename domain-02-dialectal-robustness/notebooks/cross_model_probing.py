#!/usr/bin/env python3
"""
Arabic Fake News Detection — Cross-Model Probing v6
=====================================================
Domain 2: Linguistic & Dialectal Robustness

v6 — Linguistic Perturbation Edition:
  • Takes AFND (MSA) samples and creates 2 perturbed versions:
      - Mild Dialect : common lexical swaps (Egyptian + Levantine light)
      - Heavy Dialect : aggressive multi-dialect swaps (Egyptian + Gulf + Maghrebi)
  • Runs ARBERT on all 3 versions (Original / Mild / Heavy)
  • Measures Prediction Flip Rate, Confidence Drop, and Dialect Gap
  • 5 publication-grade visualizations:
      VIZ 1 Performance Comparison Bar Chart (Acc/F1 across 3 versions)
      VIZ 2 Prediction Flip Rate Chart (% sentences that flipped label)
      VIZ 3 Confidence Drop Line Plot (mean confidence across versions)
      VIZ 4 Confusion Matrix (per version) (3-panel)
      VIZ 5 Dialect Robustness Degradation (drop magnitude)

Usage:
  python cross_model_probing_v6.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Package Check
# ──────────────────────────────────────────────────────────────────────────────
import subprocess, sys, importlib, os

def check_and_install_packages():
    required = {
        'pandas': 'pandas', 'torch': 'torch', 'transformers': 'transformers',
        'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib', 'seaborn': 'seaborn',
        'tqdm': 'tqdm', 'numpy': 'numpy', 'pyarrow': 'pyarrow',
        'fastparquet': 'fastparquet', 'scipy': 'scipy',
    }
    missing = []
    print("="*80); print("CHECKING PACKAGES"); print("="*80)
    for pkg, name in required.items():
        try: importlib.import_module(pkg); print(f"{pkg} - OK")
        except ImportError: print(f"{pkg} - MISSING"); missing.append(name)
    if missing:
        r = input(f"\nInstall missing ({', '.join(missing)})? (y/n): ").strip().lower()
        if r == 'y':
            for p in missing: subprocess.check_call([sys.executable,"-m","pip","install",p])
        else: print("Aborted."); return False
    print("\nAll packages ready!\n"); return True

if not check_and_install_packages(): sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────────────
import re, unicodedata, json, warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score,
                             matthews_corrcoef, cohen_kappa_score,
                             balanced_accuracy_score, log_loss)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.stats import pointbiserialr

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 2. Palette — fixed to spec
# ──────────────────────────────────────────────────────────────────────────────
P = {
    'navy' : '#1E3A5F',
    'teal' : '#2A9D8F',
    'orange' : '#F4A261',
    'gray' : '#64748B',
    'bg' : '#F8FAFC',
    'panel' : '#FFFFFF',
    'border' : '#E2E8F0',
    'grid' : '#EEF2F7',
    'text' : '#1E293B',
    'sub' : '#64748B',
}

# Heatmap: light gray navy (single hue, professional)
CMAP = LinearSegmentedColormap.from_list(
    'navy_ramp', ['#F8FAFC', '#C8D8EE', '#1E3A5F'])

# Per-version colour assignment
VERSION_COLORS = {
    'Original (MSA)' : P['navy'],
    'Mild Dialect' : P['teal'],
    'Heavy Dialect' : P['orange'],
}

def _style():
    plt.rcParams.update({
        'figure.facecolor' : P['bg'],
        'axes.facecolor' : P['panel'],
        'axes.edgecolor' : P['border'],
        'axes.labelcolor' : P['sub'],
        'xtick.color' : P['sub'],
        'ytick.color' : P['sub'],
        'text.color' : P['text'],
        'grid.color' : P['grid'],
        'grid.linewidth' : 0.6,
        'axes.spines.top' : False,
        'axes.spines.right': False,
        'font.family' : 'DejaVu Serif',
        'font.size' : 11,
        'axes.titlesize' : 13,
        'axes.titleweight' : 'bold',
        'axes.titlecolor' : P['text'],
        'axes.titlepad' : 14,
        'savefig.dpi' : 300,
        'savefig.bbox' : 'tight',
        'savefig.facecolor': P['bg'],
    })

_style()

def _clean_ax(ax):
    ax.set_facecolor(P['panel'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(P['border'])
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color=P['grid'], linewidth=0.5, alpha=0.9)

def _save(fig, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {os.path.basename(path)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Config
# ──────────────────────────────────────────────────────────────────────────────
CFG = {
    'model_id' : 'zhafyz/arabic-fake-news-arbert-afnd',
    'data_path' : 'afnd_clean.parquet',
    'text_col' : '', # auto-detect
    'label_col' : '', # auto-detect
    'sample_size': 2000,
    'batch_size' : 8,
    'threshold' : 0.5,
    'use_gpu' : True,
    'output_dir' : './results_v6',
    'perturb_n' : 500, # how many samples to perturb (subset for speed)
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Linguistic Perturbation Dictionaries
# ──────────────────────────────────────────────────────────────────────────────
#
# MILD = common everyday swaps that still sound natural
# (Egyptian + light Levantine — widely understood pan-Arabic)
# HEAVY = aggressive multi-dialect, unusual forms
# (Egyptian + Gulf + Maghrebi — maximum deviation from MSA)
#
# Each entry: (MSA_pattern, dialect_replacement, dialect_tag)
#

# ──────────────────────────────────────────────────────────────────────────────
# Perturbation Rules — per dialect, then combined
# ──────────────────────────────────────────────────────────────────────────────
#
# DIALECT_TARGET options:
# 'Egyptian' — Egyptian swaps only
# 'Levantine' — Levantine swaps only
# 'Gulf' — Gulf swaps only
# 'Maghrebi' — Maghrebi swaps only
# 'All' — All dialects applied in sequence (strongest perturbation)
#
DIALECT_TARGET = 'All' # Change here to target a specific dialect

# ── Egyptian ──────────────────────────────────────────────────────────────────
EGY_MILD = [
    (r'\bلم\b', 'مش', 'Egyptian'),
    (r'\bليس\b', 'مش', 'Egyptian'),
    (r'\bلست\b', 'أنا مش', 'Egyptian'),
    (r'\bماذا\b', 'إيه', 'Egyptian'),
    (r'\bكيف\b', 'إزاي', 'Egyptian'),
    (r'\bأين\b', 'فين', 'Egyptian'),
    (r'\bمتى\b', 'إمتى', 'Egyptian'),
    (r'\bلماذا\b', 'ليه', 'Egyptian'),
    (r'\bأنت\b', 'إنت', 'Egyptian'),
    (r'\bنحن\b', 'إحنا', 'Egyptian'),
    (r'\bأنتم\b', 'إنتوا', 'Egyptian'),
    (r'\bيريد\b', 'عاوز', 'Egyptian'),
    (r'\bأريد\b', 'عاوز', 'Egyptian'),
    (r'\bتريد\b', 'عاوز', 'Egyptian'),
    (r'\bيذهب\b', 'يروح', 'Egyptian'),
    (r'\bيأتي\b', 'ييجي', 'Egyptian'),
    (r'\bيقول\b', 'بيقول', 'Egyptian'),
    (r'\bيعرف\b', 'بيعرف', 'Egyptian'),
    (r'\bالآن\b', 'دلوقتي', 'Egyptian'),
    (r'\bاليوم\b', 'النهارده', 'Egyptian'),
    (r'\bغداً\b', 'بكره', 'Egyptian'),
    (r'\bأمس\b', 'إمبارح', 'Egyptian'),
    (r'\bجداً\b', 'أوي', 'Egyptian'),
    (r'\bهكذا\b', 'كده', 'Egyptian'),
    (r'\bلكن\b', 'بس', 'Egyptian'),
]
EGY_HEAVY = EGY_MILD + [
    (r'\bشيء\b', 'حاجة', 'Egyptian'),
    (r'\bشيئاً\b', 'حاجة', 'Egyptian'),
    (r'\bكثيراً\b', 'كتير', 'Egyptian'),
    (r'\bقليلاً\b', 'شوية', 'Egyptian'),
    (r'\bجميل\b', 'حلو', 'Egyptian'),
    (r'\bمعه\b', 'معاه', 'Egyptian'),
    (r'\bبعد\b', 'بعدين', 'Egyptian'),
    (r'\bقبل\b', 'قبل كده', 'Egyptian'),
    (r'\bبالطبع\b', 'أكيد', 'Egyptian'),
    (r'\bبالتأكيد\b','أكيد', 'Egyptian'),
    (r'\bسريعاً\b', 'بسرعة', 'Egyptian'),
    (r'\bلا أعرف\b', 'معرفش', 'Egyptian'),
]

# ── Levantine ─────────────────────────────────────────────────────────────────
LEV_MILD = [
    (r'\bلقد\b', 'صار', 'Levantine'),
    (r'\bلذلك\b', 'متل هيك', 'Levantine'),
    (r'\bلأن\b', 'لأنو', 'Levantine'),
    (r'\bحيث\b', 'وين', 'Levantine'),
    (r'\bأيضاً\b', 'كمان', 'Levantine'),
    (r'\bأيضا\b', 'كمان', 'Levantine'),
    (r'\bلكن\b', 'بس', 'Levantine'),
    (r'\bعندما\b', 'لما', 'Levantine'),
    (r'\bحتى\b', 'لحتى', 'Levantine'),
    (r'\bبعد ذلك\b', 'بعدين', 'Levantine'),
    (r'\bهذا\b', 'هاد', 'Levantine'),
    (r'\bهذه\b', 'هاي', 'Levantine'),
    (r'\bنحن\b', 'نحنا', 'Levantine'),
    (r'\bسوف\b', 'رح', 'Levantine'),
    (r'\bمن أجل\b', 'منشان', 'Levantine'),
    (r'\bربما\b', 'يمكن', 'Levantine'),
]
LEV_HEAVY = LEV_MILD + [
    (r'\bليس\b', 'مو', 'Levantine'),
    (r'\bماذا\b', 'شو', 'Levantine'),
    (r'\bأين\b', 'وين', 'Levantine'),
    (r'\bمتى\b', 'إيمتى', 'Levantine'),
    (r'\bلماذا\b', 'ليش', 'Levantine'),
    (r'\bيريد\b', 'بدو', 'Levantine'),
    (r'\bتريد\b', 'بدها', 'Levantine'),
    (r'\bكثيراً\b', 'كتير', 'Levantine'),
    (r'\bالآن\b', 'هلق', 'Levantine'),
    (r'\bشيء\b', 'شي', 'Levantine'),
    (r'\bبالطبع\b', 'طبعاً', 'Levantine'),
]

# ── Gulf ──────────────────────────────────────────────────────────────────────
GLF_MILD = [
    (r'\bلقد\b', 'چان', 'Gulf'),
    (r'\bلذلك\b', 'عيل', 'Gulf'),
    (r'\bأيضاً\b', 'بعد', 'Gulf'),
    (r'\bأيضا\b', 'بعد', 'Gulf'),
    (r'\bلكن\b', 'بس', 'Gulf'),
    (r'\bعندما\b', 'لما', 'Gulf'),
    (r'\bحتى\b', 'لين', 'Gulf'),
    (r'\bبعد ذلك\b', 'بعدين', 'Gulf'),
    (r'\bهذه\b', 'هذي', 'Gulf'),
    (r'\bنحن\b', 'إحنا', 'Gulf'),
    (r'\bمن أجل\b', 'عشان', 'Gulf'),
]
GLF_HEAVY = GLF_MILD + [
    (r'\bماذا\b', 'وش', 'Gulf'),
    (r'\bأين\b', 'وين', 'Gulf'),
    (r'\bلماذا\b', 'ليش', 'Gulf'),
    (r'\bيريد\b', 'يبغى', 'Gulf'),
    (r'\bكثيراً\b', 'وايد', 'Gulf'),
    (r'\bجدا\b', 'وايد', 'Gulf'),
    (r'\bالآن\b', 'الحين', 'Gulf'),
    (r'\bشيء\b', 'شي', 'Gulf'),
    (r'\bليس\b', 'مو', 'Gulf'),
    (r'\bلم يكن\b', 'ما چان', 'Gulf'),
    (r'\bجيد\b', 'زين', 'Gulf'),
    (r'\bصحيح\b', 'صح', 'Gulf'),
]

# ── Maghrebi ──────────────────────────────────────────────────────────────────
MAG_MILD = [
    (r'\bلماذا\b', 'علاش', 'Maghrebi'),
    (r'\bكيف\b', 'كيفاش', 'Maghrebi'),
    (r'\bكثيراً\b', 'بزاف', 'Maghrebi'),
    (r'\bكثيرا\b', 'بزاف', 'Maghrebi'),
    (r'\bأيضاً\b', 'برك', 'Maghrebi'),
    (r'\bلكن\b', 'وليكن', 'Maghrebi'),
    (r'\bهناك\b', 'تما', 'Maghrebi'),
]
MAG_HEAVY = MAG_MILD + [
    (r'\bنعم\b', 'إيه', 'Maghrebi'),
    (r'\bصحيح\b', 'مزيان', 'Maghrebi'),
    (r'\bجيد\b', 'مزيان', 'Maghrebi'),
    (r'\bأين\b', 'فين', 'Maghrebi'),
    (r'\bشيء\b', 'حاجة', 'Maghrebi'),
    (r'\bكيف حالك\b','كيفاش راك', 'Maghrebi'),
    (r'\bبالطبع\b', 'واش', 'Maghrebi'),
]

# ── Combined: All dialects ────────────────────────────────────────────────────
# Mild = Egyptian mild + Levantine mild + Gulf mild + Maghrebi mild
ALL_MILD = EGY_MILD + LEV_MILD + GLF_MILD + MAG_MILD
# Heavy = Egyptian heavy + Levantine heavy + Gulf heavy + Maghrebi heavy
ALL_HEAVY = EGY_HEAVY + LEV_HEAVY + GLF_HEAVY + MAG_HEAVY

# ── Selector ──────────────────────────────────────────────────────────────────
DIALECT_SWAPS = {
    'Egyptian' : (EGY_MILD, EGY_HEAVY),
    'Levantine' : (LEV_MILD, LEV_HEAVY),
    'Gulf' : (GLF_MILD, GLF_HEAVY),
    'Maghrebi' : (MAG_MILD, MAG_HEAVY),
    'All' : (ALL_MILD, ALL_HEAVY),
}

# Pick swaps based on DIALECT_TARGET
MILD_SWAPS, HEAVY_SWAPS = DIALECT_SWAPS.get(DIALECT_TARGET, DIALECT_SWAPS['All'])


def perturb_text(text: str, swaps: list) -> tuple:
    """
    Apply dialect swaps to MSA text.
    Returns (perturbed_text, list_of_dialects_used).
    """
    if not isinstance(text, str): return text, []
    dialects_used = set()
    for pattern, replacement, dialect in swaps:
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)
            dialects_used.add(dialect)
    return text, list(dialects_used)


def build_perturbation_df(df: pd.DataFrame, text_col: str, n: int) -> pd.DataFrame:
    """
    Take n MSA samples, create Mild and Heavy perturbed versions.
    Prints per-dialect swap statistics.
    """
    print(f"\nLINGUISTIC PERTURBATION (n={n}, target={DIALECT_TARGET})")
    print("-"*60)

    sample = df.sample(n=min(n, len(df)), random_state=42).copy()
    sample = sample.reset_index(drop=True)

    mild_texts, mild_dialects = [], []
    heavy_texts, heavy_dialects = [], []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=" Perturbing"):
        text = str(row[text_col])
        mt, md = perturb_text(text, MILD_SWAPS)
        ht, hd = perturb_text(text, HEAVY_SWAPS)
        mild_texts.append(mt); mild_dialects.append(md)
        heavy_texts.append(ht); heavy_dialects.append(hd)

    sample['text_original'] = sample[text_col].astype(str)
    sample['text_mild'] = mild_texts
    sample['text_heavy'] = heavy_texts
    sample['dialects_mild'] = [', '.join(sorted(d)) if d else 'None' for d in mild_dialects]
    sample['dialects_heavy']= [', '.join(sorted(d)) if d else 'None' for d in heavy_dialects]

    # Stats — overall
    mild_changed = sum(1 for o, m in zip(sample['text_original'], sample['text_mild']) if o != m)
    heavy_changed = sum(1 for o, h in zip(sample['text_original'], sample['text_heavy']) if o != h)
    print(f"Mild — texts changed : {mild_changed}/{len(sample)} ({mild_changed/len(sample)*100:.1f}%)")
    print(f"Heavy — texts changed : {heavy_changed}/{len(sample)} ({heavy_changed/len(sample)*100:.1f}%)")

    # Stats — per dialect breakdown
    if DIALECT_TARGET == 'All':
        print(f"\nPer-dialect contribution (Mild):")
        all_dialects = ['Egyptian', 'Levantine', 'Gulf', 'Maghrebi']
        for d in all_dialects:
            count = sample['dialects_mild'].str.contains(d).sum()
            print(f"{d:12s}: {count:4d} samples affected ({count/len(sample)*100:.1f}%)")

    return sample

# ──────────────────────────────────────────────────────────────────────────────
# 5. Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def _load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        for engine in ['pyarrow', 'fastparquet']:
            try: return pd.read_parquet(path, engine=engine)
            except Exception: pass
    for enc in ['utf-8', 'utf-8-sig', 'iso-8859-6', 'windows-1256', 'cp1256']:
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    raise RuntimeError(f"Cannot load: {path}")

def load_data():
    print(f"\nLoading: {CFG['data_path']}")
    df = _load(CFG['data_path'])
    print(f"{len(df):,} rows × {len(df.columns)} cols")

    tc = CFG['text_col']
    lc = CFG['label_col']

    if not tc or tc not in df.columns:
        for c in ['text', 'tweet', 'content', 'news', 'article', 'sentence', 'body', 'title']:
            if c in df.columns: tc = c; break
    if not lc or lc not in df.columns:
        for c in ['label', 'class', 'target', 'category', 'veracity', 'is_fake', 'fake']:
            if c in df.columns: lc = c; break

    if not tc or not lc:
        print(f"Available columns: {df.columns.tolist()}")
        if not tc: tc = input("Text column name: ").strip()
        if not lc: lc = input("Label column name: ").strip()

    df = df.dropna(subset=[tc, lc])
    n = CFG['sample_size']
    if n > 0 and n < len(df):
        df = df.sample(n=n, random_state=42)
        print(f"Sampled {n:,} rows")

    print(f"Text col: '{tc}' | Label col: '{lc}'")
    return df, tc, lc

# ──────────────────────────────────────────────────────────────────────────────
# 7. Label Normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_labels(df, lc):
    uniq = df[lc].unique(); lmap = {}; rev = {}
    for lbl in uniq:
        s = str(lbl).lower().strip()
        if any(w in s for w in ['fake', 'not_credible', 'false', 'mislead', 'lie', 'fraud', 'hoax']):
            lmap[lbl] = 1; rev[1] = 'Fake'
        elif any(w in s for w in ['real', 'credible', 'true', 'legit', 'fact', 'honest']):
            lmap[lbl] = 0; rev[0] = 'Real'
        else:
            try: n = float(lbl); lmap[lbl] = int(n); rev[int(n)] = f'Class {int(n)}'
            except: pass
    if not lmap:
        for i, l in enumerate(uniq): lmap[l] = i; rev[i] = str(l)

    df = df.copy()
    df['label_encoded'] = df[lc].map(lmap)
    bad = df['label_encoded'].isna()
    if bad.any(): df = df[~bad].copy()
    df['label_encoded'] = df['label_encoded'].astype(int)
    print(f"Labels: { {v: (df['label_encoded']==k).sum() for k, v in rev.items()} }")
    return df, lmap, rev

# ──────────────────────────────────────────────────────────────────────────────
# 8. Inference
# ──────────────────────────────────────────────────────────────────────────────
def infer(texts, tokenizer, model, device):
    model.eval()
    preds, confs, probs_list = [], [], []
    bs = CFG['batch_size']; th = CFG['threshold']
    for i in tqdm(range(0, len(texts), bs), desc=" Inference", leave=False):
        batch = [clean(str(t)) for t in texts[i:i+bs]]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                           max_length=256, padding=True,
                           add_special_tokens=True).to(device)
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits, dim=-1)
            mx, ids = torch.max(p, dim=-1)
            ids[mx < th] = -1
        preds.extend(ids.cpu().numpy())
        confs.extend(mx.cpu().numpy())
        probs_list.extend(p.cpu().numpy())
    return np.array(preds), np.array(confs), np.array(probs_list)

# ──────────────────────────────────────────────────────────────────────────────
# 9. Metrics
# ──────────────────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred, y_prob, confs=None):
    m = {}
    mask = y_pred != -1
    yt = y_true[mask]; yp = y_pred[mask]
    m['uncertainty_rate'] = float((~mask).sum() / len(y_true))
    if len(yt) == 0: return m

    m['accuracy'] = float(accuracy_score(yt, yp))
    m['balanced_accuracy'] = float(balanced_accuracy_score(yt, yp))
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
    m['precision'] = float(p); m['recall'] = float(r); m['f1'] = float(f)
    m['mcc'] = float(matthews_corrcoef(yt, yp))
    m['kappa'] = float(cohen_kappa_score(yt, yp))
    m['per_class'] = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
    m['confusion_matrix'] = confusion_matrix(yt, yp)

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob.flatten()
            m['roc_auc'] = float(roc_auc_score(y_true, pos))
            m['log_loss'] = float(log_loss(y_true, y_prob))
        except: m['roc_auc'] = None

    if confs is not None:
        certain_confs = confs[mask]
        m['mean_confidence'] = float(np.mean(certain_confs)) if len(certain_confs) else 0.0
    return m

# ──────────────────────────────────────────────────────────────────────────────
# 10. Flip Rate Analysis
# ──────────────────────────────────────────────────────────────────────────────
def compute_flip_rates(preds_orig, preds_mild, preds_heavy):
    """
    Flip rate = % of samples where prediction changed vs original.
    Only counts certain predictions (not -1).
    """
    def flip_rate(a, b):
        valid = (a != -1) & (b != -1)
        if valid.sum() == 0: return 0.0
        return float((a[valid] != b[valid]).sum() / valid.sum())

    return {
        'mild_flip_rate' : flip_rate(preds_orig, preds_mild),
        'heavy_flip_rate': flip_rate(preds_orig, preds_heavy),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZATIONS — 5 charts per spec
# ──────────────────────────────────────────────────────────────────────────────

def viz1_performance_comparison(results, out_dir):
    """
    VIZ 1 — Grouped bar chart: Accuracy + F1 for Original / Mild / Heavy.
    Colors: Navy / Teal / Orange per spec.
    """
    versions = ['Original (MSA)', 'Mild Dialect', 'Heavy Dialect']
    accs = [results[v]['metrics'].get('accuracy', 0) for v in versions]
    f1s = [results[v]['metrics'].get('f1', 0) for v in versions]
    cols = [VERSION_COLORS[v] for v in versions]

    x = np.arange(len(versions)); w = 0.32
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(P['bg']); _clean_ax(ax)

    b1 = ax.bar(x - w/2, accs, w, color=cols, alpha=0.88, edgecolor='white', lw=0.8, label='Accuracy')
    b2 = ax.bar(x + w/2, f1s, w, color=cols, alpha=0.45, edgecolor='white', lw=0.8, label='F1-Score')

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.006, f'{h:.3f}',
                ha='center', fontsize=8.5, color=P['sub'], fontweight='bold')

    ax.axhline(0.9, color=P['gray'], linestyle=(0,(5,4)), lw=0.8, alpha=0.6, label='0.9 reference')
    ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylim(0.60, 1.10); ax.set_ylabel('Score', fontsize=10, color=P['sub'])

    handles = [mpatches.Patch(color=P['navy'], alpha=0.88, label='Original (MSA)'),
               mpatches.Patch(color=P['teal'], alpha=0.88, label='Mild Dialect'),
               mpatches.Patch(color=P['orange'], alpha=0.88, label='Heavy Dialect'),
               mpatches.Patch(color=P['gray'], alpha=0.88, label='Accuracy (solid)'),
               mpatches.Patch(color=P['gray'], alpha=0.45, label='F1-Score (faded)')]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=3,
              bbox_to_anchor=(0.5, 1.08), loc='upper center')
    ax.set_title('VIZ 1 — Performance Comparison Across Dialect Versions',
                 color=P['text'], pad=30)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'VIZ1_performance_comparison.png'))


def viz2_flip_rate(flip_data, out_dir):
    """
    VIZ 2 — Horizontal bar chart: Prediction Flip Rate for Mild / Heavy.
    Colors: Teal for Mild, Orange for Heavy.
    """
    labels = ['Mild Dialect\n(light swaps)', 'Heavy Dialect\n(aggressive swaps)']
    rates = [flip_data['mild_flip_rate'], flip_data['heavy_flip_rate']]
    colors = [P['teal'], P['orange']]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(P['bg']); _clean_ax(ax)
    ax.grid(True, axis='x', color=P['grid'], linewidth=0.5, alpha=0.9)
    ax.grid(False, axis='y')

    bars = ax.barh(labels, rates, color=colors, alpha=0.88,
                   edgecolor='white', lw=0.8, height=0.45)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{rate*100:.1f}%', va='center', fontsize=11,
                fontweight='bold', color=P['sub'])

    ax.set_xlabel('Prediction Flip Rate (% sentences that changed label)', fontsize=10, color=P['sub'])
    ax.set_xlim(0, max(rates)*1.3 + 0.05)
    ax.axvline(0.1, color=P['gray'], linestyle=(0,(4,3)), lw=0.9, alpha=0.7, label='10% threshold')
    ax.axvline(0.25, color=P['orange'], linestyle=(0,(4,3)), lw=0.9, alpha=0.7, label='25% critical')
    ax.legend(fontsize=9, frameon=False)
    ax.set_title('VIZ 2 — Prediction Flip Rate: Syntactic Weakness Analysis',
                 color=P['text'])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'VIZ2_flip_rate.png'))


def viz3_confidence_drop(results, out_dir):
    """
    VIZ 3 — Line plot: Mean confidence across Original / Mild / Heavy.
    Line: Navy. Points: Teal.
    """
    versions = ['Original (MSA)', 'Mild Dialect', 'Heavy Dialect']
    confs = [results[v]['metrics'].get('mean_confidence', 0) for v in versions]
    x = np.arange(len(versions))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(P['bg']); _clean_ax(ax)

    ax.plot(x, confs, color=P['navy'], lw=2.5, zorder=2)
    ax.fill_between(x, confs, min(confs)*0.97, color=P['navy'], alpha=0.08)
    ax.scatter(x, confs, color=P['teal'], s=90, zorder=3, edgecolors='white', lw=1.5)

    for xi, conf in zip(x, confs):
        ax.text(xi, conf + 0.003, f'{conf:.3f}', ha='center', fontsize=10,
                fontweight='bold', color=P['sub'])

    # Drop annotations
    for i in range(1, len(versions)):
        drop = confs[0] - confs[i]
        if drop > 0.005:
            ax.annotate(f'−{drop:.3f}',
                        xy=(i, confs[i]), xytext=(i + 0.12, confs[i] + 0.015),
                        fontsize=8.5, color=P['orange'],
                        arrowprops=dict(arrowstyle='->', color=P['orange'], lw=1.1))

    ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylabel('Mean Prediction Confidence', fontsize=10, color=P['sub'])
    ymin = min(confs) - 0.04; ymax = max(confs) + 0.04
    ax.set_ylim(max(0.5, ymin), min(1.02, ymax))
    ax.set_title('VIZ 3 — Confidence Drop Across Dialect Versions',
                 color=P['text'])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'VIZ3_confidence_drop.png'))


def viz4_confusion_matrices(results, rev, out_dir):
    """
    VIZ 4 — 3-panel confusion matrix (Original / Mild / Heavy).
    Gradient: light gray (#F8FAFC) navy (#1E3A5F).
    """
    versions = ['Original (MSA)', 'Mild Dialect', 'Heavy Dialect']
    class_labels = list(rev.values())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(P['bg'])

    for ax, ver in zip(axes, versions):
        cm = results[ver]['metrics'].get('confusion_matrix')
        if cm is None:
            ax.set_visible(False); continue

        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, annot=False, cmap=CMAP,
                    xticklabels=class_labels, yticklabels=class_labels,
                    linewidths=2, linecolor=P['bg'], vmin=0,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    ax=ax)
        n = cm.shape[0]
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                fg = '#F8FAFC' if val / max(cm.max(), 1) > 0.5 else P['text']
                ax.text(j+0.5, i+0.33, str(val), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=fg)
                ax.text(j+0.5, i+0.67, f'{cm_pct[i,j]:.1f}%', ha='center', va='center',
                        fontsize=9, color=fg, alpha=0.8)
        ax.tick_params(length=0)
        ax.set_xlabel('Predicted', fontsize=9, color=P['sub'])
        ax.set_ylabel('Actual', fontsize=9, color=P['sub'])
        color = VERSION_COLORS[ver]
        ax.set_title(ver, fontsize=11, fontweight='bold', color=color, pad=10)

    fig.suptitle('VIZ 4 — Confusion Matrices: Original / Mild / Heavy Dialect',
                 fontsize=13, fontweight='bold', color=P['text'], y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'VIZ4_confusion_matrices.png'))


def viz5_degradation(results, out_dir):
    """
    VIZ 5 — Robustness Degradation Chart: drop in Acc + F1 from Original.
    Colors: Teal for Mild, Orange for Heavy.
    """
    versions = ['Mild Dialect', 'Heavy Dialect']
    colors = [P['teal'], P['orange']]
    orig_acc = results['Original (MSA)']['metrics'].get('accuracy', 0)
    orig_f1 = results['Original (MSA)']['metrics'].get('f1', 0)

    drops_acc = [orig_acc - results[v]['metrics'].get('accuracy', 0) for v in versions]
    drops_f1 = [orig_f1 - results[v]['metrics'].get('f1', 0) for v in versions]

    x = np.arange(len(versions)); w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(P['bg']); _clean_ax(ax)

    b1 = ax.bar(x - w/2, drops_acc, w, color=colors, alpha=0.88,
                edgecolor='white', lw=0.8, label='Δ Accuracy')
    b2 = ax.bar(x + w/2, drops_f1, w, color=colors, alpha=0.45,
                edgecolor='white', lw=0.8, label='Δ F1-Score')

    for b, val in zip(list(b1)+list(b2), drops_acc+drops_f1):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + 0.002,
                f'−{abs(val):.3f}', ha='center', fontsize=9,
                color=P['sub'], fontweight='bold')

    ax.axhline(0, color=P['border'], lw=0.8)
    ax.axhline(0.05, color=P['teal'], linestyle=(0,(4,3)), lw=0.9, alpha=0.8, label='5% mild concern')
    ax.axhline(0.15, color=P['orange'], linestyle=(0,(4,3)), lw=0.9, alpha=0.8, label='15% moderate')
    ax.axhline(0.30, color=P['navy'], linestyle=(0,(4,3)), lw=0.9, alpha=0.8, label='30% critical')

    ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylabel('Performance Drop (Original − Dialect Version)', fontsize=10, color=P['sub'])
    ax.legend(fontsize=9, frameon=False, ncol=3)
    ax.set_title('VIZ 5 — Dialect Robustness Degradation Chart',
                 color=P['text'])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'VIZ5_degradation.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 12. Summary + JSON
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(results, flip_data):
    print("\n" + "="*80)
    print("PERTURBATION SUMMARY")
    print("="*80)
    print(f"{'Version':22s} {'Acc':>8} {'F1':>8} {'MCC':>8} {'AvgConf':>10}")
    print("" + "-"*60)
    for ver in ['Original (MSA)', 'Mild Dialect', 'Heavy Dialect']:
        m = results[ver]['metrics']
        print(f"{ver:22s} {m.get('accuracy',0):>8.4f} {m.get('f1',0):>8.4f} "
              f"{m.get('mcc',0):>8.4f} {m.get('mean_confidence',0):>10.4f}")
    print("="*80)
    print(f"\nMild flip rate : {flip_data['mild_flip_rate']*100:.2f}% "
          + ("HIGH" if flip_data['mild_flip_rate'] > 0.25 else
             "MODERATE" if flip_data['mild_flip_rate'] > 0.10 else "LOW"))
    print(f"Heavy flip rate : {flip_data['heavy_flip_rate']*100:.2f}% "
          + ("HIGH" if flip_data['heavy_flip_rate'] > 0.25 else
             "MODERATE" if flip_data['heavy_flip_rate'] > 0.10 else "LOW"))


def save_json_summary(results, flip_data, out_dir, ts):
    summary = {
        'timestamp': ts,
        'model': CFG['model_id'],
        'dataset': CFG['data_path'],
        'sample_size': CFG['sample_size'],
        'perturbation_n': CFG['perturb_n'],
        'flip_rates': flip_data,
        'versions': {}
    }
    for ver in ['Original (MSA)', 'Mild Dialect', 'Heavy Dialect']:
        m = dict(results[ver]['metrics'])
        for k in ['confusion_matrix', 'per_class', 'fpr', 'tpr']:
            m.pop(k, None)
        summary['versions'][ver] = m

    path = os.path.join(out_dir, f'perturbation_summary_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 13. Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = CFG['output_dir']
    viz_dir = os.path.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Device
    if CFG['use_gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\nCPU mode")

    # ── Load model ──
    print(f"\nLoading model: {CFG['model_id']}")
    print("First time may take a few minutes...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG['model_id'])
        model = AutoModelForSequenceClassification.from_pretrained(
                        CFG['model_id']).to(device)
        print("Model loaded!")
    except Exception as e:
        print(f"Model load failed: {e}"); return

    # ── Load data ──
    try:
        df, tc, lc = load_data()
    except Exception as e:
        print(f"Data load failed: {e}"); return

    df, lmap, rev = normalize_labels(df, lc)

    # ── Build perturbation dataset ──
    pert_df = build_perturbation_df(df, tc, CFG['perturb_n'])
    y_true = pert_df['label_encoded'].values

    # ── Run inference on all 3 versions ──
    results = {}
    for ver_name, text_col in [
        ('Original (MSA)', 'text_original'),
        ('Mild Dialect', 'text_mild'),
        ('Heavy Dialect', 'text_heavy'),
    ]:
        print(f"\nInference — {ver_name}")
        texts = pert_df[text_col].tolist()
        preds, confs, probs = infer(texts, tokenizer, model, device)
        metrics = calc_metrics(y_true, preds, probs, confs)
        results[ver_name] = {
            'preds' : preds,
            'confs' : confs,
            'metrics': metrics,
        }
        certain = (preds != -1).sum()
        correct = (y_true[preds != -1] == preds[preds != -1]).sum()
        print(f"Accuracy : {correct/certain*100:.2f}% ({correct}/{certain})")
        print(f"F1 : {metrics.get('f1',0):.4f}")
        print(f"Conf avg : {metrics.get('mean_confidence',0):.4f}")

    # ── Flip rate ──
    flip_data = compute_flip_rates(
        results['Original (MSA)']['preds'],
        results['Mild Dialect']['preds'],
        results['Heavy Dialect']['preds'],
    )

    # ── Save predictions CSV ──
    pert_df['pred_original'] = results['Original (MSA)']['preds']
    pert_df['pred_mild'] = results['Mild Dialect']['preds']
    pert_df['pred_heavy'] = results['Heavy Dialect']['preds']
    pert_df['conf_original'] = results['Original (MSA)']['confs']
    pert_df['conf_mild'] = results['Mild Dialect']['confs']
    pert_df['conf_heavy'] = results['Heavy Dialect']['confs']
    pert_df['flipped_mild'] = (pert_df['pred_original'] != pert_df['pred_mild'])
    pert_df['flipped_heavy'] = (pert_df['pred_original'] != pert_df['pred_heavy'])
    csv_path = os.path.join(out_dir, f'perturbation_predictions_{ts}.csv')
    pert_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nPredictions CSV: {csv_path}")

    # ── Visualizations ──
    print("\nGenerating visualizations...")
    viz1_performance_comparison(results, viz_dir)
    viz2_flip_rate(flip_data, viz_dir)
    viz3_confidence_drop(results, viz_dir)
    viz4_confusion_matrices(results, rev, viz_dir)
    viz5_degradation(results, viz_dir)

    # ── Summary ──
    print_summary(results, flip_data)
    save_json_summary(results, flip_data, out_dir, ts)

    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nDone! Output: {out_dir}")
    print(f"Visualizations : {viz_dir}/")


if __name__ == "__main__":
    main()