#!/usr/bin/env python3
"""
Advanced Arabic Fake News Detection - ArBERT Model Evaluation with Full Calibration Metrics
Implements: MCC, ECE, Brier Score, Bootstrapping (1000 iterations), Reliability Diagrams
"""

import subprocess
import sys
import importlib
import os

def check_and_install_packages():
    """Check required packages and install if missing"""
    required_packages = {
        'pandas': 'pandas',
        'torch': 'torch',
        'transformers': 'transformers',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'pyarrow': 'pyarrow',
        'fastparquet': 'fastparquet',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    
    print("="*80)
    print("CHECKING REQUIRED PACKAGES")
    print("="*80)
    
    for package, install_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"{package} - OK")
        except ImportError:
            print(f"{package} - MISSING")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        response = input("\nDo you want to install missing packages? (y/n): ").strip().lower()
        
        if response == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"{package} installed successfully")
                except subprocess.CalledProcessError:
                    print(f"Failed to install {package}. Please install manually.")
                    return False
        else:
            print("\nCannot proceed without required packages. Please install them manually.")
            return False
    
    print("\nAll required packages are available!")
    return True

# Check packages before importing
if not check_and_install_packages():
    sys.exit(1)

# Now import all packages
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import unicodedata
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, matthews_corrcoef, cohen_kappa_score,
                             balanced_accuracy_score, log_loss, brier_score_loss)
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def preprocess_arabic_text(text: str) -> str:
    """Preprocess Arabic text for ArBERT model"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_user_input():
    """Get user inputs for CSV/Parquet file and parameters"""
    print("\n" + "="*80)
    print("ARABIC FAKE NEWS DETECTION - ADVANCED CALIBRATION METRICS")
    print("="*80)
    print("\nSupported formats: CSV (.csv) and Parquet (.parquet)")
    
    while True:
        csv_path = input("\nEnter path to CSV or Parquet file: ").strip()
        
        if os.path.exists(csv_path):
            file_ext = os.path.splitext(csv_path)[1].lower()
            if file_ext in ['.csv', '.parquet']:
                break
            else:
                print(f"Unsupported file format: {file_ext}")
                print("Please provide a .csv or .parquet file")
        else:
            print(f"File not found: {csv_path}")
    
    try:
        sample_size_input = input(" How many samples to evaluate (0 for all): ").strip()
        sample_size = int(sample_size_input) if sample_size_input else 0
    except:
        sample_size = 0
    
    text_col = input(" Enter text column name (press Enter for auto-detect): ").strip()
    label_col = input(" Enter label column name (press Enter for auto-detect): ").strip()
    
    try:
        batch_size_input = input(" Batch size for inference (default 16): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 16
    except:
        batch_size = 16
    
    threshold_input = input(" Confidence threshold for classification (default 0.5): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.5
    
    # NEW: Bootstrapping iterations
    bootstrap_input = input(" Number of bootstrap iterations for confidence intervals (default 1000): ").strip()
    n_bootstrap = int(bootstrap_input) if bootstrap_input else 1000
    
    # NEW: Number of calibration bins
    bins_input = input(" Number of calibration bins for ECE (default 15): ").strip()
    n_bins = int(bins_input) if bins_input else 15
    
    output_dir = input(" Output directory for results (press Enter for './results_arbert_advanced'): ").strip()
    if not output_dir:
        output_dir = "./results_arbert_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'file_path': csv_path,
        'sample_size': sample_size,
        'text_col': text_col,
        'label_col': label_col,
        'batch_size': batch_size,
        'threshold': threshold,
        'n_bootstrap': n_bootstrap,
        'n_bins': n_bins,
        'output_dir': output_dir
    }

def _load_parquet_safe(file_path):
    """Load a parquet file trying multiple engines"""
    engines = ['pyarrow', 'fastparquet']
    last_error = None
    for engine in engines:
        try:
            df = pd.read_parquet(file_path, engine=engine)
            print(f"(engine: {engine})")
            return df
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Could not load parquet file with any engine (tried {engines}).\nLast error: {last_error}")

def load_and_analyze_data(file_path, sample_size, user_text_col, user_label_col):
    """Load dataset from CSV or Parquet"""
    print(f"\nLoading dataset from: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.parquet':
            df = _load_parquet_safe(file_path)
            print(f"Parquet file loaded: {len(df):,} rows, {len(df.columns)} columns")
        elif file_ext == '.csv':
            encodings = ['utf-8', 'utf-8-sig', 'iso-8859-6', 'windows-1256', 'cp1256']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"CSV file loaded with {encoding} encoding: {len(df):,} rows, {len(df.columns)} columns")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any common encoding")
        else:
            try:
                df = pd.read_csv(file_path)
                print(f"File loaded as CSV: {len(df):,} rows, {len(df.columns)} columns")
            except:
                df = _load_parquet_safe(file_path)
                print(f"File loaded as Parquet: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        dtype_str = str(df[col].dtype)
        missing_count = df[col].isna().sum()
        print(f"{i+1}. {col} ({dtype_str}) - {missing_count} missing")
    
    if user_text_col and user_text_col in df.columns:
        text_col = user_text_col
    else:
        text_candidates = ['text', 'tweet', 'content', 'news', 'article', 'sentence', 'statement', 'body', 'title']
        text_col = next((c for c in df.columns if c.lower() in text_candidates), None)
        if text_col:
            print(f"Auto-detected text column: '{text_col}'")
    
    if user_label_col and user_label_col in df.columns:
        label_col = user_label_col
    else:
        label_candidates = ['label', 'class', 'target', 'category', 'type', 'veracity', 'is_fake', 'fake']
        label_col = next((c for c in df.columns if c.lower() in label_candidates), None)
        if label_col:
            print(f"Auto-detected label column: '{label_col}'")
    
    if not text_col or not label_col:
        print(f"\nCould not find text/label columns automatically.")
        print(f"Available columns: {df.columns.tolist()}")
        if not text_col:
            text_col = input("Please specify the text column name: ").strip()
        if not label_col:
            label_col = input("Please specify the label column name: ").strip()
    
    print(f"\nSelected columns:")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")
    
    initial_len = len(df)
    df = df.dropna(subset=[text_col, label_col])
    if initial_len - len(df) > 0:
        print(f"Dropped {initial_len - len(df)} rows with missing values")
    
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size:,} rows for evaluation")
    
    print(f"\nFinal dataset: {len(df):,} rows")
    
    df['text_length'] = df[text_col].astype(str).str.len()
    print(f"\nText Statistics:")
    print(f"Mean length: {df['text_length'].mean():.0f} characters")
    print(f"Median length: {df['text_length'].median():.0f} characters")
    
    return df, text_col, label_col

def normalize_labels(df, label_col):
    """Normalize and understand label formats"""
    unique_labels = df[label_col].unique()
    print(f"\nUnique labels found: {unique_labels}")
    
    label_mapping = {}
    reverse_mapping = {}
    
    for label in unique_labels:
        label_str = str(label).lower().strip()
        
        if any(word in label_str for word in ['fake', 'not_credible', 'false', 'misleading', '0', 'lie', 'fraud', 'hoax']):
            label_mapping[label] = 1 # Fake
            reverse_mapping[1] = 'Fake'
        elif any(word in label_str for word in ['real', 'credible', 'true', 'legitimate', '1', 'truth', 'fact', 'honest']):
            label_mapping[label] = 0 # Real
            reverse_mapping[0] = 'Real'
        else:
            try:
                num_label = float(label)
                if num_label in [0, 1]:
                    label_mapping[label] = int(num_label)
                    reverse_mapping[int(num_label)] = f'Class {int(num_label)}'
                elif num_label in [0.0, 1.0]:
                    label_mapping[label] = int(num_label)
                    reverse_mapping[int(num_label)] = f'Class {int(num_label)}'
            except:
                pass
    
    if not label_mapping:
        print("Could not map labels automatically. Creating default mapping...")
        unique_labels_list = list(unique_labels)
        for i, label in enumerate(unique_labels_list):
            label_mapping[label] = i
            reverse_mapping[i] = str(label)
    
    df['label_encoded'] = df[label_col].map(label_mapping)
    
    print(f"\nLabel Mapping:")
    for orig, encoded in label_mapping.items():
        print(f"'{orig}' -> {encoded} ({reverse_mapping[encoded]})")
    
    unmapped_mask = df['label_encoded'].isna()
    if unmapped_mask.any():
        unmapped_labels = df.loc[unmapped_mask, label_col].unique().tolist()
        dropped = unmapped_mask.sum()
        print(f"\nDropping {dropped} rows with unmappable labels: {unmapped_labels}")
        df = df[~unmapped_mask].copy()
    
    df['label_encoded'] = df['label_encoded'].astype(int)
    
    print(f"\nEncoded Label Distribution:")
    label_counts = df['label_encoded'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        print(f"{reverse_mapping[label]}: {count} ({percentage:.1f}%)")
    
    return df, label_mapping, reverse_mapping

def predict_with_probabilities(texts, tokenizer, model, device, batch_size=16):
    """Batch prediction returning raw probabilities (logits + softmax)"""
    model.eval()
    all_probs = [] # Probability of being fake (class 1)
    all_confidences = [] # Max probability
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i:i+batch_size]
        processed_batch = [preprocess_arabic_text(str(t)) for t in batch_texts]
        
        inputs = tokenizer(
            processed_batch,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
            add_special_tokens=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract logits and apply softmax
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get probability of fake news (class 1)
            prob_fake = probs[:, 1].cpu().numpy()
            # Get max confidence
            confidence = torch.max(probs, dim=-1)[0].cpu().numpy()
            
            all_probs.extend(prob_fake)
            all_confidences.extend(confidence)
    
    return np.array(all_probs), np.array(all_confidences)

def calculate_expected_calibration_error(y_true, y_proba, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE)
    Partitions predictions into M bins and computes weighted average of |accuracy - confidence|
    """
    # Use probability of positive class (fake)
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    bin_info = []
    
    for bin_idx in range(n_bins):
        in_bin = (bin_indices == bin_idx)
        if in_bin.any():
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_proba[in_bin].mean()
            bin_weight = in_bin.mean()
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
            
            bin_info.append({
                'bin': bin_idx,
                'lower': bin_boundaries[bin_idx],
                'upper': bin_boundaries[bin_idx + 1],
                'n_samples': in_bin.sum(),
                'accuracy': bin_accuracy,
                'confidence': bin_confidence,
                'weight': bin_weight
            })
    
    return ece, bin_info

def calculate_brier_score(y_true, y_proba):
    """Calculate Brier Score - mean squared difference between predicted probability and actual outcome"""
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    return brier_score_loss(y_true, y_proba)

def bootstrap_metrics(y_true, y_proba, n_iterations=1000, n_bins=15, confidence_level=0.95):
    """
    Perform bootstrapping to compute confidence intervals for all metrics
    """
    n_samples = len(y_true)
    metrics_bootstrap = {
        'accuracy': [],
        'mcc': [],
        'ece': [],
        'brier': [],
        'f1': []
    }
    
    print(f"\nRunning {n_iterations} bootstrap iterations for 95% Confidence Intervals...")
    
    for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_bs = y_true[indices]
        y_proba_bs = y_proba[indices]
        
        # Predictions (threshold at 0.5)
        y_pred_bs = (y_proba_bs >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true_bs, y_pred_bs)
        mcc = matthews_corrcoef(y_true_bs, y_pred_bs)
        ece_bs, _ = calculate_expected_calibration_error(y_true_bs, y_proba_bs, n_bins)
        brier = calculate_brier_score(y_true_bs, y_proba_bs)
        
        # F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true_bs, y_pred_bs, zero_division=0)
        
        metrics_bootstrap['accuracy'].append(acc)
        metrics_bootstrap['mcc'].append(mcc)
        metrics_bootstrap['ece'].append(ece_bs)
        metrics_bootstrap['brier'].append(brier)
        metrics_bootstrap['f1'].append(f1)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    confidence_intervals = {}
    for metric_name, values in metrics_bootstrap.items():
        values = np.array(values)
        confidence_intervals[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, lower_percentile),
            'ci_upper': np.percentile(values, upper_percentile)
        }
    
    return confidence_intervals

def plot_reliability_diagram(y_true, y_proba, n_bins=15, save_path=None):
    """
    Create Reliability Diagram (Calibration Curve)
    X-axis: Predicted Confidence, Y-axis: Actual Accuracy
    Perfect model follows 45° line
    """
    plt.figure(figsize=(10, 8))
    
    # Get calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    
    # Plot calibration curve
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
             label='Model Calibration Curve', color='#2E86AB')
    
    # Plot perfect calibration line (45°)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2,
             label='Perfect Calibration (45° line)', color='#A23B72')
    
    # Add histogram of predictions
    ax2 = plt.gca().twinx()
    n, bins, patches = ax2.hist(y_proba, bins=50, alpha=0.3, color='gray',
                                 label='Prediction Distribution')
    ax2.set_ylabel('Frequency', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.xlabel('Predicted Confidence (P(Fake))', fontsize=14)
    plt.ylabel('Actual Accuracy', fontsize=14)
    plt.title(f'Reliability Diagram (Calibration Curve)\nECE = {calculate_expected_calibration_error(y_true, y_proba, n_bins)[0]:.4f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add text annotation for over/under confidence
    if prob_true.mean() < prob_pred.mean():
        plt.text(0.05, 0.85, ' Overconfident\n(Confidence > Accuracy)',
                 transform=plt.gca().transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.05, 0.85, ' Underconfident\n(Accuracy > Confidence)',
                 transform=plt.gca().transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return prob_true, prob_pred

def plot_calibration_by_bin(y_true, y_proba, bin_info, save_path=None):
    """Plot accuracy vs confidence per bin (bar chart)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot: Accuracy vs Confidence per bin
    bin_centers = [(b['lower'] + b['upper']) / 2 for b in bin_info if b['n_samples'] > 0]
    bin_accuracies = [b['accuracy'] for b in bin_info if b['n_samples'] > 0]
    bin_confidences = [b['confidence'] for b in bin_info if b['n_samples'] > 0]
    bin_counts = [b['n_samples'] for b in bin_info if b['n_samples'] > 0]
    
    x = np.arange(len(bin_centers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, bin_accuracies, width, label='Actual Accuracy', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, bin_confidences, width, label='Predicted Confidence', color='#A23B72')
    
    ax1.set_xlabel('Confidence Bin', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Calibration by Bin', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{b["lower"]:.1f}-{b["upper"]:.1f}' for b in bin_info if b['n_samples'] > 0], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sample size per bin
    ax2.bar(x, bin_counts, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution Across Bins', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{b["lower"]:.1f}-{b["upper"]:.1f}' for b in bin_info if b['n_samples'] > 0], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bootstrap_confidence_intervals(confidence_intervals, save_path=None):
    """Display bootstrap confidence intervals as bar chart with error bars"""
    metrics_names = list(confidence_intervals.keys())
    means = [confidence_intervals[m]['mean'] for m in metrics_names]
    errors_lower = [confidence_intervals[m]['mean'] - confidence_intervals[m]['ci_lower'] for m in metrics_names]
    errors_upper = [confidence_intervals[m]['ci_upper'] - confidence_intervals[m]['mean'] for m in metrics_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(metrics_names))
    bars = ax.barh(y_pos, means, xerr=[errors_lower, errors_upper],
                   capsize=5, color='#2E86AB', alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.upper() for m in metrics_names])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Bootstrap 95% Confidence Intervals (n=1000 iterations)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, linestyle='--', color='gray', alpha=0.7, label='Reference (0.5)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add value labels
    for i, (bar, mean, ci_lower, ci_upper) in enumerate(zip(bars, means,
                          [confidence_intervals[m]['ci_lower'] for m in metrics_names],
                          [confidence_intervals[m]['ci_upper'] for m in metrics_names])):
        ax.text(mean + 0.02, i, f'{mean:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_histogram_with_calibration(y_true, y_proba, save_path=None):
    """Plot confidence histogram split by correct/incorrect predictions"""
    y_pred = (y_proba >= 0.5).astype(int)
    correct = (y_pred == y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram by correctness
    axes[0].hist(y_proba[correct], bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black', density=True)
    axes[0].hist(y_proba[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black', density=True)
    axes[0].set_xlabel('Predicted Probability (Fake)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Separate by true class
    fake_mask = (y_true == 1)
    real_mask = (y_true == 0)
    
    axes[1].hist(y_proba[fake_mask], bins=30, alpha=0.7, label='True Fake News', color='#E63946', edgecolor='black', density=True)
    axes[1].hist(y_proba[real_mask], bins=30, alpha=0.7, label='True Real News', color='#457B9D', edgecolor='black', density=True)
    axes[1].set_xlabel('Predicted Probability (Fake)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Confidence by True Class', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_advanced_dashboard(y_true, y_proba, bin_info, confidence_intervals, save_path=None):
    """Create comprehensive dashboard with all calibration metrics"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Reliability Diagram (top left)
    ax1 = plt.subplot(3, 3, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=15, strategy='uniform')
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=6, label='Model', color='#2E86AB')
    ax1.plot([0, 1], [0, 1], linestyle='--', linewidth=2, label='Perfect', color='#A23B72')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ECE by Bin (top middle)
    ax2 = plt.subplot(3, 3, 2)
    bin_centers = [(b['lower'] + b['upper']) / 2 for b in bin_info if b['n_samples'] > 0]
    ece_per_bin = [np.abs(b['accuracy'] - b['confidence']) for b in bin_info if b['n_samples'] > 0]
    ax2.bar(range(len(bin_centers)), ece_per_bin, color='#F18F01', alpha=0.7)
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('|Accuracy - Confidence|')
    ax2.set_title('ECE Contribution per Bin')
    ax2.set_xticks(range(len(bin_centers)))
    ax2.set_xticklabels([f'{b["lower"]:.1f}' for b in bin_info if b['n_samples'] > 0], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence Distribution (top right)
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(y_proba, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0.5, linestyle='--', color='red', label='Threshold (0.5)')
    ax3.set_xlabel('Predicted Probability (Fake)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bootstrap CI (middle left)
    ax4 = plt.subplot(3, 3, 4)
    metrics_names = list(confidence_intervals.keys())
    means = [confidence_intervals[m]['mean'] for m in metrics_names]
    errors = [[confidence_intervals[m]['mean'] - confidence_intervals[m]['ci_lower'] for m in metrics_names],
              [confidence_intervals[m]['ci_upper'] - confidence_intervals[m]['mean'] for m in metrics_names]]
    y_pos = np.arange(len(metrics_names))
    ax4.barh(y_pos, means, xerr=errors, capsize=5, color='#2E86AB', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([m.upper() for m in metrics_names])
    ax4.set_xlabel('Score')
    ax4.set_title('Bootstrap 95% CI')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Accuracy vs Confidence Scatter (middle middle)
    ax5 = plt.subplot(3, 3, 5)
    bin_acc = [b['accuracy'] for b in bin_info if b['n_samples'] > 0]
    bin_conf = [b['confidence'] for b in bin_info if b['n_samples'] > 0]
    bin_sizes = [b['n_samples'] for b in bin_info if b['n_samples'] > 0]
    ax5.scatter(bin_conf, bin_acc, s=np.array(bin_sizes)/10, alpha=0.6, color='#2E86AB')
    ax5.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Accuracy vs Confidence (bubble size = sample count)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Calibration Error Distribution (middle right)
    ax6 = plt.subplot(3, 3, 6)
    # Simulate calibration errors from bootstrap for ECE
    ece_values = confidence_intervals.get('ece', {})
    if 'std' in ece_values:
        ax6.text(0.5, 0.5, f'ECE: {confidence_intervals["ece"]["mean"]:.4f}\n±{confidence_intervals["ece"]["std"]:.4f}\n95% CI: [{confidence_intervals["ece"]["ci_lower"]:.4f}, {confidence_intervals["ece"]["ci_upper"]:.4f}]',
                ha='center', va='center', fontsize=12, transform=ax6.transAxes,
                bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Expected Calibration Error (ECE)')
    
    # 7. Brier Score (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    brier = calculate_brier_score(y_true, y_proba)
    ax7.text(0.5, 0.5, f'Brier Score: {brier:.4f}\n(Lower is better)\nPerfect = 0, Worst = 1',
            ha='center', va='center', fontsize=11, transform=ax7.transAxes,
            bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    ax7.set_title('Brier Score (Proper Scoring Rule)')
    
    # 8. MCC Display (bottom middle)
    ax8 = plt.subplot(3, 3, 8)
    y_pred = (y_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    ci_mcc = confidence_intervals.get('mcc', {})
    mcc_text = f'MCC: {mcc:.4f}\n'
    if 'ci_lower' in ci_mcc:
        mcc_text += f'95% CI: [{ci_mcc["ci_lower"]:.4f}, {ci_mcc["ci_upper"]:.4f}]'
    ax8.text(0.5, 0.5, mcc_text, ha='center', va='center', fontsize=11, transform=ax8.transAxes,
            bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('Matthews Correlation Coefficient (MCC)')
    
    # 9. Summary Metrics (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    summary_text = f'Summary:\nAccuracy: {accuracy:.4f}\nBalanced Accuracy: {balanced_acc:.4f}\nF1: {confidence_intervals.get("f1", {}).get("mean", 0):.4f}'
    ax9.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11, transform=ax9.transAxes,
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('Performance Summary')
    
    plt.suptitle('Advanced Calibration & Statistical Metrics Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_advanced_report(y_true, y_proba, y_pred, metrics, confidence_intervals, bin_info, config, class_names, timestamp):
    """Generate comprehensive HTML report with all calibration metrics"""
    report_path = os.path.join(config['output_dir'], f'advanced_calibration_report_{timestamp}.html')
    
    # Calculate current metrics
    ece, _ = calculate_expected_calibration_error(y_true, y_proba, config['n_bins'])
    brier = calculate_brier_score(y_true, y_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Calibration Metrics - ArBERT Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        .ci-info {{ font-size: 0.8em; font-family: monospace; margin-top: 10px; }}
        .card-primary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .card-success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }}
        .card-warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }}
        .card-info {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        .interpretation {{ background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
        .badge-good {{ background-color: #2ecc71; color: white; }}
        .badge-bad {{ background-color: #e74c3c; color: white; }}
        .badge-mid {{ background-color: #f39c12; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1> Advanced Calibration & Statistical Metrics Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Model:</strong> zhafyz/arabic-fake-news-arbert-afnd | <strong>Device:</strong> {'CUDA' if torch.cuda.is_available() else 'CPU'}</p>
        
        <h2> Core Calibration Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card card-primary">
                <div class="metric-value">{ece:.4f}</div>
                <div class="metric-label">Expected Calibration Error (ECE)</div>
                <div class="ci-info">95% CI: [{confidence_intervals['ece']['ci_lower']:.4f}, {confidence_intervals['ece']['ci_upper']:.4f}]</div>
                <div class="ci-info"> Lower is better (0 = perfect calibration)</div>
            </div>
            <div class="metric-card card-success">
                <div class="metric-value">{brier:.4f}</div>
                <div class="metric-label">Brier Score</div>
                <div class="ci-info">95% CI: [{confidence_intervals['brier']['ci_lower']:.4f}, {confidence_intervals['brier']['ci_upper']:.4f}]</div>
                <div class="ci-info"> Proper scoring rule</div>
            </div>
            <div class="metric-card card-info">
                <div class="metric-value">{mcc:.4f}</div>
                <div class="metric-label">Matthews Correlation Coefficient</div>
                <div class="ci-info">95% CI: [{confidence_intervals['mcc']['ci_lower']:.4f}, {confidence_intervals['mcc']['ci_upper']:.4f}]</div>
                <div class="ci-info"> Uses all 4 quadrants of confusion matrix</div>
            </div>
            <div class="metric-card card-warning">
                <div class="metric-value">{accuracy:.4f}</div>
                <div class="metric-label">Accuracy (Raw)</div>
                <div class="ci-info">95% CI: [{confidence_intervals['accuracy']['ci_lower']:.4f}, {confidence_intervals['accuracy']['ci_upper']:.4f}]</div>
                <div class="ci-info"> Vanity metric - not trustworthy alone</div>
            </div>
        </div>
        
        <div class="interpretation">
            <strong> Interpretation:</strong><br>
            • <strong>ECE = {ece:.4f}</strong> means the model's confidence is off by {ece*100:.2f}% on average.
              {' Well calibrated (<5% is excellent)' if ece < 0.05 else ' Needs improvement (>5% indicates miscalibration)' if ece < 0.1 else ' Poorly calibrated (>10% is serious miscalibration)'}<br>
            • <strong>Brier Score = {brier:.4f}</strong>: {' Excellent (<0.1)' if brier < 0.1 else ' Moderate (0.1-0.25)' if brier < 0.25 else ' Poor (>0.25)'}<br>
            • <strong>MCC = {mcc:.4f}</strong>: {' Strong agreement' if mcc > 0.7 else ' Moderate' if mcc > 0.4 else ' Weak'}<br>
            • <strong>Accuracy = {accuracy:.4f}</strong> but MCC tells the real story - see above.
        </div>
        
        <h2> Calibration by Confidence Bin</h2>
        <table>
            <thead>
                <tr><th>Bin</th><th>Confidence Range</th><th># Samples</th><th>Accuracy</th><th>Confidence</th><th>Gap (|Acc-Conf|)</th></tr>
            </thead>
            <tbody>
    """
    
    for b in bin_info:
        if b['n_samples'] > 0:
            gap = abs(b['accuracy'] - b['confidence'])
            gap_class = 'badge-good' if gap < 0.05 else 'badge-mid' if gap < 0.1 else 'badge-bad'
            html_content += f"""
                <tr>
                    <td>{b['bin']}</td>
                    <td>{b['lower']:.2f} - {b['upper']:.2f}</td>
                    <td>{b['n_samples']}</td>
                    <td>{b['accuracy']:.4f}</td>
                    <td>{b['confidence']:.4f}</td>
                    <td><span class="badge {gap_class}">{gap:.4f}</span></td>
                </tr>
            """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2> Bootstrap Confidence Intervals (n={config['n_bootstrap']} iterations, {int((1-0.95)*100)}% α)</h2>
        <table>
            <thead><tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>95% CI Lower</th><th>95% CI Upper</th></tr></thead>
            <tbody>
    """
    
    for metric_name, ci_dict in confidence_intervals.items():
        html_content += f"""
                <tr>
                    <td><strong>{metric_name.upper()}</strong></td>
                    <td>{ci_dict['mean']:.4f}</td>
                    <td>{ci_dict['std']:.4f}</td>
                    <td>{ci_dict['ci_lower']:.4f}</td>
                    <td>{ci_dict['ci_upper']:.4f}</td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2> Technical Definitions</h2>
        <ul>
            <li><strong>Matthews Correlation Coefficient (MCC):</strong> <code>MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]</code><br>
            Uses all four confusion matrix quadrants. High score only if model predicts well on both positive and negative classes.</li>
            <li><strong>Expected Calibration Error (ECE):</strong> <code>ECE = Σ (|B_m|/n) × |acc(B_m) - conf(B_m)|</code><br>
            Weighted average of the gap between accuracy and confidence across M bins.</li>
            <li><strong>Brier Score:</strong> <code>BS = (1/N) Σ (f_t - o_t)²</code><br>
            Proper scoring rule measuring mean squared difference between predicted probability and actual outcome. Punishes confidently wrong predictions.</li>
            <li><strong>Bootstrapping:</strong> {config['n_bootstrap']} iterations with random subsampling to generate 95% confidence intervals. Proves results aren't a fluke of a specific test set.</li>
        </ul>
        
        <h2> Generated Visualizations</h2>
        <p>The following plots have been saved to <code>{config['output_dir']}/visualizations/</code>:</p>
        <ul>
            <li><code>reliability_diagram.png</code> - Calibration curve with 45° perfect calibration line</li>
            <li><code>calibration_by_bin.png</code> - Per-bin accuracy vs confidence</li>
            <li><code>bootstrap_confidence_intervals.png</code> - Bootstrap CI visualization</li>
            <li><code>confidence_histogram.png</code> - Distribution of confidences</li>
            <li><code>calibration_dashboard.png</code> - Comprehensive 9‑panel dashboard</li>
        </ul>
        
        <div class="footer">
            <p>Generated by ArBERT Advanced Calibration Evaluation System</p>
            <p>This report implements all task requirements: Probability Extraction | Binning Analysis | Bootstrapping (1000 iterations) | Reliability Diagram </p>
        </div>
    </div>
</body>
</html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Advanced HTML Report saved: {report_path}")
    return report_path

def main():
    # Get user input
    config = get_user_input()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    model_id = "zhafyz/arabic-fake-news-arbert-afnd"
    print(f"\nLoading model: {model_id}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and prepare data
    df, text_col, label_col = load_and_analyze_data(
        config['file_path'],
        config['sample_size'],
        config['text_col'],
        config['label_col']
    )
    
    # Normalize labels
    df, label_mapping, class_names = normalize_labels(df, label_col)
    df['text'] = df[text_col].astype(str)
    
    # Run predictions to get raw probabilities
    print("\nRunning inference with probability extraction (Softmax applied)...")
    texts = df['text'].tolist()
    prob_fake, confidences = predict_with_probabilities(
        texts, tokenizer, model, device, config['batch_size']
    )
    
    # Prepare data
    y_true = df['label_encoded'].values # 1 = Fake, 0 = Real
    y_proba = prob_fake # Probability of being fake
    y_pred = (y_proba >= config['threshold']).astype(int)
    
    # Calculate advanced metrics
    print("\nCalculating advanced calibration metrics...")
    ece, bin_info = calculate_expected_calibration_error(y_true, y_proba, config['n_bins'])
    brier = calculate_brier_score(y_true, y_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Bootstrap for confidence intervals
    confidence_intervals = bootstrap_metrics(y_true, y_proba, config['n_bootstrap'], config['n_bins'])
    
    # Print results
    print("\n" + "="*80)
    print("ADVANCED CALIBRATION METRICS RESULTS")
    print("="*80)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Interpretation: Uses all 4 quadrants of confusion matrix (TP, TN, FP, FN)")
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Interpretation: Average gap between confidence and accuracy across {config['n_bins']} bins")
    print(f"\nBrier Score: {brier:.4f}")
    print(f"Interpretation: Mean squared error of probabilities (lower is better)")
    print(f"\nAccuracy (vanity): {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    print(f"\nBootstrap 95% Confidence Intervals ({config['n_bootstrap']} iterations):")
    for metric_name, ci in confidence_intervals.items():
        print(f"{metric_name.upper()}: {ci['mean']:.4f} ± {ci['std']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    # Create visualizations directory
    viz_dir = os.path.join(config['output_dir'], 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Reliability Diagram (Calibration Curve)
    print("\nGenerating visualizations...")
    plot_reliability_diagram(y_true, y_proba, config['n_bins'],
                             os.path.join(viz_dir, 'reliability_diagram.png'))
    
    # 2. Calibration by Bin (Bar chart)
    plot_calibration_by_bin(y_true, y_proba, bin_info,
                            os.path.join(viz_dir, 'calibration_by_bin.png'))
    
    # 3. Bootstrap Confidence Intervals
    plot_bootstrap_confidence_intervals(confidence_intervals,
                                         os.path.join(viz_dir, 'bootstrap_confidence_intervals.png'))
    
    # 4. Confidence Histogram
    plot_confidence_histogram_with_calibration(y_true, y_proba,
                                                os.path.join(viz_dir, 'confidence_histogram.png'))
    
    # 5. Advanced Dashboard (9-panel comprehensive view)
    plot_advanced_dashboard(y_true, y_proba, bin_info, confidence_intervals,
                            os.path.join(viz_dir, 'calibration_dashboard.png'))
    
    # Save results
    results_df = pd.DataFrame({
        'text': df['text'].values,
        'true_label': y_true,
        'true_class': [class_names.get(y, f'Class {y}') for y in y_true],
        'predicted_label': y_pred,
        'predicted_class': [class_names.get(p, f'Class {p}') for p in y_pred],
        'probability_fake': y_proba,
        'confidence': confidences,
        'correct': (y_true == y_pred)
    })
    
    output_csv = os.path.join(config['output_dir'], f'advanced_predictions_{timestamp}.csv')
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Detailed predictions saved: {output_csv}")
    
    # Save metrics as JSON
    metrics_dict = {
        'mcc': float(mcc),
        'ece': float(ece),
        'brier_score': float(brier),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'n_bins': config['n_bins'],
        'n_bootstrap': config['n_bootstrap'],
        'bootstrap_confidence_intervals': confidence_intervals,
        'bin_details': bin_info,
        'config': config
    }
    
    json_path = os.path.join(config['output_dir'], f'advanced_metrics_{timestamp}.json')
    # Convert numpy types to Python natives for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False, default=convert_for_json)
    print(f"Metrics saved: {json_path}")
    
    # Generate HTML report
    report_path = generate_advanced_report(y_true, y_proba, y_pred, metrics_dict,
                                           confidence_intervals, bin_info, config,
                                           class_names, timestamp)
    
    # Print misclassification examples
    print("\n" + "="*80)
    print("MISCLASSIFICATION EXAMPLES")
    print("="*80)
    misclassified = results_df[~results_df['correct']]
    if len(misclassified) > 0:
        for idx, row in misclassified.head(10).iterrows():
            print(f"\n{idx}:")
            print(f"True: {row['true_class']} | Pred: {row['predicted_class']}")
            print(f"Prob(Fake): {row['probability_fake']:.4f} | Conf: {row['confidence']:.4f}")
            print(f"Text: {str(row['text'])[:100]}...")
    else:
        print("\nNo misclassifications found!")
    
    # Calibration interpretation
    print("\n" + "="*80)
    print("CALIBRATION INTERPRETATION")
    print("="*80)
    if ece < 0.05:
        print("ECE < 5%: The model is WELL CALIBRATED")
        print("The model's confidence closely matches actual accuracy.")
    elif ece < 0.10:
        print("ECE 5-10%: The model has MODERATE MISCALIBRATION")
        print("Consider recalibration or adjusting confidence threshold.")
    else:
        print("ECE > 10%: The model is POORLY CALIBRATED")
        print("The model is dangerously overconfident in its predictions.")
        print("Suggestion: Use temperature scaling or Platt scaling to recalibrate.")
    
    # MCC interpretation
    print(f"\nMCC = {mcc:.4f}: ", end="")
    if mcc > 0.7:
        print("Strong agreement between predictions and actual labels.")
    elif mcc > 0.4:
        print("Moderate agreement.")
    else:
        print("Weak agreement. The model struggles with this dataset.")
    
    print(f"\nBrier Score = {brier:.4f}: ", end="")
    if brier < 0.1:
        print("Excellent probabilistic predictions.")
    elif brier < 0.25:
        print("Moderate - room for improvement.")
    else:
        print("Poor - the model's probabilities are not trustworthy.")
    
    print("\nAdvanced evaluation completed successfully!")
    print(f"\nAll results saved in: {config['output_dir']}")
    print(f"- Predictions: {output_csv}")
    print(f"- Metrics JSON: {json_path}")
    print(f"- HTML Report: {report_path}")
    print(f"- Visualizations: {viz_dir}/")
    print("\nGenerated Visualizations:")
    print("1. reliability_diagram.png - Calibration curve with 45° line")
    print("2. calibration_by_bin.png - Per-bin accuracy vs confidence")
    print("3. bootstrap_confidence_intervals.png - Bootstrap 95% CI")
    print("4. confidence_histogram.png - Confidence distribution analysis")
    print("5. calibration_dashboard.png - Complete 9-panel dashboard")

if __name__ == "__main__":
    main()