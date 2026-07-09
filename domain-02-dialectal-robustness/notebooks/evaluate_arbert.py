#!/usr/bin/env python3
"""
Arabic Fake News Detection - ArBERT Model Evaluation
This script uses the model from: https://huggingface.co/zhafyz/arabic-fake-news-arbert-afnd
Supports both CSV and Parquet file formats
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
        'pyarrow': 'pyarrow', # For parquet support
        'fastparquet': 'fastparquet' # Fallback parquet engine
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
                             balanced_accuracy_score, log_loss)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def preprocess_arabic_text(text: str) -> str:
    """Preprocess Arabic text for ArBERT model"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_user_input():
    """Get user inputs for CSV/Parquet file and parameters"""
    print("\n" + "="*80)
    print("ARABIC FAKE NEWS DETECTION - ArBERT MODEL")
    print("="*80)
    print("\nSupported formats: CSV (.csv) and Parquet (.parquet)")
    
    # Get file path (CSV or Parquet)
    while True:
        csv_path = input("\nEnter path to CSV or Parquet file: ").strip()
        
        if os.path.exists(csv_path):
            # Check file extension
            file_ext = os.path.splitext(csv_path)[1].lower()
            if file_ext in ['.csv', '.parquet']:
                break
            else:
                print(f"Unsupported file format: {file_ext}")
                print("Please provide a .csv or .parquet file")
        else:
            print(f"File not found: {csv_path}")
    
    # Get sample size
    try:
        sample_size_input = input(" How many samples to evaluate (0 for all): ").strip()
        sample_size = int(sample_size_input) if sample_size_input else 0
    except:
        sample_size = 0
    
    # Get text column name
    text_col = input(" Enter text column name (press Enter for auto-detect): ").strip()
    
    # Get label column name
    label_col = input(" Enter label column name (press Enter for auto-detect): ").strip()
    
    # Get batch size
    try:
        batch_size_input = input(" Batch size for inference (default 16): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 16
    except:
        batch_size = 16
    
    # Get confidence threshold
    try:
        threshold_input = input(" Confidence threshold for classification (default 0.5): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.5
    except:
        threshold = 0.5
    
    # Get output directory
    output_dir = input(" Output directory for results (press Enter for './results_arbert'): ").strip()
    if not output_dir:
        output_dir = "./results_arbert"
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'file_path': csv_path,
        'sample_size': sample_size,
        'text_col': text_col,
        'label_col': label_col,
        'batch_size': batch_size,
        'threshold': threshold,
        'output_dir': output_dir
    }

def _load_parquet_safe(file_path):
    """Load a parquet file trying multiple engines to handle version conflicts."""
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
    raise RuntimeError(
        f"Could not load parquet file with any engine (tried {engines}).\n"
        f"Last error: {last_error}\n"
        "Try running: pip install --upgrade pyarrow OR pip install fastparquet"
    )


def load_and_analyze_data(file_path, sample_size, user_text_col, user_label_col):
    """Load dataset from CSV or Parquet with comprehensive analysis"""
    print(f"\nLoading dataset from: {file_path}")
    
    # Check file extension and load accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.parquet':
            df = _load_parquet_safe(file_path)
            print(f"Parquet file loaded: {len(df):,} rows, {len(df.columns)} columns")
        elif file_ext == '.csv':
            # Try different encodings for CSV
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
            # Try both formats
            try:
                df = pd.read_csv(file_path)
                print(f"File loaded as CSV: {len(df):,} rows, {len(df.columns)} columns")
            except:
                df = _load_parquet_safe(file_path)
                print(f"File loaded as Parquet: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    # Show column information
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        dtype_str = str(df[col].dtype)
        missing_count = df[col].isna().sum()
        print(f"{i+1}. {col} ({dtype_str}) - {missing_count} missing")
    
    # Auto-detect or use user-specified columns
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
    
    # Drop rows with missing values
    initial_len = len(df)
    df = df.dropna(subset=[text_col, label_col])
    if initial_len - len(df) > 0:
        print(f"Dropped {initial_len - len(df)} rows with missing values")
    
    # Sample if needed
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size:,} rows for evaluation")
    
    print(f"\nFinal dataset: {len(df):,} rows")
    
    # Analyze text lengths
    df['text_length'] = df[text_col].astype(str).str.len()
    print(f"\nText Statistics:")
    print(f"Mean length: {df['text_length'].mean():.0f} characters")
    print(f"Median length: {df['text_length'].median():.0f} characters")
    print(f"Min length: {df['text_length'].min()} characters")
    print(f"Max length: {df['text_length'].max():,} characters")
    
    # Show length distribution
    length_bins = [0, 50, 100, 200, 500, 1000, 5000, float('inf')]
    length_labels = ['<50', '50-100', '100-200', '200-500', '500-1000', '1000-5000', '>5000']
    df['length_category'] = pd.cut(df['text_length'], bins=length_bins, labels=length_labels)
    print(f"\nText Length Distribution:")
    for cat, count in df['length_category'].value_counts().sort_index().items():
        print(f"{cat}: {count} ({count/len(df)*100:.1f}%)")
    
    return df, text_col, label_col

def normalize_labels(df, label_col):
    """Normalize and understand label formats"""
    unique_labels = df[label_col].unique()
    print(f"\nUnique labels found: {unique_labels}")
    
    # Create label mapping
    label_mapping = {}
    reverse_mapping = {}
    
    # Try to map string labels to integers
    for label in unique_labels:
        label_str = str(label).lower().strip()
        
        # Check for fake/not_credible indicators
        if any(word in label_str for word in ['fake', 'not_credible', 'false', 'misleading', '0', 'lie', 'fraud', 'hoax']):
            label_mapping[label] = 1 # Fake
            reverse_mapping[1] = 'Fake'
        # Check for real/credible indicators
        elif any(word in label_str for word in ['real', 'credible', 'true', 'legitimate', '1', 'truth', 'fact', 'honest']):
            label_mapping[label] = 0 # Real
            reverse_mapping[0] = 'Real'
        else:
            # For numeric labels that are already 0/1
            try:
                num_label = float(label)
                if num_label in [0, 1]:
                    label_mapping[label] = int(num_label)
                    reverse_mapping[int(num_label)] = f'Class {int(num_label)}'
                elif num_label in [0.0, 1.0]:
                    label_mapping[label] = int(num_label)
                    reverse_mapping[int(num_label)] = f'Class {int(num_label)}'
            except:
                # Unknown format, try to infer
                pass
    
    if not label_mapping:
        print("Could not map labels automatically. Creating default mapping...")
        unique_labels_list = list(unique_labels)
        for i, label in enumerate(unique_labels_list):
            label_mapping[label] = i
            reverse_mapping[i] = str(label)
    
    # Apply mapping
    df['label_encoded'] = df[label_col].map(label_mapping)
    
    print(f"\nLabel Mapping:")
    for orig, encoded in label_mapping.items():
        print(f"'{orig}' -> {encoded} ({reverse_mapping[encoded]})")

    # Drop rows whose label could not be mapped (e.g. 'undecided')
    unmapped_mask = df['label_encoded'].isna()
    if unmapped_mask.any():
        unmapped_labels = df.loc[unmapped_mask, label_col].unique().tolist()
        dropped = unmapped_mask.sum()
        print(f"\nDropping {dropped} rows with unmappable labels: {unmapped_labels}")
        print(f"(These are neither 'fake' nor 'real' — excluded from evaluation)")
        df = df[~unmapped_mask].copy()
    df['label_encoded'] = df['label_encoded'].astype(int)

    print(f"\nEncoded Label Distribution:")
    label_counts = df['label_encoded'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        print(f"{reverse_mapping[label]}: {count} ({percentage:.1f}%)")

    return df, label_mapping, reverse_mapping

def predict_with_confidence(texts, tokenizer, model, device, batch_size=16, threshold=0.5):
    """Batch prediction with confidence scoring"""
    model.eval()
    predictions = []
    confidences = []
    probabilities = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i:i+batch_size]
        processed_batch = [preprocess_arabic_text(str(t)) for t in batch_texts]
        
        # Tokenize
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
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Get predictions with confidence threshold
            max_probs, pred_ids = torch.max(probs, dim=-1)
            
            # Apply threshold (if confidence < threshold, mark as uncertain)
            confident_mask = max_probs >= threshold
            final_preds = pred_ids.clone()
            final_preds[~confident_mask] = -1 # -1 indicates uncertain
            
            predictions.extend(final_preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(confidences), np.array(probabilities)

def calculate_advanced_metrics(y_true, y_pred, y_proba, predictions_with_uncertainty=None, confidences=None):
    """Calculate comprehensive metrics including uncertainty handling"""
    # Remove uncertain predictions for certain metrics
    if predictions_with_uncertainty is not None:
        certain_mask = predictions_with_uncertainty != -1
        y_true_certain = y_true[certain_mask]
        y_pred_certain = predictions_with_uncertainty[certain_mask]
        uncertainty_rate = (~certain_mask).sum() / len(y_true)
    else:
        y_true_certain = y_true
        y_pred_certain = y_pred
        uncertainty_rate = 0
    
    metrics = {}
    
    # Standard metrics on certain predictions
    if len(y_true_certain) > 0:
        metrics['accuracy'] = accuracy_score(y_true_certain, y_pred_certain)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_certain, y_pred_certain)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y_true_certain, y_pred_certain, average='binary', zero_division=0
        )
        metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro'], _ = precision_recall_fscore_support(
            y_true_certain, y_pred_certain, average='macro', zero_division=0
        )
        metrics['mcc'] = matthews_corrcoef(y_true_certain, y_pred_certain)
        metrics['kappa'] = cohen_kappa_score(y_true_certain, y_pred_certain)
        metrics['per_class'] = precision_recall_fscore_support(y_true_certain, y_pred_certain,
                                                                average=None, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_true_certain, y_pred_certain)
    
    metrics['uncertainty_rate'] = uncertainty_rate
    
    # Metrics using probabilities
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            # For binary classification, take probability of positive class
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                positive_proba = y_proba[:, 1]
            else:
                positive_proba = y_proba.flatten()
            metrics['roc_auc'] = roc_auc_score(y_true, positive_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except Exception as e:
            print(f"Could not calculate ROC-AUC or Log Loss: {e}")
            metrics['roc_auc'] = None
            metrics['log_loss'] = None
    
    # Calibration metrics
    if confidences is not None and len(y_true_certain) > 0:
        try:
            # Expected Calibration Error (simplified)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(confidences, bin_boundaries) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            ece = 0.0
            for bin_idx in range(n_bins):
                in_bin = (bin_indices == bin_idx)
                if in_bin.any():
                    bin_accuracy = y_true_certain[in_bin].mean()
                    bin_confidence = confidences[in_bin].mean()
                    ece += np.abs(bin_accuracy - bin_confidence) * in_bin.mean()
            metrics['ece'] = ece
        except:
            metrics['ece'] = None
    
    return metrics

def plot_detailed_visualizations(df_results, metrics, config, class_names):
    """Generate comprehensive visualizations"""
    viz_dir = os.path.join(config['output_dir'], 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall confidence
    correct_mask = df_results['correct'] & ~df_results['uncertain']
    incorrect_mask = ~df_results['correct'] & ~df_results['uncertain']
    
    if correct_mask.any():
        axes[0].hist(df_results[correct_mask]['confidence'], bins=20, alpha=0.7,
                     label='Correct', color='green', edgecolor='black')
    if incorrect_mask.any():
        axes[0].hist(df_results[incorrect_mask]['confidence'], bins=20, alpha=0.7,
                     label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence by class
    for label in df_results['label_encoded'].unique():
        label_data = df_results[df_results['label_encoded'] == label]
        if len(label_data) > 0:
            class_name = class_names[label] if label < len(class_names) else f'Class {label}'
            axes[1].hist(label_data['confidence'], bins=20, alpha=0.7,
                        label=f'{class_name} (n={len(label_data)})', edgecolor='black')
    axes[1].set_xlabel('Confidence Score', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Confidence Distribution by Class', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    if metrics.get('roc_auc') is not None and metrics['roc_auc'] is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(df_results['label_encoded'], df_results['probability_positive'])
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Performance by Text Length
    plt.figure(figsize=(10, 6))
    
    # Create length bins
    df_results['length_bin'] = pd.cut(df_results['text_length'], bins=10)
    length_performance = df_results[~df_results['uncertain']].groupby('length_bin').agg({
        'correct': 'mean',
    }).reset_index()
    
    # Remove NaN bins
    length_performance = length_performance.dropna()
    
    if len(length_performance) > 0:
        plt.plot(range(len(length_performance)), length_performance['correct'],
                marker='o', linewidth=2, markersize=8)
        plt.xlabel('Text Length Bin', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance by Text Length', fontsize=14, fontweight='bold')
        plt.xticks(range(len(length_performance)),
                   [f'{int(bin.left)}-{int(bin.right)}' for bin in length_performance['length_bin']],
                   rotation=45)
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'performance_by_length.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Uncertainty Analysis (if available)
    if 'uncertain' in df_results.columns and df_results['uncertain'].any():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Uncertainty by class
        uncertainty_by_class = df_results.groupby('label_encoded')['uncertain'].mean()
        axes[0].bar(range(len(uncertainty_by_class)), uncertainty_by_class.values)
        axes[0].set_xlabel('Class', fontsize=11)
        axes[0].set_ylabel('Uncertainty Rate', fontsize=11)
        axes[0].set_title('Uncertainty Rate by Class', fontsize=12)
        axes[0].set_xticks(range(len(uncertainty_by_class)))
        axes[0].set_xticklabels([class_names[i] if i < len(class_names) else f'Class {i}'
                                 for i in uncertainty_by_class.index])
        axes[0].set_ylim([0, 1])
        
        # Confidence of uncertain predictions
        uncertain_data = df_results[df_results['uncertain']]
        if len(uncertain_data) > 0:
            axes[1].hist(uncertain_data['confidence'], bins=20, color='orange', edgecolor='black')
            axes[1].set_xlabel('Confidence Score', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title(f'Confidence of Uncertain Predictions (n={len(uncertain_data)})', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {viz_dir}")

def generate_comprehensive_report(df_results, metrics, config, class_names, timestamp):
    """Generate detailed HTML report"""
    report_path = os.path.join(config['output_dir'], f'arbert_report_{timestamp}.html')
    
    # Calculate additional stats
    total_samples = len(df_results)
    correct_predictions = df_results['correct'].sum()
    accuracy_pct = correct_predictions / total_samples * 100
    
    # Determine if uncertain predictions exist
    has_uncertain = 'uncertain' in df_results.columns
    if has_uncertain:
        uncertain_count = df_results['uncertain'].sum()
        certain_count = total_samples - uncertain_count
        certain_accuracy = df_results[df_results['uncertain'] == False]['correct'].mean() if certain_count > 0 else 0
    else:
        uncertain_count = 0
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArBERT Fake News Detection - Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #2ecc71;
            color: white;
        }}
        .badge-danger {{
            background-color: #e74c3c;
            color: white;
        }}
        .badge-warning {{
            background-color: #f39c12;
            color: white;
        }}
        .example {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 10px;
            margin: 10px 0;
            font-family: monospace;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> ArBERT Fake News Detection - Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2> Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>File Path</td><td>{config['file_path']}</td></tr>
            <tr><td>Sample Size</td><td>{total_samples:,}</td></tr>
            <tr><td>Confidence Threshold</td><td>{config['threshold']}</td></tr>
            <tr><td>Batch Size</td><td>{config['batch_size']}</td></tr>
            <tr><td>Device</td><td>{'CUDA' if torch.cuda.is_available() else 'CPU'}</td></tr>
        </table>
        
        <h2> Overall Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{accuracy_pct:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['f1']:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['balanced_accuracy']:.3f}</div>
                <div class="metric-label">Balanced Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['mcc']:.3f}</div>
                <div class="metric-label">MCC</div>
            </div>
        </div>
        
        <h2> Per-Class Performance</h2>
        <table>
            <thead>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
            </thead>
            <tbody>
    """
    
    for i, (p, r, f1, s) in enumerate(zip(*metrics['per_class'])):
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{p:.3f}</td>
                    <td>{r:.3f}</td>
                    <td>{f1:.3f}</td>
                    <td>{s}</td>
                </tr>
        """
    
    if has_uncertain and uncertain_count > 0:
        html_content += f"""
                <tr style="background-color: #fff3cd;">
                    <td colspan="5"><strong>Uncertain Predictions: {uncertain_count} ({uncertain_count/total_samples*100:.1f}%)</strong></td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2> Advanced Metrics</h2>
        <table>
            <tr><td>Precision (Macro)</td><td>{metrics['precision_macro']:.4f}</td></tr>
            <tr><td>Recall (Macro)</td><td>{metrics['recall_macro']:.4f}</td></tr>
            <tr><td>F1-Score (Macro)</td><td>{metrics['f1_macro']:.4f}</td></tr>
            <tr><td>Matthews Correlation Coefficient</td><td>{metrics['mcc']:.4f}</td></tr>
            <tr><td>Cohen's Kappa</td><td>{metrics['kappa']:.4f}</td></tr>
    """
    
    if metrics.get('roc_auc') and metrics['roc_auc'] is not None:
        html_content += f"""
            <tr><td>ROC-AUC</td><td>{metrics['roc_auc']:.4f}</td></tr>
            <tr><td>Log Loss</td><td>{metrics['log_loss']:.4f}</td></tr>
        """
    
    if metrics.get('ece') and metrics['ece'] is not None:
        html_content += f"""
            <tr><td>Expected Calibration Error (ECE)</td><td>{metrics['ece']:.4f}</td></tr>
        """
    
    html_content += f"""
        </table>
        
        <h2> Confusion Matrix</h2>
        <pre>
{metrics['confusion_matrix']}
        </pre>
        
        <h2> Sample Predictions</h2>
        <table>
            <thead>
                <tr><th>Text (truncated)</th><th>True Label</th><th>Predicted</th><th>Confidence</th><th>Status</th></tr>
            </thead>
            <tbody>
    """
    
    for _, row in df_results.head(20).iterrows():
        status_class = 'badge-success' if row['correct'] else 'badge-danger'
        status_text = 'Correct' if row['correct'] else 'Incorrect'
        if has_uncertain and row.get('uncertain', False):
            status_class = 'badge-warning'
            status_text = '? Uncertain'
        
        html_content += f"""
                <tr>
                    <td>{str(row['text'])[:100]}...</td>
                    <td>{class_names[row['label_encoded']] if row['label_encoded'] < len(class_names) else row['label_encoded']}</td>
                    <td>{class_names[row['predicted']] if row['predicted'] >= 0 and row['predicted'] < len(class_names) else 'Uncertain'}</td>
                    <td>{row['confidence']:.3f}</td>
                    <td><span class="badge {status_class}">{status_text}</span></td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by ArBERT Fake News Detection System</p>
            <p>Model: zhafyz/arabic-fake-news-arbert-afnd</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML Report saved: {report_path}")
    return report_path

def main():
    # Get user input
    config = get_user_input()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model from HuggingFace
    model_id = "zhafyz/arabic-fake-news-arbert-afnd"
    print(f"\nLoading model: {model_id}")
    print("Downloading model (first time may take a few minutes)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your internet connection and try again.")
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
    
    # Run predictions
    print("\nRunning predictions with confidence threshold...")
    texts = df['text'].tolist()
    predictions, confidences, probabilities = predict_with_confidence(
        texts, tokenizer, model, device,
        config['batch_size'], config['threshold']
    )
    
    # Prepare results
    y_true = df['label_encoded'].values
    
    # Note: ArBERT model uses standard mapping (0=fake, 1=real)
    # Adjust if needed based on model's actual output
    y_pred = predictions
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_advanced_metrics(y_true, y_pred, probabilities, predictions, confidences)
    
    # Create results dataframe
    df_results = df.copy()
    df_results['predicted'] = y_pred
    df_results['confidence'] = confidences
    df_results['correct'] = (y_true == y_pred) & (y_pred != -1)
    df_results['uncertain'] = (y_pred == -1)
    
    # Add probability of positive class (for ROC)
    if len(probabilities.shape) > 1 and probabilities.shape[1] >= 2:
        df_results['probability_positive'] = probabilities[:, 1]
    else:
        df_results['probability_positive'] = confidences
    
    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    total = len(df_results)
    correct = df_results['correct'].sum()
    uncertain = df_results['uncertain'].sum()
    certain = total - uncertain
    
    if certain > 0:
        print(f"\nOverall Accuracy (certain predictions only): {correct/certain*100:.2f}% ({correct}/{certain})")
    else:
        print(f"\nNo certain predictions (all below threshold)")
    
    if uncertain > 0:
        print(f"Uncertain predictions (below threshold): {uncertain} ({uncertain/total*100:.2f}%)")
    
    print(f"Binary F1-Score: {metrics['f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Matthews Correlation: {metrics['mcc']:.4f}")
    
    if metrics.get('roc_auc') and metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nPer-class Performance:")
    for i, (p, r, f1, s) in enumerate(zip(*metrics['per_class'])):
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        print(f"{class_name}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}, Support={s}")
    
    print(f"\nConfusion Matrix (certain predictions only):")
    cm = metrics['confusion_matrix']
    print(f"[[{cm[0,0]:5d} {cm[0,1]:5d}]")
    print(f"[{cm[1,0]:5d} {cm[1,1]:5d}]]")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_detailed_visualizations(df_results, metrics, config, class_names)
    
    # Save results
    output_csv = os.path.join(config['output_dir'], f'arbert_predictions_{timestamp}.csv')
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Detailed predictions saved: {output_csv}")
    
    # Save metrics as JSON
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            metrics_serializable[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            metrics_serializable[k] = int(v)
        else:
            metrics_serializable[k] = v
    
    json_path = os.path.join(config['output_dir'], f'arbert_metrics_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {json_path}")
    
    # Generate HTML report
    report_path = generate_comprehensive_report(df_results, metrics, config, class_names, timestamp)
    
    # Show misclassification examples
    print("\n" + "="*80)
    print("TOP MISCLASSIFICATION EXAMPLES")
    print("="*80)
    misclassified = df_results[~df_results['correct'] & ~df_results['uncertain']]
    
    if len(misclassified) > 0:
        for idx, row in misclassified.head(10).iterrows():
            print(f"\nExample {idx}:")
            print(f"True: {class_names[row['label_encoded']] if row['label_encoded'] < len(class_names) else row['label_encoded']}")
            print(f"Pred: {class_names[row['predicted']] if row['predicted'] >= 0 else 'Uncertain'}")
            print(f"Conf: {row['confidence']:.3f}")
            print(f"Text: {str(row['text'])[:100]}...")
    else:
        if uncertain > 0:
            print("\nNo misclassifications found on certain predictions!")
            print(f"(Note: {uncertain} uncertain predictions were excluded)")
        else:
            print("\nNo misclassifications found!")
    
    print("\nEvaluation completed successfully!")
    print(f"\nAll results saved in: {config['output_dir']}")
    print(f"- Predictions: {output_csv}")
    print(f"- Metrics: {json_path}")
    print(f"- Report: {report_path}")
    print(f"- Visualizations: {config['output_dir']}/visualizations/")

if __name__ == "__main__":
    main()