#!/usr/bin/env python3
"""
Arabic Fake News Detection - Marbertv2 Model Evaluation
This script uses the model from: https://huggingface.co/zhafyz/arabic-fake-news-marbertv2-tweets
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pyarabic.araby as araby
import re
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, matthews_corrcoef, cohen_kappa_score)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def clean_arabic_tweet(text: str) -> str:
    """Clean Arabic tweet text"""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = text.replace('#', ' ')
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = re.sub(r'[^\u0600-\u06FF0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_user_input():
    """Get user inputs for CSV file and parameters"""
    print("\n" + "="*80)
    print("ARABIC FAKE NEWS DETECTION - MARBERTV2 MODEL")
    print("="*80)
    
    csv_path = input("\nEnter path to CSV file: ").strip()
    while not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        csv_path = input("Please enter valid CSV path: ").strip()
    
    try:
        sample_size = int(input("How many samples to evaluate (0 for all)? ").strip())
    except:
        sample_size = 0
    
    text_col = input("Enter text column name (press Enter for auto-detect): ").strip()
    label_col = input("Enter label column name (press Enter for auto-detect): ").strip()
    
    try:
        batch_size = int(input("Batch size for inference (default 32): ").strip() or "32")
    except:
        batch_size = 32
    
    output_dir = input("Output directory for results (press Enter for './results'): ").strip()
    if not output_dir:
        output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'csv_path': csv_path,
        'sample_size': sample_size,
        'text_col': text_col,
        'label_col': label_col,
        'batch_size': batch_size,
        'output_dir': output_dir
    }

def load_data(csv_path, sample_size, user_text_col, user_label_col):
    """Load and prepare dataset"""
    print(f"\nLoading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    if user_text_col and user_text_col in df.columns:
        text_col = user_text_col
    else:
        text_col = next((c for c in df.columns if c.lower() in ['text', 'tweet', 'content', 'news', 'article']), None)
    
    if user_label_col and user_label_col in df.columns:
        label_col = user_label_col
    else:
        label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'target', 'category']), None)
    
    if not text_col or not label_col:
        raise ValueError(f"Could not find text/label columns. Found: {df.columns.tolist()}")
    
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")
    
    df = df.dropna(subset=[text_col, label_col])
    
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows")
    
    print(f"Final dataset size: {len(df)} rows")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    return df, text_col, label_col

def parse_labels(df, label_col):
    """Parse and understand label mapping"""
    unique_labels = df[label_col].unique()
    print(f"\nUnique labels found: {unique_labels}")
    
    label_mapping = {}
    for label in unique_labels:
        label_str = str(label).lower()
        if label_str in ['fake', 'not_credible', 'false', '0']:
            label_mapping[label] = 0
        elif label_str in ['real', 'credible', 'true', '1']:
            label_mapping[label] = 1
        else:
            print(f"Unknown label format: {label}")
    
    if label_mapping:
        df['label_encoded'] = df[label_col].map(label_mapping)
        print(f"Label mapping: {label_mapping}")
        return df, label_mapping
    
    if df[label_col].dtype in ['int64', 'float64']:
        df['label_encoded'] = df[label_col]
        return df, {0: 'Fake', 1: 'Real'}
    
    return df, None

def predict_batch(texts, tokenizer, model, device, batch_size=32):
    """Batch prediction with progress bar"""
    model.eval()
    predictions = []
    confidences = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        cleaned_batch = [clean_arabic_tweet(str(t)) for t in batch_texts]
        
        inputs = tokenizer(
            cleaned_batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_preds = torch.argmax(probs, dim=-1)
            batch_conf = torch.max(probs, dim=-1)[0]
            
            predictions.extend(batch_preds.cpu().numpy())
            confidences.extend(batch_conf.cpu().numpy())
    
    return np.array(predictions), np.array(confidences)

def calculate_detailed_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro'], _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted'], _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['per_class'] = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {output_path}")

def plot_confidence_distribution(confidences_correct, confidences_incorrect, output_path):
    """Plot confidence distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(confidences_correct, bins=20, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(confidences_incorrect, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution saved: {output_path}")

def plot_roc_curve(y_true, y_proba, output_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {output_path}")

def analyze_misclassifications(df, y_true, y_pred, confidences, text_col, label_col, n_samples=20):
    """Detailed analysis of misclassified samples"""
    df_results = df.copy()
    df_results['predicted'] = y_pred
    df_results['confidence'] = confidences
    df_results['correct'] = (y_true == y_pred)
    
    misclassified = df_results[~df_results['correct']]
    
    print("\n" + "="*80)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*80)
    print(f"Total misclassified: {len(misclassified)}/{len(df_results)} ({len(misclassified)/len(df_results)*100:.2f}%)")
    
    print("\nMisclassifications by original label:")
    for label in sorted(df_results['label_encoded'].unique()):
        mask = (df_results['label_encoded'] == label) & (~df_results['correct'])
        count = mask.sum()
        total = (df_results['label_encoded'] == label).sum()
        if total > 0:
            print(f"Label {label}: {count}/{total} ({count/total*100:.2f}%)")
    
    print(f"\nTop {min(n_samples, len(misclassified))} Misclassified Examples:")
    print("-"*80)
    for idx, row in misclassified.head(n_samples).iterrows():
        print(f"\nExample {idx}:")
        print(f"True Label: {row[label_col]}")
        print(f"Predicted: {row['predicted']} (confidence: {row['confidence']:.3f})")
        print(f"Text: {str(row[text_col])[:150]}...")
    
    print(f"\nMisclassification Confidence Stats:")
    print(f"Mean confidence: {misclassified['confidence'].mean():.3f}")
    print(f"Median confidence: {misclassified['confidence'].median():.3f}")
    print(f"Std confidence: {misclassified['confidence'].std():.3f}")
    
    return df_results

def perform_advanced_analysis(df_results, text_col, output_dir):
    """Perform advanced error analysis including structural binning, KDE, correlation, and TF-IDF analysis"""
    
    print("\n" + "="*80)
    print("ADVANCED STRUCTURAL & LINGUISTIC ANALYSIS")
    print("="*80)
    
    if 'word_count' not in df_results.columns:
        df_results['word_count'] = df_results[text_col].astype(str).apply(lambda x: len(x.split()))
    if 'text_length' not in df_results.columns:
        df_results['text_length'] = df_results[text_col].astype(str).apply(len)
    
    # 1) STRUCTURAL BINNING
    print("\nSTRUCTURAL BINNING ANALYSIS")
    print("-"*40)
    
    df_results['length_bin'] = pd.qcut(df_results['word_count'], q=5, duplicates='drop')
    length_analysis = df_results.groupby('length_bin')['correct'].mean().reset_index()
    print("\nAccuracy by text length quintile:")
    print(length_analysis)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=length_analysis['length_bin'].astype(str), y=length_analysis['correct'])
    plt.xticks(rotation=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Word Count Quintiles")
    plt.title("Structural Binning: Accuracy Across Text Length Groups")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'structural_binning_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Structural binning plot saved")
    
    # 2) KDE of Confidence
    print("\nCONFIDENCE DISTRIBUTION (KDE)")
    print("-"*40)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_results[df_results['correct'] == True]['confidence'],
                label='Correct Predictions', fill=True)
    sns.kdeplot(df_results[df_results['correct'] == False]['confidence'],
                label='Incorrect Predictions', fill=True)
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimation (KDE) of Confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_kde.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"KDE plot saved")
    
    # 3) Feature Correlation Analysis
    print("\nFEATURE CORRELATION ANALYSIS")
    print("-"*40)
    
    df_results['url_count'] = df_results[text_col].astype(str).apply(
        lambda x: len(re.findall(r'http\S+|www\S+', str(x)))
    )
    df_results['punctuation_count'] = df_results[text_col].astype(str).apply(
        lambda x: len(re.findall(r'[!?.,؛،]', str(x)))
    )
    
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    df_results['emoji_count'] = df_results[text_col].astype(str).apply(
        lambda x: len(emoji_pattern.findall(str(x)))
    )
    df_results['is_error'] = (~df_results['correct']).astype(int)
    
    corr_features = ['text_length', 'word_count', 'confidence', 'url_count', 'punctuation_count', 'emoji_count', 'is_error']
    existing_features = [f for f in corr_features if f in df_results.columns]
    
    corr_matrix = df_results[existing_features].corr(method='spearman')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("Spearman Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved")
    
    print("\nKey correlations with error (is_error):")
    if 'is_error' in corr_matrix.columns:
        error_corr = corr_matrix['is_error'].sort_values(ascending=False)
        for feat, corr_val in error_corr.items():
            if feat != 'is_error':
                print(f"{feat}: {corr_val:.3f}")
    
    # 4) TF-IDF Trigger Word Analysis
    print("\nTF-IDF TRIGGER WORD ANALYSIS")
    print("-"*40)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    error_texts = df_results[df_results['correct'] == False][text_col].astype(str)
    
    if len(error_texts) > 0:
        vectorizer = TfidfVectorizer(max_features=3000, stop_words=None)
        X_error = vectorizer.fit_transform(error_texts)
        error_scores = np.asarray(X_error.mean(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        
        tfidf_df = pd.DataFrame({
            'word': feature_names,
            'error_tfidf_score': error_scores
        })
        
        top_trigger_words = tfidf_df.sort_values(by='error_tfidf_score', ascending=False).head(20)
        
        print("\nTop 20 Trigger Words in Error Set:")
        print(top_trigger_words.to_string(index=False))
        
        top_trigger_words.to_csv(os.path.join(output_dir, 'top_trigger_words.csv'),
                                  index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='error_tfidf_score', y='word', data=top_trigger_words)
        plt.title("Top Trigger Words in Error Set (TF-IDF)")
        plt.xlabel("TF-IDF Score")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trigger_words_tfidf.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trigger words plot saved")
    else:
        print("No error samples found, skipping TF-IDF analysis")
    
    return df_results

def generate_report(metrics, config, output_dir):
    """Generate comprehensive report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"marbertv2_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ARABIC FAKE NEWS DETECTION - MARBERTV2 MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        for key, value in config.items():
            f.write(f" {key}: {value}\n")
        
        f.write("\n\nPERFORMANCE METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision (binary): {metrics['precision']:.4f}\n")
        f.write(f"Recall (binary): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score (binary): {metrics['f1']:.4f}\n\n")
        
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n\n")
        
        f.write(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['kappa']:.4f}\n")
        
        if metrics.get('roc_auc'):
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-"*40 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"[[{cm[0,0]:5d} {cm[0,1]:5d}]\n")
        f.write(f" [{cm[1,0]:5d} {cm[1,1]:5d}]]\n")
        
        f.write("\n\nPER-CLASS METRICS:\n")
        f.write("-"*40 + "\n")
        for i, (p, r, f1, s) in enumerate(zip(*metrics['per_class'])):
            f.write(f"\nClass {i}:\n")
            f.write(f" Precision: {p:.4f}\n")
            f.write(f" Recall: {r:.4f}\n")
            f.write(f" F1-Score: {f1:.4f}\n")
            f.write(f" Support: {s}\n")
    
    print(f"\nDetailed report saved: {report_path}")
    return report_path

def main():
    config = get_user_input()
    
    model_id = "zhafyz/arabic-fake-news-marbertv2-tweets"
    print(f"\nLoading model: {model_id}")
    print("This may take a few minutes on first run...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    
    df, text_col, label_col = load_data(
        config['csv_path'],
        config['sample_size'],
        config['text_col'],
        config['label_col']
    )
    
    df, label_mapping = parse_labels(df, label_col)
    if label_mapping is None:
        print("Could not parse labels. Please ensure labels are 'fake'/'real' or 0/1")
        return
    
    unique_labels = df['label_encoded'].unique()
    if len(unique_labels) != 2:
        print(f"Expected binary classification, but found {len(unique_labels)} classes")
    
    print("\nRunning predictions...")
    texts = df[text_col].tolist()
    predictions, confidences = predict_batch(
        texts, tokenizer, model, device, config['batch_size']
    )
    
    y_true = df['label_encoded'].values
    y_pred = 1 - predictions
    y_proba = confidences
    
    print("\nCalculating metrics...")
    metrics = calculate_detailed_metrics(y_true, y_pred, y_proba)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Binary F1-Score: {metrics['f1']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Matthews Correlation: {metrics['mcc']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nPer-class Performance:")
    class_names = ['Fake', 'Real'] if 0 in unique_labels else ['Class 0', 'Class 1']
    for i, (p, r, f1, s) in enumerate(zip(*metrics['per_class'])):
        print(f"{class_names[i]}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, Support={s}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"[[{cm[0,0]:5d} {cm[0,1]:5d}]")
    print(f"[{cm[1,0]:5d} {cm[1,1]:5d}]]")
    
    print("\nGenerating visualizations...")
    
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(config['output_dir'], 'marbertv2_confusion_matrix.png')
    )
    
    correct_mask = (y_true == y_pred)
    plot_confidence_distribution(
        confidences[correct_mask],
        confidences[~correct_mask],
        os.path.join(config['output_dir'], 'marbertv2_confidence_dist.png')
    )
    
    if metrics.get('roc_auc'):
        plot_roc_curve(
            y_true, y_proba,
            os.path.join(config['output_dir'], 'marbertv2_roc_curve.png')
        )
    
    df_results = analyze_misclassifications(
        df, y_true, y_pred, confidences, text_col, label_col
    )
    
    # Original Failure Mode Analysis
    print("\n" + "="*80)
    print("ADVANCED FAILURE MODE ANALYSIS")
    print("="*80)
    
    df_results["text_length"] = df_results[text_col].astype(str).apply(len)
    df_results["word_count"] = df_results[text_col].astype(str).apply(lambda x: len(x.split()))
    df_results["is_error"] = (df_results["label_encoded"] != df_results["predicted"])
    
    errors = df_results[df_results["is_error"] == True].copy()
    
    high_conf_failures = errors[errors["confidence"] > 0.90]
    print(f"\nHigh Confidence Failures: {len(high_conf_failures)}")
    high_conf_failures.to_csv(os.path.join(config['output_dir'], "high_confidence_failures.csv"), index=False, encoding='utf-8-sig')
    
    short_context_failures = errors[errors["word_count"] < 5]
    print(f"Short Context Failures: {len(short_context_failures)}")
    short_context_failures.to_csv(os.path.join(config['output_dir'], "short_context_failures.csv"), index=False, encoding='utf-8-sig')
    
    trigger_words = ["السعودية", "الرياض", "شركة", "ريال", "انفجار", "التواصل", "عاجل", "رسمي"]
    def contains_trigger_word(text):
        text = str(text)
        return any(word in text for word in trigger_words)
    
    errors["trigger_word_error"] = errors[text_col].apply(contains_trigger_word)
    trigger_word_failures = errors[errors["trigger_word_error"] == True]
    print(f"Trigger Word Dependency: {len(trigger_word_failures)}")
    trigger_word_failures.to_csv(os.path.join(config['output_dir'], "trigger_word_failures.csv"), index=False, encoding='utf-8-sig')
    
    calibration_failures = errors[(errors["confidence"] >= 0.5) & (errors["confidence"] <= 0.8)]
    print(f"Calibration Instability: {len(calibration_failures)}")
    calibration_failures.to_csv(os.path.join(config['output_dir'], "calibration_failures.csv"), index=False, encoding='utf-8-sig')
    
    error_summary = pd.DataFrame({
        "Error_Type": ["High Confidence Failures", "Short Context Failures", "Trigger Word Dependency", "Calibration Instability"],
        "Count": [len(high_conf_failures), len(short_context_failures), len(trigger_word_failures), len(calibration_failures)]
    })
    
    print("\nERROR TYPE SUMMARY:")
    print(error_summary)
    
    summary_path = os.path.join(config['output_dir'], "error_type_summary.csv")
    error_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Error_Type", y="Count", data=error_summary)
    plt.title("Failure Mode Distribution")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], "failure_mode_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop High Confidence Failures:")
    for idx, row in high_conf_failures.head(10).iterrows():
        print("\n" + "-"*60)
        print(f"Confidence: {row['confidence']:.4f}")
        print(f"True Label: {row['label_encoded']}")
        print(f"Predicted : {row['predicted']}")
        print(f"Text: {str(row[text_col])[:200]}")
    
    # NEW ADVANCED ANALYSIS
    df_results = perform_advanced_analysis(df_results, text_col, config['output_dir'])
    
    print("\nAdvanced Failure Analysis Completed.")
    
    output_csv = os.path.join(config['output_dir'], 'marbertv2_predictions.csv')
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nDetailed predictions saved: {output_csv}")
    
    report_path = generate_report(metrics, config, config['output_dir'])
    
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
    json_path = os.path.join(config['output_dir'], 'marbertv2_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved: {json_path}")
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()