#!/usr/bin/env python3
"""
Fake News Detection — TF-IDF + Logistic Regression Baseline

Runs on train_clean.csv, valid_clean.csv, and test_clean.csv in the same folder.
Prints both majority and TF-IDF+LR results, and saves two figures:
- fig_confusion_matrix.png
- fig_baseline_compare.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train_clean.csv")
valid_path = os.path.join(base_dir, "valid_clean.csv")
test_path = os.path.join(base_dir, "test_clean.csv")

print(f"Loading datasets from:\n{train_path}\n{valid_path}\n{test_path}\n")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Test shape:  {test_df.shape}\n")

TEXT_COL = "clean_statement"
LABEL_COL = "label"

X_train, y_train = train_df[TEXT_COL], train_df[LABEL_COL]
X_valid, y_valid = valid_df[TEXT_COL], valid_df[LABEL_COL]
X_test, y_test = test_df[TEXT_COL], test_df[LABEL_COL]

# Majority baseline
majority_class = y_train.value_counts().idxmax()
y_pred_majority = np.full(len(y_test), majority_class)
maj_acc = accuracy_score(y_test, y_pred_majority)
maj_f1 = f1_score(y_test, y_pred_majority, average="macro")

print("=== Majority Baseline ===")
print(f"Most frequent class: {majority_class}")
print(f"Test Accuracy: {maj_acc:.4f}")
print(f"Test Macro-F1: {maj_f1:.4f}\n")

# TF-IDF + LR baseline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=2, max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode"
    )),
    ("lr", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        multi_class="ovr",
        C=1.0
    ))
])

model.fit(X_train, y_train)

y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)

valid_acc = accuracy_score(y_valid, y_pred_valid)
valid_f1 = f1_score(y_valid, y_pred_valid, average="macro")
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average="macro")

print("=== TF-IDF + Logistic Regression ===")
print(f"Validation Accuracy: {valid_acc:.4f} | Macro-F1: {valid_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# Output plots
def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Figure 1: Confusion Matrix
def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# Figure 2: Baseline Comparison
def plot_baseline_compare(maj_acc, maj_f1, test_acc, test_f1, out_path):
    metrics = ["Accuracy", "Macro-F1"]
    majority = [maj_acc, maj_f1]
    tfidf_lr = [test_acc, test_f1]

    idx = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(idx - width / 2, majority, width, label="Majority")
    plt.bar(idx + width / 2, tfidf_lr, width, label="TF-IDF+LR")

    plt.xticks(idx, metrics)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Baseline Comparison (Test)")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

label_order = sorted(y_train.unique().tolist())
cm = confusion_matrix(y_test, y_pred_test, labels=label_order)
plot_confusion_matrix(cm, label_order, os.path.join(base_dir, "fig_confusion_matrix.png"))
plot_baseline_compare(maj_acc, maj_f1, test_acc, test_f1, os.path.join(base_dir, "fig_baseline_compare.png"))
