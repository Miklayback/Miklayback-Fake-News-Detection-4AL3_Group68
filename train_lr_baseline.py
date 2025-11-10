#!/usr/bin/env python3
"""
Fake News Detection — TF-IDF + Logistic Regression Baseline
============================================================
Runs automatically on train_clean.csv, valid_clean.csv, and test_clean.csv
in the same directory. Prints both majority and TF-IDF + LR baseline results.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Automatically find CSVs in the current folder ===
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train_clean.csv")
valid_path = os.path.join(base_dir, "valid_clean.csv")
test_path  = os.path.join(base_dir, "test_clean.csv")

print(f"Loading datasets from:\n{train_path}\n{valid_path}\n{test_path}\n")

# === Load data ===
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df  = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Test shape:  {test_df.shape}\n")

# === Define columns ===
TEXT_COL = "clean_statement"
LABEL_COL = "label"

X_train, y_train = train_df[TEXT_COL], train_df[LABEL_COL]
X_valid, y_valid = valid_df[TEXT_COL], valid_df[LABEL_COL]
X_test,  y_test  = test_df[TEXT_COL],  test_df[LABEL_COL]

# === Majority baseline ===
majority_class = y_train.value_counts().idxmax()
y_pred_majority = np.full(len(y_test), majority_class)
maj_acc = accuracy_score(y_test, y_pred_majority)
maj_f1  = f1_score(y_test, y_pred_majority, average="macro")

print("=== Majority Baseline ===")
print(f"Most frequent class: {majority_class}")
print(f"Test Accuracy: {maj_acc:.4f}")
print(f"Test Macro-F1: {maj_f1:.4f}\n")

# === TF-IDF + Logistic Regression ===
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,1),          # unigram baseline
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

# Evaluate on validation and test
y_pred_valid = model.predict(X_valid)
y_pred_test  = model.predict(X_test)

valid_acc = accuracy_score(y_valid, y_pred_valid)
valid_f1  = f1_score(y_valid, y_pred_valid, average="macro")
test_acc  = accuracy_score(y_test, y_pred_test)
test_f1   = f1_score(y_test, y_pred_test, average="macro")

print("=== TF-IDF + Logistic Regression ===")
print(f"Validation Accuracy: {valid_acc:.4f} | Macro-F1: {valid_f1:.4f}")
print(f"Test Accuracy:       {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0))

print("""
Model summary:
- Text vectorization: TF-IDF (unigrams)
- Classifier: Logistic Regression
- Loss function: Cross-entropy (log loss) with L2 regularization
- Optimization: Coordinate descent via liblinear solver
- Class weighting: Balanced by inverse class frequency
""")
