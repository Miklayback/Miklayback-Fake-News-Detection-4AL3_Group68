#!/usr/bin/env python3
"""
train_svm_improved.py

TF-IDF + Linear SVM with small, validated tweaks using cross-validation.
- Loads train_clean.csv, valid_clean.csv, test_clean.csv (from prepare_data.py)
- Uses 5-fold Stratified CV on the *training* split to select hyperparameters
- (Optionally) refits the best model on train+valid before final test evaluation
- Prints Validation & Test metrics, and saves:
    * fig_confusion_matrix_svm.png
    * fig_svm_vs_majority.png

Why this should help
--------------------
1) Add bigrams (1,2) to capture short phrases (negations & cue words) that
   unigrams miss. This often improves separability in sparse text.
2) Tune C (margin strength) — balances large-margin vs misclassification.
3) Validate whether class_weight balancing helps Macro-F1 for this dataset.
4) Use CV to stabilize hyperparameter selection (less dependent on one split).

Usage
-----
python train_svm_improved.py \
    --cv-folds 5 \
    --refit-on-train-valid 1

(Defaults are fine to start.)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ------------------------
# CLI
# ------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="TF-IDF + LinearSVC (improved) with CV hyperparameter selection"
    )
    p.add_argument("--cv-folds", type=int, default=5, help="Number of Stratified CV folds for hyperparam search")
    p.add_argument("--refit-on-train-valid", type=int, default=1, choices=[0,1],
                   help="If 1, refit best model on train+valid before testing; if 0, refit on train only")
    p.add_argument("--min-df", type=int, default=2, help="TfidfVectorizer min_df")
    p.add_argument("--max-df", type=float, default=0.95, help="TfidfVectorizer max_df")
    p.add_argument("--max-iter", type=int, default=5000, help="LinearSVC max_iter")
    p.add_argument("--seed", type=int, default=42, help="Random seed for CV shuffling")
    return p

# ------------------------
# Helpers
# ------------------------

def _ensure_dir_for(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_confusion_matrix(cm, labels, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
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
    _ensure_dir_for(out_path)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_compare_majority_vs_model(maj_acc, maj_f1, model_acc, model_f1, out_path):
    metrics = ["Accuracy", "Macro-F1"]
    m1 = [maj_acc, maj_f1]
    m2 = [model_acc, model_f1]
    idx = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(idx - width/2, m1, width, label="Majority")
    plt.bar(idx + width/2, m2, width, label="SVM (improved)")
    plt.xticks(idx, metrics)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Majority vs Improved SVM (Test)")
    plt.legend()
    plt.tight_layout()
    _ensure_dir_for(out_path)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ------------------------
# Main
# ------------------------

def main():
    args = build_argparser().parse_args()
    SEED = args.seed

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_CSV = os.path.join(BASE_DIR, "train_clean.csv")
    VALID_CSV = os.path.join(BASE_DIR, "valid_clean.csv")
    TEST_CSV  = os.path.join(BASE_DIR, "test_clean.csv")

    TEXT_COL = "clean_statement"
    LABEL_COL = "label"

    # Load data
    print(f"Loading datasets from:\n{TRAIN_CSV}\n{VALID_CSV}\n{TEST_CSV}\n")
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape:  {test_df.shape}\n")

    X_train, y_train = train_df[TEXT_COL], train_df[LABEL_COL]
    X_valid, y_valid = valid_df[TEXT_COL], valid_df[LABEL_COL]
    X_test,  y_test  = test_df[TEXT_COL],  test_df[LABEL_COL]

    labels_sorted = sorted(y_train.unique().tolist())

    # Majority (sanity check)
    majority_class = y_train.value_counts().idxmax()
    y_pred_majority = np.full(len(y_test), majority_class)
    maj_acc = accuracy_score(y_test, y_pred_majority)
    maj_f1  = f1_score(y_test, y_pred_majority, average="macro")
    print("=== Majority Baseline (Test) ===")
    print(f"Most frequent class: {majority_class}")
    print(f"Test Accuracy: {maj_acc:.4f}")
    print(f"Test Macro-F1: {maj_f1:.4f}\n")

    # Search space (kept small & effective)
    ngram_candidates = [(1,1), (1,2)]  # allow CV to decide if bigrams help
    C_candidates = [0.5, 1.0, 2.0]
    cw_candidates = [None, "balanced"]

    print("=== Cross-Validation on TRAIN split ===")
    print(f"CV folds: {args.cv_folds}, seed: {SEED}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)

    best_cfg = None
    best_cv_f1 = -1.0
    best_model = None

    for ngram in ngram_candidates:
        for C in C_candidates:
            for cw in cw_candidates:
                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        ngram_range=ngram,
                        min_df=args.min_df,
                        max_df=args.max_df,
                        sublinear_tf=True,
                        lowercase=True,
                        strip_accents="unicode",
                    )),
                    ("svm", LinearSVC(C=C, class_weight=cw, max_iter=args.max_iter))
                ])

                cv_scores = cross_val_score(
                    pipe, X_train, y_train,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=None
                )
                mean_f1 = cv_scores.mean()
                std_f1 = cv_scores.std()
                print(
                    f"SVM CV — ngram={ngram}, C={C}, class_weight={cw} -> "
                    f"CV Macro-F1={mean_f1:.4f} (±{std_f1:.4f})"
                )

                if mean_f1 > best_cv_f1:
                    best_cv_f1 = mean_f1
                    best_cfg = {"ngram_range": ngram, "C": C, "class_weight": cw}
                    best_model = pipe

    print("\n=== Selected by CV (TRAIN) ===")
    print(best_cfg)
    print(f"Best CV Macro-F1: {best_cv_f1:.4f}\n")

    # Optional: refit on train+valid to use more data before final test
    if args.refit_on_train_valid:
        print("Refitting best model on TRAIN+VALID ...")
        X_refit = pd.concat([X_train, X_valid], axis=0)
        y_refit = pd.concat([y_train, y_valid], axis=0)
        best_model.fit(X_refit, y_refit)
    else:
        print("Refitting best model on TRAIN only ...")
        best_model.fit(X_train, y_train)

    # Evaluate on VALID (for visibility) and TEST (final)
    print("\n=== Evaluation (VALID) ===")
    valid_pred = best_model.predict(X_valid)
    valid_acc = accuracy_score(y_valid, valid_pred)
    valid_f1  = f1_score(y_valid, valid_pred, average="macro")
    print(f"Valid Accuracy: {valid_acc:.4f} | Macro-F1: {valid_f1:.4f}")

    print("\n=== Evaluation (TEST) ===")
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1  = f1_score(y_test, test_pred, average="macro")
    print(f"Test Accuracy:  {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
    print("\nClassification Report (SVM, Test):")
    print(classification_report(y_test, test_pred, zero_division=0))

    # Save confusion matrix
    cm = confusion_matrix(y_test, test_pred, labels=labels_sorted)
    plot_confusion_matrix(cm, labels_sorted,
                          os.path.join(BASE_DIR, "fig_confusion_matrix_svm.png"),
                          title="Confusion Matrix (Test) — Improved SVM")

    # Compare to majority
    plot_compare_majority_vs_model(maj_acc, maj_f1, test_acc, test_f1,
                                   os.path.join(BASE_DIR, "fig_svm_vs_majority.png"))

if __name__ == "__main__":
    main()
