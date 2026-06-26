"""
model_training.py

Loads the gold feature store + label store across a training time window,
trains Logistic Regression and Random Forest models, evaluates both on a
held-out out-of-time (OOT) test window, selects the best model by AUC,
and saves the winning model (+ metadata) to the model bank.

Train window : 2023-01-01 to 2024-06-01  (18 months)
OOT test     : 2024-07-01 to 2024-12-01  (6 months)

This script is designed to be called directly (python utils/model_training.py)
or imported and called from an Airflow task.
"""

import os
import glob
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-06-01"
TEST_START  = "2024-07-01"
TEST_END    = "2024-12-01"

GOLD_FEATURE_STORE_DIR = "datamart/gold/feature_store/"
GOLD_LABEL_STORE_DIR   = "datamart/gold/label_store/"
MODEL_BANK_DIR         = "model_bank/"

# Columns that must NEVER be used as model inputs
ID_COLS = ["Customer_ID", "snapshot_date"]


# ─────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────
def load_gold_table(directory, prefix, date_list):
    """Load and concatenate gold parquet partitions for a list of snapshot dates."""
    frames = []
    for date_str in date_list:
        partition_name = f"{prefix}_{date_str.replace('-', '_')}.parquet"
        filepath = os.path.join(directory, partition_name)
        if os.path.exists(filepath):
            frames.append(pd.read_parquet(filepath))
        else:
            print(f"  [warn] missing partition: {filepath}")
    if not frames:
        raise FileNotFoundError(f"No partitions found in {directory} for given date range")
    return pd.concat(frames, ignore_index=True)


def generate_month_list(start_date_str, end_date_str):
    """Generate list of first-of-month date strings between start and end (inclusive)."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


def build_dataset(label_date_list, all_feature_dates):
    """
    Join feature store + label store into one X, y dataframe.

    label_date_list   : snapshot dates to use for the LABEL window (i.e. loans
                         that reached MOB==6 within this window).
    all_feature_dates : the full available range of feature store partitions.
                         We must load the FULL range here (not just label_date_list)
                         because a label dated e.g. 2024-07-01 (MOB6) corresponds to
                         a loan that originated 6 months earlier, around 2024-01-01.
                         The feature row we need to join against lives in the
                         2024-01-01 partition, not 2024-07-01. Loading only the
                         label window's dates would silently produce zero matches.

    Critical join logic: the label store's 'snapshot_date' is the date the loan
    reached MOB==6 — it is NOT the date to join features on. Instead we join on
    'feature_snapshot_date', which is the loan's origination date — the actual
    point-in-time when the bank had to make its lending decision and when
    feature values are valid. This prevents temporal leakage (joining on the
    MOB6 date would mean using features from 6 months in the future relative
    to the application).
    """
    features_df = load_gold_table(GOLD_FEATURE_STORE_DIR, "gold_feature_store", all_feature_dates)
    labels_df = load_gold_table(GOLD_LABEL_STORE_DIR, "gold_label_store", label_date_list)

    merged = labels_df.merge(
        features_df,
        left_on=["Customer_ID", "feature_snapshot_date"],
        right_on=["Customer_ID", "snapshot_date"],
        how="inner",
        suffixes=("", "_feat")
    )
    print(f"  merged rows: {len(merged)}  (labels: {len(labels_df)}, features loaded: {len(features_df)})")
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────
def prepare_X_y(df):
    """Split a merged dataframe into model-ready X (numeric only) and y (label)."""
    y = df["label"].astype(int)

    drop_cols = set(ID_COLS + ["loan_id", "label", "label_def", "feature_snapshot_date", "snapshot_date_feat"])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # Keep only numeric columns (one-hot encoding already done at gold layer,
    # so any remaining object columns are unexpected — log and drop them)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  [info] dropping non-numeric columns not yet encoded: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Replace any remaining inf/-inf (e.g. from earlier ratio division edge cases) with 0
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    return X, y, X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────
# Training + evaluation
# ─────────────────────────────────────────────────────────────────────────
def evaluate(model, X, y, scaler=None, needs_scaling=False):
    X_eval = scaler.transform(X) if needs_scaling else X
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    return {
        "auc": roc_auc_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


def train_and_select_best_model():
    print("=" * 70)
    print("STEP 1: Build training and OOT test datasets")
    print("=" * 70)

    train_dates = generate_month_list(TRAIN_START, TRAIN_END)
    test_dates = generate_month_list(TEST_START, TEST_END)

    # Feature partitions must cover the FULL available history, since a label's
    # feature_snapshot_date (loan origination) can fall well before the label's
    # own snapshot_date (MOB6 observation date). We use the earliest possible
    # origination date (data starts 2023-01-01) through the end of the test window.
    all_feature_dates = generate_month_list("2023-01-01", TEST_END)

    print(f"Train window: {TRAIN_START} to {TRAIN_END} ({len(train_dates)} months)")
    train_df = build_dataset(train_dates, all_feature_dates)

    print(f"OOT test window: {TEST_START} to {TEST_END} ({len(test_dates)} months)")
    test_df = build_dataset(test_dates, all_feature_dates)

    X_train, y_train, feature_cols = prepare_X_y(train_df)
    X_test, y_test, _ = prepare_X_y(test_df)

    # Ensure test set has exactly the same columns as train (handles any edge-case mismatch)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    print(f"\nTrain set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set (OOT): {X_test.shape[0]} rows")
    print(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")

    print("\n" + "=" * 70)
    print("STEP 2: Train candidate models")
    print("=" * 70)

    results = {}

    # ── Logistic Regression (needs scaling) ──
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    lr_train_metrics = evaluate(lr_model, X_train, y_train, scaler, needs_scaling=True)
    lr_test_metrics = evaluate(lr_model, X_test, y_test, scaler, needs_scaling=True)
    results["logistic_regression"] = {
        "model": lr_model, "scaler": scaler,
        "train_metrics": lr_train_metrics, "test_metrics": lr_test_metrics
    }
    print(f"  Train AUC: {lr_train_metrics['auc']:.4f} | OOT Test AUC: {lr_test_metrics['auc']:.4f}")

    # ── Random Forest (no scaling needed) ──
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_train_metrics = evaluate(rf_model, X_train, y_train)
    rf_test_metrics = evaluate(rf_model, X_test, y_test)
    results["random_forest"] = {
        "model": rf_model, "scaler": None,
        "train_metrics": rf_train_metrics, "test_metrics": rf_test_metrics
    }
    print(f"  Train AUC: {rf_train_metrics['auc']:.4f} | OOT Test AUC: {rf_test_metrics['auc']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 3: Select best model (by OOT test AUC)")
    print("=" * 70)

    best_name = max(results, key=lambda k: results[k]["test_metrics"]["auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    print(f"OOT Test metrics: {json.dumps(best['test_metrics'], indent=2)}")

    print("\n" + "=" * 70)
    print("STEP 4: Save best model to model bank")
    print("=" * 70)

    os.makedirs(MODEL_BANK_DIR, exist_ok=True)
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    artefact = {
        "model": best["model"],
        "scaler": best["scaler"],
        "model_name": best_name,
        "feature_cols": feature_cols,
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "train_metrics": best["train_metrics"],
        "test_metrics": best["test_metrics"],
        "trained_at": model_version,
    }

    model_path = os.path.join(MODEL_BANK_DIR, f"model_{model_version}.pkl")
    joblib.dump(artefact, model_path)
    print(f"Saved best model artefact to: {model_path}")

    # Also save a "latest" pointer for easy retrieval by inference script
    latest_path = os.path.join(MODEL_BANK_DIR, "model_latest.pkl")
    joblib.dump(artefact, latest_path)
    print(f"Saved latest model pointer to: {latest_path}")

    # Save a human-readable metadata summary alongside
    metadata = {
        "model_name": best_name,
        "trained_at": model_version,
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "n_features": len(feature_cols),
        "train_metrics": best["train_metrics"],
        "test_metrics": best["test_metrics"],
        "all_candidates": {
            name: {"train_metrics": r["train_metrics"], "test_metrics": r["test_metrics"]}
            for name, r in results.items()
        }
    }
    metadata_path = os.path.join(MODEL_BANK_DIR, f"model_{model_version}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return artefact


if __name__ == "__main__":
    train_and_select_best_model()
