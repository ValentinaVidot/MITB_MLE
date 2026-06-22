"""
model_inference.py

Loads the latest trained model from the model bank and scores customers
across each monthly snapshot date, saving predictions as a gold table
(datamart/gold/model_predictions/).

This is designed to be run AFTER model_training.py has produced a model
in model_bank/model_latest.pkl. It can be called directly or from an
Airflow task, and supports being run for a single date (for backfill-style
orchestration) or a full date range.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

GOLD_FEATURE_STORE_DIR     = "datamart/gold/feature_store/"
GOLD_PREDICTIONS_DIR       = "datamart/gold/model_predictions/"
MODEL_BANK_DIR             = "model_bank/"


def load_latest_model():
    """Load the most recently trained model artefact from the model bank."""
    model_path = os.path.join(MODEL_BANK_DIR, "model_latest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run model_training.py first."
        )
    artefact = joblib.load(model_path)
    print(f"Loaded model: {artefact['model_name']} (trained at {artefact['trained_at']})")
    return artefact


def score_snapshot(snapshot_date_str, artefact=None):
    """
    Score all customers in a single feature store snapshot date and save
    predictions as a gold table partition.
    """
    if artefact is None:
        artefact = load_latest_model()

    model = artefact["model"]
    scaler = artefact["scaler"]
    feature_cols = artefact["feature_cols"]
    model_name = artefact["model_name"]
    model_version = artefact["trained_at"]

    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(GOLD_FEATURE_STORE_DIR, partition_name)
    if not os.path.exists(filepath):
        print(f"  [warn] no feature store partition for {snapshot_date_str}, skipping")
        return None

    df = pd.read_parquet(filepath)
    print(f"{snapshot_date_str} | loaded {len(df)} customers from feature store")

    # ── Build X in the exact same column order/shape as training ──
    X = df.reindex(columns=feature_cols, fill_value=0)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Drop any non-numeric leftovers the same way training did
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        # re-add as zero columns so shape always matches feature_cols exactly
        for c in non_numeric:
            X[c] = 0
        X = X[feature_cols]

    X_eval = scaler.transform(X) if scaler is not None else X

    predicted_proba = model.predict_proba(X_eval)[:, 1]
    predicted_label = model.predict(X_eval)

    predictions_df = pd.DataFrame({
        "Customer_ID": df["Customer_ID"],
        "snapshot_date": df["snapshot_date"],
        "predicted_proba": predicted_proba,
        "predicted_label": predicted_label,
        "model_name": model_name,
        "model_version": model_version,
    })

    os.makedirs(GOLD_PREDICTIONS_DIR, exist_ok=True)
    out_partition = f"gold_predictions_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(GOLD_PREDICTIONS_DIR, out_partition)
    predictions_df.to_parquet(out_path, index=False)
    print(f"  saved {len(predictions_df)} predictions to {out_path}")
    print(f"  predicted default rate: {predicted_label.mean():.3f}")

    return predictions_df


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


def run_inference_for_range(start_date_str, end_date_str):
    """Score every monthly snapshot in a date range and save each as a gold partition."""
    artefact = load_latest_model()
    date_list = generate_month_list(start_date_str, end_date_str)

    print(f"\nRunning inference for {len(date_list)} months: {start_date_str} to {end_date_str}")
    all_predictions = []
    for date_str in date_list:
        preds = score_snapshot(date_str, artefact=artefact)
        if preds is not None:
            all_predictions.append(preds)

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        print(f"\nTotal predictions across all months: {len(combined)}")
        return combined
    else:
        print("\nNo predictions were generated.")
        return None


if __name__ == "__main__":
    # By default, score the full available history (2023-01-01 to 2024-12-01)
    run_inference_for_range("2023-01-01", "2024-12-01")
