"""
model_monitoring.py

Monitors the deployed model's performance and stability across time:

1. PERFORMANCE (vs ground truth): wherever a prediction's snapshot_date
   matches a matured label (i.e. that loan has reached MOB==6 and a label
   exists), compute real AUC / accuracy / precision / recall / f1.

2. STABILITY (drift, no ground truth needed): compute the Population
   Stability Index (PSI) of each month's predicted-probability distribution
   against the distribution seen during training. A high PSI means the
   model is scoring the current population very differently than what
   it was trained on — an early warning sign even before performance
   visibly degrades.

3. PREDICTED VS ACTUAL DEFAULT RATE: simple, intuitive view for business
   stakeholders — is the model's predicted default rate tracking the real
   default rate over time?

Saves all monitoring results as a gold table (datamart/gold/model_monitoring/)
and produces matplotlib visualisations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

GOLD_PREDICTIONS_DIR = "datamart/gold/model_predictions/"
GOLD_LABEL_STORE_DIR  = "datamart/gold/label_store/"
GOLD_MONITORING_DIR   = "datamart/gold/model_monitoring/"
MODEL_BANK_DIR        = "model_bank/"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def generate_month_list(start_date_str, end_date_str):
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


def load_predictions(date_str):
    path = os.path.join(GOLD_PREDICTIONS_DIR, f"gold_predictions_{date_str.replace('-', '_')}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_all_labels(label_date_range):
    """
    Load and concatenate ALL label store partitions across the given range.

    This is necessary because label partitions are named by their MOB6
    'snapshot_date' (when the loan observation was made), but we need to
    look up labels by 'feature_snapshot_date' (loan origination date) to
    match them against predictions. Since a label's origination date can
    fall many months before its own partition's snapshot_date, we must
    load broadly across the whole label date range and then filter by
    feature_snapshot_date afterward.
    """
    frames = []
    for date_str in label_date_range:
        path = os.path.join(GOLD_LABEL_STORE_DIR, f"gold_label_store_{date_str.replace('-', '_')}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calculate_psi(reference_scores, current_scores, n_bins=10):
    """
    Population Stability Index between a reference distribution (e.g. training
    set scores) and a current distribution (e.g. this month's predicted scores).

    PSI interpretation (industry standard thresholds):
      < 0.1  : no significant shift
      0.1-0.25 : moderate shift, worth monitoring
      > 0.25 : significant shift, model may need review/retraining
    """
    # Bin edges based on the reference distribution's quantiles
    bin_edges = np.quantile(reference_scores, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0  # not enough variation to compute meaningfully

    ref_counts, _ = np.histogram(reference_scores, bins=bin_edges)
    cur_counts, _ = np.histogram(current_scores, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) / division by 0 with a small epsilon
    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────
# Main monitoring routine
# ─────────────────────────────────────────────────────────────────────────
def run_monitoring(start_date_str="2023-01-01", end_date_str="2024-12-01"):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    print("=" * 70)
    print("MODEL MONITORING")
    print("=" * 70)

    # ── Load model metadata for the reference (training) score distribution ──
    model_path = os.path.join(MODEL_BANK_DIR, "model_latest.pkl")
    artefact = joblib.load(model_path)
    model_name = artefact["model_name"]
    model_version = artefact["trained_at"]
    train_start, train_end = artefact["train_window"]

    print(f"Monitoring model: {model_name} (trained at {model_version})")
    print(f"Training window was: {train_start} to {train_end}\n")

    # Reference distribution = predicted probabilities during the TRAINING window
    train_dates = generate_month_list(train_start, train_end)
    reference_preds = []
    for d in train_dates:
        p = load_predictions(d)
        if p is not None:
            reference_preds.append(p["predicted_proba"])
    reference_scores = pd.concat(reference_preds, ignore_index=True).values if reference_preds else np.array([0.3])
    print(f"Reference distribution built from {len(reference_scores)} training-period predictions\n")

    # ── Load ALL label partitions once upfront (label partitions span the
    #    full 24 months since loans reaching MOB6 are spread across that range) ──
    full_label_date_range = generate_month_list("2023-01-01", "2024-12-01")
    all_labels = load_all_labels(full_label_date_range)
    print(f"Loaded {len(all_labels)} total matured labels across all partitions\n")

    # ── Walk through every month and compute monitoring stats ──
    date_list = generate_month_list(start_date_str, end_date_str)
    monitoring_rows = []

    for date_str in date_list:
        preds = load_predictions(date_str)
        if preds is None:
            print(f"{date_str} | no predictions found, skipping")
            continue

        row = {
            "snapshot_date": date_str,
            "model_name": model_name,
            "model_version": model_version,
            "n_customers_scored": len(preds),
            "predicted_default_rate": float(preds["predicted_label"].mean()),
            "mean_predicted_proba": float(preds["predicted_proba"].mean()),
        }

        # ── Stability: PSI vs training reference ──
        row["psi"] = calculate_psi(reference_scores, preds["predicted_proba"].values)
        if row["psi"] < 0.1:
            row["psi_flag"] = "stable"
        elif row["psi"] < 0.25:
            row["psi_flag"] = "moderate_shift"
        else:
            row["psi_flag"] = "significant_shift"

        # ── Performance: compare against matured labels where available ──
        # A label's feature_snapshot_date == loan origination date == the date
        # the loan was scored. We filter the full label set to find labels
        # whose origination matches this prediction snapshot.
        actual_default_rate = None
        if len(all_labels) > 0 and "feature_snapshot_date" in all_labels.columns:
            matched_labels = all_labels[all_labels["feature_snapshot_date"].astype(str) == date_str]
            if len(matched_labels) > 0:
                merged = matched_labels.merge(preds, on="Customer_ID", how="inner")
                if len(merged) > 0:
                    y_true = merged["label"].astype(int)
                    y_pred = merged["predicted_label"].astype(int)
                    y_proba = merged["predicted_proba"]
                    actual_default_rate = float(y_true.mean())
                    row["n_matured_labels"] = len(merged)
                    row["actual_default_rate"] = actual_default_rate
                    try:
                        row["auc"] = roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else None
                    except Exception:
                        row["auc"] = None
                    row["accuracy"] = accuracy_score(y_true, y_pred)
                    row["precision"] = precision_score(y_true, y_pred, zero_division=0)
                    row["recall"] = recall_score(y_true, y_pred, zero_division=0)
                    row["f1"] = f1_score(y_true, y_pred, zero_division=0)

        if actual_default_rate is None:
            row["n_matured_labels"] = 0
            row["actual_default_rate"] = None
            row["auc"] = None
            row["accuracy"] = None
            row["precision"] = None
            row["recall"] = None
            row["f1"] = None

        monitoring_rows.append(row)

        perf_str = f"AUC={row['auc']:.3f}" if row["auc"] is not None else "AUC=n/a (label not matured yet)"
        print(f"{date_str} | predicted_rate={row['predicted_default_rate']:.3f} | "
              f"PSI={row['psi']:.4f} ({row['psi_flag']}) | {perf_str}")

    monitoring_df = pd.DataFrame(monitoring_rows)

    # ── Save as gold monitoring table (one combined file, also partitioned) ──
    os.makedirs(GOLD_MONITORING_DIR, exist_ok=True)
    combined_path = os.path.join(GOLD_MONITORING_DIR, "gold_model_monitoring_all.parquet")
    monitoring_df.to_parquet(combined_path, index=False)
    print(f"\nSaved combined monitoring table to: {combined_path}")

    for date_str in monitoring_df["snapshot_date"]:
        sub = monitoring_df[monitoring_df["snapshot_date"] == date_str]
        out_path = os.path.join(GOLD_MONITORING_DIR, f"gold_model_monitoring_{date_str.replace('-', '_')}.parquet")
        sub.to_parquet(out_path, index=False)

    return monitoring_df


# ─────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────
def plot_monitoring_results(monitoring_df, output_dir="datamart/gold/model_monitoring/"):
    """Generate and save monitoring visualisations as PNG files."""
    monitoring_df = monitoring_df.sort_values("snapshot_date")
    dates = monitoring_df["snapshot_date"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # ── Plot 1: Predicted vs Actual default rate ──
    ax = axes[0]
    ax.plot(dates, monitoring_df["predicted_default_rate"], marker='o', label="Predicted default rate", color="#0D9488")
    actual_mask = monitoring_df["actual_default_rate"].notna()
    if actual_mask.any():
        ax.plot(dates[actual_mask], monitoring_df.loc[actual_mask, "actual_default_rate"],
                marker='s', label="Actual default rate (matured)", color="#E05252")
    ax.set_title("Predicted vs Actual Default Rate Over Time")
    ax.set_ylabel("Default Rate")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: PSI over time ──
    ax = axes[1]
    ax.plot(dates, monitoring_df["psi"], marker='o', color="#F59E0B")
    ax.axhline(0.1, color="green", linestyle="--", alpha=0.5, label="Stable threshold (0.10)")
    ax.axhline(0.25, color="red", linestyle="--", alpha=0.5, label="Significant shift threshold (0.25)")
    ax.set_title("Population Stability Index (PSI) Over Time")
    ax.set_ylabel("PSI")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: AUC over time (where matured) ──
    ax = axes[2]
    auc_mask = monitoring_df["auc"].notna()
    if auc_mask.any():
        ax.plot(dates[auc_mask], monitoring_df.loc[auc_mask, "auc"], marker='o', color="#1E3A5F")
        ax.axhline(0.7, color="gray", linestyle="--", alpha=0.5, label="Acceptable threshold (0.70)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No matured labels available yet", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Model AUC Over Time (vs Matured Ground Truth)")
    ax.set_ylabel("AUC")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "monitoring_charts.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved monitoring charts to: {out_path}")
    plt.close()

    return out_path


if __name__ == "__main__":
    monitoring_df = run_monitoring()
    plot_monitoring_results(monitoring_df)
