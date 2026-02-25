"""
main.py
=======
Headless training + analysis pipeline for the IMU Gait Fall-Risk project.
Run this directly to train and save all models without starting the web app.

Usage:
    python main.py
"""

import os
import sys
import time

DATA_DIR   = os.path.join("data", "UCI-HAR Dataset")
OUTPUT_DIR = "output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Check dataset is present ──────────────────────────────────────────
    if not os.path.exists(DATA_DIR):
        print("[ERROR] UCI HAR dataset not found. Run: python download_data.py")
        sys.exit(1)

    # ── 2. Load data ──────────────────────────────────────────────────────────
    from src.data_loader import load_uci_har, make_fall_risk_dataset
    print("Loading subjects...")
    t0 = time.time()
    X_all, y_all, subjects_all = load_uci_har(DATA_DIR)

    print("\nCreating fall-risk dataset...")
    X_mob, y_risk, subj_mob, y_activity = make_fall_risk_dataset(
        X_all, y_all, subjects_all)

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    from src.feature_extraction import build_feature_dataframe
    print("\nExtracting features...")
    features_df = build_feature_dataframe(X_mob, y_risk, y_activity, subj_mob)

    # ── 4. Train RF + XGBoost ─────────────────────────────────────────────────
    from src.model import (train_classical_models, train_lstm,
                            plot_model_comparison)
    print("\nTraining RF + XGBoost models...")
    classical_metrics, _, _ = train_classical_models(features_df, OUTPUT_DIR)

    # ── 5. Time-series analysis ───────────────────────────────────────────────
    from src.time_series import run_full_time_series_analysis
    import numpy as np
    from src.preprocessing import extract_imu_components

    # Pick a representative walking window for ACF demo
    walking_idx = (features_df["activity"] == 1)
    sample_row  = features_df[walking_idx].index[0] if walking_idx.any() else 0
    sample_win  = X_mob[sample_row]
    _, _, _, _, _, _, acc_mag, _ = extract_imu_components(sample_win)

    print("\nRunning time-series analysis...")
    run_full_time_series_analysis(features_df, acc_mag)

    # ── 6. Train LSTM ─────────────────────────────────────────────────────────
    print("\nTraining LSTM model...")
    lstm_metrics = train_lstm(X_mob, y_risk, subj_mob, OUTPUT_DIR)

    # ── 7. Model comparison chart ─────────────────────────────────────────────
    all_metrics = {
        "rf":   classical_metrics["rf"],
        "xgb":  classical_metrics["xgb"],
        "lstm": lstm_metrics,
    }
    plot_model_comparison(all_metrics, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  All outputs saved to  '{OUTPUT_DIR}/'")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
