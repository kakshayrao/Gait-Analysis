# =============================================================================
# main.py — Entry Point
# =============================================================================
# Gait-Based Parkinson Detection and Real-Time Fall Risk Monitoring
# Using Vertical Ground Reaction Force (vGRF) Signals
#
# Runs the full pipeline:
#   1. Load PhysioNet gaitpdb data
#   2. Extract gait features
#   3. Time-series analysis (ACF, ADF test, Trend/Seasonality decomposition)
#   4. Train & evaluate: Random Forest, XGBoost, LSTM
#   5. Simulate real-time monitoring
#   6. Compare walking patterns (slow vs fast gait)
# =============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project modules ──────────────────────────────────────────────────────────
from src.data_loader import load_all_subjects
from src.feature_extraction import (
    build_feature_dataframe,
    smooth_signal,
    normalize_signal,
    segment_into_windows,
)
from src.model import train_and_evaluate, train_lstm
from src.live_monitoring import simulate_live_monitoring
from src.time_series_analysis import check_stationarity, decompose_gait


# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join("data", "gait-in-parkinsons-disease-1.0.0")
OUTPUT_DIR = os.path.join("output")
FS         = 100          # sampling frequency (Hz)
WINDOW     = 300          # 3 s window
STEP       = 150          # 50 % overlap


def ensure_dirs():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_stride_intervals(df):
    """Extract stride intervals from a subject DataFrame."""
    from scipy.signal import find_peaks
    sig = normalize_signal(smooth_signal(df["total_force"].values))
    peaks, _ = find_peaks(sig, height=np.mean(sig) * 0.5,
                          distance=int(FS * 0.4))
    if len(peaks) < 4:
        return None
    return np.diff(peaks) / FS


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 : Load Data
# ─────────────────────────────────────────────────────────────────────────────
def step1_load_data():
    print("\n" + "=" * 60)
    print("  STEP 1 — DATA LOADING")
    print("=" * 60)

    if not os.path.isdir(DATA_DIR):
        print(f"\n  [ERROR] Data directory '{DATA_DIR}' not found.")
        print("  Please download the PhysioNet gaitpdb dataset and place")
        print("  the .txt files inside the 'data/' folder.")
        print("  Dataset: https://physionet.org/content/gaitpdb/1.0.0/")
        sys.exit(1)

    subjects = load_all_subjects(DATA_DIR)
    if not subjects:
        print("  [ERROR] No valid subject files loaded. Exiting.")
        sys.exit(1)
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 : Preprocessing + Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def step2_extract_features(subjects):
    print("\n" + "=" * 60)
    print("  STEP 2 — PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 60)

    features_df = build_feature_dataframe(subjects, WINDOW, STEP, FS)
    print(f"\n  Feature DataFrame shape: {features_df.shape}")
    print(features_df.describe().round(4).to_string())
    return features_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 : Time-Series Analysis
#   3a. Autocorrelation of stride intervals
#   3b. Stationarity check (ADF test)
#   3c. Trend and seasonality decomposition
# ─────────────────────────────────────────────────────────────────────────────
def step3_time_series_analysis(subjects):
    print("\n" + "=" * 60)
    print("  STEP 3 — TIME-SERIES ANALYSIS")
    print("=" * 60)

    healthy_subj   = next((s for s in subjects if s[1] == 0), None)
    parkinson_subj = next((s for s in subjects if s[1] == 1), None)

    # ── 3.0  Raw signal plots ────────────────────────────────────────────
    for subj, tag in [(healthy_subj, "Healthy"), (parkinson_subj, "Parkinson")]:
        if subj is None:
            continue
        df, label, fname = subj
        sig = smooth_signal(df["total_force"].values)
        time_ax = np.arange(len(sig)) / FS

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time_ax, sig, linewidth=0.5, color="#1f77b4")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total vGRF")
        ax.set_title(f"Raw vGRF Signal — {tag} ({fname})",
                     fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"raw_signal_{tag.lower()}.png"),
                    dpi=150)
        plt.close()
        print(f"  → Saved raw_signal_{tag.lower()}.png")

    # ── 3a. Autocorrelation of stride intervals ──────────────────────────
    print("\n  [3a] Autocorrelation of Stride Intervals")
    for subj, tag in [(healthy_subj, "Healthy"), (parkinson_subj, "Parkinson")]:
        if subj is None:
            continue
        df, _, fname = subj
        intervals = _get_stride_intervals(df)
        if intervals is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        max_lag = min(40, len(intervals) - 1)
        acf = [np.corrcoef(intervals[:-lag], intervals[lag:])[0, 1]
               for lag in range(1, max_lag + 1)]
        ax.bar(range(1, max_lag + 1), acf, color="#4C72B0", alpha=0.7)
        ax.set_xlabel("Lag (strides)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Stride Interval ACF — {tag}",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"autocorrelation_{tag.lower()}.png"),
            dpi=150,
        )
        plt.close()
        print(f"  → Saved autocorrelation_{tag.lower()}.png")

    # ── 3b. Stationarity check (ADF test) ────────────────────────────────
    print("\n  [3b] Augmented Dickey-Fuller Test")
    for subj, tag in [(healthy_subj, "Healthy"), (parkinson_subj, "Parkinson")]:
        if subj is None:
            continue
        df, _, _ = subj
        intervals = _get_stride_intervals(df)
        if intervals is not None:
            check_stationarity(intervals, label=tag)

    # ── 3c. Trend & Seasonality decomposition ────────────────────────────
    print("\n  [3c] Trend and Seasonality Decomposition")
    for subj, tag in [(healthy_subj, "Healthy"), (parkinson_subj, "Parkinson")]:
        if subj is None:
            continue
        df, _, _ = subj
        intervals = _get_stride_intervals(df)
        if intervals is not None:
            decompose_gait(intervals, label=tag, output_dir=OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 : Train & Evaluate Models (RF, XGBoost, LSTM)
# ─────────────────────────────────────────────────────────────────────────────
def step4_train_models(features_df):
    print("\n" + "=" * 60)
    print("  STEP 4 — MACHINE LEARNING MODELS")
    print("=" * 60)

    # 4a. Random Forest + XGBoost
    print("\n  [4a] Random Forest & XGBoost")
    rf_model, feature_names = train_and_evaluate(features_df, OUTPUT_DIR)

    # 4b. LSTM
    print("\n  [4b] LSTM Sequence Modeling")
    train_lstm(features_df, OUTPUT_DIR)

    return rf_model, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 : Fall Risk Prediction (Live Monitoring)
# ─────────────────────────────────────────────────────────────────────────────
def step5_live_monitoring(subjects, model, feature_names):
    print("\n" + "=" * 60)
    print("  STEP 5a — FALL RISK PREDICTION (Live Monitoring)")
    print("=" * 60)

    healthy_subj   = next((s for s in subjects if s[1] == 0), None)
    parkinson_subj = next((s for s in subjects if s[1] == 1), None)

    for subj, tag in [(healthy_subj, "Healthy"),
                      (parkinson_subj, "Parkinson")]:
        if subj is None:
            print(f"  [SKIP] No {tag} subject available.")
            continue
        df, label, fname = subj
        df = df.ffill().fillna(0)

        simulate_live_monitoring(
            model=model,
            feature_names=feature_names,
            signal_left=df["total_left"].values,
            signal_right=df["total_right"].values,
            signal_total=df["total_force"].values,
            subject_label=tag,
            output_dir=OUTPUT_DIR,
            fs=FS,
            window_size=WINDOW,
            step=50,
        )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5b : Compare Walking Patterns (Slow vs Fast Gait)
# ─────────────────────────────────────────────────────────────────────────────
def step5b_walk_pattern_comparison(features_df):
    """
    Compare walking patterns by splitting subjects into slow vs fast
    walkers based on median cadence.

    Note: The PhysioNet gaitpdb dataset contains only walking trials.
    Running data is not available, so we use cadence-based stratification
    as a proxy for different locomotion intensities.
    """
    print("\n" + "=" * 60)
    print("  STEP 5b — WALKING PATTERN COMPARISON (Slow vs Fast Gait)")
    print("=" * 60)

    median_cadence = features_df["cadence"].median()
    features_df = features_df.copy()
    features_df["gait_speed"] = np.where(
        features_df["cadence"] >= median_cadence, "Fast", "Slow"
    )

    print(f"\n  Median cadence (split threshold): {median_cadence:.2f} steps/min")
    print(f"  Fast walkers:  {(features_df['gait_speed'] == 'Fast').sum()} windows")
    print(f"  Slow walkers:  {(features_df['gait_speed'] == 'Slow').sum()} windows")

    # Compare key features between slow and fast walkers
    compare_features = ["stride_time", "variability", "symmetry", "cv_force"]
    fig, axes = plt.subplots(1, len(compare_features), figsize=(18, 5))

    for i, col in enumerate(compare_features):
        slow = features_df.loc[features_df["gait_speed"] == "Slow", col]
        fast = features_df.loc[features_df["gait_speed"] == "Fast", col]
        axes[i].boxplot(
            [slow, fast],
            labels=["Slow Gait", "Fast Gait"],
            patch_artist=True,
            boxprops=dict(facecolor="#FFD580"),
        )
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Walking Pattern Comparison: Slow vs Fast Gait",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "walk_pattern_comparison.png"), dpi=150)
    plt.close()
    print(f"  → Saved walk_pattern_comparison.png")

    # Show Parkinson prevalence in each group
    for speed in ["Slow", "Fast"]:
        group = features_df[features_df["gait_speed"] == speed]
        pd_ratio = group["label"].mean()
        print(f"  {speed} gait → Parkinson prevalence: {pd_ratio:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  GAIT-BASED PARKINSON DETECTION & FALL RISK MONITORING  ║")
    print("║  Using Vertical Ground Reaction Force (vGRF) Signals    ║")
    print("╚" + "═" * 58 + "╝\n")

    ensure_dirs()

    # 1. Load
    subjects = step1_load_data()

    # 2. Features
    features_df = step2_extract_features(subjects)

    # 3. Time-series analysis (ACF, ADF, Decomposition)
    step3_time_series_analysis(subjects)

    # 4. ML models (RF, XGBoost, LSTM)
    model, feature_names = step4_train_models(features_df)

    # 5a. Fall risk prediction (live monitoring)
    step5_live_monitoring(subjects, model, feature_names)

    # 5b. Walking pattern comparison
    step5b_walk_pattern_comparison(features_df)

    print("\n" + "=" * 60)
    print("  ✓  ALL STEPS COMPLETE")
    print(f"  Plots saved to:  {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60 + "\n")

    # ── Clinical Summary ─────────────────────────────────────────────────
    print("  CLINICAL RELEVANCE SUMMARY")
    print("  " + "─" * 40)
    print("  • Increased stride variability is a hallmark of")
    print("    Parkinson's Disease and correlates with fall risk.")
    print("  • Reduced cadence indicates slower, less steady gait.")
    print("  • Gait asymmetry may signal unilateral motor deficit.")
    print("  • Slow-gait subjects show higher Parkinson prevalence,")
    print("    confirming bradykinesia as a PD biomarker.")
    print("  • A system like this could be deployed on wearable")
    print("    insole sensors for continuous monitoring of PD")
    print("    patients, alerting caregivers to elevated fall risk.")
    print()


if __name__ == "__main__":
    main()
