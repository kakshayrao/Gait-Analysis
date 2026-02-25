"""
src/time_series.py
==================
Time-series analysis of IMU gait data:
  1. Autocorrelation of acc magnitude signal
  2. ADF stationarity test
  3. STL trend + seasonality decomposition of stride time series
  4. per-activity feature comparison (box plots)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf


OUTPUT_DIR = "output"


# ─── 1. Autocorrelation ───────────────────────────────────────────────────────

def plot_acf_steps(acc_mag_signal: np.ndarray,
                   label: str = "Gait",
                   nlags: int = 40,
                   save_path: str | None = None):
    """
    Plot autocorrelation of the acc magnitude signal (gait regularity).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(acc_mag_signal, lags=nlags, ax=ax, alpha=0.05)
    ax.set_title(f"Autocorrelation of Acc Magnitude — {label}", fontsize=13)
    ax.set_xlabel("Lag (samples @ 50 Hz)")
    ax.set_ylabel("ACF")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR,
                                 f"acf_{label.lower().replace(' ', '_')}.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved {os.path.basename(save_path)}")
    return save_path


# ─── 2. ADF Stationarity Test ─────────────────────────────────────────────────

def run_adf_test(signal: np.ndarray, label: str = "Signal") -> dict:
    """
    Augmented Dickey-Fuller stationarity test.

    Returns dict with test stat, p-value, critical values, verdict.
    """
    result = adfuller(signal, autolag="AIC")
    adf_stat   = result[0]
    p_value    = result[1]
    crit_vals  = result[4]
    stationary = p_value < 0.05

    print(f"\n  ADF Test — {label}")
    print(f"  {'─'*40}")
    print(f"  ADF Statistic  : {adf_stat:.4f}")
    print(f"  p-value        : {p_value:.6f}")
    print(f"  Stationary     : {'YES [OK]' if stationary else 'NO [WARNING]'}")
    for cv_key, cv_val in crit_vals.items():
        print(f"  Critical ({cv_key:3s})  : {cv_val:.4f}")

    return {
        "label": label,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "critical_values": crit_vals,
        "stationary": stationary,
    }


# ─── 3. STL Decomposition ─────────────────────────────────────────────────────

def plot_stl_decomposition(stride_series: np.ndarray,
                            label: str = "Gait",
                            period: int = 10,
                            save_path: str | None = None):
    """
    STL decomposition of a stride_time series into trend + seasonal + residual.

    Parameters
    ----------
    stride_series : 1-D array of stride times (seconds) across consecutive windows
    period        : seasonal period (default 10 = ~10 stride cycles)
    """
    if len(stride_series) < period * 2:
        print(f"  Skipping STL for '{label}' — too few observations.")
        return None

    stl = STL(stride_series, period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    components = {
        "Observed":   stride_series,
        "Trend":      res.trend,
        "Seasonal":   res.seasonal,
        "Residual":   res.resid,
    }
    for ax, (name, data) in zip(axes, components.items()):
        ax.plot(data, linewidth=1.0, color="#2196F3" if name != "Residual" else "#FF5722")
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_title(f"STL Decomposition — {label}", fontsize=13)
    axes[-1].set_xlabel("Window index")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR,
                                 f"stl_{label.lower().replace(' ', '_')}.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved {os.path.basename(save_path)}")
    return save_path


# ─── 4. Activity Feature Comparison ──────────────────────────────────────────

def plot_feature_comparison(features_df,
                             features_to_plot: list | None = None,
                             save_path: str | None = None):
    """
    Box plots comparing key gait features across activity types
    (Walking flat, Upstairs, Downstairs).
    """
    import pandas as pd

    activity_map = {1: "Walking", 2: "Upstairs", 3: "Downstairs"}
    df = features_df.copy()
    df["Activity"] = df["activity"].map(activity_map)

    if features_to_plot is None:
        features_to_plot = ["stride_time", "cadence", "sma",
                             "jerk", "mean_gyro", "spectral_entropy"]

    n = len(features_to_plot)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    palette = {"Walking": "#4CAF50", "Upstairs": "#FF9800", "Downstairs": "#F44336"}

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        groups = [df[df["Activity"] == act][feat].dropna().values
                  for act in ["Walking", "Upstairs", "Downstairs"]]
        parts = ax.boxplot(groups, patch_artist=True,
                           labels=["Walking", "Upstairs", "Downstairs"],
                           medianprops=dict(color="white", linewidth=2))
        colors = [palette["Walking"], palette["Upstairs"], palette["Downstairs"]]
        for patch, color in zip(parts["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gait Features by Activity Type", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "feature_comparison_by_activity.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  -> Saved {os.path.basename(save_path)}")
    return save_path


def run_full_time_series_analysis(features_df, X_sample: np.ndarray):
    """
    Run all time-series analyses and save plots to output/.

    Parameters
    ----------
    features_df : DataFrame from build_feature_dataframe()
    X_sample    : one representative acc magnitude window (128,)
                  used for ACF demo
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. ACF on a sample walking sequence
    plot_acf_steps(X_sample, label="Walking")

    # 2. ADF test per activity group
    for act_id, act_name in [(1, "Walking"), (2, "Upstairs"), (3, "Downstairs")]:
        subset = features_df[features_df["activity"] == act_id]["stride_time"].dropna()
        if len(subset) > 10:
            run_adf_test(subset.values, label=act_name)

    # 3. STL decomposition of stride time per activity
    for act_id, act_name in [(1, "Walking"), (2, "Upstairs"), (3, "Downstairs")]:
        subset = features_df[features_df["activity"] == act_id]["stride_time"].dropna()
        if len(subset) >= 20:
            plot_stl_decomposition(subset.values, label=act_name)

    # 4. Feature comparison box plots
    plot_feature_comparison(features_df)
