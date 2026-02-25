# =============================================================================
# time_series_analysis.py
# =============================================================================
# Time-series analysis methods for gait signals:
#   - Autocorrelation of stride intervals
#   - Stationarity check (Augmented Dickey-Fuller test)
#   - Trend and seasonality decomposition of gait cycles
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def check_stationarity(stride_intervals: np.ndarray, label: str = ""):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on stride intervals.

    Null hypothesis: the series has a unit root (non-stationary).
    If p-value < 0.05 -> reject H0 -> series is stationary.

    Parameters
    ----------
    stride_intervals : 1-D array of stride durations (seconds).
    label : str -- descriptive label for console output.

    Returns
    -------
    dict with ADF statistic, p-value, and stationarity verdict.
    """
    if len(stride_intervals) < 10:
        print(f"  [{label}] Too few intervals ({len(stride_intervals)}) "
              "for ADF test -- skipping.")
        return None

    result = adfuller(stride_intervals, autolag="AIC")
    adf_stat = result[0]
    p_value  = result[1]
    is_stationary = p_value < 0.05

    print(f"\n  ADF Test -- {label}")
    print(f"  {'-' * 40}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.6f}")
    print(f"  Stationary    : {'YES [OK]' if is_stationary else 'NO ✗'}")
    for key, val in result[4].items():
        print(f"  Critical ({key}) : {val:.4f}")

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "is_stationary": is_stationary,
    }


def decompose_gait(stride_intervals: np.ndarray,
                   label: str = "",
                   output_dir: str = "output"):
    """
    Decompose stride intervals into trend, seasonal, and residual
    components using additive seasonal decomposition.

    The 'period' is estimated from the data -- typically corresponds
    to a few gait cycles.

    Parameters
    ----------
    stride_intervals : 1-D array of stride durations (seconds).
    label : str -- descriptive label for the plot title.
    output_dir : str -- directory to save the decomposition plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(stride_intervals) < 20:
        print(f"  [{label}] Too few intervals for decomposition -- skipping.")
        return

    # Use a period that captures ~10 strides as one "season"
    period = min(10, len(stride_intervals) // 3)
    if period < 2:
        period = 2

    decomposition = seasonal_decompose(
        stride_intervals, model="additive", period=period
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(stride_intervals, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Observed")
    axes[0].set_title(f"Gait Cycle Decomposition -- {label}",
                      fontsize=14, fontweight="bold")

    axes[1].plot(decomposition.trend, color="#ff7f0e", linewidth=1.2)
    axes[1].set_ylabel("Trend")

    axes[2].plot(decomposition.seasonal, color="#2ca02c", linewidth=0.8)
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(decomposition.resid, color="#d62728", linewidth=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Stride Index")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_label = label.replace(" ", "_").lower()
    save_path = os.path.join(output_dir, f"decomposition_{safe_label}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved decomposition_{safe_label}.png")
