# =============================================================================
# feature_extraction.py
# =============================================================================
# Segments vGRF signals into fixed-length windows and extracts interpretable
# gait features used for Parkinson's Disease classification.
#
# Features per window
# -------------------
#   stride_time     : mean time between consecutive force peaks (seconds)
#   cadence         : steps per minute  (60 / stride_time)
#   variability     : standard deviation of stride intervals
#   symmetry        : left-right force symmetry  (ratio of RMS forces)
#   mean_force      : mean total vertical ground reaction force
#   std_force       : standard deviation of total force
#   cv_force        : coefficient of variation of total force
# =============================================================================

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ── Signal Pre-processing Helpers ────────────────────────────────────────────

def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    mn, mx = signal.min(), signal.max()
    if mx - mn == 0:
        return np.zeros_like(signal)
    return (signal - mn) / (mx - mn)


# ── Windowing ────────────────────────────────────────────────────────────────

def segment_into_windows(signal: np.ndarray,
                         window_size: int = 300,
                         step: int = 150) -> list:
    """
    Slide a fixed-size window across the signal.

    Parameters
    ----------
    window_size : int
        Number of samples per window (300 = 3 s at 100 Hz).
    step : int
        Hop size between windows (150 = 50 % overlap).

    Returns
    -------
    List of 1-D numpy arrays.
    """
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start : start + window_size])
    return windows


# ── Per-Window Feature Computation ───────────────────────────────────────────

def compute_window_features(left_win: np.ndarray,
                            right_win: np.ndarray,
                            total_win: np.ndarray,
                            fs: int = 100):
    """
    Extract gait features from one window of data.

    Returns None if insufficient peaks are found for stride computation.
    """
    # --- Detect peaks in the total force signal ----
    # Height threshold: half the mean force avoids noise spikes
    height = np.mean(total_win) * 0.5
    peaks, _ = find_peaks(total_win, height=height, distance=int(fs * 0.4))

    if len(peaks) < 3:
        return None  # need at least 3 peaks to compute variability

    # Stride intervals (time between consecutive peaks)
    intervals = np.diff(peaks) / fs  # seconds

    stride_time = np.mean(intervals)
    cadence     = 60.0 / stride_time if stride_time > 0 else 0.0
    variability = np.std(intervals)

    # Gait symmetry: ratio of RMS(left) / RMS(right)
    rms_left  = np.sqrt(np.mean(left_win ** 2))
    rms_right = np.sqrt(np.mean(right_win ** 2))
    if rms_right == 0:
        symmetry = 0.0
    else:
        symmetry = rms_left / rms_right  # 1.0 = perfectly symmetric

    # Basic statistical features of the total force
    mean_force = np.mean(total_win)
    std_force  = np.std(total_win)
    cv_force   = std_force / mean_force if mean_force > 0 else 0.0

    return {
        "stride_time": stride_time,
        "cadence":     cadence,
        "variability": variability,
        "symmetry":    symmetry,
        "mean_force":  mean_force,
        "std_force":   std_force,
        "cv_force":    cv_force,
    }


# ── Build Feature DataFrame from All Subjects ───────────────────────────────

def build_feature_dataframe(subjects: list,
                            window_size: int = 300,
                            step: int = 150,
                            fs: int = 100) -> pd.DataFrame:
    """
    Process every subject's signals and create one row per window.

    Parameters
    ----------
    subjects : list of (DataFrame, label, filename) tuples
        As returned by data_loader.load_all_subjects().

    Returns
    -------
    pd.DataFrame with columns:
        stride_time, cadence, variability, symmetry,
        mean_force, std_force, cv_force, label
    """
    rows = []

    for df, label, fname in subjects:
        # Handle any NaN values by forward-fill then zero-fill
        df = df.ffill().fillna(0)

        # Extract the three core signals
        left_signal  = smooth_signal(df["total_left"].values)
        right_signal = smooth_signal(df["total_right"].values)
        total_signal = smooth_signal(df["total_force"].values)

        # Normalize per subject
        left_norm  = normalize_signal(left_signal)
        right_norm = normalize_signal(right_signal)
        total_norm = normalize_signal(total_signal)

        # Segment into windows
        left_wins  = segment_into_windows(left_norm,  window_size, step)
        right_wins = segment_into_windows(right_norm, window_size, step)
        total_wins = segment_into_windows(total_norm, window_size, step)

        for lw, rw, tw in zip(left_wins, right_wins, total_wins):
            feats = compute_window_features(lw, rw, tw, fs)
            if feats is not None:
                feats["label"] = label
                feats["subject_id"] = fname
                rows.append(feats)

    feature_df = pd.DataFrame(rows)
    print(f"Extracted {len(feature_df)} feature windows "
          f"(Healthy: {(feature_df['label'] == 0).sum()}, "
          f"Parkinson: {(feature_df['label'] == 1).sum()})")
    return feature_df
