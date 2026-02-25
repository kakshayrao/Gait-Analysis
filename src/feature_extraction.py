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


# -- Signal Pre-processing Helpers --------------------------------------------

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


# -- Windowing ----------------------------------------------------------------

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


# -- Helpers ------------------------------------------------------------------

def _spectral_entropy(signal: np.ndarray) -> float:
    """Compute normalised spectral entropy via FFT power spectrum."""
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    total = fft_vals.sum()
    if total == 0:
        return 0.0
    psd = fft_vals / total                      # normalised PSD
    psd = psd[psd > 0]                          # avoid log(0)
    entropy = -np.sum(psd * np.log2(psd))
    max_entropy = np.log2(len(fft_vals))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# -- Per-Window Feature Computation -------------------------------------------

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

    # -- NEW: Additional discriminative features --------------------------
    # Peak force (heel-strike intensity)
    peak_force = float(np.max(total_win))

    # RMS of total force
    rms_force = float(np.sqrt(np.mean(total_win ** 2)))

    # Range of total force
    range_force = float(np.max(total_win) - np.min(total_win))

    # Swing-stance ratio (fraction of samples below mean -> swing phase)
    swing_mask = total_win < mean_force
    swing_stance_ratio = float(swing_mask.sum()) / len(total_win)

    # Spectral entropy (frequency-domain irregularity)
    spec_entropy = _spectral_entropy(total_win)

    # Stride regularity via autocorrelation at dominant stride lag
    if len(intervals) >= 2:
        dominant_lag = int(round(np.median(np.diff(peaks))))
        if 0 < dominant_lag < len(total_win) // 2:
            acf = np.corrcoef(total_win[:-dominant_lag],
                              total_win[dominant_lag:])[0, 1]
            stride_regularity = float(acf) if np.isfinite(acf) else 0.0
        else:
            stride_regularity = 0.0
    else:
        stride_regularity = 0.0

    # -- NEW: High-discriminability PD biomarkers -------------------------

    # 1. Freeze-of-Gait (FoG) index: power in freeze band (3-8 Hz)
    #    divided by locomotion band (0.5-3 Hz). High in PD freezers.
    freqs = np.fft.rfftfreq(len(total_win), d=1.0/fs)
    fft_power = np.abs(np.fft.rfft(total_win)) ** 2
    loco_mask   = (freqs >= 0.5) & (freqs < 3.0)
    freeze_mask = (freqs >= 3.0) & (freqs < 8.0)
    loco_power   = fft_power[loco_mask].sum()
    freeze_power = fft_power[freeze_mask].sum()
    fog_index = freeze_power / (loco_power + 1e-8)

    # 2. Mean jerk: rate of change of total force -- PD has more irregular
    jerk = float(np.mean(np.abs(np.diff(total_win))))

    # 3. Force skewness -- PD gait tends to drag, shifting the distribution
    force_mean = np.mean(total_win)
    force_std  = np.std(total_win) + 1e-8
    skewness = float(np.mean(((total_win - force_mean) / force_std) ** 3))

    # 4. Force kurtosis -- measures impulsiveness of heel strikes
    kurtosis = float(np.mean(((total_win - force_mean) / force_std) ** 4))

    # 5. Mean heel-strike width (peak prominence width proxy)
    if len(peaks) >= 2:
        half_heights = total_win[peaks] * 0.5
        widths = []
        for p, hh in zip(peaks, half_heights):
            left_crossings  = np.where(total_win[:p] < hh)[0]
            right_crossings = np.where(total_win[p:] < hh)[0]
            l = left_crossings[-1]  if len(left_crossings)  > 0 else 0
            r = p + right_crossings[0] if len(right_crossings) > 0 else len(total_win) - 1
            widths.append(r - l)
        mean_strike_width = float(np.mean(widths)) / fs
    else:
        mean_strike_width = 0.0

    # 6. Left-right peak force asymmetry
    left_peaks, _  = find_peaks(left_win,  height=np.mean(left_win)*0.4,
                                 distance=int(fs * 0.4))
    right_peaks, _ = find_peaks(right_win, height=np.mean(right_win)*0.4,
                                 distance=int(fs * 0.4))
    lp_mean = np.mean(left_win[left_peaks])   if len(left_peaks)  > 0 else 0.0
    rp_mean = np.mean(right_win[right_peaks]) if len(right_peaks) > 0 else 0.0
    peak_asymmetry = abs(lp_mean - rp_mean) / (lp_mean + rp_mean + 1e-8)

    # 7. Stride interval coefficient of variation (already have std; add CV)
    stride_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-8))

    return {
        "stride_time":        stride_time,
        "cadence":            cadence,
        "variability":        variability,
        "symmetry":           symmetry,
        "mean_force":         mean_force,
        "std_force":          std_force,
        "cv_force":           cv_force,
        "peak_force":         peak_force,
        "rms_force":          rms_force,
        "range_force":        range_force,
        "swing_stance_ratio": swing_stance_ratio,
        "spectral_entropy":   spec_entropy,
        "stride_regularity":  stride_regularity,
        # New high-discriminability features
        "fog_index":          float(fog_index),
        "jerk":               jerk,
        "skewness":           skewness,
        "kurtosis":           kurtosis,
        "mean_strike_width":  mean_strike_width,
        "peak_asymmetry":     peak_asymmetry,
        "stride_cv":          stride_cv,
    }


# -- Build Feature DataFrame from All Subjects -------------------------------

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
    pd.DataFrame with feature columns + label + subject_id
    """
    rows = []

    for df, label, fname in subjects:
        df = df.ffill().fillna(0)

        left_signal  = smooth_signal(df["total_left"].values)
        right_signal = smooth_signal(df["total_right"].values)
        total_signal = smooth_signal(df["total_force"].values)

        left_norm  = normalize_signal(left_signal)
        right_norm = normalize_signal(right_signal)
        total_norm = normalize_signal(total_signal)

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

