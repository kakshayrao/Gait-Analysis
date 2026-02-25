"""
src/feature_extraction.py
==========================
Extracts biomechanically-meaningful gait features from IMU windows.

Each UCI HAR window is 128 samples at 50 Hz (= 2.56 seconds).
Features are computed from the acc and gyro magnitude signals.
"""

import numpy as np
from scipy.signal import find_peaks
from src.preprocessing import extract_imu_components

FS = 50   # Hz


# ─── Spectral helpers ────────────────────────────────────────────────────────

def _spectral_entropy(signal: np.ndarray) -> float:
    """Normalised spectral entropy from FFT power spectrum."""
    fft_power = np.abs(np.fft.rfft(signal)) ** 2
    total = fft_power.sum()
    if total == 0:
        return 0.0
    psd = fft_power / total
    psd = psd[psd > 0]
    entropy = -np.sum(psd * np.log2(psd))
    return float(entropy / np.log2(len(fft_power))) if len(fft_power) > 1 else 0.0


def _dominant_frequency(signal: np.ndarray, fs: float = FS) -> float:
    """Frequency (Hz) at the maximum power in the FFT spectrum."""
    fft_power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    # Exclude DC component
    fft_power[0] = 0
    return float(freqs[np.argmax(fft_power)])


# ─── Per-window feature computation ──────────────────────────────────────────

def compute_imu_features(window: np.ndarray, fs: float = FS) -> dict | None:
    """
    Compute 12 gait features from one IMU window.

    Parameters
    ----------
    window : np.ndarray  shape (128, 6)
        [body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z]
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict of features, or None if insufficient peaks for stride computation.
    """
    (acc_x, acc_y, acc_z,
     gyro_x, gyro_y, gyro_z,
     acc_mag, gyro_mag) = extract_imu_components(window)

    # ── Stride / Cadence ────────────────────────────────────────────────────
    height_threshold = np.mean(acc_mag) * 0.8
    peaks, _ = find_peaks(acc_mag,
                           height=height_threshold,
                           distance=int(fs * 0.3))   # min 0.3 s between steps

    if len(peaks) < 2:
        return None   # not enough steps detected in this window

    intervals = np.diff(peaks) / fs          # seconds between consecutive steps
    stride_time  = float(np.mean(intervals))
    cadence      = 60.0 / stride_time if stride_time > 0 else 0.0
    step_variance = float(np.var(intervals))

    # ── Symmetry index ──────────────────────────────────────────────────────
    # Energy in first half vs second half of window
    half = len(acc_mag) // 2
    e1 = float(np.sum(acc_mag[:half] ** 2))
    e2 = float(np.sum(acc_mag[half:] ** 2))
    symmetry_index = e1 / (e2 + 1e-8)

    # ── Signal Magnitude Area ────────────────────────────────────────────────
    sma = float(np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)))

    # ── Gyroscope features ───────────────────────────────────────────────────
    mean_gyro = float(np.mean(gyro_mag))

    # ── Jerk (rate of change of acceleration) ────────────────────────────────
    jerk = float(np.mean(np.abs(np.diff(acc_mag))))

    # ── Frequency domain ─────────────────────────────────────────────────────
    dom_freq       = _dominant_frequency(acc_mag, fs)
    spec_entropy   = _spectral_entropy(acc_mag)

    # ── Autocorrelation at lag-1 ─────────────────────────────────────────────
    if len(acc_mag) > 1:
        acf = np.corrcoef(acc_mag[:-1], acc_mag[1:])[0, 1]
        autocorr = float(acf) if np.isfinite(acf) else 0.0
    else:
        autocorr = 0.0

    # ── RMS acceleration ─────────────────────────────────────────────────────
    rms_acc = float(np.sqrt(np.mean(acc_mag ** 2)))

    # ── Tilt angle (mean inclination from vertical) ──────────────────────────
    acc_xy = np.sqrt(acc_x ** 2 + acc_y ** 2) + 1e-8
    tilt_rad = np.arctan2(np.abs(acc_z), acc_xy)
    tilt_angle = float(np.degrees(np.mean(tilt_rad)))

    # ── Stride coefficient of variation ──────────────────────────────────────
    stride_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-8))

    return {
        "stride_time":    stride_time,
        "cadence":        cadence,
        "step_variance":  step_variance,
        "symmetry_index": symmetry_index,
        "sma":            sma,
        "mean_gyro":      mean_gyro,
        "jerk":           jerk,
        "dominant_freq":  dom_freq,
        "spectral_entropy": spec_entropy,
        "autocorr_lag1":  autocorr,
        "rms_acc":        rms_acc,
        "tilt_angle":     tilt_angle,
        "stride_cv":      stride_cv,
    }


# ─── Build feature DataFrame from all windows ─────────────────────────────────

def build_feature_dataframe(X: np.ndarray,
                            y_risk: np.ndarray,
                            y_activity: np.ndarray,
                            subjects: np.ndarray):
    """
    Extract features from all windows into a pandas DataFrame.

    Parameters
    ----------
    X          : (N, 128, 6)  raw IMU windows
    y_risk     : (N,)         binary fall risk labels (0=low, 1=high)
    y_activity : (N,)         original activity labels (1-3)
    subjects   : (N,)         subject IDs

    Returns
    -------
    pd.DataFrame with feature columns + label + subject_id + activity
    """
    import pandas as pd

    rows = []
    skipped = 0
    for i in range(len(X)):
        feats = compute_imu_features(X[i])
        if feats is None:
            skipped += 1
            continue
        feats["label"]      = int(y_risk[i])
        feats["activity"]   = int(y_activity[i])
        feats["subject_id"] = int(subjects[i])
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"Feature extraction: {len(df)} windows  "
          f"(skipped {skipped} with <2 peaks)")
    print(f"  Low risk: {(df['label']==0).sum()}  "
          f"High risk: {(df['label']==1).sum()}")
    return df
