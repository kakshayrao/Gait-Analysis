"""
src/preprocessing.py
====================
Signal cleaning and utility functions for IMU data.

UCI HAR raw signals are already:
  - Sampled at 50 Hz
  - Pre-segmented into 128-sample (2.56 s) windows with 50% overlap
  - Noise-filtered at source

This module provides additional utility helpers used during feature extraction.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


FS = 50          # sampling frequency (Hz)
WINDOW = 128     # samples per window  (2.56 s)


# ─── Filtering ────────────────────────────────────────────────────────────────

def butter_bandpass(lowcut: float = 0.3,
                    highcut: float = 20.0,
                    fs: float = FS,
                    order: int = 4):
    """Design a Butterworth bandpass filter (returns SOS form)."""
    nyq = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", output="sos")


_SOS = butter_bandpass()      # pre-build once


def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Apply bandpass filter to a 1-D signal."""
    return sosfiltfilt(_SOS, signal).astype(np.float32)


# ─── Magnitude helpers ────────────────────────────────────────────────────────

def magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Euclidean magnitude: sqrt(x^2 + y^2 + z^2)."""
    return np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(np.float32)


def extract_imu_components(X_window: np.ndarray):
    """
    Split a single window (128, 6) into named IMU components.

    Parameters
    ----------
    X_window : np.ndarray  shape (128, 6)
        Columns: [body_acc_x, body_acc_y, body_acc_z,
                  body_gyro_x, body_gyro_y, body_gyro_z]

    Returns
    -------
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z : np.ndarray(128,) each
    acc_mag, gyro_mag : np.ndarray(128,)
    """
    acc_x, acc_y, acc_z   = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    gyro_x, gyro_y, gyro_z = X_window[:, 3], X_window[:, 4], X_window[:, 5]

    acc_mag  = magnitude(acc_x, acc_y, acc_z)
    gyro_mag = magnitude(gyro_x, gyro_y, gyro_z)

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, acc_mag, gyro_mag


def normalize_windows(X: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each channel across all windows (global statistics).

    Parameters
    ----------
    X : np.ndarray  shape (N, 128, 6)

    Returns
    -------
    X_norm : np.ndarray  shape (N, 128, 6)
    """
    X_flat = X.reshape(-1, X.shape[-1])          # (N*128, 6)
    mu  = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    return ((X - mu) / std).astype(np.float32)
