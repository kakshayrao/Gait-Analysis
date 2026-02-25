"""
src/data_loader.py
==================
Loads the UCI HAR raw Inertial Signals dataset.

Default data path: data/UCI-HAR Dataset   (note the hyphen)

UCI HAR structure (raw signals):
  train/Inertial Signals/body_acc_x_train.txt  (128 timesteps per row)
  train/Inertial Signals/body_gyro_x_train.txt
  ...
  train/subject_train.txt   (subject ID per window)
  train/y_train.txt         (activity label 1-6 per window)
"""

import os
import numpy as np

DEFAULT_DATA_DIR = os.path.join("data", "UCI-HAR Dataset")

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# UCI HAR raw signal channels we use
SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def _load_txt(path: str) -> np.ndarray:
    """Load a whitespace-delimited text file as a float32 array."""
    return np.loadtxt(path, dtype=np.float32)


def _load_split(data_dir: str, split: str) -> tuple:
    """
    Load one split ('train' or 'test') from the UCI HAR raw signals.

    Returns
    -------
    X : np.ndarray  shape (N, 128, 6)   6 IMU channels x 128 timesteps
    y : np.ndarray  shape (N,)          activity labels  1-6
    subjects : np.ndarray shape (N,)    subject IDs      1-30
    """
    inertial_dir = os.path.join(data_dir, split, "Inertial Signals")

    channels = []
    for sig in SIGNAL_FILES:
        fpath = os.path.join(inertial_dir, f"{sig}_{split}.txt")
        channels.append(_load_txt(fpath))          # shape (N, 128)

    X = np.stack(channels, axis=-1)                # (N, 128, 6)

    y_path = os.path.join(data_dir, split, f"y_{split}.txt")
    y = _load_txt(y_path).astype(np.int32).ravel()

    subj_path = os.path.join(data_dir, split, f"subject_{split}.txt")
    subjects = _load_txt(subj_path).astype(np.int32).ravel()

    return X, y, subjects


def load_uci_har(data_dir: str = DEFAULT_DATA_DIR) -> tuple:
    """
    Load the full UCI HAR dataset (train + test splits merged).

    Returns
    -------
    X        : np.ndarray (N, 128, 6)   raw IMU windows
    y        : np.ndarray (N,)          activity labels 1-6
    subjects : np.ndarray (N,)          subject IDs
    """
    X_tr, y_tr, subj_tr = _load_split(data_dir, "train")
    X_te, y_te, subj_te = _load_split(data_dir, "test")

    X        = np.concatenate([X_tr, X_te], axis=0)
    y        = np.concatenate([y_tr, y_te], axis=0)
    subjects = np.concatenate([subj_tr, subj_te], axis=0)

    n_windows = len(y)
    counts = {ACTIVITY_NAMES[i]: int((y == i).sum()) for i in range(1, 7)}
    print(f"Loaded {n_windows} windows from {len(np.unique(subjects))} subjects.")
    for name, cnt in counts.items():
        print(f"  {name:25s}: {cnt:5d} windows")

    return X, y, subjects


def make_fall_risk_dataset(X: np.ndarray,
                           y: np.ndarray,
                           subjects: np.ndarray) -> tuple:
    """
    Filter to mobile activities and create binary fall-risk labels.

    Fall Risk mapping:
      WALKING (1)             -> 0  (Low)
      WALKING_UPSTAIRS (2)    -> 1  (High)
      WALKING_DOWNSTAIRS (3)  -> 1  (High)
      SITTING/STANDING/LAYING -> excluded

    Returns
    -------
    X_mob, y_risk, subj_mob, y_activity (original 1-3 labels)
    """
    mobile_mask = np.isin(y, [1, 2, 3])
    X_mob    = X[mobile_mask]
    y_acts   = y[mobile_mask]
    subj_mob = subjects[mobile_mask]

    y_risk = np.where(y_acts == 1, 0, 1).astype(np.int32)  # 1=low, 2,3=high

    n_low  = int((y_risk == 0).sum())
    n_high = int((y_risk == 1).sum())
    print(f"\nFall-risk dataset: {len(y_risk)} windows  "
          f"| Low risk: {n_low}  High risk: {n_high}")
    return X_mob, y_risk, subj_mob, y_acts
