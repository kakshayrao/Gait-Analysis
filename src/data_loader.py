# =============================================================================
# data_loader.py
# =============================================================================
# Loads PhysioNet "Gait in Parkinson's Disease" (gaitpdb) dataset files.
#
# Each .txt file contains tab-separated columns:
#   Column 0  : Time (seconds)
#   Columns 1-8  : Left foot sensors  (L1 – L8)
#   Columns 9-16 : Right foot sensors (R1 – R8)
#   Column 17 : Total left foot force   (provided by dataset)
#   Column 18 : Total right foot force   (provided by dataset)
#
# Filename convention:
#   GaCo*  → Healthy Control
#   GaPt*  → Parkinson's Disease patient
#   JuCo*  → Healthy Control
#   JuPt*  → Parkinson's Disease patient
#   SiCo*  → Healthy Control
#   SiPt*  → Parkinson's Disease patient
# =============================================================================

import os
import numpy as np
import pandas as pd


# Column names for the 19-column gaitpdb format
COLUMN_NAMES = (
    ["time"]
    + [f"L{i}" for i in range(1, 9)]   # 8 left-foot sensors
    + [f"R{i}" for i in range(1, 9)]   # 8 right-foot sensors
    + ["total_left", "total_right"]      # aggregated force per foot
)


def get_label_from_filename(filename: str) -> int:
    """
    Determine subject label from the filename.

    Returns
    -------
    0 : Healthy Control  (filename contains 'Co')
    1 : Parkinson's Disease  (filename contains 'Pt')
    -1 : Unknown / skipped
    """
    base = os.path.basename(filename)
    if "Co" in base:
        return 0   # Healthy
    elif "Pt" in base:
        return 1   # Parkinson
    return -1       # Unknown — will be filtered out


def load_subject_file(filepath: str) -> pd.DataFrame:
    """
    Read a single gaitpdb .txt file and return a tidy DataFrame.

    An extra column 'total_force' is computed as the sum of total_left
    and total_right for whole-body vertical ground reaction force.
    """
    # Files are tab-separated with no header row
    df = pd.read_csv(filepath, sep="\t", header=None)

    # Some files may have 19 or 20 columns; keep only the first 19
    df = df.iloc[:, :19]
    df.columns = COLUMN_NAMES

    # Derived feature: overall vertical ground reaction force
    df["total_force"] = df["total_left"] + df["total_right"]

    return df


def load_all_subjects(data_dir: str):
    """
    Load every valid .txt walking-trial file from *data_dir*.

    Returns
    -------
    subjects : list of (DataFrame, label, filename) tuples
        Only files whose label is 0 or 1 are included.
    """
    subjects = []
    txt_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".txt")
    )

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in '{data_dir}'. "
            "Please download the PhysioNet gaitpdb dataset first."
        )

    for fname in txt_files:
        label = get_label_from_filename(fname)
        if label == -1:
            continue  # skip non-subject files (e.g. README)

        filepath = os.path.join(data_dir, fname)
        try:
            df = load_subject_file(filepath)
            subjects.append((df, label, fname))
        except Exception as exc:
            print(f"  [WARN] Skipping {fname}: {exc}")

    print(f"Loaded {len(subjects)} subject files "
          f"({sum(1 for _, l, _ in subjects if l == 0)} Healthy, "
          f"{sum(1 for _, l, _ in subjects if l == 1)} Parkinson)")
    return subjects
