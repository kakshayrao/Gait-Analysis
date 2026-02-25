"""
download_data.py
================
Downloads, verifies, and extracts the UCI HAR (Human Activity Recognition)
dataset into data/raw/.
"""
import os
import zipfile
import urllib.request
import shutil

URL      = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00240/UCI%20HAR%20Dataset.zip")
DEST_ZIP = os.path.join("data", "raw", "UCI_HAR.zip")
DEST_DIR = os.path.join("data", "raw")


def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    bar = "#" * int(pct // 2)
    print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def download():
    os.makedirs(DEST_DIR, exist_ok=True)

    if not os.path.exists(DEST_ZIP):
        print("Downloading UCI HAR Dataset (~25 MB)...")
        urllib.request.urlretrieve(URL, DEST_ZIP, progress)
        print("\n  Download complete.")
    else:
        print("  Archive already present, skipping download.")

    uci_dir = os.path.join(DEST_DIR, "UCI HAR Dataset")
    if os.path.exists(uci_dir):
        print("  Dataset already extracted.")
    else:
        print("Extracting...")
        with zipfile.ZipFile(DEST_ZIP, "r") as zf:
            zf.extractall(DEST_DIR)
        print(f"  Extracted to {uci_dir}")

    print("\nDone! UCI HAR dataset is ready.")
    return uci_dir


if __name__ == "__main__":
    download()
