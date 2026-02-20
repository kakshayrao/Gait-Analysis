"""
Download the PhysioNet Gait in Parkinson's Disease (gaitpdb) dataset.
Fetches the file index page, extracts all .txt file links, and downloads
them into the data/ directory.
"""
import os
import re
import urllib.request

BASE_URL = "https://physionet.org/files/gaitpdb/1.0.0/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_file_list():
    """Scrape the directory listing page for .txt file links."""
    print(f"Fetching file list from {BASE_URL} ...")
    req = urllib.request.Request(BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    # Links look like: href="GaCo01_01.txt"
    files = re.findall(r'href="([^"]+\.txt)"', html)
    return sorted(set(files))


def download_file(filename):
    url = BASE_URL + filename
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest):
        return False  # already downloaded
    urllib.request.urlretrieve(url, dest)
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = get_file_list()
    print(f"Found {len(files)} .txt files to download.")

    for i, fname in enumerate(files, 1):
        new = download_file(fname)
        status = "downloaded" if new else "exists"
        print(f"  [{i:3d}/{len(files)}] {fname} — {status}")

    print(f"\nDone. Files saved to: {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
