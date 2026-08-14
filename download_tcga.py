
# Download TCGA Pathology Reports dataset.
# Source: Kefeli & Tatonetti (2024) - tatonetti-lab/tcga-path-reports


import os
import zipfile
import urllib.request
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "TCGA")
os.makedirs(OUT_DIR, exist_ok=True)

ZIP_URL = "https://github.com/tatonetti-lab/tcga-path-reports/raw/main/TCGA_Reports.csv.zip"
ZIP_PATH = os.path.join(OUT_DIR, "TCGA_Reports.csv.zip")
CSV_PATH = os.path.join(OUT_DIR, "TCGA_Reports.csv")


def download_with_progress(url, dest):
    print(f"Downloading: {url}")
    last_pct = -1

    def reporthook(count, block_size, total_size):
        nonlocal last_pct
        if total_size > 0:
            pct = int(count * block_size * 100 / total_size)
            pct = min(pct, 100)
            if pct != last_pct and pct % 10 == 0:
                print(f"  {pct}%...")
                last_pct = pct

    urllib.request.urlretrieve(url, dest, reporthook)
    print(f"  Saved: {dest}")


def main():
    print("=" * 60)
    print("TCGA Pathology Reports Downloader")
    print("Kefeli & Tatonetti, 2024 — fully open dataset")
    print("=" * 60)

    # Download zip
    if not os.path.exists(ZIP_PATH):
        download_with_progress(ZIP_URL, ZIP_PATH)
    else:
        print(f"Already downloaded: {ZIP_PATH}")

    # Unzip
    if not os.path.exists(CSV_PATH):
        print(f"\nUnzipping...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(OUT_DIR)
            extracted = z.namelist()
        print(f"  Extracted: {extracted}")
    else:
        print(f"Already extracted: {CSV_PATH}")

    # Inspect
    print(f"\nLoading and inspecting...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(f"\n{'='*60}")
    print(f"TCGA Reports Dataset Summary")
    print(f"{'='*60}")
    print(f"Total reports   : {len(df):,}")
    print(f"Total columns   : {len(df.columns)}")
    print(f"\nAll columns:")
    for col in df.columns:
        n_null = df[col].isna().sum()
        print(f"  {col:40s}  nulls={n_null}")

    print(f"\nSample report text (first 300 chars):")
    text_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["text", "report", "note", "content", "path"]
    )]
    if text_cols:
        sample = str(df[text_cols[0]].dropna().iloc[0])[:300]
        print(f"  [{text_cols[0]}]: {sample}...")

    print(f"\nCancer types (project_id or similar):")
    type_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["project", "cancer", "type", "disease", "site"]
    )]
    for col in type_cols[:3]:
        print(f"\n  [{col}] value counts:")
        vc = df[col].value_counts()
        for val, cnt in vc.head(15).items():
            print(f"    {str(val):40s}: {cnt}")

    print(f"\n{'='*60}")
    print(f"Raw file: {CSV_PATH}")
    print(f"Next: run  python -m src.data.preprocess_tcga")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
