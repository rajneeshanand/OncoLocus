
import os
import argparse
import pandas as pd
import numpy as np

# Column name variants seen in different GENIE BPC releases
PATIENT_ID_COLS    = ["patient_id", "patientId", "PATIENT_ID", "cpt_genie_sample_id"]
IMPRESSION_COLS    = ["img_impression", "impression", "IMPRESSION", "img_report_text"]
DATE_COLS          = ["img_scan_int", "img_scan_year", "scan_date", "img_date"]
LABEL_COLS         = ["img_overall", "overall", "OVERALL", "img_progression_label"]
CANCER_TYPE_COLS   = ["cancer_type", "cohort", "CANCER_TYPE"]

# Progression label mapping → binary (1 = progressing, 0 = not)
PROGRESSION_MAP = {
    "Progressing/Worsening/Enlarging": 1,
    "Progressing":                     1,
    "Worsening":                       1,
    "Enlarging":                       1,
    "Mixed response":                  1,
    "Mixed":                           1,
    "Stable/No change":                0,
    "Stable":                          0,
    "No change":                       0,
    "Improving/Responding":            0,
    "Improving":                       0,
    "Responding":                      0,
    "Not stated/Indeterminate":        0,
    "Indeterminate":                   0,
    "Not stated":                      0,
}

CANCER_TYPE_TOKENS = {
    "NSCLC":    "[CANCER:LUNG]",
    "BrCa":     "[CANCER:BREAST]",
    "CRC":      "[CANCER:COLORECTAL]",
    "PANC":     "[CANCER:PANCREATIC]",
    "Prostate": "[CANCER:PROSTATE]",
    "BLADDER":  "[CANCER:BLADDER]",
}


def _find_col(df, candidates):
    # Return first column name from candidates that exists in df.
    for c in candidates:
        if c in df.columns:
            return c
    return None


def preprocess_cohort(imaging_csv_path, cancer_type, verbose=True):

    df = pd.read_csv(imaging_csv_path, low_memory=False)

    if verbose:
        print(f"\n[{cancer_type}] Raw shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

    # Map column names
    pid_col    = _find_col(df, PATIENT_ID_COLS)
    imp_col    = _find_col(df, IMPRESSION_COLS)
    date_col   = _find_col(df, DATE_COLS)
    label_col  = _find_col(df, LABEL_COLS)

    missing = [name for name, col in
               [("patient_id", pid_col), ("impression", imp_col),
                ("date", date_col), ("label", label_col)]
               if col is None]
    if missing:
        print(f"  [WARNING] Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        print(f"  Skipping {cancer_type} — update PATIENT_ID_COLS etc. with correct names")
        return None

    # Rename to standard names 
    df = df.rename(columns={
        pid_col:   "patient_id",
        imp_col:   "report_text_raw",
        date_col:  "date_raw",
        label_col: "label_raw",
    })

    #  Drop rows missing key fields 
    before = len(df)
    df = df.dropna(subset=["patient_id", "report_text_raw", "label_raw"])
    if verbose:
        print(f"  Dropped {before - len(df)} rows with missing key fields")

    #  Map progression labels to binary 
    df["label"] = df["label_raw"].map(PROGRESSION_MAP)
    unmapped = df["label"].isna().sum()
    if unmapped > 0 and verbose:
        print(f"  [WARNING] {unmapped} unmapped label values:")
        print(f"    {df[df['label'].isna()]['label_raw'].value_counts().to_dict()}")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    #  Parse scan dates to days from first scan per patient 
    try:
        df["date_val"] = pd.to_numeric(df["date_raw"], errors="coerce")
        # If numeric, treat as days directly (GENIE BPC uses interval in days)
        if df["date_val"].notna().sum() > len(df) * 0.5:
            df["days_from_start"] = df.groupby("patient_id")["date_val"].transform(
                lambda x: x - x.min()
            )
        else:
            # Try parsing as date strings
            df["date_val"] = pd.to_datetime(df["date_raw"], errors="coerce")
            df["days_from_start"] = df.groupby("patient_id")["date_val"].transform(
                lambda x: (x - x.min()).dt.days
            )
    except Exception as e:
        if verbose:
            print(f"  [WARNING] Date parsing failed: {e}. Using visit order.")
        df = df.sort_values(["patient_id", "date_raw"])
        df["days_from_start"] = df.groupby("patient_id").cumcount() * 60

    df["days_from_start"] = df["days_from_start"].fillna(0).astype(float)

    #  Sort by patient and date, assign visit index 
    df = df.sort_values(["patient_id", "days_from_start"]).reset_index(drop=True)
    df["visit_index"] = df.groupby("patient_id").cumcount()

    #  Prepend cancer type token to report text 
    token = CANCER_TYPE_TOKENS.get(cancer_type, f"[CANCER:{cancer_type.upper()}]")
    df["report_text"] = token + " " + df["report_text_raw"].str.strip()

    #  Add cancer type column 
    df["cancer_type"] = cancer_type

    #  Filter patients with at least 2 visits 
    visit_counts = df.groupby("patient_id")["visit_index"].max() + 1
    valid_patients = visit_counts[visit_counts >= 2].index
    before = df["patient_id"].nunique()
    df = df[df["patient_id"].isin(valid_patients)]
    if verbose:
        print(f"  Patients with ≥2 visits: {df['patient_id'].nunique()} / {before}")
        print(f"  Total reports: {len(df)}")
        print(f"  Progression rate: {df['label'].mean():.1%}")

    return df[["patient_id", "cancer_type", "visit_index",
               "days_from_start", "report_text", "label"]]


def preprocess_all(data_dir, out_dir, verbose=True):
  
    os.makedirs(out_dir, exist_ok=True)
    all_data = {}

    cohort_dirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    for cohort in sorted(cohort_dirs):
        imaging_path = os.path.join(data_dir, cohort, "imaging_level_dataset.csv")
        if not os.path.exists(imaging_path):
            if verbose:
                print(f"[skip] {cohort}: imaging_level_dataset.csv not found")
            continue

        df = preprocess_cohort(imaging_path, cohort, verbose=verbose)
        if df is None or len(df) == 0:
            continue

        out_path = os.path.join(out_dir, f"{cohort}_processed.csv")
        df.to_csv(out_path, index=False)
        all_data[cohort] = df
        if verbose:
            print(f"  Saved: {out_path}")

    print(f"\nProcessed {len(all_data)} cohorts: {list(all_data.keys())}")
    total_patients = sum(df["patient_id"].nunique() for df in all_data.values())
    total_reports = sum(len(df) for df in all_data.values())
    print(f"Total: {total_patients} patients, {total_reports} reports")

    return all_data


def load_processed(processed_dir):
    import glob
    all_data = {}
    for path in glob.glob(os.path.join(processed_dir, "*_processed.csv")):
        cohort = os.path.basename(path).replace("_processed.csv", "")
        all_data[cohort] = pd.read_csv(path)
    return all_data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/raw")
    p.add_argument("--out_dir",   default="data/processed/real")
    args = p.parse_args()

    preprocess_all(args.data_dir, args.out_dir)
