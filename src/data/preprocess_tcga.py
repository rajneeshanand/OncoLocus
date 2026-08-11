
import os
import re
import argparse
import pandas as pd
import numpy as np

# Cancer-site keyword patterns -> TCGA project code
# More-specific patterns listed before generic fallbacks.

SITE_PATTERNS = [
    ("LUSC", [r"squamous.*lung", r"lung.*squamous", r"squamous cell.*lung",
              r"squamous.*pulmonary", r"pulmonary.*squamous"]),
    ("LUAD", [r"\blung\b", r"pulmonary", r"\bbronch", r"broncho"]),
    ("BRCA", [r"\bbreast\b"]),
    ("KIRC", [r"\bkidney\b", r"renal cell", r"renal carcinoma"]),
    ("BLCA", [r"\bbladder\b", r"urothelial", r"urinary tract"]),
    ("HNSC", [r"laryn", r"pharyn", r"\btongue\b", r"\bmandible\b",
              r"\bbuccal\b", r"oral cavity", r"head.*neck", r"neck.*dissection"]),
    ("COAD", [r"\bcolon\b", r"colorectal", r"\bsigmoid\b", r"\bcecum\b",
              r"\brectum\b", r"\brectal\b"]),
    ("UCEC", [r"\buterus\b", r"endometr", r"uterine"]),
    ("THCA", [r"\bthyroid\b"]),
    ("SARC", [r"leiomyosarcoma", r"liposarcoma", r"osteosarcoma",
              r"fibrosarcoma", r"\bsarcoma\b"]),
    ("GBM",  [r"glioblastoma", r"\bglioma\b", r"\bbrain\b"]),
    ("STAD", [r"\bstomach\b", r"\bgastric\b"]),
    ("PRAD", [r"\bprostate\b"]),
    ("SKCM", [r"\bmelanoma\b", r"\bskin\b"]),
    ("LIHC", [r"\bliver\b", r"hepato", r"hepatocell"]),
    ("CESC", [r"\bcervix\b", r"\bcervical\b"]),
    ("PAAD", [r"pancrea"]),
    ("OV",   [r"\bovary\b", r"\bovarian\b"]),
]

# Cancer types with enough reports after label filtering
VALID_CANCER_TYPES = {
    "BLCA", "HNSC", "LUAD", "BRCA", "KIRC",
    "UCEC", "COAD", "THCA", "SARC", "STAD",
    "GBM",  "PRAD", "LUSC", "LIHC", "SKCM",
    "CESC", "PAAD",
}

# Cancer type token prepended to text (same convention as GENIE BPC preprocessor)
CANCER_TYPE_TOKEN = {t: f"[CANCER:{t}]" for t in VALID_CANCER_TYPES}

# Integer ID for each cancer type (for the embedding layer)
CANCER_TYPE_IDS = {t: i for i, t in enumerate(sorted(VALID_CANCER_TYPES))}


# Tumor grade extraction (binary: low=0, high=1)

_GRADE_HIGH_PATS = [
    r"grade[ -]?(?:iii|3|high)",
    r"poorly[ -]differentiated",
    r"undifferentiated",
    r"high[ -]grade",
    r"grade iii",
    r"nuclear grade iii",
    r"histologic grade iii",
    r"fnclcc grade 3",
]

_GRADE_LOW_PATS = [
    r"grade[ -]?(?:i|1|low)",
    r"well[ -]differentiated",
    r"low[ -]grade",
    r"grade i\b",
    r"fnclcc grade 1",
    r"histologic grade i\b",
    r"nuclear grade i\b",
]


def _extract_cancer_site(text):
    t = text.lower()
    for site, pats in SITE_PATTERNS:
        for p in pats:
            if re.search(p, t):
                return site
    return "OTHER"


def _extract_grade_label(text):

    t = text.lower()
    is_high = any(re.search(p, t) for p in _GRADE_HIGH_PATS)
    is_low  = any(re.search(p, t) for p in _GRADE_LOW_PATS)

    if is_high and not is_low:
        return 1
    if is_low and not is_high:
        return 0
    # Both present (e.g., different sections at different grades):
    # take the more severe (high) as the label, same convention as clinical
    # practice where overall grade = worst component grade.
    if is_high and is_low:
        return 1
    return None  # no grade information


def preprocess_tcga(
    csv_path,
    out_dir="data/processed/tcga",
    min_count=40,
    verbose=True,
):

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    if verbose:
        print(f"Loaded {len(df):,} reports from {csv_path}")

    #  Extract patient_id from filename 
    df["patient_id"] = df["patient_filename"].str.extract(
        r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", expand=False
    ).fillna(df["patient_filename"])

    #  Extract cancer site and grade 
    df["cancer_type"] = df["text"].apply(_extract_cancer_site)
    df["label"]       = df["text"].apply(_extract_grade_label)

    if verbose:
        print("\nCancer site distribution (all reports):")
        print(df["cancer_type"].value_counts().to_string())
        n_low  = (df["label"] == 0).sum()
        n_high = (df["label"] == 1).sum()
        n_na   = df["label"].isna().sum()
        print(f"\nGrade label distribution:")
        print(f"  Low-grade  (0): {n_low:,}")
        print(f"  High-grade (1): {n_high:,}")
        print(f"  Unlabeled    : {n_na:,}")

    #  Drop unlabeled rows 
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if verbose:
        print(f"\nDropped {before - len(df):,} rows without grade information")
        print(f"Remaining: {len(df):,} reports")

    #  Filter to known cancer types with enough data 
    df = df[df["cancer_type"].isin(VALID_CANCER_TYPES)]
    counts = df["cancer_type"].value_counts()
    keep = counts[counts >= min_count].index.tolist()
    df = df[df["cancer_type"].isin(keep)]

    if verbose:
        print(f"\nAfter type filtering (>={min_count} reports): {len(df):,} reports")
        print(f"Cancer types: {sorted(df['cancer_type'].unique())}")

    #  Standard format columns 
    df = df.copy()
    df["visit_index"]    = 0
    df["days_from_start"] = 0.0
    df["report_text"] = df.apply(
        lambda r: f"{CANCER_TYPE_TOKEN.get(r['cancer_type'], '')} {str(r['text']).strip()}",
        axis=1,
    )

    #  Save per-type CSVs 
    all_data = {}
    for cancer_type in sorted(df["cancer_type"].unique()):
        sub = df[df["cancer_type"] == cancer_type][
            ["patient_id", "cancer_type", "visit_index",
             "days_from_start", "report_text", "label"]
        ].reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{cancer_type}_processed.csv")
        sub.to_csv(out_path, index=False)
        all_data[cancer_type] = sub

        if verbose:
            pos_rate = sub["label"].mean()
            print(f"  {cancer_type:6s}: {len(sub):4d} reports | "
                  f"high-grade {pos_rate:.1%} | {out_path}")

    if verbose:
        total = sum(len(v) for v in all_data.values())
        print(f"\nTotal: {total:,} reports, {len(all_data)} cancer types")
        print(f"Output: {out_dir}")

    return all_data


def load_processed_tcga(processed_dir):
    # Load already-processed TCGA CSVs (skip re-extraction)
    import glob
    all_data = {}
    for path in glob.glob(os.path.join(processed_dir, "*_processed.csv")):
        cancer_type = os.path.basename(path).replace("_processed.csv", "")
        all_data[cancer_type] = pd.read_csv(path)
    return all_data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess TCGA pathology reports")
    p.add_argument("--csv",       default="data/raw/TCGA/TCGA_Reports.csv")
    p.add_argument("--out_dir",   default="data/processed/tcga")
    p.add_argument("--min_count", type=int, default=40,
                   help="Minimum reports per cancer type to keep")
    args = p.parse_args()
    preprocess_tcga(args.csv, args.out_dir, min_count=args.min_count)
