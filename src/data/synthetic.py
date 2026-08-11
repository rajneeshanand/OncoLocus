

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CANCER_TYPES = ["NSCLC", "BrCa", "CRC", "PANC", "Prostate"]

CANCER_TYPE_TOKENS = {
    "NSCLC":    "[CANCER:LUNG]",
    "BrCa":     "[CANCER:BREAST]",
    "CRC":      "[CANCER:COLORECTAL]",
    "PANC":     "[CANCER:PANCREATIC]",
    "Prostate": "[CANCER:PROSTATE]",
}


TEMPLATES = {
    "NSCLC": [
        "Comparison to prior CT dated [[DATE]]. The right upper lobe mass measures {size} cm, {change} from {prior_size} cm on prior study. No new pulmonary nodules identified. Mediastinal lymphadenopathy {lymph}. Pleural effusion {effusion}.",
        "Interval CT chest. Target lesion in the right lower lobe measures {size} cm, previously {prior_size} cm. Findings are {overall} with treatment. No evidence of new distant metastases.",
        "CT chest/abdomen/pelvis. Primary lung lesion {size} cm x {size2} cm {change}. Hilar lymph nodes {lymph}. Liver and adrenal glands unremarkable.",
        "Restaging CT. Known NSCLC with primary lesion now {size} cm, {change} compared to prior imaging [[DATE]]. Bone scan pending for osseous metastasis evaluation.",
    ],
    "BrCa": [
        "MRI breast with contrast. The index lesion in the {side} breast measures {size} cm, {change} from {prior_size} cm. Axillary lymph nodes {lymph}. No new satellite lesions.",
        "CT chest/abdomen/pelvis for staging. Hepatic lesion {size} cm in segment {seg}, {change}. Known breast primary. No new osseous lesions.",
        "Bone scan. Increased uptake in the {bone} {change} compared to prior study [[DATE]]. Consistent with known osseous metastases from breast cancer.",
        "PET/CT. FDG-avid lesion in {side} breast {size} cm, SUVmax {suv}. Axillary lymph nodes {lymph}. Hepatic foci {change}.",
    ],
    "CRC": [
        "CT abdomen/pelvis. Post-resection surveillance. No evidence of local recurrence at the anastomotic site. Liver {liver}. No new peritoneal implants.",
        "CT chest/abdomen/pelvis. Hepatic metastases: dominant lesion segment {seg} measures {size} cm, {change} from {prior_size} cm on prior study [[DATE]].",
        "MRI liver. Multiple hepatic lesions {change}. Largest lesion {size} cm in segment {seg}. Portal lymphadenopathy {lymph}.",
        "Restaging CT. Colorectal primary status post resection. New peritoneal nodule {size} cm identified in the {location}. {overall} disease.",
    ],
    "PANC": [
        "CT abdomen/pelvis with contrast. Pancreatic head mass {size} cm {change} from prior {prior_size} cm. Superior mesenteric vein {vessel}. No new distant metastases.",
        "MRI abdomen. Known pancreatic adenocarcinoma. Primary tumor {size} cm, {change}. Celiac axis {vessel}. Liver shows {liver}.",
        "CT for restaging. Pancreatic mass now {size} cm previously {prior_size} cm. Peripancreatic lymph nodes {lymph}. No new hepatic lesions.",
        "PET/CT. Hypermetabolic pancreatic mass {size} cm, SUVmax {suv}. Peripancreatic soft tissue {change}. No definite distant metastatic disease.",
    ],
    "Prostate": [
        "MRI prostate multiparametric. Index lesion in the {zone} zone measures {size} cm, PI-RADS {pirads}. No evidence of extracapsular extension.",
        "Bone scan. No new osseous metastases. Prior lesion in the {bone} {change} compared to [[DATE]].",
        "CT abdomen/pelvis. No pelvic lymphadenopathy. No visceral metastases. PSA {psa} at time of imaging.",
        "PSMA PET/CT. Focal uptake in the right posterior zone {size} cm. Pelvic lymph node involvement {lymph}. Bone metastases {overall}.",
    ],
}

PROGRESSION_PHRASES = {
    "progressing": [
        "increased in size",
        "enlarged",
        "new lesion identified",
        "increased in number",
        "worsened",
        "progressive disease",
        "interval increase",
    ],
    "stable": [
        "stable in size",
        "unchanged",
        "no significant interval change",
        "similar to prior",
        "stable disease",
        "no new lesions",
    ],
    "improving": [
        "decreased in size",
        "reduced",
        "partial response",
        "improved",
        "responding to treatment",
        "interval decrease",
    ],
}

FILL_VALUES = {
    "size":       lambda: f"{random.uniform(1.0, 6.0):.1f}",
    "size2":      lambda: f"{random.uniform(0.8, 5.0):.1f}",
    "prior_size": lambda: f"{random.uniform(0.8, 6.5):.1f}",
    "seg":        lambda: str(random.randint(2, 8)),
    "side":       lambda: random.choice(["left", "right"]),
    "bone":       lambda: random.choice(["lumbar spine", "thoracic spine", "right iliac crest", "left femur", "sacrum"]),
    "zone":       lambda: random.choice(["peripheral", "transitional", "central"]),
    "pirads":     lambda: str(random.randint(2, 5)),
    "psa":        lambda: f"{random.uniform(0.1, 45.0):.1f}",
    "suv":        lambda: f"{random.uniform(2.0, 18.0):.1f}",
    "location":   lambda: random.choice(["right lower quadrant", "pelvis", "omentum", "mesentery"]),
    "vessel":     lambda: random.choice(["patent", "abutted", "encased without occlusion", "narrowed"]),
    "liver":      lambda: random.choice(["no lesions", "stable lesions", "new hypodensity"]),
    "lymph":      lambda: random.choice(["not enlarged", "mildly enlarged", "stable", "increased in size"]),
    "effusion":   lambda: random.choice(["absent", "small", "moderate", "stable"]),
    "overall":    lambda: random.choice(["stable", "progressing", "responding", "mixed"]),
}


def _fill_template(template, change_phrase):
    text = template.replace("{change}", change_phrase)
    for key, fn in FILL_VALUES.items():
        text = text.replace("{" + key + "}", fn())
    text = text.replace("[[DATE]]", f"20{random.randint(18,23)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
    return text


def generate_report(cancer_type, label):
    if label == 1:  # progressing
        change_phrase = random.choice(PROGRESSION_PHRASES["progressing"])
    elif label == 0 and random.random() < 0.5:
        change_phrase = random.choice(PROGRESSION_PHRASES["stable"])
    else:
        change_phrase = random.choice(PROGRESSION_PHRASES["improving"])

    template = random.choice(TEMPLATES[cancer_type])
    report = _fill_template(template, change_phrase)

    # Prepend cancer type token (Gap 2 feature)
    token = CANCER_TYPE_TOKENS[cancer_type]
    return f"{token} {report}"


def generate_patient_sequence(patient_id, cancer_type, n_visits=None):
 
    if n_visits is None:
        n_visits = random.randint(2, 8)

    # Assign an overall trajectory (mostly stable, or eventually progressing)
    trajectory = random.choices(
        ["stable_throughout", "progression_midway", "progression_late", "response_then_progress"],
        weights=[0.3, 0.3, 0.25, 0.15],
    )[0]

    start_date = datetime(random.randint(2018, 2022), random.randint(1, 12), random.randint(1, 28))
    records = []

    for visit_idx in range(n_visits):
        # Days since first visit (roughly every 6-12 weeks)
        days_elapsed = visit_idx * random.randint(42, 90)
        visit_date = start_date + timedelta(days=days_elapsed)

        # Determine label based on trajectory
        if trajectory == "stable_throughout":
            label = 0
        elif trajectory == "progression_midway":
            label = 1 if visit_idx >= n_visits // 2 else 0
        elif trajectory == "progression_late":
            label = 1 if visit_idx >= n_visits - 2 else 0
        else:  # response_then_progress
            if visit_idx < 2:
                label = 0
            elif visit_idx < n_visits - 1:
                label = 0
            else:
                label = 1

        report_text = generate_report(cancer_type, label)

        records.append({
            "patient_id":    patient_id,
            "cancer_type":   cancer_type,
            "visit_index":   visit_idx,
            "days_from_start": days_elapsed,
            "scan_date":     visit_date.strftime("%Y-%m-%d"),
            "report_text":   report_text,
            "label":         label,  # 1 = progressing, 0 = not progressing
        })

    return records


def generate_synthetic_dataset(
    n_patients_per_type=200,
    output_dir=None,
    seed=42,
):

    random.seed(seed)
    np.random.seed(seed)

    all_data = {}
    patient_counter = 0

    for cancer_type in CANCER_TYPES:
        records = []
        for _ in range(n_patients_per_type):
            pid = f"SYN-{cancer_type[:3].upper()}-{patient_counter:05d}"
            patient_records = generate_patient_sequence(pid, cancer_type)
            records.extend(patient_records)
            patient_counter += 1

        df = pd.DataFrame(records)
        all_data[cancer_type] = df

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            out_path = f"{output_dir}/{cancer_type}_synthetic.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {len(df)} records ({n_patients_per_type} patients) -> {out_path}")

    return all_data


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "synthetic")
    data = generate_synthetic_dataset(n_patients_per_type=200, output_dir=out)
    for ct, df in data.items():
        prog_rate = df.groupby("patient_id")["label"].max().mean()
        print(f"{ct}: {len(df['patient_id'].unique())} patients, "
              f"{len(df)} reports, "
              f"progression rate: {prog_rate:.1%}")
