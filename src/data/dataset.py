
# PyTorch Dataset classes for HTT.
"""
PatientSequenceDataset:
  - Each sample = one patient = a sequence of radiology reports
  - Returns tokenized reports, days-elapsed tensor, and label sequence
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

CANCER_TYPE_TO_ID = {
    "NSCLC":    0,
    "BrCa":     1,
    "CRC":      2,
    "PANC":     3,
    "Prostate": 4,
}


class PatientSequenceDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer_name="microsoft/BioGPT-Large",
        max_report_len=256,
        max_seq_len=10,
        use_last_label=True,
        cancer_type_to_id=None,
    ):
        self.max_report_len = max_report_len
        self.max_seq_len = max_seq_len
        self.use_last_label = use_last_label

        self.cancer_type_to_id = cancer_type_to_id if cancer_type_to_id is not None \
            else CANCER_TYPE_TO_ID

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Group by patient
        self.patients = []
        for pid, group in df.groupby("patient_id"):
            group = group.sort_values("visit_index").reset_index(drop=True)
            self.patients.append(group)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        group = self.patients[idx]

        # Truncate to max_seq_len visits
        group = group.iloc[: self.max_seq_len]
        n_visits = len(group)

        # Tokenize each report 
        input_ids_list = []
        attention_mask_list = []

        for _, row in group.iterrows():
            enc = self.tokenizer(
                row["report_text"],
                max_length=self.max_report_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attention_mask_list.append(enc["attention_mask"].squeeze(0))

        # Stack into (n_visits, max_report_len)
        input_ids = torch.stack(input_ids_list)
        attention_masks = torch.stack(attention_mask_list)

        # Pad to max_seq_len if needed
        pad_len = self.max_seq_len - n_visits
        if pad_len > 0:
            pad_ids = torch.zeros(pad_len, self.max_report_len, dtype=torch.long)
            pad_mask = torch.zeros(pad_len, self.max_report_len, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_masks = torch.cat([attention_masks, pad_mask], dim=0)

        # Days elapsed (normalized to 0-1 range for stability) 
        days = torch.tensor(group["days_from_start"].values, dtype=torch.float)
        days = days / 365.0  # normalize to years
        if pad_len > 0:
            days = torch.cat([days, torch.zeros(pad_len)])

        #  Sequence mask (1 for real visits, 0 for padding) 
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        seq_mask[:n_visits] = True

        #  Labels 
        labels = torch.tensor(group["label"].values, dtype=torch.long)
        if pad_len > 0:
            labels = torch.cat([labels, torch.full((pad_len,), -1, dtype=torch.long)])

        if self.use_last_label:
            # Patient-level label: did the patient ever progress?
            patient_label = torch.tensor(
                int(group["label"].max()), dtype=torch.long
            )
        else:
            patient_label = labels

        #  Cancer type 
        cancer_type_id = torch.tensor(
            self.cancer_type_to_id.get(group["cancer_type"].iloc[0], 0),
            dtype=torch.long,
        )

        return {
            "input_ids":      input_ids,        # (max_seq_len, max_report_len)
            "attention_mask": attention_masks,  # (max_seq_len, max_report_len)
            "days_elapsed":   days,             # (max_seq_len,)
            "seq_mask":       seq_mask,         # (max_seq_len,) bool
            "visit_labels":   labels,           # (max_seq_len,)
            "patient_label":  patient_label,    # scalar
            "cancer_type_id": cancer_type_id,   # scalar
            "patient_id":     group["patient_id"].iloc[0],
            "n_visits":       n_visits,
        }


def build_datasets(
    all_data,
    tokenizer_name="microsoft/BioGPT-Large",
    max_report_len=256,
    max_seq_len=10,
    train_ratio=0.7,
    val_ratio=0.15,
    # test_ratio=0.15 (remainder)
    gap2_held_out_type="PANC",
    seed=42,
    cancer_type_to_id=None,
):

    import random
    random.seed(seed)

    train_dfs, val_dfs, test_dfs = [], [], []
    gap2_df = None

    for cancer_type, df in all_data.items():
        if cancer_type == gap2_held_out_type:
            gap2_df = df
            continue

        patient_ids = df["patient_id"].unique().tolist()
        random.shuffle(patient_ids)

        n = len(patient_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = set(patient_ids[:n_train])
        val_ids = set(patient_ids[n_train: n_train + n_val])
        test_ids = set(patient_ids[n_train + n_val:])

        train_dfs.append(df[df["patient_id"].isin(train_ids)])
        val_dfs.append(df[df["patient_id"].isin(val_ids)])
        test_dfs.append(df[df["patient_id"].isin(test_ids)])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    kwargs = dict(
        tokenizer_name=tokenizer_name,
        max_report_len=max_report_len,
        max_seq_len=max_seq_len,
        cancer_type_to_id=cancer_type_to_id,
    )

    train_ds = PatientSequenceDataset(train_df, **kwargs)
    val_ds = PatientSequenceDataset(val_df, **kwargs)
    test_ds = PatientSequenceDataset(test_df, **kwargs)
    gap2_ds = PatientSequenceDataset(gap2_df, **kwargs) if gap2_df is not None else None

    print(f"Train patients: {len(train_ds)}")
    print(f"Val   patients: {len(val_ds)}")
    print(f"Test  patients: {len(test_ds)}")
    if gap2_ds:
        print(f"Gap2 ({gap2_held_out_type}) patients: {len(gap2_ds)}")

    return train_ds, val_ds, test_ds, gap2_ds
