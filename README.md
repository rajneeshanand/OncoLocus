# OncoLocus

**Hierarchical Temporal Transformer for Cancer Grade Prediction and Cross-Cancer Transfer Learning**

> A deep learning framework that reads sequences of cancer pathology reports across time and generalizes to cancer types it has never seen during training.

---

## Overview

OncoLocus addresses two gaps left open by prior cancer NLP literature:

**Gap 1 — No temporal modeling.**
Every prior cancer NLP model reads one report in isolation. OncoLocus processes a patient's full sequence of reports in chronological order, encoding the actual number of days between visits using Continuous-Time Positional Encoding.

**Gap 2 — No cross-cancer transfer.**
No prior study tests whether a model trained on some cancer types can predict outcomes for entirely unseen cancer types. OncoLocus achieves an average AUROC of **0.923** on three cancer types (thyroid, sarcoma, lung squamous cell) it never saw during training — matching in-distribution performance.

---

## Architecture

```
Patient Reports (chronological sequence)
        │
        ▼
┌─────────────────────────┐
│  Level 1: Report Encoder │   BiomedBERT + LoRA (11% params trainable)
│  Per-report → 512-dim   │   Mean pooling over non-padding tokens
└────────────┬────────────┘
             │  sequence of embeddings + actual days elapsed
             ▼
┌──────────────────────────────────┐
│  Level 2: Temporal Transformer   │   4 layers, 8 heads
│  • Continuous-Time Encoding      │   Sinusoidal over actual days (not indices)
│  • Cancer Type Conditioning      │   Learnable embedding per cancer type
│  • CLS token aggregation         │   Patient-level summary
└────────────┬─────────────────────┘
             │
             ▼
    Binary Classifier
  (high-grade / low-grade)
```

---

## Key Results

### Experiment 1 — Temporal Modeling (Synthetic Sequential Data)

| Model | Val AUROC | Zero-Shot PANC AUROC | Zero-Shot PANC F1 |
|---|---|---|---|
| **HTT (Ours)** | **0.9421** | **0.9945** | **0.9474** |
| Baseline (no temporal) | 0.8808 | 0.9490 | 0.8819 |

HTT outperforms the non-temporal baseline by **+6.13 AUROC points** on validation and **+4.55 points** on zero-shot pancreatic cancer transfer.

### Experiment 2 — Cross-Cancer Transfer on Real Data (TCGA-Reports)

**In-distribution (11 cancer types seen during training):**

| AUROC | F1 | Precision | Recall |
|---|---|---|---|
| 0.9234 | 0.9046 | 0.8812 | 0.9293 |

**Zero-shot transfer (3 cancer types never seen during training):**

| Cancer Type | AUROC | F1 | n |
|---|---|---|---|
| Thyroid Carcinoma (THCA) | 1.000 | 0.889 | 69 |
| Sarcoma (SARC) | 0.960 | 0.954 | 119 |
| Lung Squamous Cell (LUSC) | 0.808 | 0.894 | 271 |
| **Average** | **0.923** | **0.912** | 459 |

---

## Dataset

**TCGA-Reports** (Kefeli & Tatonetti, 2024)
- 9,523 free-text pathology reports from The Cancer Genome Atlas
- Fully open access — no data use agreement required
- Source: [tatonetti-lab/tcga-path-reports](https://github.com/tatonetti-lab/tcga-path-reports)

Download automatically:
```bash
python download_tcga.py
```

---

## Installation

```bash
git clone https://github.com/rajneeshanand/OncoLocus.git
cd OncoLocus
pip install -r requirements.txt
```

---

## Usage

### Preprocess TCGA data (run once)
```bash
python -m src.data.preprocess_tcga --csv data/raw/TCGA/TCGA_Reports.csv
```

### Train HTT on synthetic sequential data (Experiment 1 — Gap 1)
```bash
python train.py --use_synthetic --epochs 10 --batch_size 8
```

### Train Baseline ablation on synthetic data
```bash
python train.py --use_synthetic --baseline --output_dir results/synthetic_baseline
```

### Train HTT on TCGA real data (Experiment 2 — Gap 2)
```bash
python train_tcga.py \
    --gap2_held_out THCA LUSC SARC \
    --epochs 20 \
    --output_dir results/tcga_htt_full
```

### Train Baseline on TCGA
```bash
python train_tcga.py --baseline --output_dir results/tcga_baseline_full
```

### Quick smoke test (2 epochs, verifies everything runs)
```bash
python train.py --use_synthetic --epochs 2 --batch_size 2 --n_patients 20
```

---

## Project Structure

```
OncoLocus/
├── src/
│   ├── data/
│   │   ├── synthetic.py          # Synthetic radiology report generator
│   │   ├── preprocess_tcga.py    # TCGA grade label + cancer site extraction
│   │   ├── preprocess.py         # GENIE BPC preprocessor (future real data)
│   │   └── dataset.py            # PatientSequenceDataset + build_datasets()
│   ├── models/
│   │   ├── report_encoder.py     # Level 1: BiomedBERT + LoRA report encoder
│   │   ├── temporal_transformer.py  # Level 2: Continuous-Time Temporal Transformer
│   │   ├── htt_model.py          # Full HTT (Level 1 + Level 2)
│   │   └── baseline_model.py     # Non-temporal ablation baseline
│   ├── training/
│   │   └── trainer.py            # HTTTrainer: loss, optimizer, early stopping
│   └── evaluation/
│       └── metrics.py            # AUROC, F1, per-cancer-type breakdown
├── results/
│   ├── synthetic_htt/            # Experiment 1 HTT results
│   ├── synthetic_baseline/       # Experiment 1 Baseline results
│   ├── tcga_htt_full/            # Experiment 2 HTT results
│   └── tcga_baseline_full/       # Experiment 2 Baseline results
├── train.py                      # Entry point: synthetic / GENIE BPC training
├── train_tcga.py                 # Entry point: TCGA training
├── download_tcga.py              # Automated TCGA dataset downloader
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Model Details

| Component | Detail |
|---|---|
| Base encoder | BiomedBERT-base-uncased (Gu et al., 2021) |
| Fine-tuning | LoRA (r=16, alpha=32) — 11.1% params trainable |
| Temporal layers | 4 Transformer encoder layers, 8 heads |
| Hidden dimension | 512 |
| Time encoding | Sinusoidal over actual elapsed days (normalized to years) |
| Cancer type conditioning | Learnable embedding per cancer type |
| Optimizer | AdamW (lr=2×10⁻⁴, weight decay=0.01) |
| Training | 20 epochs max, early stopping patience=5, AMP fp16 |
| Hardware | NVIDIA RTX 5060 8 GB VRAM |

---

## Citation

If you use OncoLocus in your research, please cite:

```bibtex
@software{oncolocus2026,
  author    = {Anand, Rajneesh},
  title     = {OncoLocus: Hierarchical Temporal Transformer for Cross-Cancer Transfer Learning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/rajneeshanand/OncoLocus}
}
```

---

## Acknowledgements

- TCGA-Reports dataset: Kefeli & Tatonetti (2024), *Patterns*, Cell Press
- BiomedBERT: Gu et al. (2021), *ACM Transactions on Computing for Healthcare*
- Lehigh University, Department of Chemical and Biomolecular Engineering

---

*Manuscript under preparation. Results and discussion available upon request.*
