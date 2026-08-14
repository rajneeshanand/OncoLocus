
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.data.preprocess_tcga import (
    preprocess_tcga, load_processed_tcga, CANCER_TYPE_IDS
)
from src.data.dataset import build_datasets
from src.models.htt_model import build_model
from src.models.baseline_model import BaselineModel
from src.training.trainer import HTTTrainer
from src.evaluation.metrics import compute_metrics, format_metrics


# Default cancer types to hold out for Gap 2 experiment
# (matches paper experiments: thyroid, lung squamous, sarcoma — all unseen at train time)
DEFAULT_GAP2_HELD_OUT = ["THCA", "LUSC", "SARC"]


def parse_args():
    p = argparse.ArgumentParser(description="Train HTT on TCGA pathology reports")

    # Data
    p.add_argument("--tcga_csv",     default="data/raw/TCGA/TCGA_Reports.csv")
    p.add_argument("--processed_dir",default="data/processed/tcga",
                   help="Load pre-processed CSVs if they exist (skip re-extraction)")
    p.add_argument("--reprocess",    action="store_true",
                   help="Force re-run preprocess_tcga even if processed CSVs exist")

    # Gap 2 held-out types (space-separated, e.g., THCA GBM SKCM)
    p.add_argument("--gap2_held_out", nargs="+", default=DEFAULT_GAP2_HELD_OUT,
                   metavar="TYPE",
                   help="Cancer types completely withheld for zero-shot transfer test")

    # Model
    p.add_argument("--encoder",
                   default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                   help="HuggingFace model name for Level-1 report encoder")
    p.add_argument("--hidden_dim",   type=int, default=512)
    p.add_argument("--n_layers",     type=int, default=4)
    p.add_argument("--n_heads",      type=int, default=8)
    p.add_argument("--lora_r",       type=int, default=16)
    p.add_argument("--no_lora",      action="store_true")
    p.add_argument("--max_report_len", type=int, default=256,
                   help="Max tokens per report (256 fits 8 GB with BioGPT-Large)")

    # Training
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--grad_accum",   type=int, default=4)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--no_amp",       action="store_true")

    # Ablation
    p.add_argument("--baseline", action="store_true",
                   help="Non-temporal baseline (for ablation vs HTT)")

    # Output
    p.add_argument("--output_dir", default="results/tcga_run01")
    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


def load_tcga_data(args):
    """Return all_data dict — pre-processed CSVs if available, else re-extract."""
    if not args.reprocess and os.path.isdir(args.processed_dir):
        csvs = [f for f in os.listdir(args.processed_dir) if f.endswith("_processed.csv")]
        if csvs:
            print(f"Loading {len(csvs)} pre-processed TCGA CSVs from {args.processed_dir}")
            return load_processed_tcga(args.processed_dir)

    print("Preprocessing TCGA reports (one-time extraction) ...")
    return preprocess_tcga(args.tcga_csv, args.processed_dir)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    #  Load data 
    all_data = load_tcga_data(args)

    types_seen = sorted(all_data.keys())
    types_held = [t for t in args.gap2_held_out if t in types_seen]
    types_train = [t for t in types_seen if t not in types_held]

    print(f"\nCancer types for training  : {types_train}")
    print(f"Cancer types held out (Gap 2): {types_held}")

    # Build training-only subset of all_data
    train_data = {t: all_data[t] for t in types_train}

    #  Build PyTorch datasets 
    # TCGA: max_seq_len=1 (one pathology report per patient visit)
    # The temporal transformer gracefully handles seq_len=1 (still learns
    # cross-cancer representations via cancer_type_emb).
    print("\nBuilding datasets ...")
    ds_kwargs = dict(
        tokenizer_name=args.encoder,
        max_report_len=args.max_report_len,
        max_seq_len=1,          # single-report; no temporal sequence for TCGA
        gap2_held_out_type=None,
        seed=args.seed,
        cancer_type_to_id=CANCER_TYPE_IDS,
    )

    train_ds, val_ds, test_ds, _ = build_datasets(all_data=train_data, **ds_kwargs)

    # Build separate held-out datasets (one per cancer type).
    # Use ALL data from each held-out type — zero-shot evaluation doesn't need
    # a separate train/val split; every sample is unseen, so all are test samples.
    from src.data.dataset import PatientSequenceDataset
    gap2_datasets = {}
    for t in types_held:
        gap2_datasets[t] = PatientSequenceDataset(
            all_data[t],
            tokenizer_name=args.encoder,
            max_report_len=args.max_report_len,
            max_seq_len=1,
            cancer_type_to_id=CANCER_TYPE_IDS,
        )

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    for t, ds in gap2_datasets.items():
        print(f"  Gap2 {t}: {len(ds):,} (ALL data, zero-shot)")

    #  Build model 
    n_cancer_types = len(CANCER_TYPE_IDS)
    if args.baseline:
        print("\nBuilding Baseline (non-temporal) model ...")
        model = BaselineModel(
            encoder_model=args.encoder,
            hidden_dim=args.hidden_dim,
            lora_r=args.lora_r,
            use_lora=not args.no_lora,
        ).to(device)
        model.count_parameters()
    else:
        print("\nBuilding HTT model ...")
        model_config = {
            "encoder_model":  args.encoder,
            "hidden_dim":     args.hidden_dim,
            "n_layers":       args.n_layers,
            "n_heads":        args.n_heads,
            "lora_r":         args.lora_r,
            "use_lora":       not args.no_lora,
            "max_seq_len":    1,
            "n_cancer_types": n_cancer_types,
        }
        model = build_model(model_config, device=device)

    #  Train 
    trainer_config = {
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "epochs":           args.epochs,
        "grad_accum_steps": args.grad_accum,
        "patience":         args.patience,
        "use_amp":          not args.no_amp,
    }
    trainer = HTTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=trainer_config,
        output_dir=args.output_dir,
        device=device,
    )
    trainer.train()

    #  Load best checkpoint 
    ckpt = torch.load(
        os.path.join(args.output_dir, "checkpoint_best.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    #  Test evaluation 
    print("\n--- In-distribution Test ---")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)
    print(format_metrics(test_metrics, prefix="Test "))
    _save_metrics(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    #  Gap 2: zero-shot transfer per held-out type 
    all_gap2 = {}
    for t, ds in gap2_datasets.items():
        print(f"\n--- Gap 2 Zero-shot: {t} ---")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        m = trainer.evaluate(loader)
        print(format_metrics(m, prefix=f"{t} "))
        all_gap2[t] = {k: (None if isinstance(v, float) and v != v else v)
                       for k, v in m.items() if k != "per_cancer_type"}

    with open(os.path.join(args.output_dir, "gap2_metrics.json"), "w") as f:
        json.dump(all_gap2, f, indent=2)

    print(f"\nAll results saved to: {args.output_dir}")

    #  Summary table 
    print("\n========== GAP 2 SUMMARY ==========")
    print(f"{'Type':8s}  {'AUROC':>7s}  {'F1':>7s}")
    print("-" * 28)
    for t, m in all_gap2.items():
        auroc = m.get("auroc", float("nan"))
        f1 = m.get("f1", float("nan"))
        auroc_s = f"{auroc:.4f}" if auroc is not None else "  nan  "
        f1_s = f"{f1:.4f}" if f1 is not None else "  nan  "
        print(f"{t:8s}  {auroc_s:>7s}  {f1_s:>7s}")
  


def _save_metrics(metrics, path):
    clean = {k: (None if isinstance(v, float) and v != v else v)
             for k, v in metrics.items() if k != "per_cancer_type"}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


if __name__ == "__main__":
    main()
