# Train with synthetic data (while waiting for GENIE BPC access)
# Train with real GENIE BPC data (after download): python train.py --data_dir data/raw

import argparse
import os
import torch
import json

from src.data.synthetic import generate_synthetic_dataset
from src.data.dataset import build_datasets
from src.models.htt_model import build_model
from src.models.baseline_model import BaselineModel
from src.training.trainer import HTTTrainer
from src.evaluation.metrics import compute_metrics, format_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train Hierarchical Temporal Transformer")

    # Data
    p.add_argument("--use_synthetic", action="store_true",
                   help="Use synthetic data (default when real data not available)")
    p.add_argument("--data_dir", type=str, default="data/raw",
                   help="Path to real GENIE BPC data directory")
    p.add_argument("--n_patients", type=int, default=200,
                   help="Patients per cancer type (synthetic mode)")
    p.add_argument("--gap2_held_out", type=str, default="PANC",
                   help="Cancer type to hold out for zero-shot transfer test")

    # Model
    p.add_argument("--encoder_model", type=str, default="microsoft/BioGPT-Large")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=10)
    p.add_argument("--max_report_len", type=int, default=256)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no_amp", action="store_true")

    # Ablation
    p.add_argument("--baseline", action="store_true",
                   help="Train non-temporal baseline model (for ablation comparison)")

    # Output
    p.add_argument("--output_dir", type=str, default="results/run01")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def load_real_data(data_dir):
    import pandas as pd

    COHORTS = ["NSCLC", "BrCa", "CRC", "PANC", "Prostate", "BLADDER"]
    all_data = {}

    for cohort in COHORTS:
        imaging_path = os.path.join(data_dir, cohort, "imaging_level_dataset.csv")
        patient_path = os.path.join(data_dir, cohort, "patient_level_dataset.csv")
        cancer_path = os.path.join(data_dir, cohort, "cancer_level_dataset_index.csv")

        if not os.path.exists(imaging_path):
            print(f"  [skip] {cohort}: imaging_level_dataset.csv not found")
            continue

        df_img = pd.read_csv(imaging_path)
        print(f"  {cohort}: {len(df_img)} imaging records, columns: {list(df_img.columns)[:8]}")

        # Column mapping (will be updated once we see real column names)
        # Expected: patient_id, report_text (impression), scan_date, label
        all_data[cohort] = df_img

    if not all_data:
        raise FileNotFoundError(
            f"No imaging CSV files found in {data_dir}. "
            "Either run download_data.py first, or use --use_synthetic"
        )
    return all_data


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Load data ---
    if args.use_synthetic or not os.path.exists(args.data_dir):
        print("\nGenerating synthetic data...")
        all_data = generate_synthetic_dataset(
            n_patients_per_type=args.n_patients,
            seed=args.seed,
        )
    else:
        print(f"\nLoading real data from {args.data_dir}...")
        all_data = load_real_data(args.data_dir)

    #  Build datasets 
    print("\nBuilding datasets...")
    train_ds, val_ds, test_ds, gap2_ds = build_datasets(
        all_data=all_data,
        tokenizer_name=args.encoder_model,
        max_report_len=args.max_report_len,
        max_seq_len=args.max_seq_len,
        gap2_held_out_type=args.gap2_held_out,
        seed=args.seed,
    )

    #  Build model 
    if args.baseline:
        print("\nBuilding Baseline (non-temporal) model...")
        model = BaselineModel(
            encoder_model=args.encoder_model,
            hidden_dim=args.hidden_dim,
            lora_r=args.lora_r,
            use_lora=not args.no_lora,
        ).to(device)
        model.count_parameters()
    else:
        print("\nBuilding HTT model...")
        model_config = {
            "encoder_model":  args.encoder_model,
            "hidden_dim":     args.hidden_dim,
            "n_layers":       args.n_layers,
            "n_heads":        args.n_heads,
            "lora_r":         args.lora_r,
            "use_lora":       not args.no_lora,
            "max_seq_len":    args.max_seq_len,
        }
        model = build_model(model_config, device=device)

    #  Train 
    trainer_config = {
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "epochs":          args.epochs,
        "grad_accum_steps": args.grad_accum,
        "patience":        args.patience,
        "use_amp":         not args.no_amp,
    }
    trainer = HTTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=trainer_config,
        output_dir=args.output_dir,
        device=device,
    )
    history = trainer.train()

    #  Test evaluation 
    print("\n--- Test Set Evaluation ---")
    checkpoint = torch.load(
        os.path.join(args.output_dir, "checkpoint_best.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)
    print(format_metrics(test_metrics, prefix="Test "))

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        clean = {k: (None if isinstance(v, float) and v != v else v)
                 for k, v in test_metrics.items() if k != "per_cancer_type"}
        json.dump(clean, f, indent=2)

    #  Gap 2: Zero-shot transfer test ---
    if gap2_ds is not None:
        print(f"\n--- Gap 2: Zero-shot Transfer ({args.gap2_held_out}) ---")
        gap2_loader = DataLoader(gap2_ds, batch_size=args.batch_size, shuffle=False)
        gap2_metrics = trainer.evaluate(gap2_loader)
        print(format_metrics(gap2_metrics, prefix=f"{args.gap2_held_out} "))

        with open(os.path.join(args.output_dir, "gap2_metrics.json"), "w") as f:
            clean = {k: (None if isinstance(v, float) and v != v else v)
                     for k, v in gap2_metrics.items() if k != "per_cancer_type"}
            json.dump(clean, f, indent=2)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
