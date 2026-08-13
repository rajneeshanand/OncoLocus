
# Training loop for the HTT model.


import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from ..evaluation.metrics import compute_metrics, format_metrics


class HTTTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config=None,
        output_dir="results/",
        device=None,
    ):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        cfg = config or {}
        self.lr = cfg.get("lr", 2e-4)
        self.batch_size = cfg.get("batch_size", 8)
        self.epochs = cfg.get("epochs", 20)
        self.grad_accum = cfg.get("grad_accum_steps", 4)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.patience = cfg.get("patience", 5)
        self.use_amp = cfg.get("use_amp", True)
        self.visit_loss_weight = cfg.get("visit_loss_weight", 0.3)
        self.num_workers = cfg.get("num_workers", 0)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Training on: {self.device}")
        self.model = self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        total_steps = len(self.train_loader) * self.epochs // self.grad_accum
        warmup_steps = total_steps // 10
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.device.type == "cuda")

        self.best_auroc = 0.0
        self.best_epoch = 0
        self.history = {"train_loss": [], "val_auroc": [], "val_f1": []}

    def _to_device(self, batch):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_loss(self, batch):
        batch = self._to_device(batch)

        with autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
            patient_logits, visit_logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                days_elapsed=batch["days_elapsed"],
                seq_mask=batch["seq_mask"],
                cancer_type_ids=batch["cancer_type_id"],
            )

            # Primary: patient-level loss
            patient_loss = self.criterion(patient_logits, batch["patient_label"])

            # Auxiliary: visit-level loss (weighted)
            b, seq_len, n_cls = visit_logits.shape
            visit_loss = self.criterion(
                visit_logits.view(b * seq_len, n_cls),
                batch["visit_labels"].view(b * seq_len),
            )

            total_loss = patient_loss + self.visit_loss_weight * visit_loss

        return total_loss, patient_logits

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for step, batch in enumerate(pbar):
            loss, _ = self._compute_loss(batch)
            loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_logits, all_type_ids = [], [], []

        # Build reverse-lookup from whatever mapping is stored on the dataset.
        # Falls back to the synthetic-data names for backwards compatibility.
        try:
            rev = {v: k for k, v in loader.dataset.cancer_type_to_id.items()}
        except AttributeError:
            rev = {0: "NSCLC", 1: "BrCa", 2: "CRC", 3: "PANC", 4: "Prostate"}
        CANCER_NAMES = rev

        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = self._to_device(batch)
            with autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
                patient_logits, _ = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    days_elapsed=batch["days_elapsed"],
                    seq_mask=batch["seq_mask"],
                    cancer_type_ids=batch["cancer_type_id"],
                )
            all_labels.extend(batch["patient_label"].cpu().numpy())
            all_logits.extend(patient_logits.cpu().float().numpy())
            all_type_ids.extend(batch["cancer_type_id"].cpu().numpy())

        cancer_type_names = [CANCER_NAMES.get(int(i), "Unknown") for i in all_type_ids]
        return compute_metrics(all_labels, all_logits, cancer_types=cancer_type_names)

    def save_checkpoint(self, epoch, metrics, name="best"):
        path = os.path.join(self.output_dir, f"checkpoint_{name}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def train(self):
        print(f"\nStarting training for {self.epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            auroc = val_metrics.get("auroc", 0.0)
            f1 = val_metrics.get("f1", 0.0)

            self.history["train_loss"].append(train_loss)
            self.history["val_auroc"].append(auroc)
            self.history["val_f1"].append(f1)

            print(f"  Train loss: {train_loss:.4f}")
            print(format_metrics(val_metrics, prefix="Val "))

            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, name="best")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}. Best AUROC: {self.best_auroc:.4f} (epoch {self.best_epoch})")
                    break

        # Save training history
        history_path = os.path.join(self.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best Val AUROC: {self.best_auroc:.4f} at epoch {self.best_epoch}")
        return self.history
