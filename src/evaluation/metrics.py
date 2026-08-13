
# Evaluation metrics for HTT: AUROC (primary), F1, Precision, Recall, Accuracy,
# Specificity, and per-cancer-type breakdown for zero-shot transfer evaluation.


import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def compute_metrics(labels, logits, cancer_types=None, threshold=0.5):

    labels = np.array(labels)
    logits = np.array(logits)

    probs = softmax(logits)[:, 1]  # P(progressing)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auroc":     safe_auroc(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "accuracy":  accuracy_score(labels, preds),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    # Per-cancer-type breakdown
    if cancer_types is not None:
        cancer_types = np.array(cancer_types)
        per_type = {}
        for ct in np.unique(cancer_types):
            mask = cancer_types == ct
            if mask.sum() < 2:
                continue
            per_type[ct] = {
                "auroc": safe_auroc(labels[mask], probs[mask]),
                "f1":    f1_score(labels[mask], preds[mask], zero_division=0),
                "n":     int(mask.sum()),
            }
        metrics["per_cancer_type"] = per_type

    return metrics


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def safe_auroc(labels, probs):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, probs)


def format_metrics(metrics, prefix=""):

    lines = []
    main_keys = ["auroc", "f1", "precision", "recall", "accuracy", "specificity"]
    for k in main_keys:
        if k in metrics:
            lines.append(f"  {prefix}{k:15s}: {metrics[k]:.4f}")

    if "per_cancer_type" in metrics:
        lines.append(f"\n  {prefix}Per cancer type:")
        for ct, vals in metrics["per_cancer_type"].items():
            lines.append(
                f"    {ct:12s}  AUROC={vals['auroc']:.3f}  F1={vals['f1']:.3f}  n={vals['n']}"
            )

    return "\n".join(lines)
