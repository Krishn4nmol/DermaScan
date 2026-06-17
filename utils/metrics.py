"""
Evaluation metrics for multi-class skin lesion classification.
- Balanced accuracy
- Macro F1
- Per-class AUC + mean AUC
- Confusion matrix plot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

CLASS_NAMES = [
    "Nevi", "Melanoma", "BKL",
    "BCC", "AKIEC", "Vasc", "DF"
]


def compute_metrics(
    all_labels: List[int],
    all_preds:  List[int],
    all_probs:  np.ndarray,   # shape (N, C)
    num_classes: int = 7,
) -> dict:
    """
    Compute all evaluation metrics.

    Returns dict with: balanced_acc, macro_f1, mean_auc, per_class_auc
    """
    labels = np.array(all_labels)
    preds  = np.array(all_preds)

    balanced_acc = balanced_accuracy_score(labels, preds)
    macro_f1     = f1_score(labels, preds, average="macro", zero_division=0)

    # One-vs-rest AUC per class
    try:
        per_class_auc = roc_auc_score(
            labels, all_probs,
            multi_class="ovr", average=None,
            labels=list(range(num_classes))
        )
        mean_auc = float(np.mean(per_class_auc))
    except ValueError:
        per_class_auc = [float("nan")] * num_classes
        mean_auc      = float("nan")

    return {
        "balanced_acc":  balanced_acc,
        "macro_f1":      macro_f1,
        "mean_auc":      mean_auc,
        "per_class_auc": per_class_auc,
    }


def print_metrics(metrics: dict, epoch: int = None) -> None:
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(f"{prefix}Balanced Acc: {metrics['balanced_acc']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"Mean AUC: {metrics['mean_auc']:.4f}")


def plot_confusion_matrix(
    all_labels: List[int],
    all_preds:  List[int],
    save_path:  str = "results/confusion_matrix.png",
    class_names: List[str] = CLASS_NAMES,
) -> None:
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(
    train_losses:  List[float],
    val_losses:    List[float],
    val_bacc:      List[float],
    save_path:     str = "results/training_curves.png",
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, val_bacc, color="green", label="Val Balanced Acc")
    axes[1].set_title("Balanced Accuracy"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")
