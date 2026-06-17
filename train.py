"""
train.py — Main training script for skin lesion detection.

Usage:
  python train.py --model mobilenetv3
  python train.py --model efficientnet --epochs 30 --batch_size 16
  python train.py --model shufflenetv2 --no_preprocess --workers 0
"""

import argparse
import os
import sys
import time
import yaml
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
from models.backbones import build_model
from utils.dataset    import build_dataloaders, CLASS_LABELS
from utils.losses     import FocalLoss
from utils.metrics    import compute_metrics, print_metrics, \
                             plot_confusion_matrix, plot_training_curves


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Device selection (CPU / CUDA / ROCm)
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU — training will be slower but fully functional.")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:   # AMP (GPU only)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:                    # CPU / no AMP
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 7):
    model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_preds   = []
    all_probs   = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_probs_np = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(all_labels, all_preds, all_probs_np, num_classes)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics, all_labels, all_preds, all_probs_np


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skin Lesion Training")
    parser.add_argument("--model",        type=str, default="mobilenetv3",
                        choices=["mobilenetv3", "efficientnet", "shufflenetv2"])
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--workers",      type=int,   default=0,
                        help="DataLoader workers (0 = main process, safe on Windows)")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip hair removal + color normalization (faster)")
    parser.add_argument("--no_cbam",      action="store_true",
                        help="Disable CBAM attention (ablation)")
    parser.add_argument("--config",       type=str,   default="configs/config.yaml")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    epochs     = args.epochs     or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr         = args.lr         or cfg["optimizer"]["lr"]
    num_workers= args.workers
    num_classes= cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size"]
    ckpt_dir   = Path(cfg["checkpoints"]["dir"])
    results_dir= Path(cfg["results"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Skin Lesion Detection  |  Model: {args.model.upper()}")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"{'='*55}\n")

    set_seed(args.seed)
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    metadata_csv = "data/processed/HAM10000_metadata.csv"
    image_dir    = "data/processed/images"

    if not Path(metadata_csv).exists():
        print("[!] Dataset not found. Run: python prepare_data.py")
        sys.exit(1)

    print("Loading data ...")
    loaders, class_weights = build_dataloaders(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=args.seed,
        preprocess=not args.no_preprocess,
    )
    print(f"Train batches: {len(loaders['train'])}  "
          f"Val batches: {len(loaders['val'])}  "
          f"Test batches: {len(loaders['test'])}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        name=args.model,
        num_classes=num_classes,
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        use_cbam=not args.no_cbam,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params/1e6:.2f}M\n")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = FocalLoss(
        gamma=cfg["loss"]["gamma"],
        alpha=class_weights.to(device),
    )
    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler"]["T_0"],
        T_mult=cfg["scheduler"]["T_mult"],
    )

    # AMP scaler only for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_bacc      = 0.0
    patience_count = 0
    patience       = cfg["training"]["early_stopping_patience"]
    train_losses   = []
    val_losses_log = []
    val_bacc_log   = []
    best_ckpt      = ckpt_dir / f"{args.model}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scaler
        )
        val_loss, metrics, _, _, _ = evaluate(
            model, loaders["val"], criterion, device, num_classes
        )
        scheduler.step(epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  ", end="")
        print_metrics(metrics)
        print(f"  ({elapsed:.0f}s)")

        train_losses.append(train_loss)
        val_losses_log.append(val_loss)
        val_bacc_log.append(metrics["balanced_acc"])

        # Save best
        if metrics["balanced_acc"] > best_bacc:
            best_bacc = metrics["balanced_acc"]
            torch.save({
                "epoch":       epoch,
                "model_name":  args.model,
                "state_dict":  model.state_dict(),
                "metrics":     metrics,
                "config":      cfg,
            }, best_ckpt)
            print(f"  ✓ Best model saved (balanced_acc={best_bacc:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Test Evaluation ──────────────────────────────")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    _, test_metrics, test_labels, test_preds, _ = evaluate(
        model, loaders["test"], criterion, device, num_classes
    )
    print_metrics(test_metrics)

    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=str(results_dir / f"{args.model}_confusion_matrix.png")
    )
    plot_training_curves(
        train_losses, val_losses_log, val_bacc_log,
        save_path=str(results_dir / f"{args.model}_training_curves.png")
    )

    print(f"\nDone. Best val balanced acc: {best_bacc:.4f}")
    print(f"Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
