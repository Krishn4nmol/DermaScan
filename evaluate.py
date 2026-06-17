"""
evaluate.py — Evaluation + Grad-CAM visualization.

Usage:
  python evaluate.py --model mobilenetv3 --checkpoint checkpoints/mobilenetv3_best.pth
  python evaluate.py --model mobilenetv3 --checkpoint checkpoints/mobilenetv3_best.pth --gradcam
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

import yaml
from sklearn.metrics import classification_report

from models.backbones import build_model
from utils.dataset    import build_dataloaders, CLASS_NAMES, get_val_transforms
from utils.losses     import FocalLoss
from utils.metrics    import (compute_metrics, print_metrics,
                               plot_confusion_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Simple Grad-CAM (no external library needed)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer: torch.nn.Module):
        self.model   = model
        self.grads   = None
        self.acts    = None
        self._hooks  = []
        self._hooks.append(target_layer.register_forward_hook(self._save_act))
        self._hooks.append(target_layer.register_backward_hook(self._save_grad))

    def _save_act(self, module, inp, out):
        self.acts = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def get_target_layer(model, model_name: str):
    """Return the last conv layer of each backbone for Grad-CAM."""
    if model_name == "mobilenetv3":
        return model.features[-1]
    elif model_name == "efficientnet":
        return model.features[-1]
    elif model_name == "shufflenetv2":
        return model.conv5
    raise ValueError(f"Unknown model: {model_name}")


def overlay_gradcam(image_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        choices=["mobilenetv3", "efficientnet", "shufflenetv2"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--gradcam",    action="store_true",
                        help="Save Grad-CAM overlays for a few test samples")
    parser.add_argument("--workers",    type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg["results"]["dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(args.model,
                        num_classes=cfg["data"]["num_classes"],
                        pretrained=False,
                        dropout=cfg["model"]["dropout"],
                        use_cbam=cfg["model"]["use_cbam"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Data
    loaders, class_weights = build_dataloaders(
        metadata_csv="data/processed/HAM10000_metadata.csv",
        image_dir="data/processed/images",
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=args.workers,
    )
    criterion = FocalLoss(gamma=cfg["loss"]["gamma"],
                          alpha=class_weights.to(device))

    # Evaluate on test set
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs_np = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(all_labels, all_preds, all_probs_np,
                              cfg["data"]["num_classes"])
    print("\n── Test Results ─────────────────────────────────")
    print_metrics(metrics)
    print("\nPer-class AUC:")
    class_labels_short = ["nv","mel","bkl","bcc","akiec","vasc","df"]
    for label, auc in zip(class_labels_short, metrics["per_class_auc"]):
        print(f"  {label:6s}: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=class_labels_short,
                                 zero_division=0))

    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=str(results_dir / f"{args.model}_test_confusion_matrix.png")
    )

    # Grad-CAM visualization
    if args.gradcam:
        print("\nGenerating Grad-CAM overlays ...")
        target_layer = get_target_layer(model, args.model)
        grad_cam     = GradCAM(model, target_layer)

        transform  = get_val_transforms(cfg["data"]["image_size"])
        image_dir  = Path("data/processed/images")
        import pandas as pd
        df = pd.read_csv("data/processed/HAM10000_metadata.csv")
        samples = df.groupby("dx").first().reset_index()   # one per class

        fig, axes = plt.subplots(len(samples), 3, figsize=(10, 4 * len(samples)))
        for row_i, (_, row) in enumerate(samples.iterrows()):
            img_path = image_dir / f"{row['image_id']}.jpg"
            if not img_path.exists():
                continue
            pil_img  = Image.open(img_path).convert("RGB")
            img_np   = np.array(pil_img)
            tensor   = transform(image=img_np)["image"].unsqueeze(0).to(device)

            cam      = grad_cam(tensor)
            overlay  = overlay_gradcam(cv2.resize(img_np, (224, 224)), cam)

            axes[row_i, 0].imshow(img_np);           axes[row_i, 0].set_title(f"Original\n({row['dx']})")
            axes[row_i, 1].imshow(cam, cmap="jet");  axes[row_i, 1].set_title("CAM heatmap")
            axes[row_i, 2].imshow(overlay);          axes[row_i, 2].set_title("Overlay")
            for ax in axes[row_i]: ax.axis("off")

        plt.tight_layout()
        out_path = results_dir / f"{args.model}_gradcam.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Grad-CAM saved → {out_path}")
        grad_cam.remove_hooks()


if __name__ == "__main__":
    main()
