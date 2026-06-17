"""
compare_models.py — Train all 3 lightweight models and produce a comparison table.

Usage:
  python compare_models.py
  python compare_models.py --epochs 20 --workers 0
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["mobilenetv3", "efficientnet", "shufflenetv2"]

MODEL_PARAMS = {
    "mobilenetv3": "5.4M",
    "efficientnet": "5.3M",
    "shufflenetv2": "2.3M",
}


def run_training(model_name: str, epochs: int, batch_size: int, workers: int):
    """Run train.py as a subprocess for a given model."""
    cmd = [
        sys.executable, "train.py",
        "--model",      model_name,
        "--epochs",     str(epochs),
        "--batch_size", str(batch_size),
        "--workers",    str(workers),
    ]
    print(f"\n{'─'*55}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'─'*55}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    return elapsed, result.returncode


def load_best_metrics(model_name: str) -> dict:
    """Load metrics from saved checkpoint."""
    ckpt_path = Path("checkpoints") / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("metrics", {})


def count_params(model_name: str) -> float:
    """Count trainable parameters in millions."""
    from models.backbones import build_model
    model = build_model(model_name, pretrained=False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def benchmark_inference(model_name: str) -> float:
    """Measure average inference time (ms) over 100 forward passes on CPU."""
    from models.backbones import build_model
    model = build_model(model_name, pretrained=False).eval()
    dummy = torch.randn(1, 3, 224, 224)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    import time
    times = []
    with torch.no_grad():
        for _ in range(50):
            t = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t) * 1000)
    return sum(times) / len(times)


def make_comparison_table(results: dict, save_path: str = "results/model_comparison.csv"):
    rows = []
    for model_name, info in results.items():
        m = info.get("metrics", {})
        rows.append({
            "Model":          model_name,
            "Params (M)":     f"{info.get('params_m', '?'):.2f}",
            "Inference (ms)": f"{info.get('inference_ms', '?'):.1f}",
            "Balanced Acc":   f"{m.get('balanced_acc', 0):.4f}",
            "Macro F1":       f"{m.get('macro_f1', 0):.4f}",
            "Mean AUC":       f"{m.get('mean_auc', 0):.4f}",
            "Train Time (s)": f"{info.get('train_time_s', 0):.0f}",
        })

    df = pd.DataFrame(rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print("\n── Model Comparison ──────────────────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nSaved → {save_path}")
    return df


def plot_comparison(df: pd.DataFrame,
                    save_path: str = "results/model_comparison.png"):
    metrics = ["Balanced Acc", "Macro F1", "Mean AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, metric in zip(axes, metrics):
        values = [float(df[df["Model"] == m][metric].values[0])
                  for m in MODELS if m in df["Model"].values]
        bars = ax.bar(MODELS, values, color=colors[:len(values)], width=0.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(max(0, min(values) - 0.05), min(1, max(values) + 0.05))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9)

    plt.suptitle("Lightweight Model Comparison — Skin Lesion Detection",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=0)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, just load existing checkpoints")
    args = parser.parse_args()

    results = {}

    # ── Pre-compute params + inference speed ──────────────────────────────────
    print("Benchmarking model sizes and inference speed ...")
    for model_name in MODELS:
        params_m     = count_params(model_name)
        inference_ms = benchmark_inference(model_name)
        results[model_name] = {
            "params_m":     params_m,
            "inference_ms": inference_ms,
        }
        print(f"  {model_name:15s}  {params_m:.2f}M params  "
              f"{inference_ms:.1f}ms inference")

    # ── Train each model ──────────────────────────────────────────────────────
    if not args.skip_train:
        for model_name in MODELS:
            elapsed, rc = run_training(
                model_name, args.epochs, args.batch_size, args.workers
            )
            results[model_name]["train_time_s"] = elapsed
            if rc != 0:
                print(f"[!] Training {model_name} failed (return code {rc})")

    # ── Load best metrics from checkpoints ────────────────────────────────────
    for model_name in MODELS:
        metrics = load_best_metrics(model_name)
        results[model_name]["metrics"] = metrics

    # ── Table + plot ──────────────────────────────────────────────────────────
    df = make_comparison_table(results)
    plot_comparison(df)

    print("\n[✓] All done. Check the results/ folder.")


if __name__ == "__main__":
    main()
