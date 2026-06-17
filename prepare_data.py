"""
prepare_data.py — Organize HAM10000 dataset for training.

Steps:
1. Unzip HAM10000_images_part_1.zip and HAM10000_images_part_2.zip
   into data/processed/images/
2. Copy HAM10000_metadata.csv into data/processed/
3. Print class distribution

Download from Kaggle first:
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Place in data/raw/:
  - HAM10000_images_part_1.zip
  - HAM10000_images_part_2.zip
  - HAM10000_metadata.csv
"""

import os
import shutil
import zipfile
from pathlib import Path
import pandas as pd

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMAGE_DIR     = PROCESSED_DIR / "images"

LABEL_MAP = {
    "nv":    "Melanocytic nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign keratosis",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc":  "Vascular lesions",
    "df":    "Dermatofibroma",
}


def unzip_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    zips = list(RAW_DIR.glob("HAM10000_images_part_*.zip"))
    if not zips:
        print("\n[!] No zip files found in data/raw/")
        print("    Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("    Place HAM10000_images_part_1.zip and _part_2.zip in data/raw/\n")
        return False

    for z in sorted(zips):
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(IMAGE_DIR)
        print(f"  Done.")

    # Some Kaggle versions nest images inside a subfolder — flatten them
    for sub in IMAGE_DIR.iterdir():
        if sub.is_dir():
            for img in sub.glob("*.jpg"):
                dest = IMAGE_DIR / img.name
                if not dest.exists():
                    shutil.move(str(img), str(dest))
            # Remove empty subdir
            try:
                sub.rmdir()
            except OSError:
                pass

    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"\n{n_images} images extracted to {IMAGE_DIR}")
    return True


def copy_metadata():
    src = RAW_DIR / "HAM10000_metadata.csv"
    if not src.exists():
        # Try alternative filename from Kaggle
        alt = RAW_DIR / "hmnist_28_28_RGB.csv"
        src_candidates = list(RAW_DIR.glob("*.csv"))
        if src_candidates:
            src = src_candidates[0]
            print(f"Using metadata file: {src.name}")
        else:
            print("[!] No CSV metadata found in data/raw/")
            return False

    dest = PROCESSED_DIR / "HAM10000_metadata.csv"
    shutil.copy(src, dest)
    print(f"Metadata copied → {dest}")
    return True


def print_distribution():
    csv = PROCESSED_DIR / "HAM10000_metadata.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    print("\n── Class Distribution ─────────────────────────")
    counts = df["dx"].value_counts()
    for code, count in counts.items():
        name = LABEL_MAP.get(code, code)
        bar  = "█" * (count // 100)
        print(f"  {code:6s}  {name:25s}  {count:5d}  {bar}")
    print(f"  TOTAL: {len(df)}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  HAM10000 Data Preparation")
    print("=" * 50)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ok1 = unzip_images()
    ok2 = copy_metadata()

    if ok1 and ok2:
        print_distribution()
        print("[✓] Data ready. Run: python train.py --model mobilenetv3")
    else:
        print("\n[!] Please place the dataset files in data/raw/ and re-run.")
