"""
prepare_data.py — Organize HAM10000 dataset for training.

Handles both:
- Already extracted folders (HAM10000_images_part_1, HAM10000_images_part_2)
- Zip files (HAM10000_images_part_1.zip, HAM10000_images_part_2.zip)
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


def collect_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    part_folders = [
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2",
    ]
    for folder in part_folders:
        if folder.exists() and folder.is_dir():
            images = list(folder.glob("*.jpg"))
            print(f"Found {len(images)} images in {folder.name}, copying ...")
            for img in images:
                dest = IMAGE_DIR / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
            print(f"  Done.")

    zips = list(RAW_DIR.glob("HAM10000_images_part_*.zip"))
    for z in sorted(zips):
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(IMAGE_DIR)
        print(f"  Done.")

    for sub in IMAGE_DIR.iterdir():
        if sub.is_dir():
            for img in sub.glob("*.jpg"):
                dest = IMAGE_DIR / img.name
                if not dest.exists():
                    shutil.move(str(img), str(dest))
            try:
                sub.rmdir()
            except OSError:
                pass

    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"\n{n_images} images ready in {IMAGE_DIR}")

    if n_images == 0:
        print("[!] No images found.")
        return False
    return True


def copy_metadata():
    candidates = list(RAW_DIR.glob("*metadata*.csv")) + list(RAW_DIR.glob("HAM*.csv"))
    src = next((c for c in candidates if c.exists()), None)

    if src is None:
        print("[!] No metadata CSV found in data/raw/")
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

    ok1 = collect_images()
    ok2 = copy_metadata()

    if ok1 and ok2:
        print_distribution()
        print("[✓] Data ready. Run: python train.py --model mobilenetv3 --workers 0")
    else:
        print("\n[!] Something went wrong. Check messages above.")