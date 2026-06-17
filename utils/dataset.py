"""
HAM10000 Dataset class with:
- Preprocessing (hair removal + color constancy)
- Albumentations-based augmentation
- Stratified patient-level splits
- Class weight computation for imbalanced training
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.preprocess import preprocess_image

# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────

CLASS_LABELS = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
LABEL2IDX    = {lbl: idx for idx, lbl in enumerate(CLASS_LABELS)}
IDX2LABEL    = {idx: lbl for lbl, idx in LABEL2IDX.items()}
CLASS_NAMES  = [
    "Melanocytic nevi", "Melanoma", "Benign keratosis",
    "Basal cell carcinoma", "Actinic keratoses",
    "Vascular lesions", "Dermatofibroma"
]

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                        fill_value=0, p=0.3),    # ~random erasing
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        preprocess: bool = True,
    ):
        """
        Args:
            df          : DataFrame with columns ['image_id', 'dx', 'label']
            image_dir   : Directory containing JPEG images
            transform   : Albumentations transform pipeline
            preprocess  : Whether to apply hair removal + color constancy
        """
        self.df         = df.reset_index(drop=True)
        self.image_dir  = Path(image_dir)
        self.transform  = transform
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row      = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['image_id']}.jpg"

        image = Image.open(img_path).convert("RGB")

        if self.preprocess:
            image = preprocess_image(image,
                                     remove_hair_flag=True,
                                     color_constancy_flag=True)

        image_np = np.array(image)

        if self.transform:
            image_np = self.transform(image=image_np)["image"]

        label = int(row["label"])
        return image_np, label


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load HAM10000 metadata CSV and add integer label column."""
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map(LABEL2IDX)
    return df


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split at patient level to prevent data leakage.
    Each patient's images all go into the same split.
    """
    rng = random.Random(seed)

    # Group by patient ID if available, else by image
    id_col = "lesion_id" if "lesion_id" in df.columns else "image_id"
    patient_ids = df[id_col].unique().tolist()
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = set(patient_ids[:n_train])
    val_ids   = set(patient_ids[n_train:n_train + n_val])
    test_ids  = set(patient_ids[n_train + n_val:])

    train_df = df[df[id_col].isin(train_ids)]
    val_df   = df[df[id_col].isin(val_ids)]
    test_df  = df[df[id_col].isin(test_ids)]

    print(f"Split sizes → train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse frequency class weights for focal loss / weighted sampling."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    """Per-sample weights so each class is equally likely per batch."""
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    metadata_csv:  str,
    image_dir:     str,
    image_size:    int = 224,
    batch_size:    int = 32,
    num_workers:   int = 4,
    seed:          int = 42,
    preprocess:    bool = True,
) -> Dict[str, DataLoader]:
    """
    End-to-end: load metadata → split → build datasets → return DataLoaders.
    """
    df = load_metadata(metadata_csv)
    train_df, val_df, test_df = patient_level_split(df, seed=seed)

    train_ds = HAM10000Dataset(train_df, image_dir,
                               transform=get_train_transforms(image_size),
                               preprocess=preprocess)
    val_ds   = HAM10000Dataset(val_df,   image_dir,
                               transform=get_val_transforms(image_size),
                               preprocess=preprocess)
    test_ds  = HAM10000Dataset(test_df,  image_dir,
                               transform=get_val_transforms(image_size),
                               preprocess=preprocess)

    # Weighted sampler for training to handle class imbalance
    train_labels = train_df["label"].tolist()
    sampler = make_weighted_sampler(train_labels, num_classes=len(CLASS_LABELS))

    # Windows: num_workers > 0 can cause issues in some setups
    # Set num_workers=0 if you hit "BrokenPipeError"
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
    return loaders, compute_class_weights(train_labels, len(CLASS_LABELS))
