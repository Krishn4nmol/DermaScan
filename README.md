# Skin Lesion Detection — Lightweight Deep Learning

## Project Structure
```
skin_lesion/
├── data/
│   ├── raw/          ← Put downloaded dataset ZIPs here
│   └── processed/    ← Auto-generated after running prepare_data.py
├── models/
│   ├── backbones.py  ← MobileNetV3, EfficientNet-Lite, ShuffleNetV2 + CBAM
│   └── cbam.py       ← CBAM attention module
├── utils/
│   ├── dataset.py    ← Dataset class + augmentation
│   ├── preprocess.py ← Hair removal + color normalization
│   ├── losses.py     ← Focal loss
│   └── metrics.py    ← Balanced acc, macro F1, AUC
├── configs/
│   └── config.yaml   ← All hyperparameters in one place
├── prepare_data.py   ← Download + organize HAM10000
├── train.py          ← Main training script
├── evaluate.py       ← Evaluation + confusion matrix + Grad-CAM
├── compare_models.py ← Run all 3 backbones and compare
├── requirements.txt
└── README.md
```

## Setup (Windows, conda)

```bash
# 1. Create conda environment
conda create -n skin_lesion python=3.11 -y
conda activate skin_lesion

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download HAM10000 dataset (see prepare_data.py instructions)
python prepare_data.py

# 4. Train a single model
python train.py --model mobilenetv3

# 5. Train and compare all models
python compare_models.py

# 6. Evaluate and visualize
python evaluate.py --model mobilenetv3 --checkpoint checkpoints/mobilenetv3_best.pth
```

## Dataset Download
HAM10000 is available free from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Download and place the following in `data/raw/`:
- HAM10000_images_part_1.zip
- HAM10000_images_part_2.zip
- HAM10000_metadata.csv

Then run: `python prepare_data.py`

## Models
| Model           | Params | Notes                        |
|----------------|--------|------------------------------|
| MobileNetV3-L  | ~5.4M  | Fast, good accuracy          |
| EfficientNet-B0| ~5.3M  | Best accuracy / size tradeoff|
| ShuffleNetV2   | ~2.3M  | Fastest inference            |

All models use CBAM attention + focal loss + cosine LR schedule.

## Hardware
Tested on: AMD Ryzen 7, 32GB RAM, Radeon iGPU
Training on CPU is supported. For AMD GPU acceleration, ROCm is optional.
