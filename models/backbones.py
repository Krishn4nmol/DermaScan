"""
Lightweight CNN backbones with CBAM attention for skin lesion classification.

Supported models:
  - mobilenetv3   (MobileNetV3-Large,  ~5.4M params)
  - efficientnet  (EfficientNet-B0,    ~5.3M params)
  - shufflenetv2  (ShuffleNetV2-1.0x,  ~2.3M params)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.cbam import CBAM

# ─────────────────────────────────────────────────────────────────────────────
# Shared classification head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV3-Large + CBAM
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV3CBAM(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 dropout: float = 0.4, use_cbam: bool = True):
        super().__init__()
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.mobilenet_v3_large(weights=weights)

        # Feature extractor: everything before the adaptive pool
        self.features = base.features          # outputs (B, 960, 7, 7) for 224x224 input
        self.cbam     = CBAM(960) if use_cbam else nn.Identity()
        self.head     = ClassificationHead(960, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0 + CBAM
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 dropout: float = 0.4, use_cbam: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.efficientnet_b0(weights=weights)

        self.features = base.features          # outputs (B, 1280, 7, 7) for 224x224 input
        self.cbam     = CBAM(1280) if use_cbam else nn.Identity()
        self.head     = ClassificationHead(1280, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# ShuffleNetV2-1.0x + CBAM
# ─────────────────────────────────────────────────────────────────────────────

class ShuffleNetV2CBAM(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 dropout: float = 0.4, use_cbam: bool = True):
        super().__init__()
        weights = tv_models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.shufflenet_v2_x1_0(weights=weights)

        # ShuffleNetV2 stages
        self.conv1   = base.conv1
        self.maxpool = base.maxpool
        self.stage2  = base.stage2
        self.stage3  = base.stage3
        self.stage4  = base.stage4
        self.conv5   = base.conv5            # outputs (B, 1024, 7, 7) for 224x224 input

        self.cbam    = CBAM(1024) if use_cbam else nn.Identity()
        self.head    = ClassificationHead(1024, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.cbam(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mobilenetv3": MobileNetV3CBAM,
    "efficientnet": EfficientNetCBAM,
    "shufflenetv2": ShuffleNetV2CBAM,
}


def build_model(name: str, num_classes: int = 7, pretrained: bool = True,
                dropout: float = 0.4, use_cbam: bool = True) -> nn.Module:
    """
    Args:
        name: one of 'mobilenetv3', 'efficientnet', 'shufflenetv2'
    Returns:
        Instantiated model
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        use_cbam=use_cbam,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        m = build_model(name, pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        out = m(dummy)
        print(f"{name:15s}  params={count_parameters(m)/1e6:.2f}M  "
              f"output={tuple(out.shape)}")
