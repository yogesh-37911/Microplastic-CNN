"""
Microplastic Vision – CNN model definitions.

Provides:
  - SimpleCNN: a lightweight baseline CNN
  - Transfer-learning wrappers for torchvision models (ResNet-18, MobileNet-V2, MobileNet-V3)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models


# ── Simple baseline CNN ─────────────────────────────────────────


class SimpleCNN(nn.Module):
    """
    Lightweight 4-conv-block CNN for binary classification.
    Designed to train quickly on small datasets and modest hardware.
    """

    def __init__(self, num_classes: int = 2, input_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Transfer-learning helpers ───────────────────────────────────


def create_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with replaced final FC layer."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_mobilenet_v2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """MobileNet-V2 with replaced classifier."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def create_mobilenet_v3_small(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """MobileNet-V3 Small with an ImageNet-pretrained feature extractor."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


# ── Factory ─────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, callable] = {
    "simple_cnn": lambda nc, **kw: SimpleCNN(num_classes=nc),
    "resnet18": lambda nc, **kw: create_resnet18(num_classes=nc, pretrained=kw.get("pretrained", True)),
    "mobilenet_v2": lambda nc, **kw: create_mobilenet_v2(num_classes=nc, pretrained=kw.get("pretrained", True)),
    "mobilenet_v3_small": lambda nc, **kw: create_mobilenet_v3_small(num_classes=nc, pretrained=kw.get("pretrained", True)),
}


def create_model(
    architecture: str,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Create a model by architecture name."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[architecture](num_classes, **kwargs)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a checkpoint and return the metadata dict."""
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return checkpoint
