"""Model factory. Small enough to run on a phone, strong enough to defend at viva."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(arch: str = "mobilenet_v3_large", n_classes: int = 38, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    """Return a classifier with a fresh head.

    mobilenet_v3_large : 5.5M params, ~220 MFLOPs @224 — the deployment choice
    efficientnet_b0   : 5.3M params, stronger, slightly slower on CPU
    """
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, n_classes)
        head = m.classifier
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, n_classes)
        head = m.classifier
    else:
        raise ValueError(f"unknown arch {arch!r}")

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def target_layer_for_gradcam(model: nn.Module, arch: str = "mobilenet_v3_large"):
    """Last spatial conv block — what Grad-CAM should hook."""
    if arch == "mobilenet_v3_large":
        return model.features[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(arch)
