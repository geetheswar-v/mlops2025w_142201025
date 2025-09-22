from __future__ import annotations

import torch
from torch import nn
from torchvision import models


# Simple mapping helper; keeping explicit if/elif per requirement

def get_resnet(arch: str, num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    """Return a torchvision ResNet model for inference.

    Parameters
    ----------
    arch: one of 'resnet34','resnet50','resnet101','resnet152'
    num_classes: if provided and differs from default 1000, replace final fc layer
    pretrained: load torchvision pretrained weights
    """
    arch = arch.lower()
    if arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if num_classes is not None:
        in_features = model.fc.in_features
        if num_classes != model.fc.out_features:
            model.fc = nn.Linear(in_features, num_classes)
    return model


def prepare_model(arch: str, device: str = None, num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet(arch=arch, num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    model.eval()
    return model
