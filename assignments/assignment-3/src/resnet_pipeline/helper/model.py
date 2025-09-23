import torch
from torch import nn
from torchvision import models

def get_resnet(arch: str, num_classes: int | None = None, pretrained: bool = True, device: str | None = None) -> nn.Module:
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
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))