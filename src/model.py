"""Transfer öğrenme ile sınıflandırma modeli."""

import torch.nn as nn
from torchvision import models

from .config import CLASSES

SUPPORTED_ARCHS = ("efficientnet_b0", "resnet50", "densenet121")


def build_model(arch: str = "efficientnet_b0", num_classes: int = len(CLASSES), pretrained: bool = True):
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Desteklenmeyen mimari: {arch}. Seçenekler: {SUPPORTED_ARCHS}")
    return model


def gradcam_target_layer(model, arch: str):
    """Grad-CAM için son evrişim bloğu."""
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features
    raise ValueError(arch)
