"""PD-specific vision encoder models."""

from __future__ import annotations

from torch import nn

from ml.vision.src.models.efficientnet import EfficientNetB0Classifier
from ml.vision.src.models.small_cnn import SmallPrpdCnn


def create_vision_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if model_name == "small_prpd_cnn":
        return SmallPrpdCnn(num_classes=num_classes)
    if model_name == "efficientnet_b0":
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    supported = "small_prpd_cnn, efficientnet_b0"
    raise ValueError(f"Unsupported vision model: {model_name}. Supported: {supported}")


__all__ = ["EfficientNetB0Classifier", "SmallPrpdCnn", "create_vision_model"]
