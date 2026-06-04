"""Smoke tests for model registry."""

from __future__ import annotations

import torch

from ml.src.models.registry import create_model


def test_gru_forward_shape() -> None:
    model = create_model("gru")
    x = torch.randn(2, 7680, 20)
    y = model(x)
    assert y.shape == (2, 5)


def test_channel_first_models_forward_shape() -> None:
    for name in ["tcn", "inception_time", "resnet1d", "timesnet", "timemixer"]:
        model = create_model(name)
        x = torch.randn(2, 20, 7680)
        y = model(x)
        assert y.shape == (2, 5)


def test_time_first_official_models_forward_shape() -> None:
    for name in ["moderntcn", "patchtst", "itransformer", "units"]:
        model = create_model(name)
        x = torch.randn(2, 7680, 20)
        y = model(x)
        assert y.shape == (2, 5)
