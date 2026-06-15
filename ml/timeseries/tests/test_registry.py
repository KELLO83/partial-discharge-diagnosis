"""Smoke tests for model registry."""

from __future__ import annotations

import pytest
import torch

from ml.timeseries.src.models.registry import create_model


def create_model_or_skip(name: str):
    try:
        return create_model(name)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency is not installed: {exc}")


def test_gru_forward_shape() -> None:
    model = create_model("gru")
    x = torch.randn(2, 7680, 20)
    y = model(x)
    assert y.shape == (2, 5)


@pytest.mark.parametrize("name", ["tcn", "inception_time_small", "inception_time", "resnet1d", "timesnet", "timemixer"])
def test_channel_first_models_forward_shape(name: str) -> None:
    model = create_model_or_skip(name)
    x = torch.randn(2, 20, 7680)
    y = model(x)
    assert y.shape == (2, 5)


@pytest.mark.parametrize("name", ["moderntcn", "patchtst", "itransformer", "units"])
def test_time_first_official_models_forward_shape(name: str) -> None:
    model = create_model_or_skip(name)
    x = torch.randn(2, 7680, 20)
    y = model(x)
    assert y.shape == (2, 5)
