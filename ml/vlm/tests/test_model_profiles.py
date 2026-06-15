from __future__ import annotations

import pytest

from ml.vlm.train import DEFAULT_MODEL_PROFILE, parse_args
from ml.vlm.src.model_profiles import profile_keys, resolve_training_profile


def test_current_default_profile_uses_smolvlm2() -> None:
    profile = resolve_training_profile(DEFAULT_MODEL_PROFILE)

    assert profile.key == "smolvlm2_2b_qlora"
    assert profile.model_id == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    assert profile.load_in_4bit is True
    assert profile.train_vision_tower is False
    assert profile.train_projector is False
    assert profile.batch_size == 1
    assert profile.gradient_accumulation_steps >= 8
    assert profile.image_max_pixels == "512x512"


def test_train_cli_uses_current_default_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train.py"])

    args = parse_args()

    assert args.model_profile == DEFAULT_MODEL_PROFILE


def test_qwen3_profile_is_available_for_future_comparison() -> None:
    profile = resolve_training_profile("qwen3_vl_2b_qlora")

    assert profile.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert profile.training_strategy == "4bit_qlora_sft_text_projector_only"
    assert profile.min_vram_gb <= 8


def test_unknown_profile_reports_supported_profiles() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_training_profile("unknown")

    message = str(exc_info.value)
    assert "qwen3_vl_2b_qlora" in message
    assert "qwen2_5_vl_3b_qlora" in message
    assert "smolvlm2_2b_qlora" in profile_keys()
