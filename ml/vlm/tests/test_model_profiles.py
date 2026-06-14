from __future__ import annotations

import pytest

from ml.vlm.src.model_profiles import profile_keys, resolve_training_profile


def test_default_qwen_profile_uses_qlora_guardrails() -> None:
    profile = resolve_training_profile("qwen2_5_vl_3b_qlora")

    assert profile.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert profile.load_in_4bit is True
    assert profile.train_vision_tower is False
    assert profile.train_projector is False
    assert profile.batch_size == 1
    assert profile.gradient_accumulation_steps >= 8
    assert profile.image_max_pixels == "512x512"


def test_low_vram_profile_is_available_for_smoke_runs() -> None:
    profile = resolve_training_profile("smolvlm2_2b_qlora")

    assert profile.min_vram_gb <= 8
    assert profile.training_strategy == "4bit_qlora_sft_text_projector_only"


def test_unknown_profile_reports_supported_profiles() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_training_profile("unknown")

    message = str(exc_info.value)
    assert "qwen2_5_vl_3b_qlora" in message
    assert "smolvlm2_2b_qlora" in profile_keys()
