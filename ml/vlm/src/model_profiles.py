from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VlmTrainingProfile:
    key: str
    model_id: str
    description: str
    training_strategy: str
    min_vram_gb: int
    load_in_4bit: bool
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    precision: str
    attn_implementation: str
    image_max_pixels: str
    train_vision_tower: bool = False
    train_projector: bool = False
    target_modules: str = "all-linear"


VLM_TRAINING_PROFILES: dict[str, VlmTrainingProfile] = {
    "qwen2_5_vl_3b_qlora": VlmTrainingProfile(
        key="qwen2_5_vl_3b_qlora",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Stable fallback profile for Korean PRPD report SFT with 4-bit QLoRA.",
        training_strategy="4bit_qlora_sft_text_projector_only",
        min_vram_gb=8,
        load_in_4bit=True,
        batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        precision="fp16",
        attn_implementation="sdpa",
        image_max_pixels="512x512",
    ),
    "smolvlm2_2b_qlora": VlmTrainingProfile(
        key="smolvlm2_2b_qlora",
        model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        description="Current low-VRAM baseline profile for PRPD report SFT.",
        training_strategy="4bit_qlora_sft_text_projector_only",
        min_vram_gb=6,
        load_in_4bit=True,
        batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        precision="fp16",
        attn_implementation="sdpa",
        image_max_pixels="512x512",
    ),
    "qwen3_vl_2b_qlora": VlmTrainingProfile(
        key="qwen3_vl_2b_qlora",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        description="Future Qwen3-VL 2B comparison profile for Korean PRPD report SFT.",
        training_strategy="4bit_qlora_sft_text_projector_only",
        min_vram_gb=8,
        load_in_4bit=True,
        batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        precision="fp16",
        attn_implementation="sdpa",
        image_max_pixels="512x512",
    ),
}


def profile_keys() -> tuple[str, ...]:
    return tuple(sorted(VLM_TRAINING_PROFILES))


def resolve_training_profile(key: str) -> VlmTrainingProfile:
    try:
        return VLM_TRAINING_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(profile_keys())
        raise ValueError(f"Unsupported VLM model profile: {key}. Supported profiles: {supported}") from exc
