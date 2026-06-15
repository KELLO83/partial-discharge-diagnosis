from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.vlm.src.schema import PD_LABELS_KO

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_FALLBACK_CONFIDENCE = 0.5
DEFAULT_FALLBACK_LABEL_ID = 0
GENERIC_MODEL_TEXT = {"normal", "noise", "unknown"}
SUPPORTED_ATTN_IMPLEMENTATIONS = {"sdpa", "eager"}

RISK_LEVEL_BY_LABEL: dict[str, str] = {
    "0": "낮음",
    "1": "낮음",
    "2": "주의",
    "3": "주의",
    "4": "위험",
}
RECOMMENDED_ACTION_BY_LABEL: dict[str, str] = {
    "0": "정상 상태로 판단되며 정기 모니터링을 유지하세요.",
    "1": "센서 접촉 상태와 주변 전자기 간섭 가능성을 점검하세요.",
    "2": "절연체 표면 오염과 트래킹 흔적을 점검하세요.",
    "3": "전계 집중 부위와 고전압 접속부를 점검하세요.",
    "4": "절연체 내부 결함 가능성을 고려해 정밀 진단을 진행하세요.",
}


class VlmServiceAdapter:
    def __init__(self, context: object) -> None:
        self.context = context
        self._torch = None
        self._transformers = None
        self._pil_image = None
        self._model = None
        self._processor = None
        self._model_device = None

    def generate_report(self, tool_input: object) -> dict[str, object]:
        vote = _extract_vote(tool_input)
        try:
            parsed_report = self._run_model(tool_input)
        except Exception:
            parsed_report = _fallback_report(vote)

        label_id = _resolve_label_id(parsed_report.get("label_id"), vote)
        confidence = _resolve_confidence(parsed_report.get("confidence"), vote)
        diagnosis = _resolve_diagnosis(parsed_report.get("diagnosis"), label_id)
        return {
            "model_name": str(_model_name(self.context)),
            "model_version": str(_model_version(self.context)),
            "label_id": label_id,
            "diagnosis": diagnosis,
            "risk_level": _resolve_risk_level(parsed_report.get("risk_level"), label_id),
            "confidence": confidence,
            "reason": _resolve_reason(parsed_report.get("reason")),
            "recommended_action": _resolve_recommended_action(parsed_report.get("recommended_action"), label_id),
        }

    def _run_model(self, tool_input: object) -> dict[str, object]:
        _load_runtime_stack(self)
        image_path = Path(getattr(tool_input, "image_path"))
        if not image_path.exists():
            raise RuntimeError(f"image path does not exist: {image_path}")
        model = self._load_model()
        processor = self._load_processor()
        image = self._pil_image.open(image_path).convert("RGB")

        prompt = _build_prompt(tool_input)
        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self._model_device)
        outputs = model.generate(**inputs, max_new_tokens=_max_new_tokens(self.context), do_sample=False)
        sequence = outputs[0]
        input_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        decoded = processor.batch_decode([sequence[input_tokens:]], skip_special_tokens=True)[0]
        parsed = _parse_report(decoded)
        if parsed is None:
            raise RuntimeError(f"model output is not parseable as JSON: {decoded!r}")
        return parsed

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        checkpoint = _checkpoint_path(self.context)
        _load_runtime_stack(self)
        if checkpoint.is_dir() and (checkpoint / "adapter_config.json").exists():
            model = _load_peft_model(self, checkpoint)
        else:
            model = _load_full_model(self, checkpoint)
        self._model = model
        self._model_device = getattr(model, "device", self._torch.device("cpu"))
        return model

    def _load_processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        preprocessor = _preprocessor_path(self.context)
        _load_runtime_stack(self)
        processor_cls = getattr(self._transformers, "AutoProcessor")
        self._processor = processor_cls.from_pretrained(str(preprocessor), trust_remote_code=True)
        return self._processor


def load_adapter(context: object) -> VlmServiceAdapter:
    return VlmServiceAdapter(context)


def _load_runtime_stack(adapter: VlmServiceAdapter) -> None:
    if adapter._torch is not None and adapter._transformers is not None and adapter._pil_image is not None:
        return

    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise RuntimeError("VLM runtime requires torch and pillow for inference.") from exc
    try:
        import importlib

        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("VLM runtime requires transformers for inference.") from exc

    adapter._torch = torch
    adapter._transformers = transformers
    adapter._pil_image = PILImage


def _load_full_model(adapter: VlmServiceAdapter, checkpoint: Path | str) -> Any:
    model_cls = getattr(adapter._transformers, "AutoModelForImageTextToText")
    quantization_config = None
    if _runtime_bool(adapter.context, "load_in_4bit"):
        bitsandbytes_config = getattr(adapter._transformers, "BitsAndBytesConfig")
        quantization_config = bitsandbytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=adapter._torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return model_cls.from_pretrained(
        str(checkpoint),
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=_attn_implementation(adapter.context),
        trust_remote_code=True,
        torch_dtype=adapter._torch.float16,
    )


def _load_peft_model(adapter: VlmServiceAdapter, checkpoint: Path) -> Any:
    try:
        import importlib

        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise RuntimeError("VLM PEFT adapter inference requires peft. Install ml/vlm/requirements.txt.") from exc
    base_model = _load_full_model(adapter, _base_model_id(adapter.context, checkpoint))
    return peft.PeftModel.from_pretrained(base_model, str(checkpoint))


def _base_model_id(context: object, checkpoint: Path) -> str:
    preprocessor = _preprocessor_path(context)
    if preprocessor.exists() and preprocessor.is_file():
        payload = json.loads(preprocessor.read_text(encoding="utf-8"))
        model_id = payload.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            return model_id
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.exists():
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = payload.get("base_model_name_or_path")
        if isinstance(base_model, str) and base_model.strip():
            return base_model
    raise RuntimeError("VLM PEFT adapter is missing base model metadata.")


def _checkpoint_path(context: object) -> Path:
    checkpoint = getattr(context, "checkpoint_path", None)
    if checkpoint is None:
        raise RuntimeError("VLM checkpoint path is not configured.")
    return _to_path(checkpoint)


def _preprocessor_path(context: object) -> Path:
    preprocessor = getattr(context, "preprocessor_path", None)
    if preprocessor is not None:
        return _to_path(preprocessor)
    checkpoint = _checkpoint_path(context)
    fallback = checkpoint.parent / "processor"
    return fallback if fallback.exists() else checkpoint


def _to_path(path_value: object) -> Path:
    return path_value if isinstance(path_value, Path) else Path(str(path_value))


def _max_new_tokens(context: object) -> int:
    value = _runtime_value(context, "max_new_tokens")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_NEW_TOKENS


def _runtime_bool(context: object, key: str) -> bool:
    value = _runtime_value(context, key)
    return bool(value) if isinstance(value, bool) else False


def _runtime_value(context: object, key: str) -> object | None:
    manifest = getattr(context, "manifest", None)
    runtime = getattr(manifest, "runtime", None) if manifest is not None else None
    if isinstance(runtime, dict):
        return runtime.get(key)
    return None


def _attn_implementation(context: object) -> str:
    value = _runtime_value(context, "attn_implementation")
    if value in SUPPORTED_ATTN_IMPLEMENTATIONS:
        return str(value)
    return DEFAULT_ATTN_IMPLEMENTATION


def _parse_report(raw_text: str) -> dict[str, object] | None:
    raw = raw_text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed: Any = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_vote(tool_input: object) -> dict[str, Any]:
    votes = [
        vote
        for vote in (
            _vote_from_result(getattr(tool_input, "timeseries_result", None)),
            _vote_from_result(getattr(tool_input, "vision_result", None)),
        )
        if vote is not None
    ]
    if not votes:
        return {"label_id": DEFAULT_FALLBACK_LABEL_ID, "confidence": DEFAULT_FALLBACK_CONFIDENCE}
    votes.sort(key=lambda pair: pair[1], reverse=True)
    return {"label_id": votes[0][0], "confidence": votes[0][1]}


def _vote_from_result(model_result: object | None) -> tuple[int, float] | None:
    if model_result is None:
        return None
    label = getattr(model_result, "label_id", None)
    confidence = getattr(model_result, "confidence", None)
    if isinstance(label, int) and isinstance(confidence, int | float):
        return label, float(confidence)
    return None


def _resolve_label_id(value: object, vote: dict[str, Any]) -> int:
    if isinstance(value, int) and value in PD_LABELS_KO:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        if parsed in PD_LABELS_KO:
            return parsed
    fallback = vote.get("label_id")
    if isinstance(fallback, int) and fallback in PD_LABELS_KO:
        return fallback
    return 0


def _resolve_confidence(value: object, vote: dict[str, Any]) -> float:
    raw = value if isinstance(value, int | float) else vote.get("confidence", DEFAULT_FALLBACK_CONFIDENCE)
    confidence = float(raw)
    return max(0.0, min(confidence, 1.0))


def _resolve_diagnosis(value: object, label_id: int) -> str:
    if _is_model_specific_text(value):
        return str(value)
    return PD_LABELS_KO.get(label_id, "정상")


def _resolve_risk_level(value: object, label_id: int) -> str:
    if _is_model_specific_text(value):
        return str(value)
    return RISK_LEVEL_BY_LABEL.get(str(label_id), "확인필요")


def _resolve_reason(value: object) -> str:
    if _is_model_specific_text(value):
        return str(value)
    return "시계열/비전/지식 근거를 반영해 판단된 결과입니다."


def _resolve_recommended_action(value: object, label_id: int) -> str:
    if _is_model_specific_text(value):
        return str(value)
    return _default_recommended_action(label_id)


def _is_model_specific_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in GENERIC_MODEL_TEXT


def _default_recommended_action(label_id: int) -> str:
    return RECOMMENDED_ACTION_BY_LABEL.get(str(label_id), "추가 데이터 기반으로 재평가하세요.")


def _model_name(context: object) -> str:
    manifest = getattr(context, "manifest", None)
    return str(getattr(manifest, "model_name", "pd_vlm_reporter")) if manifest is not None else "pd_vlm_reporter"


def _model_version(context: object) -> str:
    manifest = getattr(context, "manifest", None)
    return str(getattr(manifest, "model_version", "1.0.0")) if manifest is not None else "1.0.0"


def _build_prompt(tool_input: object) -> str:
    metadata = getattr(tool_input, "safe_metadata", None)
    timeseries = getattr(tool_input, "timeseries_result", None)
    vision = getattr(tool_input, "vision_result", None)
    rag = getattr(tool_input, "rag_result", None)

    metadata_lines = _metadata_lines(metadata)
    parts = [
        "당신은 부분방전 진단 보조 모델입니다.",
        "이미지 설명을 작성하지 말고, 주어진 PRPD 이미지와 센서/과거진단 근거를 바탕으로 아래 형식의 JSON만 출력하세요.",
        '분류 번호는 0=정상, 1=노이즈, 2=표면방전, 3=코로나방전, 4=보이드방전 중 하나입니다.',
        '{"label_id": int, "diagnosis": str, "risk_level": str, "reason": str, "recommended_action": str, "confidence": 0~1}',
    ]
    if metadata_lines:
        parts.append("[설비 정보]")
        parts.extend(metadata_lines)
    if timeseries is not None:
        parts.append(_model_evidence_line("시계열", timeseries))
    if vision is not None:
        parts.append(_model_evidence_line("비전", vision))
    if rag is not None:
        parts.append(_rag_evidence_line(rag))
    return "\n".join(parts)


def _model_evidence_line(source_name: str, model_result: object) -> str:
    label = getattr(model_result, "label_id", "n/a")
    confidence = getattr(model_result, "confidence", "n/a")
    return f"[{source_name} 근거] label={label} confidence={confidence}"


def _rag_evidence_line(rag_result: object) -> str:
    document_count = len(getattr(rag_result, "documents", []))
    similar_case_count = len(getattr(rag_result, "similar_cases", []))
    return f"[RAG 문헌 근거] documents={document_count} similar_cases={similar_case_count}"


def _metadata_lines(metadata: object | None) -> list[str]:
    if metadata is None:
        return []
    return [
        f"equipment_name: {getattr(metadata, 'equipment_name', 'n/a')}",
        f"sensor_type: {getattr(metadata, 'sensor_type', 'n/a')}",
        f"temperature: {getattr(metadata, 'temperature', 'n/a')}",
        f"humidity: {getattr(metadata, 'humidity', 'n/a')}",
        f"equipment_rated_voltage: {getattr(metadata, 'equipment_rated_voltage', 'n/a')}",
        f"equipment_rated_current: {getattr(metadata, 'equipment_rated_current', 'n/a')}",
    ]


def _fallback_report(vote: dict[str, Any]) -> dict[str, object]:
    label_id = int(vote.get("label_id", DEFAULT_FALLBACK_LABEL_ID))
    return {
        "label_id": label_id,
        "diagnosis": PD_LABELS_KO.get(label_id, "정상"),
        "risk_level": RISK_LEVEL_BY_LABEL.get(str(label_id), "확인필요"),
        "confidence": float(vote.get("confidence", DEFAULT_FALLBACK_CONFIDENCE)),
        "reason": "추론 모듈이 불완전하거나 미설치되어 소스 근거 기반으로 보강 판단했습니다.",
        "recommended_action": _default_recommended_action(label_id),
    }
