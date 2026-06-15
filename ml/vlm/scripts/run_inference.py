from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.vlm.scripts.train_sft import SUPPORTED_ATTENTION_IMPLEMENTATIONS, TORCH_SDPA_ATTENTION


def run_dry_inference(
    dataset_path: Path,
    output_path: Path,
    model_id: str,
    load_in_4bit: bool,
    limit: int | None,
    index: int | None = None,
    attn_implementation: str = TORCH_SDPA_ATTENTION,
) -> int:
    records = _load_records(dataset_path)
    selected_records = _select_records(records, index, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected_records:
            assistant_text = _assistant_text(record)
            payload = _prediction_payload(
                record=record,
                raw_text=assistant_text,
                model_id=model_id,
                load_in_4bit=load_in_4bit,
                attn_implementation=attn_implementation,
                mode="dry_run_target_echo",
                cuda_peak_memory_mb=None,
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(selected_records)


def run_model_inference(
    dataset_path: Path,
    output_path: Path,
    model_id: str,
    load_in_4bit: bool,
    limit: int | None,
    max_new_tokens: int,
    index: int | None = None,
    attn_implementation: str = TORCH_SDPA_ATTENTION,
) -> int:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "VLM inference dependencies are missing. Install ml/vlm/requirements.txt, then rerun without --dry-run."
        ) from exc

    records = _load_records(dataset_path)
    selected_records = _select_records(records, index, limit)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    processor: Any = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model: Any = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected_records:
            messages = record["messages"]
            image_paths = record["images"]
            if not isinstance(image_paths, list) or not image_paths:
                raise ValueError(f"Missing image path for sample {record.get('sample_id')}")
            image = Image.open(str(image_paths[0])).convert("RGB")
            prompt = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            raw_text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            cuda_peak_memory_mb = None
            if torch.cuda.is_available():
                cuda_peak_memory_mb = round(float(torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0, 2)
            payload = _prediction_payload(
                record=record,
                raw_text=raw_text,
                model_id=model_id,
                load_in_4bit=load_in_4bit,
                attn_implementation=attn_implementation,
                mode="model_generate",
                cuda_peak_memory_mb=cuda_peak_memory_mb,
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(selected_records)


def _select_records(records: list[dict[str, Any]], index: int | None, limit: int | None) -> list[dict[str, Any]]:
    if index is not None and limit is not None:
        raise ValueError("--index and --limit cannot be used together.")
    if index is not None:
        if index < 0 or index >= len(records):
            raise IndexError(f"--index {index} is outside dataset length {len(records)}.")
        return [records[index]]
    return records[:limit] if limit is not None else records


def _prediction_payload(
    record: dict[str, Any],
    raw_text: str,
    model_id: str,
    load_in_4bit: bool,
    attn_implementation: str,
    mode: str,
    cuda_peak_memory_mb: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sample_id": str(record["sample_id"]),
        "label_id": int(record["label_id"]),
        "model_id": model_id,
        "load_in_4bit": load_in_4bit,
        "attn_implementation": attn_implementation,
        "mode": mode,
        "raw_text": raw_text,
        "cuda_peak_memory_mb": cuda_peak_memory_mb,
    }
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        payload["parse_error"] = str(exc)
        return payload
    if isinstance(parsed, dict):
        payload["parsed_json"] = parsed
    else:
        payload["parse_error"] = "Generated JSON is not an object."
    return payload


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("Each dataset line must be a JSON object.")
            records.append(parsed)
    return records


def _assistant_text(record: dict[str, Any]) -> str:
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Missing assistant message for sample {record.get('sample_id')}")
    assistant = messages[1]
    if not isinstance(assistant, dict):
        raise ValueError(f"Invalid assistant message for sample {record.get('sample_id')}")
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Assistant content must be a string for sample {record.get('sample_id')}")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default=TORCH_SDPA_ATTENTION, choices=SUPPORTED_ATTENTION_IMPLEMENTATIONS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        rows_written = run_dry_inference(
            dataset_path=args.dataset,
            output_path=args.output,
            model_id=args.model_id,
            load_in_4bit=args.load_in_4bit,
            limit=args.limit,
            index=args.index,
            attn_implementation=args.attn_implementation,
        )
    else:
        rows_written = run_model_inference(
            dataset_path=args.dataset,
            output_path=args.output,
            model_id=args.model_id,
            load_in_4bit=args.load_in_4bit,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            index=args.index,
            attn_implementation=args.attn_implementation,
        )
    print(
        json.dumps(
            {
                "rows_written": rows_written,
                "output": str(args.output),
                "model_id": args.model_id,
                "attn_implementation": args.attn_implementation,
            }
        )
    )


if __name__ == "__main__":
    main()
