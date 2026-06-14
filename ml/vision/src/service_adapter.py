from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ml.vision.src.data import image_file_to_tensor
from ml.vision.src.models import SmallPrpdCnn
from ml.vision.src.schema import DEFAULT_NUM_CLASSES, PD_LABELS_KO


class VisionServiceAdapter:
    def __init__(self, context: object) -> None:
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SmallPrpdCnn | None = None
        self._checkpoint: dict[str, Any] | None = None

    def predict_image(self, tool_input: object) -> dict[str, object]:
        checkpoint = self._load_checkpoint()
        image_size = int(checkpoint.get("image_size", 224))
        image_path = Path(getattr(tool_input, "image_path"))
        tensor = image_file_to_tensor(image_path=image_path, image_size=image_size).unsqueeze(0).to(self.device)
        model = self._load_model()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=-1)[0].cpu()
        label_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[label_id].item())
        probability_map = {str(index): float(probabilities[index].item()) for index in range(DEFAULT_NUM_CLASSES)}
        return {
            "model_name": str(checkpoint.get("model_name", "small_prpd_cnn")),
            "model_version": _model_version(self.context),
            "label_id": label_id,
            "label_name": PD_LABELS_KO.get(label_id, "unknown"),
            "confidence": confidence,
            "probabilities": probability_map,
            "evidence": {
                "model_family": "small_cnn",
                "image_size": image_size,
                "top_margin": _top_margin(probabilities),
            },
        }

    def _load_model(self) -> SmallPrpdCnn:
        if self._model is None:
            checkpoint = self._load_checkpoint()
            model = SmallPrpdCnn(num_classes=DEFAULT_NUM_CLASSES)
            model.load_state_dict(checkpoint["model_state_dict"])
            self._model = model.to(self.device)
        return self._model

    def _load_checkpoint(self) -> dict[str, Any]:
        if self._checkpoint is None:
            checkpoint_path = _checkpoint_path(self.context)
            self._checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return self._checkpoint


def load_adapter(context: object) -> VisionServiceAdapter:
    return VisionServiceAdapter(context)


def _checkpoint_path(context: object) -> Path:
    checkpoint_path = getattr(context, "checkpoint_path", None)
    if checkpoint_path is None:
        raise RuntimeError("Vision checkpoint path is not configured.")
    return Path(checkpoint_path)


def _model_version(context: object) -> str:
    manifest = getattr(context, "manifest", None)
    model_version = getattr(manifest, "model_version", None)
    return str(model_version or "unknown")


def _top_margin(probabilities: torch.Tensor) -> float:
    sorted_probabilities = torch.sort(probabilities, descending=True).values
    if len(sorted_probabilities) < 2:
        return 0.0
    return float((sorted_probabilities[0] - sorted_probabilities[1]).item())
