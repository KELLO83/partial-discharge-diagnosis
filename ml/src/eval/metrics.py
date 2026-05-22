"""Classification metrics for partial discharge experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    n_rows: int


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> ClassificationMetrics:
    labels = list(range(n_classes))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        per_class_f1=[
            float(score)
            for score in f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        n_rows=int(len(y_true)),
    )
