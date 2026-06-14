"""Classification metrics for partial discharge experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class_f1: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]
    confusion_matrix: list[list[int]]
    pd_to_normal_error_count: int
    n_rows: int


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> ClassificationMetrics:
    labels = list(range(n_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
    per_class_recall = [
        float(score)
        for score in recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    ]
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        per_class_f1=[
            float(score)
            for score in f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
        per_class_precision=[
            float(score)
            for score in precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
        per_class_recall=per_class_recall,
        confusion_matrix=matrix.tolist(),
        pd_to_normal_error_count=int(matrix[2:, 0].sum()) if matrix.shape[0] >= 5 else 0,
        n_rows=int(len(y_true)),
    )
