from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, object]:
    matrix = _confusion_matrix(y_true, y_pred, num_classes)
    recalls = _per_class_recall(matrix)
    f1_scores = _per_class_f1(matrix)
    return {
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": float(np.mean(f1_scores)) if len(f1_scores) else 0.0,
        "per_class_recall": {str(index): float(value) for index, value in enumerate(recalls)},
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            matrix[target, prediction] += 1
    return matrix


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _per_class_recall(matrix: np.ndarray) -> np.ndarray:
    denominators = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        denominators,
        out=np.zeros_like(denominators, dtype=np.float64),
        where=denominators != 0,
    )


def _per_class_f1(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=actual_positive != 0,
    )
    return np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(precision + recall) != 0,
    )
