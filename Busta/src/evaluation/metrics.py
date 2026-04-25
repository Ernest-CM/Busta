from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Dict[str, float | None]
    confusion_matrix: np.ndarray


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> EvaluationResult:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    roc_auc: float | None = None
    if y_score is not None:
        y_score = np.asarray(y_score).reshape(-1)
        if y_score.shape[0] != y_true.shape[0]:
            raise ValueError("y_score must have the same length as y_true.")
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_score))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return EvaluationResult(metrics=metrics, confusion_matrix=cm)


