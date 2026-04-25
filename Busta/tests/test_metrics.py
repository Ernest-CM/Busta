import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics


def test_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = compute_classification_metrics(y_true, y_pred)
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["precision"] == pytest.approx(1.0)
    assert result.metrics["recall"] == pytest.approx(1.0)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.metrics["roc_auc"] is None


def test_all_wrong_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    result = compute_classification_metrics(y_true, y_pred)
    assert result.metrics["accuracy"] == pytest.approx(0.0)
    assert result.metrics["precision"] == pytest.approx(0.0)
    assert result.metrics["recall"] == pytest.approx(0.0)
    assert result.metrics["f1"] == pytest.approx(0.0)


def test_confusion_matrix_layout() -> None:
    # TN=1 FP=1 FN=1 TP=1
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    result = compute_classification_metrics(y_true, y_pred)
    cm = result.confusion_matrix
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN
    assert cm[1, 1] == 1  # TP


def test_roc_auc_computed_with_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    result = compute_classification_metrics(y_true, y_pred, y_score=y_score)
    assert result.metrics["roc_auc"] is not None
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0


def test_roc_auc_none_when_single_class() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.95])
    result = compute_classification_metrics(y_true, y_pred, y_score=y_score)
    assert result.metrics["roc_auc"] is None


def test_mismatched_lengths_raise() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0])
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred)


def test_mismatched_score_length_raises() -> None:
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    y_score = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score=y_score)
