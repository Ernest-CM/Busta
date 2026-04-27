import json
from unittest.mock import patch

import numpy as np
import pytest

from src.evaluation.evaluate import load_test_split, run_evaluation
from src.evaluation.metrics import EvaluationResult


def _fake_result():
    metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1": 0.95,
        "roc_auc": 0.98,
    }
    cm = np.array([[50, 5], [3, 42]])
    return EvaluationResult(metrics=metrics, confusion_matrix=cm)


def test_run_evaluation_saves_files(tmp_path):
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict.return_value = np.array([[0.8], [0.3], [0.9], [0.2]])
    x_test = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y_test = np.array([1, 0, 1, 0])

    with patch("src.evaluation.evaluate.compute_classification_metrics", return_value=_fake_result()), \
         patch("src.evaluation.evaluate._save_confusion_matrix_png") as mock_cm, \
         patch("src.evaluation.evaluate._save_roc_curve_png") as mock_roc:
        run_evaluation(model=model, x_test=x_test, y_test=y_test, output_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
    mock_cm.assert_called_once()
    mock_roc.assert_called_once()


def test_run_evaluation_returns_metrics(tmp_path):
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict.return_value = np.array([[0.8], [0.3], [0.9], [0.2]])
    x_test = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y_test = np.array([1, 0, 1, 0])

    with patch("src.evaluation.evaluate.compute_classification_metrics", return_value=_fake_result()), \
         patch("src.evaluation.evaluate._save_confusion_matrix_png"), \
         patch("src.evaluation.evaluate._save_roc_curve_png"):
        result = run_evaluation(model=model, x_test=x_test, y_test=y_test, output_dir=tmp_path)

    assert result["accuracy"] == pytest.approx(0.95)
    assert result["roc_auc"] == pytest.approx(0.98)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["f1"] == pytest.approx(0.95)


def test_load_test_split_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_test_split(tmp_path)


def test_load_test_split_loads_arrays(tmp_path):
    x = np.ones((10, 32, 32, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    np.save(tmp_path / "X_test.npy", x)
    np.save(tmp_path / "y_test.npy", y)

    x_loaded, y_loaded = load_test_split(tmp_path)
    assert x_loaded.shape == (10, 32, 32, 3)
    assert y_loaded.shape == (10,)
    np.testing.assert_array_equal(x_loaded, x)
    np.testing.assert_array_equal(y_loaded, y)
