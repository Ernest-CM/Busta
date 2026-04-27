import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.run_benchmark import run_benchmark

_FAKE_METRICS = {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1": 0.95,
    "roc_auc": 0.98,
}


def test_benchmark_produces_json(tmp_path):
    keras_dir = tmp_path / "models"
    keras_dir.mkdir()
    (keras_dir / "model_a.keras").touch()
    (keras_dir / "model_b.keras").touch()

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    output_dir = tmp_path / "benchmark"

    with (
        patch("scripts.run_benchmark.load_test_split", return_value=(
            np.zeros((4, 32, 32, 3), dtype=np.float32),
            np.array([0, 1, 0, 1]),
        )),
        patch("scripts.run_benchmark.load_model_for_inference", return_value=MagicMock()),
        patch("scripts.run_benchmark.run_evaluation", return_value=_FAKE_METRICS),
    ):
        results = run_benchmark(processed_dir, output_dir, models_dir=keras_dir)

    assert len(results) == 2
    assert results[0]["model_name"] == "model_a"
    assert results[1]["model_name"] == "model_b"

    comparison = json.loads((output_dir / "comparison.json").read_text())
    assert len(comparison) == 2
    assert comparison[0]["accuracy"] == pytest.approx(0.95)


def test_benchmark_no_models_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        run_benchmark(tmp_path, tmp_path, models_dir=empty_dir)
