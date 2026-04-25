from pathlib import Path

import numpy as np
import pytest

# Skip all tests in this file if TensorFlow is not installed.
# TensorFlow currently requires Python ≤ 3.12; this file is skipped on 3.13+.
pytest.importorskip("tensorflow", reason="TensorFlow not installed (requires Python ≤ 3.12)")

import cv2  # noqa: E402 (cv2 is only used with TF, guard with importorskip above)


def _write_png(path: Path) -> None:
    img = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    assert ok, f"Failed to write image: {path}"


def test_model_output_shape() -> None:
    from src.models.cnn_model import build_baseline_cnn

    model = build_baseline_cnn(input_shape=(32, 32, 3))
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (4, 1)


def test_model_output_range() -> None:
    from src.models.cnn_model import build_baseline_cnn

    model = build_baseline_cnn(input_shape=(32, 32, 3))
    x = np.random.rand(8, 32, 32, 3).astype(np.float32)
    out = model.predict(x, verbose=0)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_training_smoke(tmp_path: Path) -> None:
    from src.models.train import train_baseline_model
    from src.utils.config import DataConfig, PreprocessConfig, TrainConfig

    for class_name in ("Parasitized", "Uninfected"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(6):
            _write_png(class_dir / f"{i}.png")

    data_config = DataConfig(dataset_root=tmp_path, test_size=0.2, random_state=42, stratify=True)
    preprocess_config = PreprocessConfig(image_size=(32, 32), denoise_mode="none")
    train_config = TrainConfig(
        model_dir=tmp_path / "artifacts",
        model_name="test_model.keras",
        history_name="history.json",
        evaluation_name="eval.json",
        confusion_matrix_name="cm.csv",
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model, history, metrics = train_baseline_model(
        data_config=data_config,
        preprocess_config=preprocess_config,
        train_config=train_config,
    )

    assert (tmp_path / "artifacts" / "test_model.keras").exists()
    assert (tmp_path / "artifacts" / "history.json").exists()
    assert "loss" in history
    assert "accuracy" in metrics
    assert "f1" in metrics
