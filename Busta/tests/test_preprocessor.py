from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.ingestion import ImageRecord
from src.preprocessing.preprocessor import Preprocessor
from src.utils.config import PreprocessConfig


def _write_image(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    assert ok


def test_preprocess_image_contract(tmp_path: Path) -> None:
    image = np.random.randint(0, 255, size=(80, 120, 3), dtype=np.uint8)
    image_path = tmp_path / "sample.png"
    _write_image(image_path, image)

    preprocessor = Preprocessor(PreprocessConfig(image_size=(224, 224)))
    processed = preprocessor.preprocess_image(image_path)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_denoise_reduces_noise_variance(tmp_path: Path) -> None:
    clean = np.full((224, 224, 3), 127, dtype=np.uint8)
    noise = np.random.randint(0, 50, size=(224, 224, 3), dtype=np.uint8)
    noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
    image_path = tmp_path / "noisy.png"
    _write_image(image_path, noisy)

    no_denoise = Preprocessor(PreprocessConfig(denoise_mode="none")).preprocess_image(image_path)
    median = Preprocessor(PreprocessConfig(denoise_mode="median")).preprocess_image(image_path)

    assert float(np.var(median)) <= float(np.var(no_denoise))


def test_corrupted_file_raises_by_default(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.png"
    bad_path.write_text("not an image", encoding="utf-8")

    preprocessor = Preprocessor(PreprocessConfig(on_error="raise"))
    with pytest.raises(ValueError, match="Unable to read image file"):
        preprocessor.preprocess_image(bad_path)


def test_corrupted_file_skip_mode(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.png"
    bad_path.write_text("not an image", encoding="utf-8")

    records = [ImageRecord(path=bad_path, label_name="Parasitized", label=1)]
    preprocessor = Preprocessor(PreprocessConfig(on_error="skip"))
    x, y, skipped = preprocessor.preprocess_records(records)

    assert x.shape == (0, 224, 224, 3)
    assert y.shape == (0,)
    assert skipped == [bad_path]

