import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.gradcam import generate_gradcam, overlay_heatmap  # noqa: E402


def _small_cnn():
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(4, 3, activation="relu", padding="same", name="conv2d"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def test_gradcam_output_shape():
    model = _small_cnn()
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    heatmap = generate_gradcam(model, x)
    assert heatmap.shape == (32, 32)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6


def test_overlay_output_shape_and_dtype():
    img_rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    heatmap = np.random.rand(32, 32).astype(np.float32)
    overlay = overlay_heatmap(img_rgb, heatmap)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
