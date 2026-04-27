from unittest.mock import patch

import pytest

tf = pytest.importorskip("tensorflow")

from src.models.transfer_model import build_transfer_model, unfreeze_top_layers  # noqa: E402


def _tiny_base(input_shape=(32, 32, 3), include_top=False, weights=None, **kwargs):
    """Minimal stand-in for MobileNetV2 — avoids downloading ImageNet weights in tests."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(8):
        x = tf.keras.layers.Conv2D(4, 1, padding="same", activation="relu")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def test_build_transfer_model_output_shape():
    with patch.dict("src.models.transfer_model._BASE_MODELS", {"mobilenetv2": _tiny_base}):
        model = build_transfer_model(input_shape=(32, 32, 3), base="mobilenetv2", freeze_base=True)
    assert model.output_shape == (None, 1)


def test_unfreeze_sets_trainable():
    with patch.dict("src.models.transfer_model._BASE_MODELS", {"mobilenetv2": _tiny_base}):
        model = build_transfer_model(input_shape=(32, 32, 3), base="mobilenetv2", freeze_base=True)

    base = next(lay for lay in model.layers if isinstance(lay, tf.keras.Model))
    assert not any(lay.trainable for lay in base.layers if lay.weights), "Base should be fully frozen before unfreeze"

    unfreeze_top_layers(model, n_layers=5)

    trainable_with_weights = sum(1 for lay in base.layers if lay.trainable and lay.weights)
    assert trainable_with_weights >= 5
