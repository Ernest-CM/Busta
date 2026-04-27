import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.preprocessing.augmentation import build_augmentation_pipeline  # noqa: E402


def test_pipeline_returns_keras_sequential():
    pipeline = build_augmentation_pipeline()
    assert isinstance(pipeline, tf.keras.Sequential)


def test_augmentation_output_shape():
    pipeline = build_augmentation_pipeline(seed=0)
    x = tf.constant(np.random.rand(1, 224, 224, 3).astype(np.float32))
    out = pipeline(x, training=True)
    assert tuple(out.shape) == (1, 224, 224, 3)
