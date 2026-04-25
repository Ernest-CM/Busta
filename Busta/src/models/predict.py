from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def load_model_for_inference(model_path: Path | str) -> keras.Model:
    """Load a saved Keras model for inference."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return tf.keras.models.load_model(str(model_path))

