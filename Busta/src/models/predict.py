from pathlib import Path


def load_model_for_inference(model_path):
    """Load a saved Keras model for inference. Imports TensorFlow lazily."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for model inference.") from exc
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return tf.keras.models.load_model(str(model_path))
