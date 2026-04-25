from typing import Sequence

import tensorflow as tf
from tensorflow import keras


def build_baseline_cnn(
    input_shape: Sequence[int] = (224, 224, 3),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Create a small, reproducible baseline CNN for binary malaria classification."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=tuple(input_shape)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_malaria_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def set_global_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


