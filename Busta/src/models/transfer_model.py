from typing import Sequence

import tensorflow as tf
from tensorflow import keras

# Registry maps base name → Keras application constructor.
_BASE_MODELS = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
}


def build_transfer_model(
    input_shape: Sequence[int] = (224, 224, 3),
    base: str = "mobilenetv2",
    freeze_base: bool = True,
) -> keras.Model:
    """
    Build a binary classifier on top of a pretrained base (MobileNetV2 or EfficientNetB0).

    Head: GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid).
    Compiled with Adam(1e-4), binary_crossentropy, accuracy + AUC.
    """
    if base not in _BASE_MODELS:
        raise ValueError(f"Unsupported base {base!r}. Choose from: {list(_BASE_MODELS)}")

    input_shape = tuple(input_shape)
    base_model = _BASE_MODELS[base](
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    if freeze_base:
        base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"transfer_{base}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_top_layers(model: keras.Model, n_layers: int = 30) -> None:
    """Unfreeze the last n_layers of the base sub-model for fine-tuning."""
    base_model = next((lay for lay in model.layers if isinstance(lay, keras.Model)), None)
    if base_model is None:
        raise ValueError("No sub-model found in model.layers (expected a base model).")

    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
