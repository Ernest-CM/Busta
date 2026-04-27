import tensorflow as tf

AUGMENTATION_LAYERS = ["RandomFlip", "RandomRotation", "RandomZoom", "RandomBrightness"]


def build_augmentation_pipeline(seed: int = 42) -> tf.keras.Sequential:
    """
    Return a Keras Sequential of image augmentation layers.

    Apply with training=True to activate augmentation; pass training=False (or omit)
    during inference to use the layers as identity transforms.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.15, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomBrightness(0.1, seed=seed),
        ],
        name="augmentation_pipeline",
    )
