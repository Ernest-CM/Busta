import numpy as np
import tensorflow as tf


def generate_gradcam(
    model,
    img_array: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for img_array.

    Args:
        model: compiled Keras model.
        img_array: (1, H, W, 3) float32 array (output of preprocess_single).
        layer_name: name of the Conv2D layer to target; auto-detects last Conv2D if None.

    Returns:
        (H, W) float32 heatmap with values in [0, 1].
    """
    import cv2

    if layer_name is None:
        for layer in reversed(model.layers):
            if "conv2d" in type(layer).__name__.lower():
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("No Conv2D layer found in model.")

    # Model that outputs (conv features, predictions)
    conv_layer = model.get_layer(layer_name)
    conv_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer.output)

    # Find remaining layers after the target conv layer
    conv_idx = next(
        (i for i, lay in enumerate(model.layers) if lay.name == layer_name), None
    )
    if conv_idx is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.layers.")
    post_conv_layers = model.layers[conv_idx + 1:]

    x = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(x, training=False)
        tape.watch(conv_outputs)
        z = conv_outputs
        for layer in post_conv_layers:
            z = layer(z, training=False)
        loss = z[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("GradientTape returned None — model may have no trainable weights.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)

    max_val = float(heatmap.max())
    if max_val > 0:
        heatmap /= max_val

    H, W = int(img_array.shape[1]), int(img_array.shape[2])
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    return heatmap.astype(np.float32)


def overlay_heatmap(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a jet-colormap heatmap over the original RGB image.

    Args:
        original_img_rgb: (H, W, 3) uint8 RGB array.
        heatmap: (H, W) float32 in [0, 1].
        alpha: weight given to the heatmap (0 = original only, 1 = heatmap only).

    Returns:
        (H, W, 3) uint8 overlay.
    """
    import cv2

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    orig = original_img_rgb.astype(np.float32)
    overlay = (alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * orig)
    return np.clip(overlay, 0, 255).astype(np.uint8)
