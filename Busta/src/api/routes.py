import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from src.utils.config import LABEL_MAP

api_bp = Blueprint("api", __name__)
INV_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}


@api_bp.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@api_bp.post("/predict")
def predict() -> tuple:
    model = current_app.config.get("MODEL")
    preprocessor = current_app.config.get("PREPROCESSOR")
    if model is None or preprocessor is None:
        return jsonify({"error": "Model server is not initialized."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Invalid image file."}), 400

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = preprocessor.preprocess_array(image_rgb)
    x_batch = np.expand_dims(x, axis=0)

    prob = float(model.predict(x_batch, verbose=0).reshape(-1)[0])
    pred = int(prob >= 0.5)
    return (
        jsonify(
            {
                "prediction": pred,
                "label": INV_LABEL_MAP[pred],
                "probability_parasitized": prob,
            }
        ),
        200,
    )

