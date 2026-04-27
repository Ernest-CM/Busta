import os
import tempfile
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, request

from src.api.limiter import limiter
from src.utils.config import LABEL_MAP

try:
    from src.models.gradcam import generate_gradcam, overlay_heatmap
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False

api_bp = Blueprint("api", __name__)
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB route-level cap


def _allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


def _save_upload(image_file) -> Path:
    """Save an uploaded FileStorage to a temp file; caller must delete it."""
    suffix = Path(image_file.filename).suffix.lower()
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    image_file.save(str(tmp_path))
    return tmp_path


def _check_upload_size(image_file):
    """Return a 413 JSON response if the upload exceeds the route-level cap, else None."""
    image_file.stream.seek(0, 2)
    size = image_file.stream.tell()
    image_file.stream.seek(0)
    if size > _MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large. Maximum {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413
    return None


@api_bp.get("/health")
def health() -> tuple:
    model_loaded = current_app.config.get("MODEL") is not None
    return jsonify({"status": "ok", "model_loaded": model_loaded}), 200


@api_bp.post("/predict")
@limiter.limit("10 per minute")
def predict() -> tuple:
    model = current_app.config.get("MODEL")
    preprocessor = current_app.config.get("PREPROCESSOR")
    if model is None or preprocessor is None:
        return jsonify({"error": "Model not initialized."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'."}), 400

    image_file = request.files["image"]
    if not image_file.filename or not _allowed_extension(image_file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: .png, .jpg, .jpeg, .bmp"}), 400

    size_error = _check_upload_size(image_file)
    if size_error is not None:
        return size_error

    tmp_path = None
    try:
        tmp_path = _save_upload(image_file)
        if cv2.imread(str(tmp_path)) is None:
            return jsonify({"error": "unreadable image"}), 400
        x = preprocessor.preprocess_single(tmp_path)
        prob = float(model.predict(x, verbose=0).reshape(-1)[0])
        pred = int(prob >= 0.5)
        confidence = prob if pred == 1 else 1.0 - prob
        return jsonify({
            "prediction": INV_LABEL_MAP[pred],
            "confidence": round(confidence, 4),
            "label": pred,
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@api_bp.post("/gradcam")
@limiter.limit("10 per minute")
def gradcam() -> tuple:
    if not _GRADCAM_AVAILABLE:
        return jsonify({"error": "Grad-CAM requires TensorFlow which is not installed."}), 503

    model = current_app.config.get("MODEL")
    preprocessor = current_app.config.get("PREPROCESSOR")
    if model is None or preprocessor is None:
        return jsonify({"error": "Model not initialized."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'."}), 400

    image_file = request.files["image"]
    if not image_file.filename or not _allowed_extension(image_file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: .png, .jpg, .jpeg, .bmp"}), 400

    size_error = _check_upload_size(image_file)
    if size_error is not None:
        return size_error

    tmp_path = None
    try:
        tmp_path = _save_upload(image_file)
        bgr = cv2.imread(str(tmp_path))
        if bgr is None:
            return jsonify({"error": "unreadable image"}), 400
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = preprocessor.config.image_size
        rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        x = preprocessor.preprocess_single(tmp_path)
        heatmap = generate_gradcam(model, x)
        overlay = overlay_heatmap(rgb_resized, heatmap)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", overlay_bgr)
        if not ok:
            return jsonify({"error": "Failed to encode overlay image."}), 500

        return buf.tobytes(), 200, {"Content-Type": "image/png"}
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
