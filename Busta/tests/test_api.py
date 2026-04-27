import io
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.api.app import create_app
from src.preprocessing.preprocessor import Preprocessor


def _synthetic_png() -> bytes:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = [200, 100, 50]
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.85]])
    preprocessor = Preprocessor()
    app = create_app(model=mock_model, preprocessor=preprocessor)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def rate_client():
    """Separate fixture without TESTING=True so rate limiting is enforced."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.85]])
    preprocessor = Preprocessor()
    app = create_app(model=mock_model, preprocessor=preprocessor)
    with app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_image(client):
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(_synthetic_png()), "cell.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "label" in data
    assert data["prediction"] in ("Parasitized", "Uninfected")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_no_file(client):
    resp = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_wrong_extension(client):
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"not an image"), "report.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_ui_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data


def test_predict_unreadable_image(client):
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"\x00\x01\x02\x03garbage bytes not an image"), "cell.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "unreadable image"}


def test_predict_rate_limit(rate_client):
    png = _synthetic_png()
    for _ in range(10):
        rate_client.post(
            "/api/predict",
            data={"image": (io.BytesIO(png), "cell.png")},
            content_type="multipart/form-data",
        )
    resp = rate_client.post(
        "/api/predict",
        data={"image": (io.BytesIO(png), "cell.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 429
    assert resp.get_json().get("error") == "rate limit exceeded"
