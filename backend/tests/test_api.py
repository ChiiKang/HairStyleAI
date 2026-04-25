from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _make_test_image(w: int = 100, h: int = 100) -> bytes:
    img = np.full((h, w, 3), [120, 80, 60], dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_test_mask(w: int = 100, h: int = 100) -> bytes:
    mask = np.full((h, w), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", mask)
    return buf.tobytes()


def _mock_segmenter():
    mock = MagicMock()
    mock.segment.side_effect = lambda img: np.full(img.shape[:2], 255, dtype=np.uint8)
    return mock


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_available_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "bisenet" in data["models"]


@patch("main.get_segmenter", return_value=_mock_segmenter())
def test_segment_returns_png(mock_get):
    img_bytes = _make_test_image()
    r = client.post(
        "/api/segment",
        data={"model": "bisenet"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_segment_rejects_unknown_model():
    img_bytes = _make_test_image()
    r = client.post(
        "/api/segment",
        data={"model": "nonexistent"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 400


def test_recolor_returns_png():
    img_bytes = _make_test_image()
    mask_bytes = _make_test_mask()
    r = client.post(
        "/api/recolor",
        data={"color": "#FF0000", "intensity": "80", "lift": "0"},
        files=[
            ("image", ("test.png", img_bytes, "image/png")),
            ("mask", ("mask.png", mask_bytes, "image/png")),
        ],
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_recolor_rejects_bad_color():
    img_bytes = _make_test_image()
    mask_bytes = _make_test_mask()
    r = client.post(
        "/api/recolor",
        data={"color": "not-a-color", "intensity": "80", "lift": "0"},
        files=[
            ("image", ("test.png", img_bytes, "image/png")),
            ("mask", ("mask.png", mask_bytes, "image/png")),
        ],
    )
    assert r.status_code == 400
