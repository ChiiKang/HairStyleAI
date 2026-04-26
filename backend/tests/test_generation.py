"""Tests for the hairstyle generation pipeline."""

from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from main import app

from generation.models import (
    get_model_config,
    available_generation_models,
    _build_luma_photon,
    _build_gpt_image_edit,
    _build_qwen_edit,
    _build_seedream,
    _parse_images_list,
)
from prompts.hairstyles import HAIRSTYLES, get_hairstyle_prompts

client = TestClient(app)


def _make_test_image() -> bytes:
    img = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# --- Prompt tests ---


def test_hairstyles_has_four_entries():
    assert len(HAIRSTYLES) == 4


def test_each_hairstyle_has_required_fields():
    for h in HAIRSTYLES:
        assert "id" in h
        assert "label" in h
        assert "edit_prompt" in h
        assert "generate_prompt" in h
        assert len(h["edit_prompt"]) > 20
        assert len(h["generate_prompt"]) > 20


def test_get_hairstyle_prompts_edit_mode():
    prompts = get_hairstyle_prompts(use_edit=True)
    assert len(prompts) == 4
    for p in prompts:
        assert "prompt" in p
        assert "Keep the person" in p["prompt"] or "Transform" in p["prompt"]


def test_get_hairstyle_prompts_generate_mode():
    prompts = get_hairstyle_prompts(use_edit=False)
    assert len(prompts) == 4
    for p in prompts:
        assert "prompt" in p
        assert "photorealistic" in p["prompt"].lower() or "portrait" in p["prompt"].lower()


# --- Model config tests ---


def test_all_models_available():
    models = available_generation_models()
    assert len(models) == 4
    ids = [m["id"] for m in models]
    assert "fal-ai/luma-photon/flash" in ids
    assert "openai/gpt-image-2" in ids
    assert "fal-ai/qwen-image-edit-plus" in ids


def test_get_model_config_returns_none_for_unknown():
    assert get_model_config("nonexistent") is None


def test_luma_input_uses_aspect_ratio():
    """Luma Photon Flash uses aspect_ratio, not image_size."""
    args = _build_luma_photon(prompt="test")
    assert "aspect_ratio" in args
    assert "image_size" not in args
    assert "num_images" not in args  # Luma doesn't support num_images


def test_gpt_edit_uses_image_urls_list():
    """GPT Image 2 edit uses image_urls (plural, list)."""
    args = _build_gpt_image_edit(prompt="test", image_url="https://example.com/img.png")
    assert "image_urls" in args
    assert isinstance(args["image_urls"], list)
    assert args["image_urls"] == ["https://example.com/img.png"]
    assert "image_url" not in args  # Must NOT use singular form


def test_qwen_edit_uses_image_urls_list():
    """Qwen uses image_urls (plural, list)."""
    args = _build_qwen_edit(prompt="test", image_url="https://example.com/img.png")
    assert "image_urls" in args
    assert isinstance(args["image_urls"], list)


def test_seedream_uses_image_size():
    args = _build_seedream(prompt="test")
    assert "image_size" in args
    assert args["image_size"] == "square_hd"


def test_gpt_edit_endpoint_is_edit_variant():
    """GPT Image 2 frontend ID maps to /edit endpoint."""
    config = get_model_config("openai/gpt-image-2")
    assert config["endpoint"] == "openai/gpt-image-2/edit"


def test_parse_images_list():
    result = {"images": [{"url": "https://a.com/1.png"}, {"url": "https://a.com/2.png"}]}
    urls = _parse_images_list(result)
    assert urls == ["https://a.com/1.png", "https://a.com/2.png"]


def test_parse_images_empty():
    assert _parse_images_list({}) == []
    assert _parse_images_list({"images": []}) == []


# --- API endpoint tests ---


def test_generation_models_endpoint():
    r = client.get("/api/generation-models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) == 4


def test_generate_rejects_unknown_model():
    img_bytes = _make_test_image()
    r = client.post(
        "/api/generate-hairstyles",
        data={"model": "nonexistent-model"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


@patch.dict("os.environ", {"FAL_KEY": ""}, clear=False)
def test_generate_rejects_missing_fal_key():
    img_bytes = _make_test_image()
    r = client.post(
        "/api/generate-hairstyles",
        data={"model": "fal-ai/luma-photon/flash"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 500
    assert "FAL_KEY" in r.json()["detail"]


@patch("main.generate_hairstyles")
@patch.dict("os.environ", {"FAL_KEY": "test-key"}, clear=False)
def test_generate_returns_json(mock_gen):
    mock_gen.return_value = {
        "images": ["https://cdn.fal.ai/1.png", "https://cdn.fal.ai/2.png",
                    "https://cdn.fal.ai/3.png", "https://cdn.fal.ai/4.png"],
        "labels": ["Protective Braids", "Natural Twist-Out", "Silk Press", "Bantu Knots"],
        "model": "fal-ai/luma-photon/flash",
        "duration_ms": 5000,
    }

    img_bytes = _make_test_image()
    r = client.post(
        "/api/generate-hairstyles",
        data={"model": "fal-ai/luma-photon/flash"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["images"]) == 4
    assert len(data["labels"]) == 4
    assert data["model"] == "fal-ai/luma-photon/flash"
    assert data["duration_ms"] == 5000
