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
    _build_luma_modify,
    _build_gpt_image_edit,
    _build_qwen_edit,
    _build_seedream_edit,
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
        assert len(h["edit_prompt"]) > 20


def test_get_hairstyle_prompts_edit_mode():
    prompts = get_hairstyle_prompts(use_edit=True)
    assert len(prompts) == 4
    for p in prompts:
        assert "prompt" in p
        assert "Edit only the hair" in p["prompt"]
        assert "Do NOT change" in p["prompt"]


def test_prompts_are_edit_specific():
    """All prompts must instruct the model to edit, not generate."""
    prompts = get_hairstyle_prompts()
    for p in prompts:
        assert "Only modify the hair" in p["prompt"]
        assert "face" in p["prompt"].lower()  # mentions preserving face


# --- Model config tests ---


def test_all_models_available():
    models = available_generation_models()
    assert len(models) == 9
    ids = [m["id"] for m in models]
    assert "fal-ai/luma-photon/flash" in ids
    assert "openai/gpt-image-2" in ids
    assert "fal-ai/qwen-image-edit-plus" in ids
    assert "fal-ai/flux-kontext/dev" in ids
    assert "fal-ai/flux-2/edit" in ids
    assert "xai/grok-imagine-image/edit" in ids
    assert "fal-ai/chrono-edit-lora" in ids
    assert "fal-ai/image-editing/hair-change" in ids


def test_get_model_config_returns_none_for_unknown():
    assert get_model_config("nonexistent") is None


def test_luma_modify_passes_image_url():
    """Luma Photon Modify takes image_url for editing."""
    args = _build_luma_modify(prompt="test", image_url="https://example.com/img.png")
    assert args["image_url"] == "https://example.com/img.png"
    assert args["prompt"] == "test"


def test_luma_modify_includes_required_fields():
    """Luma modify requires strength and aspect_ratio — missing these causes 422."""
    args = _build_luma_modify(prompt="test", image_url="https://example.com/img.png")
    assert "strength" in args, "Luma modify requires 'strength' field"
    assert "aspect_ratio" in args, "Luma modify requires 'aspect_ratio' field"
    assert 0 < args["strength"] <= 1.0


def test_all_model_builders_handle_required_fields():
    """Every model builder must produce a complete args dict with no missing required fields."""
    from generation.models import GENERATION_MODELS
    for model_id, config in GENERATION_MODELS.items():
        build_fn = config["build_input"]
        args = build_fn(prompt="test prompt", image_url="https://example.com/img.png")
        assert isinstance(args, dict), f"{model_id} build_input must return a dict"
        # Models use either 'prompt' or 'hair_style_prompt'
        has_prompt = "prompt" in args or "hair_style_prompt" in args
        assert has_prompt, f"{model_id} missing prompt field"
        # Image must be present in some form
        has_image = "image_url" in args or "image_urls" in args
        assert has_image, f"{model_id} missing image input"


def test_all_models_pass_image():
    """Every model must accept and pass through the image URL."""
    for model_id, config in __import__('generation.models', fromlist=['GENERATION_MODELS']).GENERATION_MODELS.items():
        build_fn = config["build_input"]
        args = build_fn(prompt="test", image_url="https://example.com/img.png")
        has_image = "image_urls" in args or "image_url" in args
        assert has_image, f"Model {model_id} does not pass image to the API"


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


def test_seedream_edit_passes_image_url():
    """Seedream edit takes image_url for editing."""
    args = _build_seedream_edit(prompt="test", image_url="https://example.com/img.png")
    assert args["image_url"] == "https://example.com/img.png"
    assert args["prompt"] == "test"


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
    assert len(data["models"]) == 9


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


@patch("main.generate_hairstyles")
@patch.dict("os.environ", {"FAL_KEY": "test-key"}, clear=False)
def test_generate_handles_null_images(mock_gen):
    """Backend must return null for failed generations, not crash."""
    mock_gen.return_value = {
        "images": [None, "https://cdn.fal.ai/2.png", None, "https://cdn.fal.ai/4.png"],
        "labels": ["Protective Braids", "Natural Twist-Out", "Silk Press", "Bantu Knots"],
        "model": "fal-ai/luma-photon/flash",
        "duration_ms": 3000,
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
    assert data["images"][0] is None  # Failed generation
    assert data["images"][1] is not None  # Successful generation
    assert data["images"][2] is None
    assert data["images"][3] is not None
