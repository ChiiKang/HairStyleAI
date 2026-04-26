"""
Generation model registry.

Each model has:
- id: the fal.ai endpoint ID used in frontend dropdown
- endpoint: the actual fal.ai endpoint to call (may differ from id for edit models)
- name: display name
- type: "edit" (takes source image) or "text-to-image" (generates from prompt only)
- cost: human-readable cost string
- build_input: function that constructs the fal.ai arguments dict
- parse_output: function that extracts image URLs from the fal.ai result

API schemas verified from fal.ai model pages (2026-04-26).
"""

from typing import Any


def _build_gpt_image_edit(prompt: str, image_url: str | None, **_: Any) -> dict:
    """GPT Image 2 edit: uses image_urls (plural, list)."""
    args: dict[str, Any] = {
        "prompt": prompt,
        "quality": "medium",
        "output_format": "png",
    }
    if image_url:
        args["image_urls"] = [image_url]
    return args


def _build_qwen_edit(prompt: str, image_url: str | None, **_: Any) -> dict:
    """Qwen Image Edit Plus: uses image_urls (plural, list)."""
    args: dict[str, Any] = {
        "prompt": prompt,
        "output_format": "png",
        "num_images": 1,
    }
    if image_url:
        args["image_urls"] = [image_url]
    return args


def _build_luma_photon(prompt: str, **_: Any) -> dict:
    """Luma Photon Flash: uses aspect_ratio (not image_size), no num_images."""
    return {
        "prompt": prompt,
        "aspect_ratio": "1:1",
    }


def _build_seedream(prompt: str, **_: Any) -> dict:
    """Seedream 5.0 Lite: text-to-image with image_size."""
    return {
        "prompt": prompt,
        "image_size": "square_hd",
        "num_images": 1,
    }


def _parse_images_list(result: dict) -> list[str]:
    """Standard parser: result has 'images' list with 'url' fields."""
    images = result.get("images", [])
    urls = []
    for img in images:
        if isinstance(img, dict) and "url" in img:
            urls.append(img["url"])
    return urls


GENERATION_MODELS = {
    "openai/gpt-image-2": {
        "id": "openai/gpt-image-2",
        "endpoint": "openai/gpt-image-2/edit",
        "name": "GPT Image 2",
        "type": "edit",
        "cost": "$0.04-0.17/img",
        "build_input": _build_gpt_image_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/qwen-image-edit-plus": {
        "id": "fal-ai/qwen-image-edit-plus",
        "endpoint": "fal-ai/qwen-image-edit-plus",
        "name": "Qwen Image Edit Plus",
        "type": "edit",
        "cost": "$0.03/MP",
        "build_input": _build_qwen_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/luma-photon/flash": {
        "id": "fal-ai/luma-photon/flash",
        "endpoint": "fal-ai/luma-photon/flash",
        "name": "Luma Photon Flash",
        "type": "text-to-image",
        "cost": "$0.005/MP",
        "build_input": _build_luma_photon,
        "parse_output": _parse_images_list,
    },
    "fal-ai/bytedance/seedream/v5/lite/text-to-image": {
        "id": "fal-ai/bytedance/seedream/v5/lite/text-to-image",
        "endpoint": "fal-ai/bytedance/seedream/v5/lite/text-to-image",
        "name": "Seedream 5.0 Lite",
        "type": "text-to-image",
        "cost": "$0.035/img",
        "build_input": _build_seedream,
        "parse_output": _parse_images_list,
    },
}


def get_model_config(model_id: str) -> dict | None:
    return GENERATION_MODELS.get(model_id)


def available_generation_models() -> list[dict]:
    return [
        {"id": m["id"], "name": m["name"], "type": m["type"], "cost": m["cost"]}
        for m in GENERATION_MODELS.values()
    ]
