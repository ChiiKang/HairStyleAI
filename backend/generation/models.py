"""
Generation model registry.

ALL models use image editing — the user's selfie is always passed as input.
No text-to-image-only models. The AI edits the hairstyle on the actual photo.

API schemas verified from fal.ai model pages (2026-04-26).
"""

from typing import Any


def _build_gpt_image_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """GPT Image 2 edit: uses image_urls (plural, list)."""
    return {
        "prompt": prompt,
        "image_urls": [image_url],
        "quality": "medium",
        "output_format": "png",
    }


def _build_qwen_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """Qwen Image Edit Plus: uses image_urls (plural, list)."""
    return {
        "prompt": prompt,
        "image_urls": [image_url],
        "output_format": "png",
        "num_images": 1,
    }


def _build_luma_modify(prompt: str, image_url: str, **_: Any) -> dict:
    """Luma Photon modify: image-to-image editing."""
    return {
        "prompt": prompt,
        "image_url": image_url,
    }


def _build_seedream_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """Seedream edit: image editing endpoint."""
    return {
        "prompt": prompt,
        "image_url": image_url,
        "num_images": 1,
    }


def _parse_images_list(result: dict) -> list[str]:
    """Standard parser: result has 'images' list with 'url' fields."""
    images = result.get("images", [])
    urls = []
    for img in images:
        if isinstance(img, dict) and "url" in img:
            urls.append(img["url"])
    # Fallback: check 'image' singular
    if not urls:
        image = result.get("image")
        if isinstance(image, dict) and "url" in image:
            urls.append(image["url"])
    return urls


GENERATION_MODELS = {
    "openai/gpt-image-2": {
        "id": "openai/gpt-image-2",
        "endpoint": "openai/gpt-image-2/edit",
        "name": "GPT Image 2",
        "cost": "$0.04-0.17/img",
        "build_input": _build_gpt_image_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/qwen-image-edit-plus": {
        "id": "fal-ai/qwen-image-edit-plus",
        "endpoint": "fal-ai/qwen-image-edit-plus",
        "name": "Qwen Image Edit Plus",
        "cost": "$0.03/MP",
        "build_input": _build_qwen_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/luma-photon/flash": {
        "id": "fal-ai/luma-photon/flash",
        "endpoint": "fal-ai/luma-photon/modify",
        "name": "Luma Photon Modify",
        "cost": "$0.005/MP",
        "build_input": _build_luma_modify,
        "parse_output": _parse_images_list,
    },
    "fal-ai/bytedance/seedream/v5/lite/text-to-image": {
        "id": "fal-ai/bytedance/seedream/v5/lite/text-to-image",
        "endpoint": "fal-ai/bytedance/seedream/v5/lite/edit",
        "name": "Seedream 5.0 Edit",
        "cost": "$0.035/img",
        "build_input": _build_seedream_edit,
        "parse_output": _parse_images_list,
    },
}


def get_model_config(model_id: str) -> dict | None:
    return GENERATION_MODELS.get(model_id)


def available_generation_models() -> list[dict]:
    return [
        {"id": m["id"], "name": m["name"], "cost": m["cost"]}
        for m in GENERATION_MODELS.values()
    ]
