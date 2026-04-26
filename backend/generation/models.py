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
    """Luma Photon modify: image-to-image editing. Requires strength + aspect_ratio."""
    return {
        "prompt": prompt,
        "image_url": image_url,
        "strength": 0.75,
        "aspect_ratio": "1:1",
    }


def _build_seedream_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """Seedream edit: image editing endpoint."""
    return {
        "prompt": prompt,
        "image_url": image_url,
        "num_images": 1,
    }


def _build_flux_kontext(prompt: str, image_url: str, **_: Any) -> dict:
    """FLUX Kontext dev: SOTA instruction editing with identity preservation."""
    return {
        "prompt": prompt,
        "image_url": image_url,
    }


def _build_flux2_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """FLUX.2 edit: next-gen editing, uses image_urls (plural, list)."""
    return {
        "prompt": prompt,
        "image_urls": [image_url],
    }


def _build_grok_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """Grok Imagine edit: cheapest flat-rate option."""
    return {
        "prompt": prompt,
        "image_url": image_url,
    }


def _build_chrono_edit(prompt: str, image_url: str, **_: Any) -> dict:
    """NVIDIA Chrono Edit LoRA: physics-aware editing."""
    return {
        "prompt": prompt,
        "image_url": image_url,
    }


def _build_hair_change(prompt: str, image_url: str, **_: Any) -> dict:
    """Dedicated hair change endpoint: free-text hairstyle prompt."""
    return {
        "image_url": image_url,
        "hair_style_prompt": prompt,
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
    "fal-ai/flux-kontext/dev": {
        "id": "fal-ai/flux-kontext/dev",
        "endpoint": "fal-ai/flux-kontext/dev",
        "name": "FLUX Kontext",
        "cost": "$0.025/MP",
        "build_input": _build_flux_kontext,
        "parse_output": _parse_images_list,
    },
    "fal-ai/flux-2/edit": {
        "id": "fal-ai/flux-2/edit",
        "endpoint": "fal-ai/flux-2/edit",
        "name": "FLUX.2 Edit",
        "cost": "$0.024/img",
        "build_input": _build_flux2_edit,
        "parse_output": _parse_images_list,
    },
    "xai/grok-imagine-image/edit": {
        "id": "xai/grok-imagine-image/edit",
        "endpoint": "xai/grok-imagine-image/edit",
        "name": "Grok Imagine Edit",
        "cost": "$0.022/img",
        "build_input": _build_grok_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/chrono-edit-lora": {
        "id": "fal-ai/chrono-edit-lora",
        "endpoint": "fal-ai/chrono-edit-lora",
        "name": "Chrono Edit LoRA",
        "cost": "$0.02/img",
        "build_input": _build_chrono_edit,
        "parse_output": _parse_images_list,
    },
    "fal-ai/image-editing/hair-change": {
        "id": "fal-ai/image-editing/hair-change",
        "endpoint": "fal-ai/image-editing/hair-change",
        "name": "Hair Change (Dedicated)",
        "cost": "$0.04/img",
        "build_input": _build_hair_change,
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
