"""
Hairstyle generation using fal.ai SDK.

Handles:
- Uploading selfie to fal CDN
- Dispatching 4 parallel generation requests (one per hairstyle)
- Collecting results with timing info
"""

import asyncio
import os
import tempfile
import time

import fal_client

from prompts.hairstyles import get_hairstyle_prompts
from .models import get_model_config


def _upload_image(image_bytes: bytes) -> str:
    """Upload image bytes to fal CDN, return the URL."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        f.flush()
        try:
            url = fal_client.upload_file(f.name)
        finally:
            os.unlink(f.name)
    return url


async def _generate_single(
    model_id: str,
    arguments: dict,
) -> dict:
    """Run a single generation via fal.ai queue (async)."""
    result = await fal_client.subscribe_async(
        model_id,
        arguments=arguments,
        with_logs=False,
    )
    return result


async def generate_hairstyles(
    image_bytes: bytes,
    model_id: str,
) -> dict:
    """
    Generate 4 hairstyle variations from a selfie.

    Args:
        image_bytes: The uploaded selfie as raw bytes.
        model_id: The fal.ai model endpoint ID.

    Returns:
        {
            "images": [url1, url2, url3, url4],
            "labels": ["Protective Braids", ...],
            "model": model_id,
            "duration_ms": 1234
        }
    """
    config = get_model_config(model_id)
    if not config:
        raise ValueError(f"Unknown generation model: {model_id}")

    is_edit = config["type"] == "edit"
    hairstyles = get_hairstyle_prompts(use_edit=is_edit)

    # Upload image to fal CDN if using an edit model
    image_url = None
    if is_edit:
        image_url = _upload_image(image_bytes)

    # Use the actual fal.ai endpoint (may differ from frontend model ID)
    endpoint = config["endpoint"]

    # Build arguments for each hairstyle
    tasks = []
    for style in hairstyles:
        build_fn = config["build_input"]
        args = build_fn(prompt=style["prompt"], image_url=image_url)
        tasks.append(_generate_single(endpoint, args))

    # Run all 4 generations concurrently
    start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration_ms = int((time.time() - start) * 1000)

    # Parse results
    parse_fn = config["parse_output"]
    images = []
    labels = []
    for i, (style, result) in enumerate(zip(hairstyles, results)):
        if isinstance(result, Exception):
            print(f"Generation failed for {style['label']}: {result}")
            images.append(None)
        else:
            urls = parse_fn(result)
            images.append(urls[0] if urls else None)
        labels.append(style["label"])

    return {
        "images": images,
        "labels": labels,
        "model": model_id,
        "duration_ms": duration_ms,
    }
