"""
Hairstyle generation using fal.ai SDK.

Handles:
- Uploading selfie to fal CDN
- Dispatching 4 parallel generation requests (one per hairstyle)
- Downloading results to local storage
- Collecting results with timing info
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone

import httpx
import fal_client

from prompts.hairstyles import get_hairstyle_prompts
from .models import get_model_config

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "generated_images")
os.makedirs(STORAGE_DIR, exist_ok=True)


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

    # All models are image-editing — always upload the selfie
    hairstyles = get_hairstyle_prompts(use_edit=True)
    image_url = _upload_image(image_bytes)

    endpoint = config["endpoint"]

    # Build arguments for each hairstyle — image_url is always passed
    tasks = []
    for style in hairstyles:
        build_fn = config["build_input"]
        args = build_fn(prompt=style["prompt"], image_url=image_url)
        tasks.append(_generate_single(endpoint, args))

    # Run all 4 generations concurrently
    start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration_ms = int((time.time() - start) * 1000)

    # Parse results and download locally
    parse_fn = config["parse_output"]
    images = []
    labels = []
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save the original selfie
    selfie_path = os.path.join(session_dir, "original.png")
    with open(selfie_path, "wb") as f:
        f.write(image_bytes)

    async with httpx.AsyncClient(timeout=30) as http:
        for i, (style, result) in enumerate(zip(hairstyles, results)):
            if isinstance(result, Exception):
                print(f"Generation failed for {style['label']}: {result}")
                images.append(None)
            else:
                urls = parse_fn(result)
                if urls:
                    # Download image from fal CDN
                    try:
                        resp = await http.get(urls[0])
                        resp.raise_for_status()
                        ext = "jpg" if "jpeg" in resp.headers.get("content-type", "") else "png"
                        filename = f"{i}_{style['id']}.{ext}"
                        filepath = os.path.join(session_dir, filename)
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        images.append(f"/api/images/{session_id}/{filename}")
                    except Exception as e:
                        print(f"Download failed for {style['label']}: {e}")
                        images.append(urls[0])  # fallback to CDN URL
                else:
                    images.append(None)
            labels.append(style["label"])

    # Save metadata
    metadata = {
        "session_id": session_id,
        "model": model_id,
        "duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images": images,
        "labels": labels,
    }
    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "images": images,
        "labels": labels,
        "model": model_id,
        "duration_ms": duration_ms,
        "session_id": session_id,
    }
