import re
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from segmentation import get_segmenter, available_models
from recolor.pipeline import clean_mask, recolor_hair

app = FastAPI(title="Hair Color Change API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode image bytes, raise 400 if invalid."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, flags)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def list_models():
    return {"models": available_models()}


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(...),
    model: str = Form("bisenet"),
):
    models = available_models()
    if model not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}. Available: {models}")

    contents = await image.read()
    img = _decode_image(contents)

    segmenter = get_segmenter(model)
    raw_mask = segmenter.segment(img)
    mask = clean_mask(raw_mask)

    _, buf = cv2.imencode(".png", mask)
    return Response(content=buf.tobytes(), media_type="image/png")


RECOLOR_METHODS = ["reinhard", "shift", "overlay"]


@app.get("/api/recolor-methods")
def list_recolor_methods():
    return {"methods": RECOLOR_METHODS, "default": "reinhard"}


@app.post("/api/recolor")
async def recolor(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    color: str = Form("#FF0000"),
    intensity: int = Form(80),
    lift: int = Form(0),
    method: str = Form("reinhard"),
):
    intensity = max(0, min(100, intensity))
    lift = max(0, min(40, lift))
    color = color.strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise HTTPException(status_code=400, detail=f"Invalid color hex: {color}")
    if method not in RECOLOR_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}. Available: {RECOLOR_METHODS}")

    img_bytes = await image.read()
    mask_bytes = await mask.read()

    img = _decode_image(img_bytes, cv2.IMREAD_COLOR)
    mask_arr = _decode_image(mask_bytes, cv2.IMREAD_GRAYSCALE)

    result = recolor_hair(img, mask_arr, color, intensity, lift, method=method)

    _, buf = cv2.imencode(".png", result)
    return Response(content=buf.tobytes(), media_type="image/png")
