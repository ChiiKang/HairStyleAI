# Hair Color Change Prototype — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React + FastAPI app that segments hair from a static image using pluggable models (BiSeNet, FASHN, SegFormer) and recolors it with luminance-preserving LAB color space blending.

**Architecture:** React/Vite frontend sends an image to a FastAPI backend. The backend runs segmentation (returning a hair mask) and recoloring (returning the final image) as two separate endpoints. Segmentation models are pluggable via an abstract base class.

**Tech Stack:** React 18, Vite, TypeScript, Tailwind CSS, FastAPI, Uvicorn, PyTorch, OpenCV, Pillow, NumPy, transformers (HuggingFace)

---

## File Map

### Backend (`backend/`)

| File | Responsibility |
|------|---------------|
| `backend/main.py` | FastAPI app, CORS, routes |
| `backend/segmentation/base.py` | Abstract segmentation interface |
| `backend/segmentation/bisenet.py` | BiSeNet CelebAMask-HQ hair segmentation |
| `backend/segmentation/fashn_parser.py` | FASHN Human Parser hair segmentation |
| `backend/segmentation/segformer.py` | SegFormer face-parsing hair segmentation |
| `backend/segmentation/__init__.py` | Model registry / factory |
| `backend/recolor/pipeline.py` | Mask cleanup + LAB recoloring |
| `backend/recolor/__init__.py` | Exports |
| `backend/requirements.txt` | Python dependencies |
| `backend/tests/test_recolor.py` | Recolor pipeline unit tests |
| `backend/tests/test_api.py` | API endpoint integration tests |

### Frontend (`frontend/src/`)

| File | Responsibility |
|------|---------------|
| `frontend/src/App.tsx` | Main layout, state orchestration |
| `frontend/src/components/ImageUploader.tsx` | Drag-and-drop image upload |
| `frontend/src/components/ModelSelector.tsx` | Segmentation model dropdown |
| `frontend/src/components/ColorPicker.tsx` | Preset swatches + custom hex input |
| `frontend/src/components/ControlPanel.tsx` | Intensity slider + lift slider |
| `frontend/src/components/ResultViewer.tsx` | Before/after + mask preview + download |
| `frontend/src/hooks/useHairRecolor.ts` | API calls, loading state, error handling |
| `frontend/src/types.ts` | Shared TypeScript types |

---

## Task 1: Backend Project Scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/main.py`

- [ ] **Step 1: Create requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
opencv-python-headless==4.10.0.84
Pillow==10.4.0
numpy>=1.26,<2.0
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
scipy>=1.12.0
pytest>=8.0.0
gdown>=5.1.0
```

- [ ] **Step 2: Create minimal FastAPI app**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hair Color Change API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Create virtual environment and install dependencies**

Run:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 4: Verify the server starts**

Run:
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

Expected: Server starts on http://localhost:8000. `GET /api/health` returns `{"status": "ok"}`.

- [ ] **Step 5: Commit**

```bash
git init
git add backend/requirements.txt backend/main.py
git commit -m "feat: scaffold FastAPI backend with health endpoint"
```

---

## Task 2: Recoloring Pipeline

**Files:**
- Create: `backend/recolor/__init__.py`
- Create: `backend/recolor/pipeline.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_recolor.py`

- [ ] **Step 1: Write failing tests for mask cleanup**

```python
# backend/tests/test_recolor.py
import numpy as np
from recolor.pipeline import clean_mask


def test_clean_mask_removes_small_islands():
    """Small isolated blobs (< 100px) should be removed."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large region — should survive
    mask[20:60, 20:60] = 255
    # Tiny 3x3 island — should be removed
    mask[80:83, 80:83] = 255

    cleaned = clean_mask(mask, min_area=100)

    assert cleaned[40, 40] == 255, "Large region should survive"
    assert cleaned[81, 81] == 0, "Small island should be removed"


def test_clean_mask_fills_small_holes():
    """Small holes inside the mask should be filled."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    # Punch a tiny 3x3 hole inside
    mask[38:41, 38:41] = 0

    cleaned = clean_mask(mask, min_area=100)

    assert cleaned[39, 39] == 255, "Small hole should be filled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend
source venv/bin/activate
python -m pytest tests/test_recolor.py -v
```

Expected: FAIL — `ImportError: cannot import name 'clean_mask'`

- [ ] **Step 3: Implement clean_mask**

```python
# backend/recolor/pipeline.py
import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    min_area: int = 100,
    feather_radius: int = 5,
) -> np.ndarray:
    """Clean a binary mask: remove small islands, fill small holes, feather edges."""
    # Threshold to pure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Remove small islands
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    # Fill small holes (invert, remove small islands in the inverse, invert back)
    inverted = cv2.bitwise_not(cleaned)
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inverted)
    for i in range(1, num_labels_inv):
        if stats_inv[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels_inv == i] = 255

    # Feather edges with Gaussian blur
    if feather_radius > 0:
        k = feather_radius * 2 + 1
        cleaned = cv2.GaussianBlur(cleaned, (k, k), 0)

    return cleaned
```

```python
# backend/recolor/__init__.py
from .pipeline import clean_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd backend
source venv/bin/activate
python -m pytest tests/test_recolor.py::test_clean_mask_removes_small_islands tests/test_recolor.py::test_clean_mask_fills_small_holes -v
```

Expected: 2 PASSED

- [ ] **Step 5: Write failing tests for recolor_hair**

Add to `backend/tests/test_recolor.py`:

```python
from recolor.pipeline import recolor_hair


def test_recolor_preserves_unmasked_areas():
    """Pixels outside the mask should be unchanged."""
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)  # BGR
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Only center is hair

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0)

    # Top-left corner should be untouched
    np.testing.assert_array_equal(result[0, 0], image[0, 0])


def test_recolor_changes_masked_area():
    """Pixels inside the mask should be different from original."""
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0)

    # Center pixel should have changed
    assert not np.array_equal(result[50, 50], image[50, 50])


def test_recolor_zero_intensity_is_noop():
    """Intensity 0 should return the original image unchanged."""
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=0, lift=0)

    np.testing.assert_array_equal(result, image)
```

- [ ] **Step 6: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_recolor.py -v
```

Expected: FAIL — `ImportError: cannot import name 'recolor_hair'`

- [ ] **Step 7: Implement recolor_hair**

Add to `backend/recolor/pipeline.py`:

```python
def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (B, G, R)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def recolor_hair(
    image: np.ndarray,
    mask: np.ndarray,
    color_hex: str,
    intensity: int = 80,
    lift: int = 0,
) -> np.ndarray:
    """
    Recolor hair using LAB color space to preserve luminance.

    Args:
        image: BGR image (OpenCV format)
        mask: grayscale mask (0-255), 255 = hair
        color_hex: target color as '#RRGGBB'
        intensity: color strength 0-100
        lift: brightness lift for dark hair 0-40
    """
    if intensity == 0:
        return image.copy()

    alpha = intensity / 100.0

    # Convert original to LAB (float for precision)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert target color to LAB
    target_bgr = np.full_like(image, hex_to_bgr(color_hex), dtype=np.uint8)
    lab_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Normalize mask to 0-1 float — this is the SINGLE blend factor
    mask_float = mask.astype(np.float32) / 255.0
    blend = alpha * mask_float  # combined intensity + mask weight

    # Apply lift to L channel for dark hair (only inside mask)
    if lift > 0:
        lift_amount = lift * 2.55  # scale 0-40 to 0-102 in LAB L range (0-255)
        l_channel = lab_image[:, :, 0]
        # Only lift dark pixels (L < 100)
        dark_mask = (l_channel < 100).astype(np.float32)
        l_lifted = l_channel + lift_amount * dark_mask * mask_float
        lab_image[:, :, 0] = np.clip(l_lifted, 0, 255)

    # Replace A and B channels with target color, blended once by (intensity * mask)
    for ch in [1, 2]:  # A and B channels
        lab_image[:, :, ch] = (
            lab_image[:, :, ch] * (1 - blend)
            + lab_target[:, :, ch] * blend
        )

    # Convert back to BGR — no second composite needed, blending was mask-aware
    lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return result
```

Update `backend/recolor/__init__.py` to also export `recolor_hair`:

```python
from .pipeline import clean_mask, recolor_hair
```

Also create an empty `backend/tests/__init__.py` if it doesn't exist yet:

```python
# backend/tests/__init__.py
```

- [ ] **Step 8: Run all recolor tests**

Run:
```bash
python -m pytest tests/test_recolor.py -v
```

Expected: 5 PASSED

- [ ] **Step 9: Commit**

```bash
git add backend/recolor/ backend/tests/
git commit -m "feat: add recoloring pipeline with LAB color space blending"
```

---

## Task 3: Segmentation Base Class and BiSeNet Model

**Files:**
- Create: `backend/segmentation/__init__.py`
- Create: `backend/segmentation/base.py`
- Create: `backend/segmentation/bisenet.py`

- [ ] **Step 1: Create abstract base class**

```python
# backend/segmentation/base.py
from abc import ABC, abstractmethod
import numpy as np


class HairSegmenter(ABC):
    """Abstract base for all hair segmentation models."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once at startup."""
        ...

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment hair from an image.

        Args:
            image: BGR image (OpenCV format), any size.

        Returns:
            Grayscale mask (0-255), same height/width as input.
            255 = hair, 0 = not hair.
        """
        ...
```

- [ ] **Step 2: Implement BiSeNet segmenter**

This uses `zllrunning/face-parsing.pytorch`. We vendor the model architecture and download pretrained weights.

```python
# backend/segmentation/bisenet.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .base import HairSegmenter

# Hair class in CelebAMask-HQ label map = 13
# Full label map: 0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
# 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose, 11=mouth, 12=u_lip, 13=hair,
# 14=hat, 15=l_lip, 16=cloth, 17=neck, 18=necklace
HAIR_LABEL = 13
# Weights hosted on Google Drive (same file referenced by zllrunning/face-parsing.PyTorch README)
GDRIVE_FILE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bisenet_79999_iter.pth")


class BiSeNetSegmenter(HairSegmenter):
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            print(f"Downloading BiSeNet weights to {MODEL_PATH}...")
            import gdown
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            print("Download complete.")

        # Import BiSeNet architecture
        # We use a minimal inline version to avoid cloning the full repo
        self.model = self._build_bisenet()
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        h, w = image.shape[:2]

        # BiSeNet expects 512x512 input
        img_resized = cv2.resize(image, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)[0]  # BiSeNet returns tuple
            parsing = output.squeeze(0).argmax(0).cpu().numpy()

        # Extract hair class
        hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255

        # Resize mask back to original dimensions
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return hair_mask

    def _build_bisenet(self):
        """
        Build BiSeNet model from the vendored third-party source.
        Requires: git clone https://github.com/zllrunning/face-parsing.PyTorch
                  into backend/third_party/face_parsing/ (done in Task 3 Step 4).
        """
        import sys
        third_party_path = os.path.join(os.path.dirname(__file__), "..", "third_party", "face_parsing")
        if third_party_path not in sys.path:
            sys.path.insert(0, third_party_path)
        from model import BiSeNet
        return BiSeNet(n_classes=19)
```

- [ ] **Step 3: Create model registry**

```python
# backend/segmentation/__init__.py
from .base import HairSegmenter
from .bisenet import BiSeNetSegmenter

# Registry of available models
_MODELS: dict[str, type[HairSegmenter]] = {
    "bisenet": BiSeNetSegmenter,
}

# Cache loaded model instances
_INSTANCES: dict[str, HairSegmenter] = {}


def get_segmenter(model_name: str) -> HairSegmenter:
    """Get a segmenter instance by name. Loads model on first call."""
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODELS.keys())}")

    if model_name not in _INSTANCES:
        instance = _MODELS[model_name]()
        instance.load()
        _INSTANCES[model_name] = instance

    return _INSTANCES[model_name]


def available_models() -> list[str]:
    return list(_MODELS.keys())
```

- [ ] **Step 4: Clone BiSeNet third-party dependency**

Run:
```bash
cd backend
git clone https://github.com/zllrunning/face-parsing.PyTorch third_party/face_parsing
```

- [ ] **Step 5: Test BiSeNet loads without crashing**

Run (in Python REPL):
```bash
cd backend
source venv/bin/activate
python -c "
from segmentation import get_segmenter
import numpy as np

seg = get_segmenter('bisenet')
# Test with a blank 256x256 image
dummy = np.zeros((256, 256, 3), dtype=np.uint8)
mask = seg.segment(dummy)
print(f'Mask shape: {mask.shape}, dtype: {mask.dtype}')
print(f'Mask range: {mask.min()}-{mask.max()}')
print('BiSeNet loaded OK')
"
```

Expected: Prints mask shape (256, 256), dtype uint8, and "BiSeNet loaded OK". The mask will be all zeros (no hair in a blank image), which is correct.

- [ ] **Step 6: Commit**

```bash
git add backend/segmentation/ backend/third_party/
git commit -m "feat: add BiSeNet hair segmentation with model registry"
```

---

## Task 4: FASHN and SegFormer Segmentation Models

**Files:**
- Create: `backend/segmentation/fashn_parser.py`
- Create: `backend/segmentation/segformer.py`
- Modify: `backend/segmentation/__init__.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add fashn-human-parser to requirements.txt**

Append to `backend/requirements.txt`:

```txt
fashn-human-parser>=0.1.0
```

- [ ] **Step 2: Implement FASHN segmenter**

```python
# backend/segmentation/fashn_parser.py
import cv2
import numpy as np

from .base import HairSegmenter

# FASHN Human Parser uses ATR label convention: hair = 2
# Verify at runtime: print(parser.predict(img)) on a test image
# If label is wrong, check model card: https://huggingface.co/fashn-ai/fashn-human-parser
HAIR_LABEL = 2


class FASHNSegmenter(HairSegmenter):
    def __init__(self):
        self.parser = None

    def load(self) -> None:
        from fashn_human_parser import FashnHumanParser
        self.parser = FashnHumanParser()

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.parser is None:
            self.load()

        h, w = image.shape[:2]

        # FASHN expects RGB PIL Image
        from PIL import Image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Run parsing — returns a numpy array of class IDs
        seg_map = self.parser.predict(pil_image)

        # Extract hair class
        hair_mask = (seg_map == HAIR_LABEL).astype(np.uint8) * 255

        # Resize if needed
        if hair_mask.shape[:2] != (h, w):
            hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return hair_mask
```

- [ ] **Step 3: Implement SegFormer segmenter**

```python
# backend/segmentation/segformer.py
import cv2
import numpy as np
import torch

from .base import HairSegmenter

# jonathandinu/face-parsing uses CelebAMask-HQ labels — hair = 13
# (17 = neck, not hair)
HAIR_LABEL = 13
MODEL_NAME = "jonathandinu/face-parsing"


class SegFormerSegmenter(HairSegmenter):
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        h, w = image.shape[:2]

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes, H/4, W/4)

        # Upsample to original size
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        parsing = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        # Extract hair class
        hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255

        return hair_mask
```

- [ ] **Step 4: Register new models in __init__.py**

Update `backend/segmentation/__init__.py`:

```python
from .base import HairSegmenter
from .bisenet import BiSeNetSegmenter
from .fashn_parser import FASHNSegmenter
from .segformer import SegFormerSegmenter

_MODELS: dict[str, type[HairSegmenter]] = {
    "bisenet": BiSeNetSegmenter,
    "fashn": FASHNSegmenter,
    "segformer": SegFormerSegmenter,
}

_INSTANCES: dict[str, HairSegmenter] = {}


def get_segmenter(model_name: str) -> HairSegmenter:
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODELS.keys())}")
    if model_name not in _INSTANCES:
        instance = _MODELS[model_name]()
        instance.load()
        _INSTANCES[model_name] = instance
    return _INSTANCES[model_name]


def available_models() -> list[str]:
    return list(_MODELS.keys())
```

- [ ] **Step 5: Install new dependency**

Run:
```bash
cd backend
source venv/bin/activate
pip install fashn-human-parser
```

- [ ] **Step 6: Smoke test each new model**

Run:
```bash
cd backend
source venv/bin/activate
python -c "
from segmentation import get_segmenter
import numpy as np

for name in ['fashn', 'segformer']:
    print(f'Testing {name}...')
    seg = get_segmenter(name)
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = seg.segment(dummy)
    print(f'  Mask shape: {mask.shape}, dtype: {mask.dtype}, range: {mask.min()}-{mask.max()}')
    print(f'  {name} OK')
"
```

Expected: Both models load and return (256, 256) uint8 masks.

- [ ] **Step 7: Commit**

```bash
git add backend/segmentation/fashn_parser.py backend/segmentation/segformer.py backend/segmentation/__init__.py backend/requirements.txt
git commit -m "feat: add FASHN and SegFormer segmentation models"
```

---

## Task 5: API Endpoints

**Files:**
- Modify: `backend/main.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

```python
# backend/tests/test_api.py
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _make_test_image(w: int = 100, h: int = 100) -> bytes:
    """Create a simple test image as PNG bytes."""
    img = np.full((h, w, 3), [120, 80, 60], dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_test_mask(w: int = 100, h: int = 100) -> bytes:
    """Create a grayscale mask as PNG bytes."""
    mask = np.full((h, w), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", mask)
    return buf.tobytes()


def _mock_segmenter():
    """Create a mock segmenter that returns an all-white mask."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend
source venv/bin/activate
python -m pytest tests/test_api.py -v
```

Expected: FAIL — `/api/models` 404, `/api/segment` 404, `/api/recolor` 404

- [ ] **Step 3: Implement API endpoints in main.py**

```python
# backend/main.py
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


@app.post("/api/recolor")
async def recolor(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    color: str = Form("#FF0000"),
    intensity: int = Form(80),
    lift: int = Form(0),
):
    # Validate inputs
    intensity = max(0, min(100, intensity))
    lift = max(0, min(40, lift))
    import re
    color = color.strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise HTTPException(status_code=400, detail=f"Invalid color hex: {color}")

    img_bytes = await image.read()
    mask_bytes = await mask.read()

    img = _decode_image(img_bytes, cv2.IMREAD_COLOR)
    mask_arr = _decode_image(mask_bytes, cv2.IMREAD_GRAYSCALE)

    result = recolor_hair(img, mask_arr, color, intensity, lift)

    _, buf = cv2.imencode(".png", result)
    return Response(content=buf.tobytes(), media_type="image/png")
```

- [ ] **Step 4: Run API tests**

Run:
```bash
cd backend
source venv/bin/activate
python -m pytest tests/test_api.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: add /api/segment and /api/recolor endpoints"
```

---

## Task 6: Frontend Project Scaffold

**Files:**
- Create: `frontend/` (via Vite scaffold)
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/types.ts`

- [ ] **Step 1: Scaffold React + Vite + TypeScript project**

Run:
```bash
cd /Users/chiikang/Desktop/OuterTech/HairColorChange
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Tailwind with Vite plugin**

Update `frontend/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

Replace `frontend/src/index.css` with:

```css
@import "tailwindcss";
```

- [ ] **Step 3: Create shared types**

```typescript
// frontend/src/types.ts
export type SegmentationModel = "bisenet" | "fashn" | "segformer";

export interface RecolorParams {
  color: string;
  intensity: number;
  lift: number;
}
```

- [ ] **Step 4: Create minimal App.tsx**

```tsx
// frontend/src/App.tsx
function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">Hair Color Change</h1>
        <p className="text-sm text-gray-400">Upload a photo and try different hair colors</p>
      </header>
      <main className="mx-auto max-w-6xl p-6">
        <p className="text-gray-500">Components coming next...</p>
      </main>
    </div>
  );
}

export default App;
```

- [ ] **Step 5: Verify frontend starts**

Run:
```bash
cd frontend
npm run dev
```

Expected: Opens on http://localhost:5173, shows "Hair Color Change" header with dark background.

- [ ] **Step 6: Commit**

```bash
cd /Users/chiikang/Desktop/OuterTech/HairColorChange
git add frontend/
git commit -m "feat: scaffold React + Vite + Tailwind frontend"
```

---

## Task 7: useHairRecolor Hook

**Files:**
- Create: `frontend/src/hooks/useHairRecolor.ts`

- [ ] **Step 1: Implement the API hook**

```typescript
// frontend/src/hooks/useHairRecolor.ts
import { useState, useCallback } from "react";
import type { SegmentationModel } from "../types";

interface HairRecolorState {
  maskUrl: string | null;
  resultUrl: string | null;
  loading: boolean;
  error: string | null;
}

export function useHairRecolor() {
  const [state, setState] = useState<HairRecolorState>({
    maskUrl: null,
    resultUrl: null,
    loading: false,
    error: null,
  });

  const processImage = useCallback(
    async (
      imageFile: File,
      model: SegmentationModel,
      color: string,
      intensity: number,
      lift: number
    ) => {
      setState((s) => ({ ...s, loading: true, error: null }));

      try {
        // Step 1: Segment
        const segForm = new FormData();
        segForm.append("image", imageFile);
        segForm.append("model", model);

        const segRes = await fetch("/api/segment", {
          method: "POST",
          body: segForm,
        });
        if (!segRes.ok) throw new Error(`Segmentation failed: ${segRes.statusText}`);

        const maskBlob = await segRes.blob();
        const maskUrl = URL.createObjectURL(maskBlob);

        // Step 2: Recolor
        const recolorForm = new FormData();
        recolorForm.append("image", imageFile);
        recolorForm.append("mask", maskBlob, "mask.png");
        recolorForm.append("color", color);
        recolorForm.append("intensity", String(intensity));
        recolorForm.append("lift", String(lift));

        const recolorRes = await fetch("/api/recolor", {
          method: "POST",
          body: recolorForm,
        });
        if (!recolorRes.ok) throw new Error(`Recoloring failed: ${recolorRes.statusText}`);

        const resultBlob = await recolorRes.blob();
        const resultUrl = URL.createObjectURL(resultBlob);

        setState({ maskUrl, resultUrl, loading: false, error: null });
      } catch (err) {
        setState((s) => ({
          ...s,
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState((s) => {
      if (s.maskUrl) URL.revokeObjectURL(s.maskUrl);
      if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
      return { maskUrl: null, resultUrl: null, loading: false, error: null };
    });
  }, []);

  return { ...state, processImage, reset };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/useHairRecolor.ts
git commit -m "feat: add useHairRecolor API hook"
```

---

## Task 8: Frontend Components

**Files:**
- Create: `frontend/src/components/ImageUploader.tsx`
- Create: `frontend/src/components/ModelSelector.tsx`
- Create: `frontend/src/components/ColorPicker.tsx`
- Create: `frontend/src/components/ControlPanel.tsx`
- Create: `frontend/src/components/ResultViewer.tsx`

- [ ] **Step 1: ImageUploader component**

```tsx
// frontend/src/components/ImageUploader.tsx
import { useCallback, useRef } from "react";

interface Props {
  onImageSelected: (file: File, previewUrl: string) => void;
  previewUrl: string | null;
}

export function ImageUploader({ onImageSelected, previewUrl }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      className="cursor-pointer rounded-xl border-2 border-dashed border-gray-700 p-8 text-center transition hover:border-gray-500"
    >
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="Uploaded"
          className="mx-auto max-h-80 rounded-lg object-contain"
        />
      ) : (
        <div className="space-y-2 text-gray-400">
          <p className="text-lg">Drop an image here or click to upload</p>
          <p className="text-sm">JPG, PNG supported</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
```

- [ ] **Step 2: ModelSelector component**

```tsx
// frontend/src/components/ModelSelector.tsx
import type { SegmentationModel } from "../types";

interface Props {
  value: SegmentationModel;
  onChange: (model: SegmentationModel) => void;
}

const MODELS: { value: SegmentationModel; label: string; desc: string }[] = [
  { value: "bisenet", label: "BiSeNet", desc: "Proven classic, fast" },
  { value: "fashn", label: "FASHN", desc: "Fashion-optimized, full body" },
  { value: "segformer", label: "SegFormer", desc: "High quality face parsing" },
];

export function ModelSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        Segmentation Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as SegmentationModel)}
        className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-white"
      >
        {MODELS.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label} — {m.desc}
          </option>
        ))}
      </select>
    </div>
  );
}
```

- [ ] **Step 3: ColorPicker component**

```tsx
// frontend/src/components/ColorPicker.tsx
interface Props {
  value: string;
  onChange: (color: string) => void;
}

const PRESETS = [
  { color: "#D4A574", label: "Honey Blonde" },
  { color: "#8B4513", label: "Auburn" },
  { color: "#FF4444", label: "Red" },
  { color: "#FF69B4", label: "Pink" },
  { color: "#9B59B6", label: "Purple" },
  { color: "#3498DB", label: "Blue" },
  { color: "#2ECC71", label: "Green" },
  { color: "#F1C40F", label: "Gold" },
  { color: "#E0E0E0", label: "Platinum" },
  { color: "#1A1A1A", label: "Black" },
];

export function ColorPicker({ value, onChange }: Props) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        Hair Color
      </label>
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.color}
            title={p.label}
            onClick={() => onChange(p.color)}
            className={`h-8 w-8 rounded-full border-2 transition ${
              value === p.color ? "border-white scale-110" : "border-gray-600"
            }`}
            style={{ backgroundColor: p.color }}
          />
        ))}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-8 cursor-pointer rounded border-0 bg-transparent"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-24 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-sm text-white"
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: ControlPanel component**

```tsx
// frontend/src/components/ControlPanel.tsx
interface Props {
  intensity: number;
  lift: number;
  onIntensityChange: (v: number) => void;
  onLiftChange: (v: number) => void;
}

export function ControlPanel({
  intensity,
  lift,
  onIntensityChange,
  onLiftChange,
}: Props) {
  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1 flex justify-between text-sm font-medium text-gray-300">
          <span>Color Intensity</span>
          <span className="text-gray-500">{intensity}%</span>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
      <div>
        <label className="mb-1 flex justify-between text-sm font-medium text-gray-300">
          <span>Dark Hair Lift</span>
          <span className="text-gray-500">{lift}%</span>
        </label>
        <input
          type="range"
          min={0}
          max={40}
          value={lift}
          onChange={(e) => onLiftChange(Number(e.target.value))}
          className="w-full"
        />
        <p className="mt-1 text-xs text-gray-500">
          Brightens dark hair so color shows better
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: ResultViewer component**

```tsx
// frontend/src/components/ResultViewer.tsx
import { useState } from "react";

interface Props {
  originalUrl: string | null;
  resultUrl: string | null;
  maskUrl: string | null;
}

type ViewMode = "result" | "before-after" | "mask";

export function ResultViewer({ originalUrl, resultUrl, maskUrl }: Props) {
  const [mode, setMode] = useState<ViewMode>("result");

  if (!resultUrl || !originalUrl) return null;

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {(["result", "before-after", "mask"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-lg px-3 py-1 text-sm ${
              mode === m
                ? "bg-white text-black"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {m === "before-after" ? "Before / After" : m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <div className="overflow-hidden rounded-xl border border-gray-800">
        {mode === "result" && (
          <img src={resultUrl} alt="Result" className="w-full object-contain" />
        )}
        {mode === "before-after" && (
          <div className="grid grid-cols-2 gap-1">
            <img src={originalUrl} alt="Before" className="w-full object-contain" />
            <img src={resultUrl} alt="After" className="w-full object-contain" />
          </div>
        )}
        {mode === "mask" && maskUrl && (
          <img src={maskUrl} alt="Hair mask" className="w-full object-contain" />
        )}
      </div>

      <a
        href={resultUrl}
        download="hair-recolored.png"
        className="inline-block rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-gray-200"
      >
        Download Result
      </a>
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: add all frontend UI components"
```

---

## Task 9: Wire Everything Together in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Assemble App.tsx with all components**

```tsx
// frontend/src/App.tsx
import { useState, useCallback } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { ModelSelector } from "./components/ModelSelector";
import { ColorPicker } from "./components/ColorPicker";
import { ControlPanel } from "./components/ControlPanel";
import { ResultViewer } from "./components/ResultViewer";
import { useHairRecolor } from "./hooks/useHairRecolor";
import type { SegmentationModel } from "./types";

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [model, setModel] = useState<SegmentationModel>("bisenet");
  const [color, setColor] = useState("#D4A574");
  const [intensity, setIntensity] = useState(80);
  const [lift, setLift] = useState(0);

  const { maskUrl, resultUrl, loading, error, processImage, reset } =
    useHairRecolor();

  const handleImageSelected = useCallback((file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
  }, []);

  const handleProcess = useCallback(() => {
    if (!imageFile) return;
    processImage(imageFile, model, color, intensity, lift);
  }, [imageFile, model, color, intensity, lift, processImage]);

  const handleReset = useCallback(() => {
    setImageFile(null);
    setPreviewUrl(null);
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">Hair Color Change</h1>
        <p className="text-sm text-gray-400">
          Upload a photo and try different hair colors
        </p>
      </header>

      <main className="mx-auto max-w-6xl p-6">
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          {/* Sidebar controls */}
          <div className="space-y-6">
            <ModelSelector value={model} onChange={setModel} />
            <ColorPicker value={color} onChange={setColor} />
            <ControlPanel
              intensity={intensity}
              lift={lift}
              onIntensityChange={setIntensity}
              onLiftChange={setLift}
            />

            <button
              onClick={handleProcess}
              disabled={!imageFile || loading}
              className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? "Processing..." : "Apply Color"}
            </button>

            {resultUrl && (
              <button
                onClick={handleReset}
                className="w-full rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-300 transition hover:bg-gray-800"
              >
                Start Over
              </button>
            )}

            {error && (
              <p className="rounded-lg bg-red-900/30 px-3 py-2 text-sm text-red-400">
                {error}
              </p>
            )}
          </div>

          {/* Main content area */}
          <div className="space-y-6">
            {!resultUrl && (
              <ImageUploader
                onImageSelected={handleImageSelected}
                previewUrl={previewUrl}
              />
            )}
            <ResultViewer
              originalUrl={previewUrl}
              resultUrl={resultUrl}
              maskUrl={maskUrl}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
```

- [ ] **Step 2: Verify frontend compiles**

Run:
```bash
cd frontend
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire up all components in App.tsx"
```

---

## Task 10: End-to-End Integration Test

**Files:** None (manual verification)

- [ ] **Step 1: Start the backend**

Run (in terminal 1):
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend**

Run (in terminal 2):
```bash
cd frontend
npm run dev
```

- [ ] **Step 3: Test the full flow**

1. Open http://localhost:5173
2. Upload a photo with visible hair
3. Select "BiSeNet" model
4. Pick a color (e.g., Red)
5. Set intensity to 80%, lift to 0%
6. Click "Apply Color"
7. Verify:
   - Mask shows hair region highlighted in white
   - Result shows recolored hair
   - Before/After view works
   - Download button saves the result
8. Try the same image with "FASHN" and "SegFormer" models
9. Compare mask quality across models

- [ ] **Step 4: Run all backend tests**

Run:
```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete hair color change prototype with multi-model support"
```
