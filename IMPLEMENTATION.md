# Hair Color Change Prototype — Implementation Summary

## Overview

An open-source static-image hair color changer with a React frontend and Python FastAPI backend. Supports multiple segmentation models for comparison and three recoloring methods with live parameter adjustment.

**Stack:** React 18 + Vite + TypeScript + Tailwind CSS | FastAPI + PyTorch + OpenCV

---

## Architecture

```
┌─────────────────────────────────┐
│     React + Vite Frontend       │
│                                 │
│  Upload → Segment → Recolor     │
│  (mask cached after segment)    │
│  Live slider/color adjustment   │
└──────────┬──────────────────────┘
           │ HTTP (multipart/form-data)
           ▼
┌─────────────────────────────────┐
│     FastAPI Backend             │
│                                 │
│  GET  /api/health               │
│  GET  /api/models               │
│  GET  /api/recolor-methods      │
│  POST /api/segment              │
│  POST /api/recolor              │
└─────────────────────────────────┘
```

### Key Design Decision: Separated Segmentation + Recoloring

- **Segmentation** (slow, ~1-3s): Model inference, runs once per image/model change
- **Recoloring** (fast, ~50ms): Pure math in LAB color space, re-runs live on parameter changes

The frontend caches the hair mask after segmentation. When the user adjusts color, intensity, lift, or method, only the recolor endpoint is called — no re-segmentation. Requests are debounced at 150ms and auto-aborted if superseded.

---

## Segmentation Models

Three pluggable models via an abstract `HairSegmenter` base class:

| Model | Source | Hair Label | Best For |
|-------|--------|-----------|----------|
| **BiSeNet** | `zllrunning/face-parsing.PyTorch` | 13 (CelebAMask-HQ) | Fast, proven baseline |
| **FASHN** | `fashn-human-parser` PyPI package | 2 (ATR convention) | Full-body, fashion photos |
| **SegFormer** | `jonathandinu/face-parsing` HuggingFace | 13 (CelebAMask-HQ) | High quality face parsing |

Models are lazily loaded on first use and cached in memory. Weights are auto-downloaded (BiSeNet via gdown from Google Drive, SegFormer from HuggingFace Hub, FASHN from PyPI package).

### Mask Cleanup Pipeline

Raw model output goes through:
1. Binary threshold
2. Remove small islands (< 100px connected components)
3. Fill small holes (inverse connected components)
4. Gaussian blur edge feathering

---

## Recoloring Methods

All methods work in **CIELAB color space** to separate luminance (L) from chrominance (A/B).

### 1. Reinhard Color Transfer (default)

Based on Reinhard et al. "Color Transfer between Images" (2001).

```
For each chrominance channel (A, B):
  1. Compute mean and std of hair region
  2. Normalize: (pixel - src_mean) / src_std
  3. Scale to target distribution: * target_std + target_mean
  4. Blend with original using intensity * mask
```

**Why it works:** Preserves the natural highlight/shadow variation in hair. Lighter strands stay lighter, darker strands stay darker. The entire color distribution shifts naturally.

### 2. Relative Shift

```
shift = target_AB - mean_hair_AB
new_AB = original_AB + shift * intensity * mask
```

Simpler than Reinhard. Moves the entire color distribution by a fixed offset. Good results, preserves relative differences between pixels.

### 3. Overlay (original method)

```
new_AB = original_AB * (1 - blend) + target_AB * blend
```

Absolute replacement — every pixel gets the same target color weighted by intensity. Looks flat and painted-on. Kept for comparison purposes.

### Dark Hair Lift

All methods support a brightness lift (0-40%) for dark hair:
- Only affects pixels with L < 100 (dark areas)
- Applied before color transfer
- Controlled via slider in the UI

---

## Project Structure

```
HairColorChange/
├── backend/
│   ├── main.py                      # FastAPI app, endpoints, validation
│   ├── requirements.txt             # Python dependencies
│   ├── segmentation/
│   │   ├── base.py                  # Abstract HairSegmenter
│   │   ├── bisenet.py               # BiSeNet (CelebAMask-HQ, label 13)
│   │   ├── fashn_parser.py          # FASHN Human Parser (ATR, label 2)
│   │   ├── segformer.py             # SegFormer face-parsing (label 13)
│   │   └── __init__.py              # Model registry + factory
│   ├── recolor/
│   │   ├── pipeline.py              # clean_mask + 3 recolor methods
│   │   └── __init__.py
│   ├── tests/
│   │   ├── test_recolor.py          # 15 tests (mask cleanup + 3 methods)
│   │   ├── test_api.py              # 6 tests (endpoints + validation)
│   │   └── __init__.py
│   ├── third_party/
│   │   └── face_parsing/            # Vendored BiSeNet architecture
│   └── models/                      # Downloaded weights (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main layout, state, auto-recolor
│   │   ├── types.ts                 # SegmentationModel, RecolorMethod
│   │   ├── hooks/
│   │   │   └── useHairRecolor.ts    # Separated segment/recolor, mask caching
│   │   └── components/
│   │       ├── ImageUploader.tsx     # Drag-and-drop upload
│   │       ├── ModelSelector.tsx     # Segmentation model dropdown
│   │       ├── ColorPicker.tsx       # Preset swatches + custom hex
│   │       ├── ControlPanel.tsx      # Intensity + lift sliders
│   │       └── ResultViewer.tsx      # Result / Before-After / Mask views
│   ├── vite.config.ts               # Tailwind plugin + /api proxy
│   └── package.json
├── docs/
│   └── superpowers/
│       ├── specs/                    # Design spec
│       └── plans/                    # Implementation plan
└── hairchangeplan.md                # Original research document
```

---

## API Reference

### GET /api/health
Returns `{"status": "ok"}`.

### GET /api/models
Returns `{"models": ["bisenet", "fashn", "segformer"]}`.

### GET /api/recolor-methods
Returns `{"methods": ["reinhard", "shift", "overlay"], "default": "reinhard"}`.

### POST /api/segment
Segments hair from an uploaded image.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| image | file | required | Input image (JPG/PNG) |
| model | string | "bisenet" | Segmentation model name |

Returns: PNG grayscale mask (255 = hair, 0 = not hair).

### POST /api/recolor
Recolors hair using a mask and target color.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| image | file | required | Original image |
| mask | file | required | Hair mask PNG |
| color | string | "#FF0000" | Target color hex |
| intensity | int | 80 | Color strength 0-100 |
| lift | int | 0 | Dark hair brightness lift 0-40 |
| method | string | "reinhard" | Recolor method |

Returns: PNG recolored image.

### Input Validation
- Unknown model → 400
- Invalid hex color → 400
- Undecoded image → 400
- Unknown recolor method → 400
- Intensity/lift clamped to valid ranges

---

## Test Coverage

**21 tests total** (all passing):

### Recolor Pipeline (15 tests)
- Mask cleanup: island removal, hole filling
- All 3 methods: preserves unmasked areas, changes masked area, zero intensity = no-op
- Reinhard-specific: preserves luminance variation
- Overlay vs Reinhard: overlay produces less variation (proves Reinhard is better)
- Shift-specific: preserves relative pixel differences
- Lift: dark pixels get brighter with lift applied

### API Endpoints (6 tests)
- Health check, model listing
- Segment returns PNG (mocked model)
- Unknown model → 400
- Recolor returns PNG
- Bad hex color → 400

---

## Running the App

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Running Tests
```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

---

## Verification History

The implementation was verified by both Claude and OpenAI Codex:

1. **Claude code-reviewer**: 9 issues found across 2 review rounds, all resolved
2. **OpenAI Codex**: 7 fixes verified pass/fail, 2 additional issues found, all resolved

Key fixes applied during review:
- Hair labels corrected: BiSeNet/SegFormer use label 13 (not 17)
- FASHN API: `FashnHumanParser().predict()` (not `.parse()`)
- BiSeNet weights: gdown from Google Drive (GitHub releases URL was broken)
- Double-masking bug: removed redundant final composite in recolor pipeline
- Hex validation: regex `#[0-9a-fA-F]{6}` (not just length check)
- API tests: mocked model loading, added validation tests

---

## Future Improvements

From the original research plan, these are next priorities:

1. **Ombre / gradient coloring** — different color from root to tip
2. **SAM 2 manual mask correction** — for hairstyles where auto-segmentation fails
3. **Alpha matting refinement** — better soft edges for curls, afros, flyaways
4. **MediaPipe browser-only mode** — no backend needed for quick demos
5. **Test set evaluation** — 30-50 diverse hair photos with scoring matrix
6. **Batch comparison** — same photo across all models side by side
