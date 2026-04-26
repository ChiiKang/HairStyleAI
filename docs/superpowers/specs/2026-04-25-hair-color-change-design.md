# Hair Color Change Prototype — Design Spec

## Goal

Build an open-source static-image hair color changer with a React frontend and Python FastAPI backend. Support multiple segmentation models for comparison. Prioritize accuracy over speed.

## Architecture

```
┌─────────────────────────────┐
│   React + Vite Frontend     │
│                             │
│  - Image upload             │
│  - Color picker / presets   │
│  - Model selector dropdown  │
│  - Intensity slider         │
│  - Before/after viewer      │
│  - Download result          │
│  - Hair mask preview        │
└──────────┬──────────────────┘
           │ HTTP (multipart/form-data)
           ▼
┌─────────────────────────────┐
│   FastAPI Backend           │
│                             │
│  POST /api/segment          │
│    - model: bisenet|fashn|  │
│      segformer              │
│    - image: uploaded file   │
│    → returns: hair mask PNG │
│                             │
│  POST /api/recolor          │
│    - image: original        │
│    - mask: hair mask        │
│    - color: hex color       │
│    - intensity: 0-100       │
│    - lift: 0-40 (dark hair) │
│    → returns: recolored PNG │
└─────────────────────────────┘
```

## Segmentation Models (pluggable)

### 1. BiSeNet face-parsing (default)
- Repo: `zllrunning/face-parsing.pytorch`
- Proven, widely used, MIT license
- Hair class = label 13 (CelebAMask-HQ label map)
- PyTorch, pretrained on CelebAMask-HQ

### 2. FASHN Human Parser
- Package: `pip install fashn-human-parser`
- SegFormer-B4 with 18 classes including hair
- Optimized for fashion/virtual try-on
- Good for full-body photos

### 3. jonathandinu SegFormer face-parsing
- HuggingFace: `jonathandinu/face-parsing`
- SegFormer fine-tuned on CelebAMask-HQ
- Has ONNX variant for potential browser use later

## Recoloring Pipeline

```
original image + hair mask
→ convert to LAB color space
→ preserve L (lightness) channel
→ for dark hair: apply controlled lift (0-40%) to L channel in masked area
→ compute single blend factor = intensity * mask (avoids double-masking)
→ replace A/B channels with target color's A/B, weighted by blend factor
→ convert back to BGR
→ output recolored image
```

### Mask Cleanup (applied before recoloring)
- Threshold raw mask probability
- Remove small islands (< 100px)
- Fill small holes
- Gaussian blur edges (feathering)

## Frontend Components

1. **ImageUploader** — drag-and-drop or click to upload
2. **ModelSelector** — dropdown: BiSeNet / FASHN / SegFormer
3. **ColorPicker** — preset swatches + custom hex input
4. **IntensitySlider** — 0-100% color strength
5. **LiftSlider** — 0-40% brightness lift for dark hair
6. **BeforeAfterViewer** — side-by-side or slider comparison
7. **MaskPreview** — toggle to see the raw hair mask
8. **DownloadButton** — save result as PNG

## API Contracts

### POST /api/segment
```json
Request: multipart/form-data
  - image: file
  - model: "bisenet" | "fashn" | "segformer"

Response: 200
  - Content-Type: image/png (grayscale mask)
```

### POST /api/recolor
```json
Request: multipart/form-data
  - image: file (original)
  - mask: file (hair mask PNG)
  - color: string (hex, e.g. "#FF4444")
  - intensity: int (0-100)
  - lift: int (0-40)

Response: 200
  - Content-Type: image/png (recolored image)
```

## Tech Stack

- **Frontend:** React 18 + Vite + TypeScript + Tailwind CSS
- **Backend:** Python 3.10+ + FastAPI + Uvicorn
- **Image processing:** OpenCV, Pillow, NumPy
- **ML:** PyTorch, transformers (HuggingFace)
- **Dev:** CORS middleware for local dev

## Project Structure

```
HairColorChange/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── ColorPicker.tsx
│   │   │   ├── IntensitySlider.tsx
│   │   │   ├── LiftSlider.tsx
│   │   │   ├── BeforeAfterViewer.tsx
│   │   │   ├── MaskPreview.tsx
│   │   │   └── DownloadButton.tsx
│   │   ├── hooks/
│   │   │   └── useHairRecolor.ts
│   │   └── types.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── backend/
│   ├── main.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── base.py          (abstract base)
│   │   ├── bisenet.py
│   │   ├── fashn_parser.py
│   │   └── segformer.py
│   ├── recolor/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── requirements.txt
│   └── models/              (downloaded weights)
├── docs/
├── test_images/
└── hairchangeplan.md
```

## Non-Goals (for prototype)

- No video/real-time processing
- No user authentication
- No database
- No ombre/gradient effects (later)
- No SAM 2 / manual mask correction (later)
- No browser-only MediaPipe mode (later)
