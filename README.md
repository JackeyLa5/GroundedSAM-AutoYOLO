# AutoYOLO

[简体中文](README_ZH.md) | English

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
  <img src="https://img.shields.io/badge/Node.js-22+-green" alt="Node.js">
  <img src="https://img.shields.io/badge/Platform-Ubuntu%20x86-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-orange" alt="GPU">
  <img src="https://img.shields.io/github/stars/JackeyLa5/GroundedSAM-AutoYOLO?style=social" alt="Stars">
</p>

```
🖼️ image/video → 🔍 Grounded-SAM detection → 🎯 instance masks → ✏️ refine → 📦 export → 🚀 YOLO → ✅ model
```

**Images or videos in → YOLO model out**, with Grounded-SAM auto-labeling (GroundingDINO + SAM2) and human-in-the-loop correction. Multi-format export, one-click YOLO training (detect & segment), video keyframe extraction, and model validation are packaged for Ubuntu x86 + NVIDIA CUDA Docker deployment.

This project is adapted from the original [VLM-AutoYOLO](https://github.com/Somnusochi/VLM-AutoYOLO) workflow. The main change is replacing the original VLM backend with Grounded-SAM / GroundingDINO + SAM2 to reduce GPU memory requirements while keeping the automatic YOLO dataset generation workflow.

![Architecture](docs/architecture_en.webp)

> See [Architecture & Workflow Documentation](docs/architecture_diagram_en.md) for detailed Mermaid diagrams.

## Key Features
- 🤖 **Grounded-SAM auto-labeling**: Open-vocabulary text detection with GroundingDINO followed by SAM2 instance segmentation
- 🎯 **Instance segmentation**: Bbox and pixel-precise mask output in one Grounded-SAM workflow, with independent BBox/Mask canvas toggles
- 🎥 **Video annotation**: Intelligent keyframe extraction (scene / motion / interval), SSIM dedup
- ✏️ **Manual refinement**: Canvas draw mode, NMS filtering, hide/show individual boxes
- 📦 **Multi-format export/import**: YOLO, YOLO-Seg, COCO JSON, Pascal VOC XML, CreateML JSON — import datasets via chunked ZIP upload (max 10GB, resume support)
- 🚀 **Training queue**: Sequential job processing with cancel support, one-click training (YOLOv8 / v11 / v26) with real-time SSE progress
- ✅ **Model validation**: Batch image / video testing, MJPEG live stream, SSE video inference
- 💾 **Smart model management**: Lazy loading, idle auto-unload, CUDA memory cleanup
- 🌐 **i18n**: English / 简体中文 / 日本語 · 🎨 **Theme**: Light / dark mode

## Built for Vision Data Pipelines

AutoYOLO is designed to close the loop for vision datasets: collect images from a camera or video source, auto-label them with Grounded-SAM, review/correct the results in the browser, and train a YOLO detection or segmentation model — all without leaving the app.

## Documentation

📚 **[User Guide (English)](docs/guide/en/README.md)** | 📚 **[用户指南 (中文)](docs/guide/README.md)**

Comprehensive guides: quick start, annotation best practices, training parameter tuning, model deployment.

## Screenshots

| Grounded-SAM Annotation & Refinement | YOLO Training |
|--------------------------------|---------------|
| ![Grounded-SAM annotation and refinement](docs/1.webp) | ![YOLO training](docs/2.webp) |

| Video Keyframe Entry | Model Validation |
|---------------------|-----------------|
| ![Video keyframe entry](docs/4.webp) | ![Model validation](docs/3.webp) |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection and Segmentation | GroundingDINO + SAM2 (Grounded-SAM) |
| Object Detection | YOLOv8 / v11 / v26 — Detect & Segment (Ultralytics) |
| Backend | Python FastAPI + PostgreSQL + SSE |
| Frontend | React + TypeScript + Vite + Tailwind CSS + antd |
| GPU Memory | CUDA expandable segments / empty_cache |
| State | Zustand + TanStack Query + ahooks |
| i18n | i18next (English / 简体中文 / 日本語) |
| Video | ffmpeg (scene / motion / interval extraction) |
| Tooling | pnpm, ESLint, Prettier, Husky, commitlint, Playwright |

## Quick Start

### Docker Deployment

> **Requirements:** Ubuntu x86_64, NVIDIA GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Docker, and Node.js 22+ / pnpm.

Run this once:

```bash
git clone https://github.com/JackeyLa5/GroundedSAM-AutoYOLO.git && cd GroundedSAM-AutoYOLO && cd frontend && pnpm install && pnpm build && cd .. && docker compose -f docker/docker-compose.yml up -d --build db backend frontend
```

Open:

```bash
http://localhost/GroundedSAM-AutoYOLO/
```

If you change frontend code later, rerun the `cd frontend && pnpm install && pnpm build` part before restarting the stack.

**Services:**

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 80 | React web UI (Nginx) |
| Backend | 8000 | FastAPI server |
| Grounded-SAM | 8002 | Grounded-SAM standalone inference service |
| Database | 5432 | PostgreSQL |

**GPU Support** — `docker/docker-compose.yml` includes NVIDIA GPU passthrough configuration and the backend image installs Grounded-SAM2 dependencies by default.

**Persistent Storage (Docker volumes):**
- `pgdata` — Database · `hf-cache` — Hugging Face cache for Grounded-SAM models · `uploads` — User images/videos · `training-data` — YOLO training outputs

**Backup / Restore:**

```bash
docker compose -f docker/docker-compose.yml exec db pg_dump -U postgres autolabeling > backup.sql
cat backup.sql | docker compose -f docker/docker-compose.yml exec -T db psql -U postgres autolabeling
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost/GroundedSAM-AutoYOLO/ |
| Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Project Structure

Full directory tree: **[docs/STRUCTURE.md](docs/STRUCTURE.md)**

## Features

### Grounded-SAM Detection and Segmentation

Enter open-vocabulary categories (e.g. `cat`, `red car`). Grounded-SAM runs GroundingDINO text detection and SAM2 instance segmentation as one annotation workflow.

- **Confidence threshold** slider (0.0–1.0, default 0.5) controls detection sensitivity
- **Mask threshold** slider (0.0–1.0, default 0.5) controls mask tightness
- Enable/disable segmentation independently — bbox-only mode skips mask extraction for faster results
- Grounded-SAM runs as a standalone HTTP service on port 8002 with its own venv (`backend/grounded-sam2-venv/`)
- Uses public GroundingDINO + SAM2 defaults; model files are cached in `~/.cache/huggingface/hub/` after first download
- Auto-starts on first use, idle auto-unload after 10 min
- Real-time loading status via SSE (`starting` → `loading` → `loaded`)
- Manual unload button to free GPU memory
- Detection records tagged with `model_type` for traceability

### Video Annotation

Upload a video, extract keyframes, select and batch-annotate.

- **Three extraction modes**: scene change, motion detection (optical flow), fixed interval
- **SSIM deduplication**: auto-removes near-duplicate frames
- **Timeline preview**: horizontal scrollable strip, click for full-size view
- **Multi-select**: check frames, select/cancel all, load to annotation queue

### Manual Annotation

Canvas-based annotation with View / Draw modes.

- Category quick-fill from history
- Grounded-SAM pre-annotation baseline → delete mistakes → draw missing boxes
- All / Best / NMS filter modes, settings saved per detection
- Hide individual boxes while inspecting dense results
- Per-frame re-detection

### History Management

- Thumbnail + category tag previews, tag-based multi-select filtering
- Click to view details, re-detect with updated labels, virtual scroll with infinite loading
- Single / batch export in **5 formats**: YOLO, YOLO-Seg, COCO JSON, Pascal VOC XML, CreateML JSON
- Format selection via dropdown menu, one-click zip download

### YOLO Training

- **Series**: YOLOv8 / v11 / v26 (n/s/m/l/x)
- **Task types**: Object Detection (Detect), Instance Segmentation (Segment)
- Segmentation training uses Grounded-SAM polygon labels and falls back to bbox when unavailable
- Tag filter + thumbnail preview for precise data selection
- Virtual scroll with "Load All" button for large datasets
- Dataset split presets (70/20/10, 80/20, 90/10, 60/20/20)
- Real-time SSE progress: Epoch / Loss / mAP50
- Rename training jobs for easier identification
- Auto ONNX export; download PT / ONNX / dataset zip

### Model Validation

- **Dual source**: trained models or externally uploaded `.pt` files
- **Conf / IoU sliders** for real-time threshold tuning
- **Batch image validation** with bounding boxes and confidence scores
- **Video validation** (three modes):
  - MJPEG live stream with interactive play/pause
  - SSE prediction stream with per-frame JSON events
  - Sync batch prediction — all frames at once
- Temporary results; export predictions as YOLO `.txt` files

### Model Management

- **Lazy loading**: Grounded-SAM loads on first use and unloads after idle (default 10 min)
- **Idle watchdog**: Grounded-SAM unloads after `MODEL_IDLE_TIMEOUT_SECONDS` of inactivity
- **SSE status**: `GET /api/v1/model/events` streams the Grounded-SAM service state
- **Manual unload**: the Grounded-SAM service has an unload endpoint
- **GPU memory**: CUDA `expandable_segments` / `empty_cache`
- **Transparent load errors**: model load/inference failures are returned to the UI instead of being hidden behind generic detection errors

### Troubleshooting: Stuck at `Loading`

If Grounded-SAM remains at `starting` or `loading`, inspect the service log and backend logs. Typical causes are CUDA OOM, missing NVIDIA runtime, driver mismatch, inaccessible model files, or a dependency import error.

Useful Docker checks:

```bash
docker compose -f docker/docker-compose.yml logs backend --tail=200
docker compose -f docker/docker-compose.yml exec backend python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mem:", torch.cuda.mem_get_info() if torch.cuda.is_available() else None)
PY
nvidia-smi
```

## API Reference

Full API documentation with request/response examples: **[docs/API.md](docs/API.md)**

## Runtime Platform

The supported deployment target is Ubuntu x86_64 + NVIDIA CUDA + Docker. Grounded-SAM and YOLO training both run inside containers.

## Highlights

- **CUDA full-pipeline GPU acceleration** — Grounded-SAM and YOLO training use NVIDIA GPUs
- **CUDA GPU memory management** — `expandable_segments:True`
- **Grounded-SAM annotation** — GroundingDINO text detection and SAM2 instance segmentation in one workflow
- **5 export formats** — YOLO, YOLO-Seg, COCO, Pascal VOC, CreateML
- **Detect & Segment training** — polygon labels auto-used when Grounded-SAM masks are available
- **Docker-only deployment** — targeted at Ubuntu x86_64 + NVIDIA CUDA
- **Grounded-SAM SSE status** — a single EventSource reports service state without polling

## Development

```bash
# Frontend
cd frontend && pnpm install && pnpm run lint && pnpm run build

# Backend
cd backend && source .venv/bin/activate
PYTHONPATH=. alembic upgrade head
python -m compileall app alembic
```

## Stargazers

[![Star History Chart](https://api.star-history.com/svg?repos=JackeyLa5/GroundedSAM-AutoYOLO&type=Date)](https://star-history.com/#JackeyLa5/GroundedSAM-AutoYOLO&Date)

## License

Code: [AGPL-3.0](LICENSE).

Third-party dependencies:
- GroundingDINO / SAM2 models — see their upstream model licenses
- Ultralytics YOLO — [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (copyleft; training/deployment may trigger obligations)

---

If this project helps you, please ⭐ [star it on GitHub](https://github.com/JackeyLa5/GroundedSAM-AutoYOLO).
