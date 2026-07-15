# AutoYOLO Architecture Design Diagram

AutoYOLO is a full-stack platform for Grounded-SAM auto-labeling, human refinement, and YOLO training. The annotation path is unified: GroundingDINO performs text-conditioned detection and SAM2 performs instance segmentation.

```mermaid
graph TD
    User((User)) --> Frontend["Frontend<br/>Vite + React + TypeScript"]
    Frontend --> API["FastAPI API"]
    API --> Detection["Detection Orchestration"]
    Detection --> GroundedSAM["Grounded-SAM Service<br/>GroundingDINO + SAM2<br/>grounded_sam2_server.py:8002"]
    Detection --> Storage[("PostgreSQL + File System")]
    API --> YOLO["YOLO Training & Validation"]
    YOLO --> Storage
    GroundedSAM --> Cache[("Hugging Face Model Cache")]
```

## Architecture Details

### Frontend

The React single-page application handles image, video, and keyframe uploads; Grounded-SAM settings; Canvas annotation refinement; dataset export; and YOLO training and validation.

### Backend

FastAPI routes receive requests, the detection service orchestrates Grounded-SAM inference and persists results, and repositories with SQLAlchemy handle structured data access. Media, annotations, and training outputs are stored on the file system.

### Grounded-SAM Service

`backend/grounded_sam2_server.py` runs as a standalone HTTP service. GroundingDINO generates bounding boxes from open-vocabulary text prompts, and SAM2 generates instance masks from those detections. Disabling segmentation returns boxes only; it does not select another model stack.

### YOLO Loop

Human-refined bounding boxes and polygons can be exported in multiple formats and used for YOLOv8, YOLOv11, or YOLOv26 detection/segmentation training and validation.

## Core Workflow

```mermaid
sequenceDiagram
    actor User
    participant UI as Frontend
    participant API as FastAPI
    participant GS as Grounded-SAM
    participant DB as Database/File System
    participant YOLO as YOLO Training & Validation

    User->>UI: Upload images, videos, or keyframes
    UI->>API: Enter an open-vocabulary prompt and start labeling
    API->>GS: GroundingDINO detection
    GS-->>API: Return bounding boxes
    opt Segmentation enabled
        API->>GS: SAM2 instance segmentation
        GS-->>API: Return masks/polygons
    end
    API-->>UI: Render Grounded-SAM annotations
    User->>UI: Add, delete, edit, and save annotations
    UI->>API: Export a dataset or create a training job
    API->>DB: Persist annotations and training data
    API->>YOLO: Train or validate
    YOLO-->>UI: Return progress and predictions
```
