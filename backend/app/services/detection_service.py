import asyncio
import logging
import time

from ..core.exceptions import AppError
from ..models.detection import Detection, ModelType
from ..repositories.detection import DetectionRepository
from ..schemas.detection import DetectionParams
from .detection_strategy import create_strategy
from .yolo_format import _canonical_class_name

logger = logging.getLogger(__name__)


async def process_detection(
    filepath: str,
    original_name: str,
    categories: list[str],
    params: DetectionParams,
    repo: DetectionRepository,
    replace_detection_id: str | None = None,
) -> Detection:
    """Orchestrates model offloading, inference strategy, and database persistence."""

    # 1. Setup the single supported automatic annotation strategy.
    strategy = create_strategy()
    strategy_kwargs = {
        "grounded_sam_text": params.grounded_sam_text,
        "use_grounded_sam_seg": params.use_grounded_sam_seg,
        "grounded_sam_threshold": params.grounded_sam_threshold,
        "grounded_sam_mask_threshold": params.grounded_sam_mask_threshold,
    }

    t0 = time.perf_counter()

    # 3. Execute inference
    try:
        result = await asyncio.to_thread(strategy.detect, filepath, categories, **strategy_kwargs)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise AppError("Inference failed", 500) from exc

    # 4. Transactional persistence
    model_type = ModelType.grounded_sam

    detection = repo.get_by_id(replace_detection_id) if replace_detection_id else None
    if detection is None:
        detection = repo.create(
            image_path=filepath,
            image_name=original_name,
            image_width=result.img_w,
            image_height=result.img_h,
            categories=categories,
            model_name="Grounded-SAM",
        )
    else:
        detection.image_width = result.img_w
        detection.image_height = result.img_h
        detection.categories = categories
        detection.model_name = "Grounded-SAM"
    detection.model_type = model_type
    detection.elapsed_ms = int((time.perf_counter() - t0) * 1000)

    polys = result.polygons
    box_dicts: list[dict] = [
        {
            "class_name": _canonical_class_name(
                b.get("class_name") or categories[0] or "object",
                categories,
            ),
            "x1": b["x1"],
            "y1": b["y1"],
            "x2": b["x2"],
            "y2": b["y2"],
            "confidence": b.get("confidence"),
            "mask_polygon": polys[i] if i < len(polys) else None,
        }
        for i, b in enumerate(result.boxes)
    ]

    if replace_detection_id:
        repo.replace_boxes(str(detection.id), box_dicts)
    else:
        repo.add_boxes(str(detection.id), box_dicts)
    repo.db.commit()
    repo.db.refresh(detection)

    return detection
