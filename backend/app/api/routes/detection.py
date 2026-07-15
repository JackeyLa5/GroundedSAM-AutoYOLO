from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...core.config import settings
from ...core.exceptions import AppError, NotFoundError
from ...models.detection import FilterMode
from ...repositories.detection import DetectionRepository
from ...schemas.common import APIResponse, BaseSchema
from ...schemas.detection import DetectionBoxOut, DetectionListItem, DetectionOut, DetectionParams
from ...services.detection_service import process_detection
from ...services.grounded_sam_client import (
    GROUNDED_SAM_URL,
    is_grounded_sam_running,
    mask_box_grounded_sam,
    stop_grounded_sam_server,
)
from ..deps import get_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detection"])


def _save_upload(file: UploadFile) -> tuple[str, str]:
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"  # type: ignore[arg-type]
    filepath = str(settings.upload_dir / safe_name)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return filepath, safe_name


@router.post("/detect", status_code=201)
async def create_detection(
    file: UploadFile = File(...),
    categories: str = Form(...),
    grounded_sam_text: str = Form(""),
    use_grounded_sam_seg: bool = Form(True),
    grounded_sam_threshold: float = Form(0.5, ge=0.0, le=1.0),
    grounded_sam_mask_threshold: float = Form(0.5, ge=0.0, le=1.0),
    replace_detection_id: str = Form(""),
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(400, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")

    try:
        cat_list: list[str] = json.loads(categories)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail="categories must be a JSON array") from exc
    if not cat_list:
        raise HTTPException(400, detail="categories cannot be empty")

    original_name = Path(file.filename).name  # type: ignore[arg-type]
    replace_id = replace_detection_id.strip() or None
    if not replace_id:
        existing = repo.get_latest_by_image_name(original_name)
        if existing:
            replace_id = str(existing.id)

    filepath, safe_name = _save_upload(file)
    if replace_id and not repo.get_by_id(replace_id):
        Path(filepath).unlink(missing_ok=True)
        raise HTTPException(404, detail=f"Detection not found: {replace_id}")

    try:
        detection = await process_detection(
            filepath=filepath,
            original_name=original_name,
            categories=cat_list,
            params=DetectionParams(
                grounded_sam_text=grounded_sam_text,
                use_grounded_sam_seg=use_grounded_sam_seg,
                grounded_sam_threshold=grounded_sam_threshold,
                grounded_sam_mask_threshold=grounded_sam_mask_threshold,
            ),
            repo=repo,
            replace_detection_id=replace_id,
        )
    except AppError as exc:
        if replace_id:
            Path(filepath).unlink(missing_ok=True)
        raise HTTPException(exc.status_code, detail=exc.detail) from exc
    if replace_id:
        Path(filepath).unlink(missing_ok=True)

    return APIResponse(
        data=DetectionOut.model_validate(detection).model_dump(by_alias=True),
    )


@router.get("/detections")
def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100000, validation_alias="pageSize"),
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    items, total = repo.list(page=page, page_size=page_size)
    return APIResponse(
        data=[DetectionListItem.model_validate(d).model_dump(by_alias=True) for d in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/detections/{detection_id}")
def get_detection(
    detection_id: str,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    return APIResponse(
        data=DetectionOut.model_validate(det).model_dump(by_alias=True),
    )


@router.post("/detections/{detection_id}/delete", status_code=204)
def delete_detection(
    detection_id: str,
    repo: DetectionRepository = Depends(get_repo),
) -> None:
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    try:
        Path(det.image_path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete image file: %s", det.image_path)
    repo.delete(det, commit=True)


@router.post("/detections/{detection_id}/boxes/{box_id}/delete", status_code=204)
def delete_box(
    detection_id: str,
    box_id: str,
    repo: DetectionRepository = Depends(get_repo),
) -> None:
    box = repo.get_box(detection_id, box_id)
    if not box:
        raise NotFoundError("DetectionBox", box_id)
    repo.delete_box(box, commit=True)


class AddBoxBody(BaseSchema):
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int


class BatchDeleteBody(BaseSchema):
    detection_ids: list[str]


class UpdateBoxBody(BaseSchema):
    x1: int
    y1: int
    x2: int
    y2: int


@router.post("/detections/delete-batch")
def delete_detections_batch(
    body: BatchDeleteBody,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    deleted = 0
    for detection_id in body.detection_ids:
        det = repo.get_by_id(detection_id)
        if not det:
            continue
        try:
            Path(det.image_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete image file: %s", det.image_path)
        repo.delete(det)
        deleted += 1
    repo.db.commit()
    return APIResponse(data={"deleted": deleted})


@router.post("/detections/{detection_id}/boxes", status_code=201)
def add_box(
    detection_id: str,
    body: AddBoxBody,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    box_dict = {
        "class_name": body.class_name,
        "x1": body.x1,
        "y1": body.y1,
        "x2": body.x2,
        "y2": body.y2,
    }
    try:
        polygon = mask_box_grounded_sam(det.image_path, box_dict)
        if polygon:
            box_dict["mask_polygon"] = polygon
    except Exception:
        logger.exception("Failed to generate SAM mask for manual box")
    boxes = repo.add_boxes(
        detection_id,
        [box_dict],
        commit=True,
    )
    return APIResponse(data=DetectionBoxOut.model_validate(boxes[0]).model_dump(by_alias=True))


class ReplaceBoxItem(BaseSchema):
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int


class ReplaceBoxesBody(BaseSchema):
    boxes: list[ReplaceBoxItem]


@router.put("/detections/{detection_id}/boxes")
def replace_boxes(
    detection_id: str,
    body: ReplaceBoxesBody,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    repo.replace_boxes(detection_id, [b.model_dump() for b in body.boxes], commit=True)
    return APIResponse(data={"ok": True, "count": len(body.boxes)})


class FilterSettingsBody(BaseSchema):
    filter_mode: FilterMode
    filter_nms_iou: float | None = None


@router.put("/detections/{detection_id}/filter-settings")
def save_filter_settings(
    detection_id: str,
    body: FilterSettingsBody,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    det.filter_mode = body.filter_mode
    det.filter_nms_iou = body.filter_nms_iou
    repo.update_detection(det, commit=True)
    return APIResponse(data={"ok": True})


@router.put("/detections/{detection_id}/boxes/{box_id}")
def update_box(
    detection_id: str,
    box_id: str,
    body: UpdateBoxBody,
    repo: DetectionRepository = Depends(get_repo),
) -> APIResponse:
    box = repo.get_box(detection_id, box_id)
    if not box:
        raise NotFoundError("DetectionBox", box_id)
    box.x1, box.y1, box.x2, box.y2 = body.x1, body.y1, body.x2, body.y2
    repo.update_box(box, commit=True)
    return APIResponse(data={"ok": True})


@router.get("/detections/{detection_id}/image")
def get_detection_image(
    detection_id: str,
    repo: DetectionRepository = Depends(get_repo),
):
    det = repo.get_by_id(detection_id)
    if not det:
        raise NotFoundError("Detection", detection_id)
    path = Path(det.image_path)
    if not path.exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(str(path))


@router.get("/model/grounded-sam/status")
def grounded_sam_status() -> APIResponse:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{GROUNDED_SAM_URL}/health", timeout=2)
        import json as _json

        data = _json.loads(resp.read())
        status = data.get("status", "unloaded")
        return APIResponse(data={"loaded": status == "loaded", "status": status})
    except Exception:
        return APIResponse(data={"loaded": False, "status": "unloaded"})


@router.get("/model/events")
async def model_events():
    """SSE endpoint that pushes Grounded-SAM model status."""

    async def event_stream():
        import urllib.request

        prev = ""
        # Fast poll during startup, slow down once all models are stable
        fast_interval = 1.5
        slow_interval = 10.0
        stable_count = 0
        prev_grounded_sam: dict = {"loaded": False, "status": "unloaded"}

        while True:
            # Grounded-SAM-2 status — keep previous value on transient health check failure.
            # Its WSGI server is single-threaded; /health can time out during inference.
            grounded_sam_data = prev_grounded_sam
            try:
                resp = urllib.request.urlopen(f"{GROUNDED_SAM_URL}/health", timeout=1)
                import json as _json

                data = _json.loads(resp.read())
                grounded_sam_data = {
                    "loaded": data.get("status") == "loaded",
                    "status": data.get("status", "unloaded"),
                }
                prev_grounded_sam = grounded_sam_data
            except Exception:
                pass

            payload = json.dumps({"groundedSam": grounded_sam_data})
            if payload != prev:
                prev = payload
                yield f"data: {payload}\n\n"

            # Use fast interval while any model is in transition, slow otherwise
            all_stable = grounded_sam_data.get("status", "") in ("loaded", "unloaded")
            if all_stable:
                stable_count += 1
            else:
                stable_count = 0

            interval = slow_interval if stable_count > 3 else fast_interval
            await asyncio.sleep(interval)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/model/grounded-sam/unload", status_code=204)
def grounded_sam_unload() -> None:
    if is_grounded_sam_running():
        stop_grounded_sam_server()
