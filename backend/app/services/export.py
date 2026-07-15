from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .coco_format import export_coco_json
from .createml_format import export_createml_json
from .voc_format import detection_to_voc
from .yolo_format import boxes_to_yolo, boxes_to_yolo_seg, detection_to_yolo

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..models.detection import Detection

FORMAT_LABELS = {
    "yolo": "YOLO",
    "yolo-seg": "YOLO Segmentation",
    "coco": "COCO JSON",
    "voc": "Pascal VOC",
    "createml": "CreateML JSON",
}


def export_single(db: Session, detection_id: str) -> tuple[str, str]:
    """Return (yolo_content, image_name) for a single detection."""
    from ..models.detection import Detection

    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det:
        raise ValueError(f"Detection not found: {detection_id}")

    class_map = _build_class_map([det])
    return detection_to_yolo(det, class_map), det.image_name


def export_single_zip(db: Session, detection_id: str, format: str = "yolo") -> bytes:
    """Export a single detection as a zip in the requested format."""
    return export_batch(db, [detection_id], format=format)


def export_batch(db: Session, detection_ids: list[str], format: str = "yolo") -> bytes:
    """Export multiple detections as a zip file in the requested format."""
    from ..models.detection import Detection

    dets: list[Detection] = []
    for det_id in detection_ids:
        det = db.query(Detection).filter(Detection.id == det_id).first()
        if det:
            dets.append(det)

    unified_map = _build_class_map(dets)

    if format == "yolo":
        return _export_yolo(dets, unified_map)
    if format == "yolo-seg":
        return _export_yolo(dets, unified_map, seg_mode=True)
    if format == "coco":
        return _export_coco(dets, unified_map)
    if format == "voc":
        return _export_voc(dets, unified_map)
    if format == "createml":
        return _export_createml(dets, unified_map)
    raise ValueError(f"Unsupported format: {format}. Supported: {list(FORMAT_LABELS)}")


def _export_yolo(
    dets: list[Detection], unified_map: dict[str, int], seg_mode: bool = False
) -> bytes:
    label_dir = "labels"
    buf = io.BytesIO()
    seen_names: dict[str, int] = {}
    samples = _group_yolo_samples(dets)
    if not samples:
        raise ValueError("No positive-target images selected for YOLO export")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample in samples:
            base = _unique_base_name(sample["image_name"], seen_names)
            if seg_mode:
                label_text = boxes_to_yolo_seg(
                    sample["boxes"],
                    sample["image_width"],
                    sample["image_height"],
                    unified_map,
                )
            else:
                label_text = boxes_to_yolo(
                    sample["boxes"],
                    sample["image_width"],
                    sample["image_height"],
                    unified_map,
                )
            zf.writestr(f"{label_dir}/{base}.txt", label_text)
            zf.write(sample["image_path"], f"images/{base}{Path(sample['image_name']).suffix}")
        names = {i: name for name, i in sorted(unified_map.items(), key=lambda x: x[1])}
        zf.writestr(
            "data.yaml",
            f"nc: {len(names)}\nnames: {json.dumps(names, ensure_ascii=False)}\n",
        )
    buf.seek(0)
    return buf.getvalue()


def _export_coco(dets: list[Detection], unified_map: dict[str, int]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", export_coco_json(dets, unified_map))
    buf.seek(0)
    return buf.getvalue()


def _export_voc(dets: list[Detection], unified_map: dict[str, int]) -> bytes:
    buf = io.BytesIO()
    seen_names: dict[str, int] = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for det in dets:
            base = _unique_base(det, seen_names)
            zf.writestr(f"{base}.xml", detection_to_voc(det, unified_map))
    buf.seek(0)
    return buf.getvalue()


def _export_createml(dets: list[Detection], unified_map: dict[str, int]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", export_createml_json(dets, unified_map))
    buf.seek(0)
    return buf.getvalue()


def _unique_base(det: Detection, seen_names: dict[str, int]) -> str:
    return _unique_base_name(det.image_name, seen_names)


def _unique_base_name(image_name: str, seen_names: dict[str, int]) -> str:
    base = Path(image_name).stem or str(uuid.uuid4())[:8]
    if base in seen_names:
        seen_names[base] += 1
        base = f"{base}_{seen_names[base]}"
    else:
        seen_names[base] = 1
    return base


def _build_class_map(detections: list[Detection]) -> dict[str, int]:
    """Build a unified class_name → class_id mapping across all given detections."""
    from .yolo_format import _get_filtered_boxes

    class_map: dict[str, int] = {}
    for sample in _group_yolo_samples(detections):
        for box in sample["boxes"]:
            if box["class_name"] not in class_map:
                class_map[box["class_name"]] = len(class_map)
    return class_map


def _group_yolo_samples(detections: list[Detection]) -> list[dict]:
    """Group detections by original image name and skip zero-target records."""
    from .yolo_format import _get_filtered_boxes

    samples: dict[str, dict] = {}
    for det in detections:
        boxes = _get_filtered_boxes(det)
        if not boxes:
            continue
        sample = samples.setdefault(
            det.image_name,
            {
                "image_path": det.image_path,
                "image_name": det.image_name,
                "image_width": det.image_width,
                "image_height": det.image_height,
                "boxes": [],
            },
        )
        sample["boxes"].extend(boxes)
    return list(samples.values())
