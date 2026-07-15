"""Shared YOLO label format utilities — single source of truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .box_filter import apply_filter

if TYPE_CHECKING:
    from ..models.detection import Detection


def _normalize_class_name(name: str) -> str:
    return name.strip().strip(".。").replace(" ", "").lower()


def _canonical_class_name(raw_name: str, categories: list[str]) -> str:
    """Constrain model phrases to the user-provided YOLO class names."""
    valid = [c.strip() for c in categories if c and c.strip()]
    if not valid:
        return raw_name
    if len(valid) == 1:
        return valid[0]

    raw_norm = _normalize_class_name(raw_name)
    for category in valid:
        if raw_norm == _normalize_class_name(category):
            return category
    for category in valid:
        cat_norm = _normalize_class_name(category)
        if cat_norm and (cat_norm in raw_norm or raw_norm in cat_norm):
            return category
    return valid[0]


def _get_filtered_boxes(detection: Detection) -> list[dict]:
    """Return boxes after applying saved filter settings, if any."""
    boxes = [
        {
            "x1": b.x1,
            "y1": b.y1,
            "x2": b.x2,
            "y2": b.y2,
            "class_name": _canonical_class_name(b.class_name, detection.categories),
            "mask_polygon": getattr(b, "mask_polygon", None),
        }
        for b in detection.boxes
    ]
    return apply_filter(boxes, detection.filter_mode, detection.filter_nms_iou)


def detection_to_yolo(detection: Detection, class_map: dict[str, int]) -> str:
    """Convert a detection record to YOLO label string.

    Format: class_id x_center y_center width height  (all normalized 0–1)
    Applies saved filter settings if present.
    """
    img_w = detection.image_width or 1
    img_h = detection.image_height or 1
    return boxes_to_yolo(_get_filtered_boxes(detection), img_w, img_h, class_map)


def boxes_to_yolo(
    boxes: list[dict],
    img_w: int,
    img_h: int,
    class_map: dict[str, int],
) -> str:
    img_w = img_w or 1
    img_h = img_h or 1
    lines: list[str] = []

    for box in boxes:
        class_id = class_map[box["class_name"]]
        x_center = ((box["x1"] + box["x2"]) / 2) / img_w
        y_center = ((box["y1"] + box["y2"]) / 2) / img_h
        bw = (box["x2"] - box["x1"]) / img_w
        bh = (box["y2"] - box["y1"]) / img_h
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(lines)


def detection_to_yolo_seg(detection: Detection, class_map: dict[str, int]) -> str:
    """Convert to YOLO segmentation format.

    Format: class_id x1 y1 x2 y2 ... xn yn  (normalized 0–1 polygon points)
    Falls back to bbox if no mask_polygon.
    """
    img_w = detection.image_width or 1
    img_h = detection.image_height or 1
    return boxes_to_yolo_seg(_get_filtered_boxes(detection), img_w, img_h, class_map)


def boxes_to_yolo_seg(
    boxes: list[dict],
    img_w: int,
    img_h: int,
    class_map: dict[str, int],
) -> str:
    img_w = img_w or 1
    img_h = img_h or 1
    lines: list[str] = []

    for box in boxes:
        class_id = class_map[box["class_name"]]
        poly = box.get("mask_polygon")
        if poly and len(poly) >= 3:
            pts = " ".join(f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in poly)
            lines.append(f"{class_id} {pts}")
        else:
            x_center = ((box["x1"] + box["x2"]) / 2) / img_w
            y_center = ((box["y1"] + box["y2"]) / 2) / img_h
            bw = (box["x2"] - box["x1"]) / img_w
            bh = (box["y2"] - box["y1"]) / img_h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(lines)
