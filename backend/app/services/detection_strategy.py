"""Grounded-SAM automatic annotation strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class DetectionResult:
    """Uniform output regardless of which model produced it."""

    def __init__(
        self,
        boxes: list[dict],
        img_w: int,
        img_h: int,
        raw_text: str = "",
        polygons: list | None = None,
    ):
        self.boxes = boxes
        self.img_w = img_w
        self.img_h = img_h
        self.raw_text = raw_text
        self.polygons = polygons or []


class DetectionStrategy(ABC):
    @abstractmethod
    def detect(self, filepath: str, categories: list[str], **kwargs) -> DetectionResult:
        """Return unified DetectionResult."""


def _grounding_prompt(categories: list[str], override_text: str = "") -> str:
    if override_text.strip():
        source = override_text
    else:
        source = ".".join(categories)
    labels = [part.strip().lower() for part in source.replace(",", ".").split(".") if part.strip()]
    return ". ".join(labels) + "." if labels else ""


class GroundedSamDetection(DetectionStrategy):
    """Grounded-SAM-2 via standalone HTTP server — detection + segmentation in one call."""

    def __init__(self, segment_fn):
        self._segment = segment_fn

    def detect(self, filepath: str, categories: list[str], **kwargs) -> DetectionResult:
        text = _grounding_prompt(categories, kwargs.get("grounded_sam_text", ""))
        use_seg = kwargs.get("use_grounded_sam_seg", True)
        threshold = kwargs.get("grounded_sam_threshold", 0.5)
        mask_threshold = kwargs.get("grounded_sam_mask_threshold", 0.5)

        boxes = self._segment(
            filepath,
            text,
            segmentation=use_seg,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )

        img = Image.open(filepath)
        w, h = img.size
        img.close()

        polygons = [b.pop("mask_polygon", None) for b in boxes]

        return DetectionResult(
            boxes=boxes,
            img_w=w,
            img_h=h,
            polygons=polygons,
        )


def create_strategy() -> DetectionStrategy:
    """Return the single supported automatic annotation backend."""
    from .grounded_sam_client import segment_grounded_sam

    return GroundedSamDetection(segment_grounded_sam)
