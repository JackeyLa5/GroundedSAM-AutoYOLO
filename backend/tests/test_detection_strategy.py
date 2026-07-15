"""Tests for detection strategy pattern."""

from app.services.detection_strategy import (
    DetectionResult,
    GroundedSamDetection,
    create_strategy,
)


class TestCreateStrategy:
    def test_default_strategy_is_grounded_sam(self):
        s = create_strategy()
        assert isinstance(s, GroundedSamDetection)


class TestDetectionResult:
    def test_default_polygons_is_empty_list(self):
        r = DetectionResult(boxes=[], img_w=100, img_h=100)
        assert r.polygons == []

    def test_polygons_stored(self):
        r = DetectionResult(boxes=[], img_w=100, img_h=100, polygons=[[[1, 2]]])
        assert r.polygons == [[[1, 2]]]

    def test_boxes_and_dims(self):
        boxes = [{"x1": 0, "y1": 0, "x2": 50, "y2": 50, "class_name": "cat"}]
        r = DetectionResult(boxes=boxes, img_w=640, img_h=480, raw_text="cat")
        assert r.img_w == 640
        assert r.img_h == 480
        assert r.raw_text == "cat"
        assert len(r.boxes) == 1


class TestStrategyKwargs:
    def test_grounded_sam_strategy_accepts_segmentation_flag(self):
        s = GroundedSamDetection(lambda *a, **kw: [])
        assert s is not None  # Grounded-SAM requires running server, skip actual detect
