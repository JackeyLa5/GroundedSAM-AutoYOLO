from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.services.export import _export_yolo


class FakeBox:
    def __init__(self, class_name: str):
        self.x1 = 10
        self.y1 = 10
        self.x2 = 50
        self.y2 = 50
        self.class_name = class_name


class FakeDetection:
    def __init__(self, image_path: Path, class_name: str):
        self.image_path = str(image_path)
        self.image_name = image_path.name
        self.image_width = 100
        self.image_height = 100
        self.categories = [class_name]
        self.filter_mode = None
        self.filter_nms_iou = None
        self.boxes = [FakeBox(class_name)]


def test_export_yolo_data_yaml_keeps_unicode_names(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100)).save(image_path)
    det = FakeDetection(image_path, "瓶子")

    data = _export_yolo([det], {"瓶子": 0})

    with zipfile.ZipFile(BytesIO(data)) as zf:
        content = zf.read("data.yaml").decode("utf-8")
    assert "瓶子" in content
    assert "\\u74f6\\u5b50" not in content
