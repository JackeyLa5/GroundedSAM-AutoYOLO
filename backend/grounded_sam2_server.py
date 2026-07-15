"""Grounded-SAM standalone server.

Runs GroundingDINO for open-vocabulary boxes, then SAM2 for optional masks.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from wsgiref.simple_server import make_server

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grounded-sam2-server")

DEVICE = "cpu"
SERVER_ARGS: argparse.Namespace | None = None


def _ensure_app_importable() -> None:
    backend_root = Path(__file__).resolve().parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _normalize_prompt(text: str) -> str:
    labels = [part.strip().lower() for part in text.replace(",", ".").split(".") if part.strip()]
    if not labels and text.strip():
        labels = [text.strip().lower()]
    return ". ".join(labels) + "." if labels else ""


def _build_sam2_model(model_id: str, checkpoint_path: str):
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2, build_sam2_hf

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"GROUNDED_SAM2_SAM2_CHECKPOINT_PATH does not exist: {ckpt_path}")
        config_name = HF_MODEL_ID_TO_FILENAMES[model_id][0]
        logger.info("Loading SAM2 checkpoint from %s", ckpt_path)
        return build_sam2(config_file=config_name, ckpt_path=str(ckpt_path), device="cpu")

    return build_sam2_hf(model_id, device="cpu")


def load_model(state: dict):
    global DEVICE

    _ensure_app_importable()
    from app.core.config import settings
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = "mps"

    grounding_model_id = SERVER_ARGS.grounding_model if SERVER_ARGS else ""
    sam2_model_id = SERVER_ARGS.sam2_model if SERVER_ARGS else ""
    sam2_checkpoint_path = SERVER_ARGS.sam2_checkpoint if SERVER_ARGS else ""
    grounding_model_id = grounding_model_id or settings.grounded_sam2_grounding_model_id
    sam2_model_id = sam2_model_id or settings.grounded_sam2_sam2_model_id
    sam2_checkpoint_path = sam2_checkpoint_path or settings.grounded_sam2_sam2_checkpoint_path

    logger.info(
        "Loading Grounded-SAM-2 to %s (grounding=%s, sam2=%s)...",
        DEVICE,
        grounding_model_id,
        sam2_model_id,
    )
    state["status"] = "loading"
    state["device"] = DEVICE

    try:
        local_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
            "TRANSFORMERS_OFFLINE"
        ) == "1"
        processor = AutoProcessor.from_pretrained(grounding_model_id, local_files_only=local_only)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_id,
            local_files_only=local_only,
        ).to(DEVICE)
        grounding_model.eval()

        sam2_model = _build_sam2_model(sam2_model_id, sam2_checkpoint_path)
        if DEVICE != "cpu":
            sam2_model = sam2_model.to(DEVICE)
        sam2_model.eval()

        state["processor"] = processor
        state["grounding_model"] = grounding_model
        state["sam2_predictor"] = SAM2ImagePredictor(sam2_model)
        state["status"] = "loaded"
        state["grounding_model_id"] = grounding_model_id
        state["sam2_model_id"] = sam2_model_id
        logger.info("Grounded-SAM-2 ready on %s", DEVICE)
    except Exception:
        logger.exception("Failed to load Grounded-SAM-2")
        state["status"] = "error"
        state["error"] = str(sys.exc_info()[1])


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    try:
        import cv2
    except ImportError:
        return []

    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return []
    return [[float(p[0]), float(p[1])] for p in largest.squeeze(1).tolist()]


def segment(
    processor,
    grounding_model,
    sam2_predictor,
    image_bytes: bytes,
    text: str = "",
    segmentation: bool = True,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> list[dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = _normalize_prompt(text)
    if not prompt:
        return []

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=mask_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes_tensor = results.get("boxes")
    if boxes_tensor is None or len(boxes_tensor) == 0:
        return []

    boxes = boxes_tensor.detach().cpu().numpy()
    confidences = results.get("scores", [])
    labels = results.get("labels", [])

    masks: np.ndarray | None = None
    if segmentation:
        sam2_predictor.set_image(np.array(image))
        with torch.no_grad():
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
        if masks is not None and masks.ndim == 4:
            masks = masks.squeeze(1)

    boxes_out: list[dict] = []
    for i, bbox in enumerate(boxes.tolist()):
        score = float(confidences[i]) if i < len(confidences) else 0.0
        class_name = str(labels[i]) if i < len(labels) and labels[i] else text or "object"
        poly = _mask_to_polygon(masks[i]) if masks is not None and i < len(masks) else []
        boxes_out.append(
            {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
                "confidence": score,
                "mask_polygon": poly if poly else None,
                "class_name": class_name,
            }
        )
    return boxes_out


def mask_box(
    sam2_predictor,
    image_bytes: bytes,
    bbox: list[float],
) -> list:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    sam2_predictor.set_image(np.array(image))
    boxes = np.array([bbox], dtype=np.float32)
    with torch.no_grad():
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    if masks is not None and masks.ndim == 4:
        masks = masks.squeeze(1)
    if masks is None or len(masks) == 0:
        return []
    return _mask_to_polygon(masks[0])


def _parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    import re

    if "boundary=" not in content_type:
        return {}
    boundary = content_type.split("boundary=", 1)[1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    boundary_bytes = boundary.encode()

    parts = body.split(b"--" + boundary_bytes)
    result: dict[str, bytes] = {}
    for part in parts:
        if part in (b"--\r\n", b"--"):
            break
        if not part.startswith(b"\r\n"):
            continue
        part = part[2:]
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_block = part[:header_end].decode("latin-1")
        value = part[header_end + 4 :]
        if value.endswith(b"\r\n"):
            value = value[:-2]
        disp_match = re.search(r'name="([^"]*)"', headers_block)
        if disp_match:
            result[disp_match.group(1)] = value
    return result


_CORS_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "*"),
]


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")

    if environ.get("REQUEST_METHOD") == "OPTIONS":
        start_response("204 No Content", _CORS_HEADERS)
        return [b""]

    if path == "/health":
        status = server_state.get("status", "unloaded")
        body = json.dumps(
            {
                "status": status,
                "backend": "grounded-sam2",
                "device": server_state.get("device", ""),
                "error": server_state.get("error", ""),
                "grounding_model_id": server_state.get("grounding_model_id", ""),
                "sam2_model_id": server_state.get("sam2_model_id", ""),
            }
        )
        start_response("200 OK", [("Content-Type", "application/json")] + _CORS_HEADERS)
        return [body.encode()]

    if path == "/segment" and environ.get("REQUEST_METHOD") == "POST":
        try:
            if server_state.get("status") != "loaded":
                resp = json.dumps({"error": "Model is not loaded yet"})
                start_response(
                    "503 Service Unavailable",
                    [("Content-Type", "application/json")] + _CORS_HEADERS,
                )
                return [resp.encode()]

            length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(length)
            form = _parse_multipart(body, environ.get("CONTENT_TYPE", ""))
            boxes = segment(
                server_state["processor"],
                server_state["grounding_model"],
                server_state["sam2_predictor"],
                form.get("file", b""),
                form.get("text", b"").decode() or "",
                form.get("segmentation", b"true").decode().lower() != "false",
                float(form.get("threshold", b"0.5").decode() or "0.5"),
                float(form.get("mask_threshold", b"0.5").decode() or "0.5"),
            )
            start_response("200 OK", [("Content-Type", "application/json")] + _CORS_HEADERS)
            return [json.dumps({"boxes": boxes}).encode()]
        except Exception as exc:
            logger.exception("Segment failed")
            start_response(
                "500 Internal Server Error",
                [("Content-Type", "application/json")] + _CORS_HEADERS,
            )
            return [json.dumps({"error": str(exc)}).encode()]

    if path == "/mask-box" and environ.get("REQUEST_METHOD") == "POST":
        try:
            if server_state.get("status") != "loaded":
                resp = json.dumps({"error": "Model is not loaded yet"})
                start_response(
                    "503 Service Unavailable",
                    [("Content-Type", "application/json")] + _CORS_HEADERS,
                )
                return [resp.encode()]

            length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(length)
            form = _parse_multipart(body, environ.get("CONTENT_TYPE", ""))
            bbox = json.loads(form.get("box", b"[]").decode() or "[]")
            polygon = mask_box(
                server_state["sam2_predictor"],
                form.get("file", b""),
                bbox,
            )
            start_response("200 OK", [("Content-Type", "application/json")] + _CORS_HEADERS)
            return [json.dumps({"mask_polygon": polygon}).encode()]
        except Exception as exc:
            logger.exception("Mask box failed")
            start_response(
                "500 Internal Server Error",
                [("Content-Type", "application/json")] + _CORS_HEADERS,
            )
            return [json.dumps({"error": str(exc)}).encode()]

    start_response("404 Not Found", [("Content-Type", "text/plain")] + _CORS_HEADERS)
    return [b"Not found"]


server_state = {}


def main():
    import threading

    global SERVER_ARGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--grounding-model", default="")
    parser.add_argument("--sam2-model", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    SERVER_ARGS = parser.parse_args()

    server_state["status"] = "starting"
    httpd = make_server("127.0.0.1", SERVER_ARGS.port, application)
    logger.info("Grounded-SAM-2 server listening on http://127.0.0.1:%d", SERVER_ARGS.port)

    threading.Thread(target=load_model, args=(server_state,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
