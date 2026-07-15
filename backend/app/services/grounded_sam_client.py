"""Client for the standalone Grounded-SAM server."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from app.core.config import settings

GROUNDED_SAM_PORT = settings.grounded_sam2_port
GROUNDED_SAM_URL = f"http://127.0.0.1:{GROUNDED_SAM_PORT}"
_grounded_sam_process: subprocess.Popen | None = None
_last_activity: float = 0.0
_watchdog_thread: threading.Thread | None = None
_watchdog_stop: threading.Event | None = None


def _bump_activity() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def _ensure_watchdog() -> None:
    """Start idle watchdog if not already running."""
    global _watchdog_thread, _watchdog_stop, _last_activity
    if _watchdog_thread is not None:
        return

    from app.core.config import settings

    _last_activity = time.monotonic()
    _watchdog_stop = threading.Event()
    stop_event = _watchdog_stop

    def _loop():
        while not stop_event.is_set():
            stop_event.wait(timeout=30)
            if stop_event.is_set():
                return
            if not is_grounded_sam_running():
                continue
            idle = time.monotonic() - _last_activity
            if idle >= settings.model_idle_timeout_seconds:
                logger.info("Grounded-SAM-2 idle for %.0fs, auto-unloading...", idle)
                stop_grounded_sam_server()

    _watchdog_thread = threading.Thread(
        target=_loop, daemon=True, name="grounded-sam2-idle-watchdog"
    )
    _watchdog_thread.start()
    logger.info(
        "Grounded-SAM-2 idle watchdog started (timeout=%ds)",
        settings.model_idle_timeout_seconds,
    )


def is_grounded_sam_running() -> bool:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{GROUNDED_SAM_URL}/health", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _is_grounded_sam_loaded() -> bool:
    """Check both that the server is alive AND the model has finished loading."""
    import json
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{GROUNDED_SAM_URL}/health", timeout=2)
        if resp.status != 200:
            return False
        data = json.loads(resp.read())
        return data.get("status") == "loaded"
    except Exception:
        return False


def start_grounded_sam_server() -> None:
    global _grounded_sam_process
    if _is_grounded_sam_loaded():
        return

    server_script = Path(__file__).resolve().parent.parent.parent / "grounded_sam2_server.py"
    grounded_sam_venv = Path(__file__).resolve().parent.parent.parent / "grounded-sam2-venv"
    venv_python = grounded_sam_venv / "bin" / "python3"
    if venv_python.exists():
        python = str(venv_python)
    else:
        python = sys.executable
        logger.warning(
            "grounded-sam2-venv not found at %s, falling back to %s",
            venv_python,
            python,
        )

    env = os.environ.copy()

    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        log_dir = settings.project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    grounded_sam_log_path = log_dir / "grounded_sam2_server.log"
    grounded_sam_log = open(grounded_sam_log_path, "a")  # noqa: SIM115
    logger.info(
        "Starting Grounded-SAM-2 server on port %d (logging to %s)...",
        GROUNDED_SAM_PORT,
        grounded_sam_log_path,
    )
    _grounded_sam_process = subprocess.Popen(
        [
            python,
            str(server_script),
            "--port",
            str(GROUNDED_SAM_PORT),
            "--grounding-model",
            settings.grounded_sam2_grounding_model_id,
            "--sam2-model",
            settings.grounded_sam2_sam2_model_id,
            "--sam2-checkpoint",
            settings.grounded_sam2_sam2_checkpoint_path,
        ],
        stdout=grounded_sam_log,
        stderr=grounded_sam_log,
        env=env,
    )
    # Wait for model to be fully loaded (not just server alive)
    for _ in range(settings.grounded_sam2_load_timeout_seconds):
        time.sleep(1)
        if _is_grounded_sam_loaded():
            logger.info("Grounded-SAM-2 server ready")
            return
    raise RuntimeError(
        "Grounded-SAM-2 model did not load within "
        f"{settings.grounded_sam2_load_timeout_seconds}s"
    )


def stop_grounded_sam_server() -> None:
    global _grounded_sam_process, _watchdog_thread, _watchdog_stop
    if _watchdog_stop:
        _watchdog_stop.set()
        _watchdog_thread = None
        _watchdog_stop = None
    if _grounded_sam_process:
        _grounded_sam_process.terminate()
        _grounded_sam_process.wait(timeout=5)
        _grounded_sam_process = None
    elif is_grounded_sam_running():
        # Process reference lost (e.g. after uvicorn restart) — kill via port
        import signal

        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{GROUNDED_SAM_PORT}"], capture_output=True, text=True
            )
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str:
                    os.kill(int(pid_str), signal.SIGTERM)
        except Exception:
            logger.exception(
                "Failed to kill Grounded-SAM-2 process on port %d", GROUNDED_SAM_PORT
            )


def segment_grounded_sam(
    image_path: str | Path,
    text: str = "",
    segmentation: bool = True,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> list[dict]:
    """Call Grounded-SAM-2 server to detect + segment. Returns list of box dicts."""
    import urllib.request

    _bump_activity()
    _ensure_watchdog()

    if not _is_grounded_sam_loaded():
        start_grounded_sam_server()

    boundary = "----FormBoundary" + os.urandom(8).hex()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    parts: list[bytes] = []

    # File part
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image_bytes
        + b"\r\n"
    )

    # Text part
    if text:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'
            ).encode()
        )

    # Segmentation part
    if not segmentation:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="segmentation"\r\n\r\n'
            f"false\r\n".encode()
        )

    # Threshold part
    if threshold != 0.5:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="threshold"\r\n\r\n'
            f"{threshold}\r\n".encode()
        )

    # Mask threshold part
    if mask_threshold != 0.5:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="mask_threshold"\r\n\r\n'
            f"{mask_threshold}\r\n".encode()
        )

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    import json

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("boxes", [])


def mask_box_grounded_sam(
    image_path: str | Path,
    box: dict,
) -> list:
    """Call SAM2 with a user-provided bbox and return a mask polygon."""
    import json
    import urllib.request

    _bump_activity()
    _ensure_watchdog()

    if not _is_grounded_sam_loaded():
        start_grounded_sam_server()

    boundary = "----FormBoundary" + os.urandom(8).hex()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image_bytes
        + b"\r\n",
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="box"\r\n\r\n'
            f"{json.dumps(bbox)}\r\n"
        ).encode(),
    ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/mask-box",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("mask_polygon") or []
