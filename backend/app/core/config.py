from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# ── CUDA memory allocator tuning ───────────────────
# Use expandable segments (PyTorch 2.1+) for dynamic memory management;
# much better than the legacy max_split_size_mb for varying tensor sizes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Database
    database_url: str = ""

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        # Fallback to local SQLite if no DATABASE_URL is provided in env
        db_path = self.project_root / "autolabeling.db"
        return f"sqlite:///{db_path}"

    # Grounded-SAM model settings
    device: str = ""
    grounded_sam2_grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    grounded_sam2_sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    grounded_sam2_sam2_checkpoint_path: str = ""
    grounded_sam2_port: int = 8002

    @property
    def resolved_device(self) -> str:
        from .gpu_memory import _detect_device

        return _detect_device(self.device)

    # Model idle timeout (seconds) — auto-unload to free GPU memory
    model_idle_timeout_seconds: int = 600
    grounded_sam2_load_timeout_seconds: int = 600
    log_dir: str = "logs"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Upload
    max_upload_size_mb: int = 20
    max_import_size_mb: int = 10240  # 10 GB
    max_video_upload_size_mb: int = 500

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def upload_dir(self) -> Path:
        d = self.project_root / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
