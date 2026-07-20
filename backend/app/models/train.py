from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, default=None)
    model_variant: Mapped[str] = mapped_column(String(32), default="yolo26n")
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    batch: Mapped[int] = mapped_column(Integer, default=16)
    train_ratio: Mapped[float] = mapped_column(Float, default=0.7)
    val_ratio: Mapped[float] = mapped_column(Float, default=0.2)
    task_type: Mapped[str] = mapped_column(String(16), default="segment")
    class_map: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
    )
    metrics: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    model_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    onnx_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    detection_links: Mapped[List["TrainingDetection"]] = relationship(
        "TrainingDetection",
        back_populates="training_job",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_training_jobs_status", "status"),)

    @property
    def detection_ids(self) -> list[str]:
        return [str(link.detection_id) for link in self.detection_links]


class TrainingDetection(Base):
    """Many-to-many link between TrainingJob and Detection."""

    __tablename__ = "training_detections"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    training_job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    detection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("detections.id", ondelete="CASCADE"),
        nullable=False,
    )

    training_job: Mapped[TrainingJob] = relationship(
        "TrainingJob",
        back_populates="detection_links",
    )

    __table_args__ = (
        Index("ix_training_detections_job_id", "training_job_id"),
        Index("ix_training_detections_detection_id", "detection_id"),
    )
