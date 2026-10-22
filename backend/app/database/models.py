"""
Database Models
------------------
Tables: Detection, NovelObject, NovelCluster, VerifiedObject, KnownClass,
ModelVersion, TrainingJob.

IDs use UUID strings (not autoincrement ints) so that image/crop filenames
can safely embed the primary key without a round-trip to the DB first.
"""
import enum
import uuid

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Boolean,
    ForeignKey,
    JSON,
    Enum as SAEnum,
    Text,
)
from sqlalchemy.orm import relationship

from app.database.database import Base
from app.utils.time_utils import utc_now_naive


def gen_uuid() -> str:
    return str(uuid.uuid4())


class NoveltyStatus(str, enum.Enum):
    KNOWN = "KNOWN"
    SUSPICIOUS = "SUSPICIOUS"
    UNKNOWN = "UNKNOWN"


class VerificationStatus(str, enum.Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class TrainingJobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Detection(Base):
    """One row per detected bounding box, from either /detect/image or /detect/frame."""

    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=gen_uuid)
    image_path = Column(String, nullable=True)  # source frame/image on disk, if persisted
    source = Column(String, default="image")  # "image" | "webcam"

    class_name = Column(String, nullable=True)  # populated only when KNOWN
    confidence = Column(Float, nullable=True)  # detector confidence for the raw class guess
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    novelty_score = Column(Float, default=0.0)
    novelty_status = Column(SAEnum(NoveltyStatus), default=NoveltyStatus.KNOWN)
    novelty_reason = Column(String, nullable=True)

    model_version = Column(String, default="yolov8n-base")
    timestamp = Column(DateTime, default=utc_now_naive)

    novel_object_id = Column(String, ForeignKey("novel_objects.id"), nullable=True)
    novel_object = relationship("NovelObject", back_populates="detections")


class NovelObject(Base):
    """A persisted crop of something flagged UNKNOWN — the Novel Object Repository."""

    __tablename__ = "novel_objects"

    id = Column(String, primary_key=True, default=gen_uuid)
    image_path = Column(String, nullable=False)  # path to the saved crop
    detected_at = Column(DateTime, default=utc_now_naive)

    novelty_score = Column(Float, default=0.0)

    ovd_label = Column(String, nullable=True)
    ovd_confidence = Column(Float, nullable=True)
    ovd_top_predictions = Column(JSON, nullable=True)  # [{label, confidence}, ...]

    embedding = Column(JSON, nullable=True)  # feature vector used for clustering

    cluster_id = Column(String, ForeignKey("novel_clusters.id"), nullable=True)
    cluster = relationship("NovelCluster", back_populates="objects")

    occurrence_count = Column(Integer, default=1)
    verification_status = Column(SAEnum(VerificationStatus), default=VerificationStatus.PENDING)
    confirmed_label = Column(String, nullable=True)

    detections = relationship("Detection", back_populates="novel_object")


class NovelCluster(Base):
    """A group of visually-similar NovelObjects, produced by DBSCAN/HDBSCAN."""

    __tablename__ = "novel_clusters"

    id = Column(String, primary_key=True, default=gen_uuid)
    object_count = Column(Integer, default=0)
    suggested_label = Column(String, nullable=True)
    average_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now_naive)

    objects = relationship("NovelObject", back_populates="cluster")


class VerifiedObject(Base):
    """A human-confirmed label, queued as training data for continual learning."""

    __tablename__ = "verified_objects"

    id = Column(String, primary_key=True, default=gen_uuid)
    novel_object_id = Column(String, ForeignKey("novel_objects.id"), nullable=False)
    image_path = Column(String, nullable=False)
    confirmed_label = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now_naive)

    # Set once this example has been consumed by a completed training job,
    # so the same examples aren't silently reused/double-counted.
    used_in_training_job_id = Column(String, ForeignKey("training_jobs.id"), nullable=True)


class KnownClass(Base):
    """A class the active detector can recognize — base COCO classes plus any learned ones."""

    __tablename__ = "known_classes"

    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, unique=True, nullable=False)
    source = Column(String, default="base")  # "base" | "continual_learning"
    example_count = Column(Integer, default=0)
    added_at = Column(DateTime, default=utc_now_naive)


class ModelVersion(Base):
    """A snapshot of the detector after a (re)training run."""

    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=gen_uuid)
    version_id = Column(String, unique=True, nullable=False)  # e.g. "v2-2026-09-24"
    created_at = Column(DateTime, default=utc_now_naive)
    classes = Column(JSON, nullable=False)  # list[str]
    dataset_size = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)  # only set if a real eval was run
    model_path = Column(String, nullable=True)
    is_active = Column(Boolean, default=False)


class TrainingJob(Base):
    """An administrator-triggered continual-learning run. Never started automatically."""

    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    target_class = Column(String, nullable=False)
    dataset_size = Column(Integer, default=0)
    status = Column(SAEnum(TrainingJobStatus), default=TrainingJobStatus.QUEUED)
    error = Column(Text, nullable=True)
    resulting_model_version = Column(String, nullable=True)
    started_at = Column(DateTime, default=utc_now_naive)
    completed_at = Column(DateTime, nullable=True)
