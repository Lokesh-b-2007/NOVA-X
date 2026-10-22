"""
Pydantic Schemas
------------------
Request/response models for every API endpoint. Kept separate from the
SQLAlchemy models (database/models.py) so the DB schema and the wire
format can evolve independently.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class NoveltyResult(BaseModel):
    status: str  # "KNOWN" | "SUSPICIOUS" | "UNKNOWN"
    novelty_score: float
    known_similarity: Optional[float] = None
    reason: str


class OVDPrediction(BaseModel):
    label: str
    confidence: float


class DetectionOut(BaseModel):
    id: str
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    bbox: BoundingBox
    novelty: NoveltyResult
    ovd: Optional[OVDPrediction] = None
    ovd_top_predictions: Optional[List[OVDPrediction]] = None
    model_version: str
    novel_object_id: Optional[str] = None


class ImageDetectionResponse(BaseModel):
    detections: List[DetectionOut]
    known_count: int
    unknown_count: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Novel Object Repository / Human Verification
# ---------------------------------------------------------------------------

class NovelObjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    image_path: str
    detected_at: datetime
    novelty_score: float
    ovd_label: Optional[str] = None
    ovd_confidence: Optional[float] = None
    ovd_top_predictions: Optional[List[Dict[str, Any]]] = None
    cluster_id: Optional[str] = None
    occurrence_count: int
    verification_status: str
    confirmed_label: Optional[str] = None


class ConfirmLabelRequest(BaseModel):
    label: str


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

class ClusterOut(BaseModel):
    cluster_id: str
    object_count: int
    suggested_label: Optional[str] = None
    average_confidence: float


# ---------------------------------------------------------------------------
# Known Classes / Model Versions / Continual Learning
# ---------------------------------------------------------------------------

class KnownClassOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    source: str
    example_count: int
    added_at: datetime


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version_id: str
    created_at: datetime
    classes: List[str]
    dataset_size: int
    accuracy: Optional[float] = None
    model_path: Optional[str] = None
    is_active: bool


class TrainRequest(BaseModel):
    target_class: str


class TrainingJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    target_class: str
    dataset_size: int
    status: str
    error: Optional[str] = None
    resulting_model_version: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Dashboard / Analytics
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    total_images_analyzed: int
    known_objects_detected: int
    unknown_objects_detected: int
    novel_objects_discovered: int
    confirmed_new_classes: int
    active_model_version: str


class AnalyticsResponse(BaseModel):
    known_vs_unknown: Dict[str, int]
    unknown_over_time: List[Dict[str, Any]]
    novelty_score_distribution: List[Dict[str, Any]]
    most_frequent_unknown_classes: List[Dict[str, Any]]
    cluster_distribution: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class KnownMetrics(BaseModel):
    precision: float
    recall: float
    map_50: float
    map_50_95: float


class OODMetrics(BaseModel):
    auroc: float
    aupr: float
    fpr95: float


class UnknownMetrics(BaseModel):
    unknown_recall: float
    unknown_precision: float


class OVDMetrics(BaseModel):
    top1_accuracy: float
    top5_accuracy: float


class EvaluationResponse(BaseModel):
    # Each block is None (not a fabricated number) when the labeled eval
    # data required to compute it isn't present under data/known_dataset/eval/.
    known_metrics: Optional[KnownMetrics] = None
    ood_metrics: Optional[OODMetrics] = None
    unknown_metrics: Optional[UnknownMetrics] = None
    ovd_metrics: Optional[OVDMetrics] = None
    ood_sample_count: Optional[int] = None
    ood_ground_truth_unknown_count: Optional[int] = None
    ovd_sample_count: Optional[int] = None
    ovd_sample_source: Optional[str] = None
    evaluation_caveats: List[str] = Field(default_factory=list)
    evaluated_at: datetime
    notes: str
