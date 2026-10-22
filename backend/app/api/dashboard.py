"""
GET /api/dashboard — top-level summary stats for the Dashboard page.
All numbers are computed live from the database; nothing here is mocked.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.schemas.schemas import DashboardStats

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard(db: Session = Depends(get_db)):
    # NOTE: "images analyzed" is tracked at the detection-row level (one row
    # per detected object per call to /api/detect/*), which is unambiguous
    # and requires no extra table. A dedicated ImageUpload table could be
    # added later if a strict 1-row-per-upload count is needed.
    known_count = db.query(db_models.Detection).filter(
        db_models.Detection.novelty_status == db_models.NoveltyStatus.KNOWN
    ).count()
    unknown_count = db.query(db_models.Detection).filter(
        db_models.Detection.novelty_status == db_models.NoveltyStatus.UNKNOWN
    ).count()
    novel_objects = db.query(db_models.NovelObject).count()
    confirmed_classes = db.query(db_models.KnownClass).filter(
        db_models.KnownClass.source == "continual_learning"
    ).count()
    active_model = db.query(db_models.ModelVersion).filter_by(is_active=True).first()

    total_detections = db.query(db_models.Detection).count()

    return DashboardStats(
        total_images_analyzed=total_detections,
        known_objects_detected=known_count,
        unknown_objects_detected=unknown_count,
        novel_objects_discovered=novel_objects,
        confirmed_new_classes=confirmed_classes,
        active_model_version=active_model.version_id if active_model else "base (no fine-tune yet)",
    )
