"""
Known Classes / Model Versioning / Continual Learning Endpoints
-------------------------------------------------------------------
GET  /api/known-classes
GET  /api/models
POST /api/models/train
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.schemas.schemas import KnownClassOut, ModelVersionOut, TrainRequest, TrainingJobOut
from app.services import continual_learning
from app.config import settings

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/known-classes", response_model=list[KnownClassOut])
def list_known_classes(db: Session = Depends(get_db)):
    existing = {c.name for c in db.query(db_models.KnownClass).all()}
    # Seed rows for the base configured classes if this is a fresh DB.
    for name in settings.KNOWN_CLASSES:
        if name and name not in existing:
            db.add(db_models.KnownClass(name=name, source="base"))
    db.commit()
    return db.query(db_models.KnownClass).order_by(db_models.KnownClass.added_at).all()


@router.get("/models", response_model=list[ModelVersionOut])
def list_model_versions(db: Session = Depends(get_db)):
    return db.query(db_models.ModelVersion).order_by(db_models.ModelVersion.created_at.desc()).all()


@router.post("/models/train", response_model=TrainingJobOut)
def train_new_class(payload: TrainRequest, db: Session = Depends(get_db)):
    """
    Manually triggered by an administrator from the Continual Learning page,
    after enough objects for `target_class` have been human-verified.
    Never called automatically by the detection pipeline.
    """
    try:
        job = continual_learning.start_training_job(db, payload.target_class)
    except continual_learning.InsufficientDataError as e:
        raise HTTPException(400, str(e))
    return job


@router.get("/training-jobs", response_model=list[TrainingJobOut])
def list_training_jobs(db: Session = Depends(get_db)):
    return db.query(db_models.TrainingJob).order_by(db_models.TrainingJob.started_at.desc()).all()


@router.get("/training-jobs/pending-size")
def pending_dataset_size(target_class: str, db: Session = Depends(get_db)):
    size = continual_learning.get_pending_dataset_size(db, target_class)
    return {
        "target_class": target_class,
        "pending_examples": size,
        "minimum_required": settings.MIN_DATASET_SIZE_FOR_TRAINING,
        "ready": size >= settings.MIN_DATASET_SIZE_FOR_TRAINING,
    }
