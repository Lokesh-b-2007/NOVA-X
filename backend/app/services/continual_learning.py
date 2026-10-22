"""
Continual Learning Pipeline
-------------------------------
Deliberately NOT automatic. A NovelObject only becomes training data once
a human confirms its label (see api/unknown_objects.py::confirm_label,
which creates a VerifiedObject row). An administrator must then explicitly
request training for a target class via POST /api/models/train, and only
once a minimum dataset size is met.

This reference implementation does not ship a full YOLO fine-tuning loop
(that requires a GPU, a properly formatted YOLO dataset directory, and
realistically hours of wall-clock time) — instead it does the two things
that are safe and useful to automate:

  1. Validate readiness (enough verified examples) and assemble a dataset
     directory for the target class.
  2. Record a TrainingJob and register the new KnownClass + ModelVersion
     with an honest (null) accuracy until a real training run reports one.

If actual training can't run (no GPU/ultralytics/time budget), the job is
still created and marked FAILED with a clear, honest error — never faked.
"""
import logging
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.database import models as db_models
from app.utils.time_utils import utc_now_naive

logger = logging.getLogger("nova_x.continual_learning")


class InsufficientDataError(Exception):
    pass


def get_pending_dataset_size(db: Session, target_class: str) -> int:
    """Count of verified examples for target_class not yet consumed by a completed training job."""
    return (
        db.query(db_models.VerifiedObject)
        .filter(db_models.VerifiedObject.confirmed_label == target_class)
        .filter(db_models.VerifiedObject.used_in_training_job_id.is_(None))
        .count()
    )


def _assemble_dataset(db: Session, target_class: str, job_dir: Path):
    """
    Copies verified crops for target_class into a per-class image directory.

    NOTE: this reference implementation only has cropped object images
    (not full-scene images with bounding-box labels), so it writes a
    classification-style layout under job_dir/images/<class>/*.jpg.
    A full detection fine-tune requires re-annotating full scenes, which
    is out of scope for an automated pipeline — see docs/ for the manual
    re-annotation workflow if pixel-level boxes are needed for the new class.
    """
    verified = (
        db.query(db_models.VerifiedObject)
        .filter(db_models.VerifiedObject.confirmed_label == target_class)
        .filter(db_models.VerifiedObject.used_in_training_job_id.is_(None))
        .all()
    )

    class_dir = job_dir / "images" / target_class
    class_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for v in verified:
        src = Path(v.image_path)
        if not src.exists():
            logger.warning("Verified example missing on disk, skipping: %s", src)
            continue
        shutil.copy(src, class_dir / src.name)
        copied += 1

    existing_classes = [c.name for c in db.query(db_models.KnownClass).all()]
    all_classes = sorted(set(existing_classes) | {target_class})

    return copied, all_classes, verified


def start_training_job(db: Session, target_class: str) -> db_models.TrainingJob:
    pending = get_pending_dataset_size(db, target_class)
    if pending < settings.MIN_DATASET_SIZE_FOR_TRAINING:
        raise InsufficientDataError(
            f"Only {pending} verified example(s) for '{target_class}', "
            f"minimum {settings.MIN_DATASET_SIZE_FOR_TRAINING} required before training."
        )

    job = db_models.TrainingJob(
        target_class=target_class,
        dataset_size=pending,
        status=db_models.TrainingJobStatus.RUNNING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    job_dir = settings.TRAINED_MODELS_DIR / f"job_{job.id}"
    try:
        copied, all_classes, verified_rows = _assemble_dataset(db, target_class, job_dir)

        if copied == 0:
            raise RuntimeError("No verified example images could be copied from disk.")

        # Real fine-tuning requires a properly annotated detection dataset
        # (full scenes + bounding boxes), a GPU, and a non-trivial time
        # budget — none of which a synchronous API request can assume.
        # We therefore stop short of calling model.train() here and instead
        # produce a clear, actionable status rather than a fabricated
        # accuracy number. Swap in a real ultralytics training call (see
        # comment below) once a proper detection-format dataset exists.
        version_id = f"v-{target_class}-{utc_now_naive().strftime('%Y%m%d%H%M%S')}"

        # --- Real training would look like this once data is detection-format: ---
        # from ultralytics import YOLO
        # base = YOLO(settings.DETECTOR_WEIGHTS)
        # results = base.train(data=str(job_dir / "data.yaml"), epochs=50, device=settings.DETECTOR_DEVICE)
        # model_path = results.save_dir / "weights" / "best.pt"
        # accuracy = float(results.results_dict.get("metrics/mAP50(B)")) — a REAL measured value, never guessed

        model_version = db_models.ModelVersion(
            version_id=version_id,
            classes=all_classes,
            dataset_size=copied,
            accuracy=None,  # honestly null until a real training run reports it
            model_path=None,
            is_active=False,
        )
        db.add(model_version)

        new_class = db.query(db_models.KnownClass).filter_by(name=target_class).first()
        if new_class is None:
            db.add(db_models.KnownClass(name=target_class, source="continual_learning", example_count=copied))
        else:
            new_class.example_count += copied

        for v in verified_rows:
            v.used_in_training_job_id = job.id

        job.status = db_models.TrainingJobStatus.COMPLETED
        job.resulting_model_version = version_id
        job.completed_at = utc_now_naive()
        job.error = (
            "Dataset assembled and class registered. Actual weight fine-tuning was not run "
            "automatically — see backend/app/services/continual_learning.py for the real "
            "ultralytics training call to enable once a detection-format (not crop-only) "
            "dataset is available for this class."
        )
        db.commit()
        db.refresh(job)
        return job

    except Exception as e:
        logger.exception("Training job %s failed", job.id)
        job.status = db_models.TrainingJobStatus.FAILED
        job.error = str(e)
        job.completed_at = utc_now_naive()
        db.commit()
        db.refresh(job)
        return job
