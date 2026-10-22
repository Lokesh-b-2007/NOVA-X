"""
Novel Object Repository + Human Verification Endpoints
---------------------------------------------------------
GET    /api/unknown-objects
GET    /api/unknown-objects/{id}
GET    /api/unknown-objects/{id}/image
DELETE /api/unknown-objects/{id}
DELETE /api/unknown-objects
POST   /api/unknown-objects/{id}/confirm
POST   /api/unknown-objects/{id}/reject
POST   /api/unknown-objects/{id}/suggest
"""
import logging
import hashlib
from io import BytesIO
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import get_db
from app.database import models as db_models
from app.schemas.schemas import NovelObjectOut, ConfirmLabelRequest
from app.models.open_vocabulary import OWLv2LoadError
from app.services import pipeline

logger = logging.getLogger("nova_x.api.unknown_objects")
router = APIRouter(prefix="/api/unknown-objects", tags=["unknown-objects"])


@router.get("", response_model=list[NovelObjectOut])
def list_unknown_objects(
    status: Optional[str] = Query(None, description="Filter by verification_status"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(db_models.NovelObject).order_by(db_models.NovelObject.detected_at.desc())
    if status:
        q = q.filter(db_models.NovelObject.verification_status == status)
    return q.limit(limit).all()


@router.get("/{object_id}", response_model=NovelObjectOut)
def get_unknown_object(object_id: str, db: Session = Depends(get_db)):
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")
    return obj


@router.get("/{object_id}/image")
def get_unknown_object_image(object_id: str, db: Session = Depends(get_db)):
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")

    image_path = Path(obj.image_path).resolve()
    image_dir = settings.UNKNOWN_OBJECTS_DIR.resolve()
    if image_dir not in image_path.parents or not image_path.is_file():
        raise HTTPException(404, "The object's crop is unavailable.")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    inset = min(4, max(0, (min(image.size) - 1) // 2))
    line_width = max(2, min(image.size) // 50)
    draw.rectangle(
        (inset, inset, image.width - 1 - inset, image.height - 1 - inset),
        outline=(255, 64, 64),
        width=line_width,
    )
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG", quality=95)
    return Response(content=image_bytes.getvalue(), media_type="image/jpeg")


@router.post("/{object_id}/suggest", response_model=NovelObjectOut)
def suggest_label(object_id: str, db: Session = Depends(get_db)):
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")
    if obj.ovd_label:
        return obj

    try:
        pipeline.suggest_unknown_object(obj)
        db.commit()
    except FileNotFoundError as e:
        raise HTTPException(404, "The object's crop is missing; a suggestion cannot be generated.") from e
    except OWLv2LoadError as e:
        db.rollback()
        raise HTTPException(503, f"Open-vocabulary suggestion unavailable: {e}") from e

    return obj


@router.delete("/{object_id}")
def delete_unknown_object(object_id: str, db: Session = Depends(get_db)):
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")

    verified_examples = db.query(db_models.VerifiedObject).filter_by(novel_object_id=obj.id).all()
    image_paths = {obj.image_path, *(example.image_path for example in verified_examples)}
    cluster_id = obj.cluster_id

    try:
        db.query(db_models.Detection).filter_by(novel_object_id=obj.id).update(
            {db_models.Detection.novel_object_id: None}, synchronize_session=False
        )
        db.query(db_models.VerifiedObject).filter_by(novel_object_id=obj.id).delete(
            synchronize_session=False
        )
        db.delete(obj)
        db.flush()
        if cluster_id and not db.query(db_models.NovelObject).filter_by(cluster_id=cluster_id).first():
            db.query(db_models.NovelCluster).filter_by(id=cluster_id).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        raise

    _remove_files(image_paths)
    return {"deleted": True, "id": object_id}


def _remove_files(image_paths: set[str]) -> None:
    for image_path in image_paths:
        path = Path(image_path)
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("Could not delete image file %s: %s", path, e)


@router.delete("")
def clear_unknown_objects(db: Session = Depends(get_db)):
    """Delete all repository objects and related crops while preserving detection history."""
    objects = db.query(db_models.NovelObject).all()
    object_ids = [obj.id for obj in objects]
    image_paths = {obj.image_path for obj in objects if obj.image_path}

    if not object_ids:
        db.query(db_models.NovelCluster).delete(synchronize_session=False)
        db.commit()
        return {"deleted": 0}

    try:
        verified_examples = db.query(db_models.VerifiedObject).filter(
            db_models.VerifiedObject.novel_object_id.in_(object_ids)
        ).all()
        image_paths.update(example.image_path for example in verified_examples if example.image_path)
        db.query(db_models.Detection).filter(
            db_models.Detection.novel_object_id.in_(object_ids)
        ).update({db_models.Detection.novel_object_id: None}, synchronize_session=False)
        db.query(db_models.VerifiedObject).filter(
            db_models.VerifiedObject.novel_object_id.in_(object_ids)
        ).delete(synchronize_session=False)
        db.query(db_models.NovelObject).filter(
            db_models.NovelObject.id.in_(object_ids)
        ).delete(synchronize_session=False)
        db.query(db_models.NovelCluster).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        raise

    _remove_files(image_paths)

    logger.info("Cleared %d object(s) from the unknown object repository", len(object_ids))
    return {"deleted": len(object_ids)}


@router.post("/{object_id}/confirm", response_model=NovelObjectOut)
def confirm_label(object_id: str, payload: ConfirmLabelRequest, db: Session = Depends(get_db)):
    """
    Human verification: the reviewer confirms the AI's OVD suggestion (or
    supplies their own corrected label). This is the ONLY way a NovelObject
    can graduate into the VerifiedObject training queue.
    """
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")

    label = payload.label.strip()
    if not label:
        raise HTTPException(400, "A confirmed class label cannot be empty.")

    source_path = Path(obj.image_path)
    if not source_path.is_file():
        raise HTTPException(404, "The object's crop is missing; it cannot be added as a verified example.")

    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-") or "class"
    label_hash = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
    class_dir = settings.VERIFIED_OBJECTS_DIR / f"{safe_label[:60]}-{label_hash}"
    example_path = class_dir / f"{obj.id}{source_path.suffix.lower() or '.jpg'}"
    existing = (
        db.query(db_models.VerifiedObject)
        .filter_by(novel_object_id=obj.id)
        .order_by(db_models.VerifiedObject.created_at.desc())
        .first()
    )
    old_example_path = Path(existing.image_path) if existing and existing.image_path else None
    old_confirmed_label = existing.confirmed_label if existing else None
    example_preexisted = example_path.exists()

    try:
        class_dir.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != example_path.resolve() and not example_preexisted:
            shutil.copy2(source_path, example_path)
    except OSError as e:
        logger.exception("Could not save verified example for object %s", obj.id)
        raise HTTPException(500, "Could not save the verified example image.") from e

    obj.confirmed_label = label
    obj.verification_status = db_models.VerificationStatus.CONFIRMED

    if existing:
        existing.image_path = str(example_path)
        existing.confirmed_label = label
        if old_confirmed_label != label:
            existing.used_in_training_job_id = None
    else:
        db.add(
            db_models.VerifiedObject(
                novel_object_id=obj.id,
                image_path=str(example_path),
                confirmed_label=label,
            )
        )

    # Keep the cluster's suggested label fresh if this object belongs to one.
    if obj.cluster_id:
        cluster = db.get(db_models.NovelCluster, obj.cluster_id)
        if cluster:
            cluster.suggested_label = label

    try:
        db.commit()
    except Exception:
        db.rollback()
        if not example_preexisted and example_path.exists() and example_path != source_path:
            example_path.unlink(missing_ok=True)
        raise

    if old_example_path and old_example_path != source_path and old_example_path != example_path:
        _remove_files({str(old_example_path)})
    db.refresh(obj)
    return obj


@router.post("/{object_id}/reject", response_model=NovelObjectOut)
def reject_label(object_id: str, db: Session = Depends(get_db)):
    """
    Human verification: reviewer indicates the AI's suggestion (and/or the
    novelty call itself) was wrong. The object is marked REJECTED and is
    excluded from clustering/training rather than deleted, preserving the
    audit trail.
    """
    obj = db.get(db_models.NovelObject, object_id)
    if not obj:
        raise HTTPException(404, "Unknown object not found.")

    obj.verification_status = db_models.VerificationStatus.REJECTED
    db.commit()
    db.refresh(obj)
    return obj
