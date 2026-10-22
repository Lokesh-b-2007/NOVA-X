"""
Detection Endpoints
--------------------
POST /api/detect/image  - full-resolution image upload (Image Detection page)
POST /api/detect/frame  - lower-resolution webcam frame (Live Detection page)

Both endpoints run the same pipeline; /frame additionally downsizes the
image first. Webcam requests include a stream ID so detections are tracked
across frames and each object is only recorded once per camera session.
"""
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.detector import DetectorLoadError
from app.services import pipeline
from app.utils import image_utils
from app.schemas.schemas import ImageDetectionResponse
from app.config import settings

logger = logging.getLogger("nova_x.api.detect")
router = APIRouter(prefix="/api/detect", tags=["detection"])


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        image = image_utils.decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        result = pipeline.process_image(db, image, source="image")
    except DetectorLoadError as e:
        # Never fake a result if the model can't load — surface a real error.
        raise HTTPException(503, f"Detector unavailable: {e}")

    return result


@router.post("/frame", response_model=ImageDetectionResponse)
async def detect_frame(
    file: UploadFile = File(...),
    stream_id: str = Form(None),
    db: Session = Depends(get_db),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded frame is empty.")

    try:
        image = image_utils.decode_image_bytes(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    image = image_utils.resize_for_streaming(image, max_width=settings.MAX_FRAME_WIDTH)

    try:
        result = pipeline.process_image(db, image, source="webcam", stream_id=stream_id)
    except DetectorLoadError as e:
        raise HTTPException(503, f"Detector unavailable: {e}")

    return result
