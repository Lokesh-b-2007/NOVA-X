"""
Detection Pipeline Orchestrator
-----------------------------------
Implements the exact flow from the spec:

    image -> known-class detector -> per-box OOD/novelty scoring
        -> KNOWN: return as-is, update the embedding bank
        -> UNKNOWN/SUSPICIOUS: crop -> OWLv2 open-vocabulary guess
            -> persist/merge into the Novel Object Repository
            -> extract embedding for later clustering

This module is the ONLY place that decides what counts as "known" vs
"unknown" (the novelty module) and it is called BEFORE the open-vocabulary
model — OVD is only ever a secondary semantic guess for something already
flagged unknown by the OOD module, never a substitute for it.
"""
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models as db_models
from app.models.detector import get_detector
from app.models.open_vocabulary import get_recognizer, OWLv2LoadError
from app.services.novelty_detection import NoveltyOutcome, NoveltyScorer, get_default_bank
from app.utils import image_utils

logger = logging.getLogger("nova_x.pipeline")

_scorer = NoveltyScorer(method=settings.OOD_METHOD)

_DETECTOR_CLASS_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "mobile telephone": "phone",
}


def _registered_class_match(
    raw_class: str,
    registered_classes: dict[str, tuple[str, str]],
) -> tuple[Optional[str], Optional[str]]:
    normalized = raw_class.strip().casefold()
    match = registered_classes.get(normalized)
    if match:
        return match

    alias = _DETECTOR_CLASS_ALIASES.get(normalized)
    if alias:
        match = registered_classes.get(alias)
        if match:
            return match
    return None, None


class _LiveDetectionTrack:
    def __init__(self, raw_class: str, bbox: tuple, detection: dict):
        self.raw_class = raw_class
        self.bbox = bbox
        self.detection = detection
        self.missed_frames = 0
        self.last_seen_frame = 0


class _LiveDetectionTracker:
    """Associates detections across webcam frames and keeps one record per object."""

    MAX_MISSED_FRAMES = 5
    STREAM_TTL_SECONDS = 600
    MAX_STREAMS = 128

    def __init__(self):
        self._streams = {}
        self._lock = threading.Lock()

    def match(self, stream_id: str, raw_detections: list) -> dict:
        now = time.monotonic()
        with self._lock:
            self._expire_streams(now)
            if stream_id not in self._streams and len(self._streams) >= self.MAX_STREAMS:
                oldest = min(
                    self._streams,
                    key=lambda key: self._streams[key].get("last_seen", 0),
                )
                del self._streams[oldest]
            stream = self._streams.setdefault(stream_id, {"tracks": {}, "frame": 0})
            stream["last_seen"] = now
            stream["frame"] += 1
            tracks = stream["tracks"]

            for track in tracks.values():
                track.missed_frames += 1

            candidates = []
            for detection_index, raw in enumerate(raw_detections):
                for track_id, track in tracks.items():
                    if raw.class_name != track.raw_class:
                        continue
                    overlap = _intersection_over_union(raw.bbox, track.bbox)
                    distance = _normalized_center_distance(raw.bbox, track.bbox)
                    if overlap >= 0.25:
                        candidates.append((overlap, detection_index, track_id))
                    elif distance <= 0.75:
                        candidates.append((0.25 - distance / 10, detection_index, track_id))

            matches = {}
            used_tracks = set()
            for _, detection_index, track_id in sorted(candidates, reverse=True):
                if detection_index in matches or track_id in used_tracks:
                    continue
                raw = raw_detections[detection_index]
                track = tracks[track_id]
                track.bbox = tuple(raw.bbox)
                track.detection["bbox"] = _bbox_dict(raw.bbox)
                track.missed_frames = 0
                track.last_seen_frame = stream["frame"]
                matches[detection_index] = track
                used_tracks.add(track_id)

            for track_id in [
                track_id for track_id, track in tracks.items()
                if track.missed_frames > self.MAX_MISSED_FRAMES
            ]:
                del tracks[track_id]
            return matches

    def add(self, stream_id: str, raw, detection: dict) -> None:
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return
            track = _LiveDetectionTrack(raw.class_name, tuple(raw.bbox), detection)
            track.last_seen_frame = stream["frame"]
            stream["tracks"][detection["id"]] = track

    def detections(self, stream_id: str) -> list:
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return []
            return [
                track.detection
                for track in sorted(
                    stream["tracks"].values(),
                    key=lambda item: item.last_seen_frame,
                    reverse=True,
                )
            ]

    def _expire_streams(self, now: float) -> None:
        expired = [
            stream_id for stream_id, stream in self._streams.items()
            if now - stream.get("last_seen", now) > self.STREAM_TTL_SECONDS
        ]
        for stream_id in expired:
            del self._streams[stream_id]

def _bbox_dict(bbox: tuple) -> dict:
    return {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}


def _intersection_over_union(first: tuple, second: tuple) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _normalized_center_distance(first: tuple, second: tuple) -> float:
    first_width = max(first[2] - first[0], 1)
    first_height = max(first[3] - first[1], 1)
    second_width = max(second[2] - second[0], 1)
    second_height = max(second[3] - second[1], 1)
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    scale = max(first_width, second_width, first_height, second_height)
    return (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5 / scale


_live_tracker = _LiveDetectionTracker()


def process_image(
    db: Session,
    image: np.ndarray,
    source: str = "image",
    stream_id: str = None,
) -> dict:
    """
    Runs the full pipeline on a single BGR image and returns a dict matching
    schemas.ImageDetectionResponse. Raises DetectorLoadError if the known-
    class detector can't load — the caller (API layer) turns that into a 503.
    """
    start = time.time()
    detector = get_detector()
    bank = get_default_bank()
    registered_classes = {
        name.strip().casefold(): (name, "base")
        for name in settings.KNOWN_CLASSES
        if name.strip()
    }
    for known_class in db.query(db_models.KnownClass).all():
        registered_classes[known_class.name.strip().casefold()] = (
            known_class.name,
            known_class.source,
        )

    raw_detections = detector.detect(image)  # may raise DetectorLoadError — not caught here
    live_matches = _live_tracker.match(stream_id, raw_detections) if stream_id else {}

    detections_out = []
    new_live_tracks = []
    known_count = 0
    unknown_count = 0

    for index, raw in enumerate(raw_detections):
        if index in live_matches:
            continue
        crop = image_utils.crop_box(image, raw.bbox)
        embedding = detector.extract_embedding(crop)

        registered_name, registered_source = _registered_class_match(raw.class_name, registered_classes)
        is_allowlisted_known = registered_name is not None
        is_learned_class = registered_source == "continual_learning"
        output_class_name = registered_name if is_allowlisted_known else None
        # Base classes still pass the OOD check. A registered learned class
        # is recognized when the detector emits its label or a known alias.
        if is_learned_class:
            # Continual-learning examples register semantic labels, not new
            # YOLO weights. Only mark it known when YOLO itself emits that
            # label (or a supported alias such as COCO's "cell phone").
            outcome = NoveltyOutcome(
                status="KNOWN",
                novelty_score=round(1.0 - raw.confidence, 4),
                known_similarity=None,
                reason=f"Detector label matches learned known class '{registered_name}'",
            )
        else:
            outcome = _scorer.score(
                detector_confidence=raw.confidence if is_allowlisted_known else 0.0,
                embedding=embedding,
                bank=bank if is_allowlisted_known else None,
            )

        detection_row = db_models.Detection(
            source=source,
            class_name=output_class_name,
            confidence=raw.confidence if is_allowlisted_known else None,
            bbox_x1=raw.bbox[0], bbox_y1=raw.bbox[1], bbox_x2=raw.bbox[2], bbox_y2=raw.bbox[3],
            novelty_score=outcome.novelty_score,
            novelty_status=db_models.NoveltyStatus(outcome.status),
            novelty_reason=outcome.reason,
            model_version=detector.version_tag,
        )

        detection_out = {
            "id": None,  # filled in after DB insert below
            "class_name": output_class_name,
            "confidence": raw.confidence if is_allowlisted_known else None,
            "bbox": {"x1": raw.bbox[0], "y1": raw.bbox[1], "x2": raw.bbox[2], "y2": raw.bbox[3]},
            "novelty": {
                "status": outcome.status,
                "novelty_score": outcome.novelty_score,
                "known_similarity": outcome.known_similarity,
                "reason": outcome.reason,
            },
            "ovd": None,
            "ovd_top_predictions": None,
            "model_version": detector.version_tag,
            "novel_object_id": None,
        }

        if outcome.status == "KNOWN":
            known_count += 1
            if embedding is not None and is_allowlisted_known:
                bank.update(output_class_name, embedding)
        else:
            unknown_count += 1
            novel_object = _handle_unknown(
                db,
                crop,
                embedding,
                outcome.novelty_score,
            )
            detection_row.novel_object_id = novel_object.id
            detection_out["novel_object_id"] = novel_object.id
            if novel_object.ovd_label:
                detection_out["ovd"] = {
                    "label": novel_object.ovd_label,
                    "confidence": novel_object.ovd_confidence,
                }
                detection_out["ovd_top_predictions"] = novel_object.ovd_top_predictions

        db.add(detection_row)
        db.flush()  # assign detection_row.id without committing yet
        detection_out["id"] = detection_row.id
        detections_out.append(detection_out)
        if stream_id:
            new_live_tracks.append((raw, detection_out))

    db.commit()

    if stream_id:
        for raw, detection in new_live_tracks:
            _live_tracker.add(stream_id, raw, detection)
        detections_out = _live_tracker.detections(stream_id)
        known_count = sum(item["novelty"]["status"] == "KNOWN" for item in detections_out)
        unknown_count = len(detections_out) - known_count

    return {
        "detections": detections_out,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "processing_time_ms": (time.time() - start) * 1000,
    }


def _handle_unknown(
    db: Session,
    crop: np.ndarray,
    embedding,
    novelty_score: float,
) -> db_models.NovelObject:
    """
    Persists a crop to the Novel Object Repository and runs OWLv2 once to
    suggest a label. Never invents a label if OWLv2 cannot load.
    """
    object_id = str(uuid.uuid4())
    crop_path = settings.UNKNOWN_OBJECTS_DIR / f"{object_id}.jpg"
    image_utils.save_crop(crop, crop_path)

    novel_object = db_models.NovelObject(
        id=object_id,
        image_path=str(crop_path),
        novelty_score=novelty_score,
        embedding=embedding.tolist() if embedding is not None else None,
        occurrence_count=1,
        verification_status=db_models.VerificationStatus.PENDING,
    )

    try:
        _apply_open_vocabulary_suggestion(novel_object, crop)
    except OWLv2LoadError as e:
        logger.warning("OWLv2 unavailable, storing unknown object without a label guess: %s", e)

    db.add(novel_object)
    db.flush()
    return novel_object


def suggest_unknown_object(novel_object: db_models.NovelObject) -> db_models.NovelObject:
    """Run OWLv2 for an existing unknown crop that has no saved suggestion."""
    crop_path = settings.UNKNOWN_OBJECTS_DIR / Path(novel_object.image_path).name
    if not crop_path.is_file():
        raise FileNotFoundError(f"Unknown-object crop is missing: {crop_path}")

    crop = image_utils.decode_image_bytes(crop_path.read_bytes())
    _apply_open_vocabulary_suggestion(novel_object, crop)
    return novel_object


def _apply_open_vocabulary_suggestion(
    novel_object: db_models.NovelObject,
    crop: np.ndarray,
) -> None:
    recognizer = get_recognizer()
    result = recognizer.recognize(image_utils.bgr_to_rgb(crop))
    novel_object.ovd_label = result["label"]
    novel_object.ovd_confidence = result["confidence"]
    novel_object.ovd_top_predictions = result["top_predictions"]
