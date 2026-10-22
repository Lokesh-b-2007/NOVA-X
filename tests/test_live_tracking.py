"""Tests for deduplicating detections across webcam frames."""
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.database import models  # noqa: E402
from app.database.database import Base  # noqa: E402
from app.models.detector import RawDetection  # noqa: E402
from app.services import pipeline  # noqa: E402


def test_webcam_tracks_known_and_unknown_objects_once(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(settings, "UNKNOWN_OBJECTS_DIR", tmp_path / "unknown_objects")

    class FakeDetector:
        version_tag = "test"

        detections = []

        def detect(self, image):
            return self.detections

        def extract_embedding(self, crop):
            return None

    class FakeBank:
        def is_empty(self):
            return False

        def update(self, class_name, embedding):
            pass

    score_calls = []
    bank = FakeBank()

    detector = FakeDetector()
    class FakeRecognizer:
        calls = 0

        def recognize(self, crop):
            self.calls += 1
            return {
                "label": "chair",
                "confidence": 0.8,
                "top_predictions": [{"label": "chair", "confidence": 0.8}],
            }

    recognizer = FakeRecognizer()
    monkeypatch.setattr(pipeline, "get_detector", lambda: detector)
    monkeypatch.setattr(pipeline, "get_recognizer", lambda: recognizer)
    monkeypatch.setattr(pipeline, "get_default_bank", lambda: bank)
    monkeypatch.setattr(
        pipeline._scorer,
        "score",
        lambda detector_confidence, **kwargs: (
            score_calls.append((detector_confidence, kwargs.get("bank")))
            or SimpleNamespace(
                status="KNOWN" if detector_confidence >= 0.8 else "UNKNOWN",
                novelty_score=0.0 if detector_confidence >= 0.8 else 0.9,
                known_similarity=None,
                reason="test",
            )
        ),
    )

    stream_id = str(uuid.uuid4())
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detections = [
        RawDetection("person", 0.95, (10, 10, 40, 40)),
        RawDetection("chair", 0.70, (60, 10, 90, 40)),
    ]
    first_frame = pipeline.process_image(db, image, source="webcam", stream_id=stream_id)
    first_ids = {item["class_name"] or "unknown": item["id"] for item in first_frame["detections"]}
    assert len(first_frame["detections"]) == 2
    assert first_frame["detections"][1]["ovd"]["label"] == "chair"

    detector.detections = [
        RawDetection("person", 0.95, (12, 10, 42, 40)),
        RawDetection("chair", 0.70, (62, 10, 92, 40)),
    ]
    second_frame = pipeline.process_image(db, image, source="webcam", stream_id=stream_id)
    assert {item["class_name"] or "unknown": item["id"] for item in second_frame["detections"]} == first_ids
    assert db.query(models.Detection).count() == 2
    assert db.query(models.NovelObject).count() == 1
    assert recognizer.calls == 1

    detector.detections = [
        RawDetection("person", 0.95, (14, 10, 44, 40)),
        RawDetection("chair", 0.70, (64, 10, 94, 40)),
        RawDetection("dog", 0.95, (10, 60, 40, 90)),
        RawDetection("cup", 0.70, (60, 60, 90, 90)),
    ]
    third_frame = pipeline.process_image(db, image, source="webcam", stream_id=stream_id)
    assert len(third_frame["detections"]) == 4
    assert db.query(models.Detection).count() == 4
    assert db.query(models.NovelObject).count() == 2
    assert third_frame["known_count"] == 2
    assert third_frame["unknown_count"] == 2
    assert score_calls == [
        (0.95, bank),
        (0.0, None),
        (0.95, bank),
        (0.0, None),
    ]
    assert recognizer.calls == 2

    db.close()
    engine.dispose()


def test_registered_phone_maps_detector_cell_phone_to_known(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        models.KnownClass(
            name="phone",
            source="continual_learning",
            example_count=31,
        )
    )
    db.commit()

    class FakeDetector:
        version_tag = "test"

        def detect(self, image):
            return [RawDetection("cell phone", 0.42, (10, 10, 40, 40))]

        def extract_embedding(self, crop):
            return None

    class FakeBank:
        def is_empty(self):
            return True

    def unexpected_novelty_scoring(**kwargs):
        raise AssertionError("Registered continual-learning class should be recognized directly.")

    monkeypatch.setattr(pipeline, "get_detector", lambda: FakeDetector())
    monkeypatch.setattr(pipeline, "get_default_bank", lambda: FakeBank())
    monkeypatch.setattr(pipeline._scorer, "score", unexpected_novelty_scoring)

    result = pipeline.process_image(
        db,
        np.zeros((100, 100, 3), dtype=np.uint8),
        source="image",
    )

    assert result["known_count"] == 1
    assert result["unknown_count"] == 0
    assert result["detections"][0]["class_name"] == "phone"
    assert result["detections"][0]["novelty"]["status"] == "KNOWN"
    assert db.query(models.NovelObject).count() == 0
    detection_row = db.query(models.Detection).one()
    assert detection_row.class_name == "phone"
    assert detection_row.novel_object_id is None

    db.close()
    engine.dispose()
