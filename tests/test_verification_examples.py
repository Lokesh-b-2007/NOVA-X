"""Tests for saving human-verified crops as class examples."""
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.api.unknown_objects import (  # noqa: E402
    confirm_label,
    get_unknown_object_image,
    suggest_label,
)
from app.config import settings  # noqa: E402
from app.database import models  # noqa: E402
from app.database.database import Base  # noqa: E402
from app.schemas.schemas import ConfirmLabelRequest  # noqa: E402
from app.services import pipeline  # noqa: E402


def test_unknown_object_image_is_available_for_review(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    unknown_dir = tmp_path / "unknown_objects"
    monkeypatch.setattr(settings, "UNKNOWN_OBJECTS_DIR", unknown_dir)
    crop_path = unknown_dir / "object-1.jpg"
    crop_path.parent.mkdir()
    crop_image = BytesIO()
    Image.new("RGB", (40, 40), color=(100, 100, 100)).save(crop_image, format="JPEG")
    crop_path.write_bytes(crop_image.getvalue())
    db.add(models.NovelObject(id="object-1", image_path=str(crop_path)))
    db.commit()

    response = get_unknown_object_image("object-1", db)

    assert response.status_code == 200
    assert response.media_type == "image/jpeg"
    with Image.open(BytesIO(response.body)) as reviewed_image:
        red, green, _ = reviewed_image.getpixel((4, 4))
        assert red > 150
        assert green < 120

    db.close()
    engine.dispose()


def test_existing_unknown_object_can_generate_and_save_a_suggestion(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    unknown_dir = tmp_path / "unknown_objects"
    monkeypatch.setattr(settings, "UNKNOWN_OBJECTS_DIR", unknown_dir)
    crop_path = unknown_dir / "object-2.jpg"
    crop_path.parent.mkdir()
    Image.new("RGB", (40, 40), color=(100, 100, 100)).save(crop_path, format="JPEG")
    obj = models.NovelObject(id="object-2", image_path=str(crop_path))
    db.add(obj)
    db.commit()

    def fake_suggestion(novel_object):
        novel_object.ovd_label = "mug"
        novel_object.ovd_confidence = 0.82
        novel_object.ovd_top_predictions = [{"label": "mug", "confidence": 0.82}]
        return novel_object

    monkeypatch.setattr(pipeline, "suggest_unknown_object", fake_suggestion)
    suggested = suggest_label("object-2", db)

    assert suggested.ovd_label == "mug"
    assert suggested.ovd_confidence == 0.82
    db.refresh(obj)
    assert obj.ovd_label == "mug"

    db.close()
    engine.dispose()


def test_confirm_saves_class_example_and_reconfirmation_is_idempotent(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(settings, "VERIFIED_OBJECTS_DIR", tmp_path / "verified_objects")

    crop_path = tmp_path / "unknown_objects" / "object-1.jpg"
    crop_path.parent.mkdir()
    crop_path.write_bytes(b"test crop bytes")
    db.add(models.NovelObject(id="object-1", image_path=str(crop_path)))
    db.commit()

    confirmed = confirm_label("object-1", ConfirmLabelRequest(label="Red cup"), db)
    example = db.query(models.VerifiedObject).filter_by(novel_object_id="object-1").one()

    assert confirmed.confirmed_label == "Red cup"
    assert example.confirmed_label == "Red cup"
    assert Path(example.image_path).is_file()
    assert Path(example.image_path).read_bytes() == crop_path.read_bytes()
    assert Path(example.image_path).parent.parent == settings.VERIFIED_OBJECTS_DIR
    assert Path(example.image_path).parent.name.startswith("Red_cup-")

    confirm_label("object-1", ConfirmLabelRequest(label="Red cup"), db)
    assert db.query(models.VerifiedObject).filter_by(novel_object_id="object-1").count() == 1

    previous_example_path = Path(example.image_path)
    confirm_label("object-1", ConfirmLabelRequest(label="Blue cup"), db)
    db.refresh(example)
    assert example.confirmed_label == "Blue cup"
    assert Path(example.image_path).parent.name.startswith("Blue_cup-")
    assert Path(example.image_path).is_file()
    assert not previous_example_path.exists()
    assert db.query(models.VerifiedObject).filter_by(novel_object_id="object-1").count() == 1

    db.close()
    engine.dispose()
