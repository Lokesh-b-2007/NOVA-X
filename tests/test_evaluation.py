"""Tests for cached known-detector evaluation results."""
import hashlib
import json
import sys
from pathlib import Path
import csv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.database import models  # noqa: E402
from app.database.database import Base  # noqa: E402
from app.services import evaluation  # noqa: E402


def test_known_metrics_load_when_cache_matches_current_setup(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    data_yaml = eval_dir / "data.yaml"
    data_yaml.write_text("path: coco128\n", encoding="utf-8")
    cache_path = eval_dir / "known_metrics.json"
    monkeypatch.setattr(evaluation, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(evaluation, "KNOWN_METRICS_CACHE", cache_path)
    monkeypatch.setattr(settings, "DETECTOR_WEIGHTS", "yolov8n.pt")
    monkeypatch.setattr(settings, "KNOWN_CLASSES", ["person", "car"])

    metrics = {
        "precision": 0.9,
        "recall": 0.8,
        "map_50": 0.85,
        "map_50_95": 0.7,
    }
    cache_path.write_text(
        json.dumps({
            "weights": "yolov8n.pt",
            "data_yaml_sha256": hashlib.sha256(data_yaml.read_bytes()).hexdigest(),
            "known_classes": ["car", "person"],
            **metrics,
        }),
        encoding="utf-8",
    )

    assert evaluation.evaluate_known_detector() == metrics


def test_known_metrics_reject_stale_cache(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    data_yaml = eval_dir / "data.yaml"
    data_yaml.write_text("path: coco128\n", encoding="utf-8")
    cache_path = eval_dir / "known_metrics.json"
    cache_path.write_text(
        json.dumps({
            "weights": "different.pt",
            "data_yaml_sha256": hashlib.sha256(data_yaml.read_bytes()).hexdigest(),
            "known_classes": sorted(settings.KNOWN_CLASSES),
            "precision": 0.9,
            "recall": 0.8,
            "map_50": 0.85,
            "map_50_95": 0.7,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(evaluation, "KNOWN_METRICS_CACHE", cache_path)

    assert evaluation.evaluate_known_detector() is None


def test_known_metrics_are_unavailable_without_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "KNOWN_METRICS_CACHE", tmp_path / "missing.json")

    assert evaluation.evaluate_known_detector() is None


def test_ood_metrics_and_unknown_counts_load_from_current_cache(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    data_yaml = eval_dir / "data.yaml"
    data_yaml.write_text("path: coco128\n", encoding="utf-8")
    cache_path = eval_dir / "ood_metrics.json"
    monkeypatch.setattr(evaluation, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(evaluation, "OOD_METRICS_CACHE", cache_path)
    monkeypatch.setattr(settings, "DETECTOR_WEIGHTS", "yolov8n.pt")
    monkeypatch.setattr(settings, "KNOWN_CLASSES", ["person"])
    monkeypatch.setattr(settings, "OOD_METHOD", "hybrid")
    monkeypatch.setattr(settings, "NOVELTY_UNKNOWN_THRESHOLD", 0.7)
    monkeypatch.setattr(settings, "NOVELTY_SUSPICIOUS_THRESHOLD", 0.4)
    cache_path.write_text(json.dumps({
        "weights": "yolov8n.pt",
        "data_yaml_sha256": hashlib.sha256(data_yaml.read_bytes()).hexdigest(),
        "known_classes": ["person"],
        "ood_method": "hybrid",
        "unknown_threshold": 0.7,
        "suspicious_threshold": 0.4,
        "feature_scoring_revision": 2,
        "auroc": 0.82,
        "aupr": 0.71,
        "fpr95": 0.32,
        "unknown_recall": 0.68,
        "unknown_precision": 0.77,
        "matched_detections": 91,
        "ground_truth_unknown_objects": 55,
        "embedding_extraction_available": True,
    }), encoding="utf-8")

    assert evaluation.evaluate_ood(db) == {"auroc": 0.82, "aupr": 0.71, "fpr95": 0.32}
    assert evaluation.evaluate_unknown_detection(db) == {
        "unknown_recall": 0.68,
        "unknown_precision": 0.77,
    }
    assert evaluation.get_ood_sample_counts(db) == {
        "ood": 91,
        "unknown": 55,
        "embedding_extraction_available": True,
    }
    db.close()
    engine.dispose()


def test_ood_metrics_reject_cache_when_scoring_settings_change(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    data_yaml = eval_dir / "data.yaml"
    data_yaml.write_text("path: coco128\n", encoding="utf-8")
    cache_path = eval_dir / "ood_metrics.json"
    monkeypatch.setattr(evaluation, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(evaluation, "OOD_METRICS_CACHE", cache_path)
    monkeypatch.setattr(settings, "KNOWN_CLASSES", ["person"])
    cache_path.write_text(json.dumps({
        "weights": settings.DETECTOR_WEIGHTS,
        "data_yaml_sha256": hashlib.sha256(data_yaml.read_bytes()).hexdigest(),
        "known_classes": ["person"],
        "ood_method": "confidence",
        "unknown_threshold": settings.NOVELTY_UNKNOWN_THRESHOLD,
        "suspicious_threshold": settings.NOVELTY_SUSPICIOUS_THRESHOLD,
    }), encoding="utf-8")

    monkeypatch.setattr(settings, "OOD_METHOD", "hybrid")
    assert evaluation.evaluate_ood(db) is None
    db.close()
    engine.dispose()


def test_ovd_metrics_use_human_confirmed_labels_and_aliases(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    novel_object = models.NovelObject(
        id="novel-phone",
        image_path="phone.jpg",
        ovd_top_predictions=[
            {"label": "cell phone", "confidence": 0.8},
            {"label": "tablet", "confidence": 0.2},
        ],
    )
    db.add(novel_object)
    db.add(models.VerifiedObject(
        id="verified-phone",
        novel_object_id="novel-phone",
        image_path="verified-phone.jpg",
        confirmed_label="phone",
    ))
    db.commit()

    assert evaluation.evaluate_ovd(db) == {"top1_accuracy": 1.0, "top5_accuracy": 1.0}
    assert evaluation.get_ovd_sample_count(db) == 1
    assert evaluation.get_ovd_sample_source(db) == "human-confirmed crops"
    db.close()
    engine.dispose()


def test_ovd_uses_folder_labeled_csv_when_no_verified_examples_exist(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    db.add(models.NovelObject(
        id="novel-cup",
        image_path="cup.jpg",
        ovd_top_predictions=[{"label": "mug", "confidence": 0.7}],
    ))
    db.commit()
    with (tmp_path / "ovd_labels.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["novel_object_id", "true_label"])
        writer.writeheader()
        writer.writerow({"novel_object_id": "novel-cup", "true_label": "mug"})

    assert evaluation.evaluate_ovd(db) == {"top1_accuracy": 1.0, "top5_accuracy": 1.0}
    assert evaluation.get_ovd_sample_count(db) == 1
    assert evaluation.get_ovd_sample_source(db) == "custom folder-labeled photos"
    db.close()
    engine.dispose()
