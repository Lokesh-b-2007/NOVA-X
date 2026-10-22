"""
Model Evaluation
------------------
Computes the metrics required by the spec:
    Known detection : precision, recall, mAP@50, mAP@50-95  (via Ultralytics val)
    OOD             : AUROC, AUPR, FPR95                    (via sklearn, from
                          COCO128 ground-truth-matched boxes or labeled CSV data)
    Unknown         : unknown recall / precision             (derived from the
                      same labeled eval set)
    Open-Vocabulary : top-1 / top-5 accuracy                 (from a labeled
                      OVD eval set)

HONESTY REQUIREMENT: this module never fabricates a metric. If the required
labeled evaluation data isn't present on disk, the corresponding metric
block is returned as None with an explanatory note, rather than a made-up
number.

Evaluation assets, all under data/:
    data/known_dataset/eval/data.yaml         - Ultralytics validation definition
    data/known_dataset/eval/known_metrics.json - output from run_known_eval.py
    data/known_dataset/eval/ood_metrics.json   - output from run_ood_eval.py
    data/known_dataset/eval/ood_labels.csv    - optional manual OOD labels
    data/known_dataset/eval/ovd_labels.csv    - optional manual OVD labels
"""
import csv
import hashlib
import json
import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models as db_models

logger = logging.getLogger("nova_x.evaluation")

EVAL_DIR = settings.KNOWN_DATASET_DIR / "eval"
KNOWN_METRICS_CACHE = EVAL_DIR / "known_metrics.json"
OOD_METRICS_CACHE = EVAL_DIR / "ood_metrics.json"


def evaluate_known_detector() -> Optional[dict]:
    """Return cached Ultralytics metrics from scripts/run_known_eval.py."""
    if not KNOWN_METRICS_CACHE.is_file():
        return None

    try:
        cached = json.loads(KNOWN_METRICS_CACHE.read_text(encoding="utf-8"))
        data_yaml = EVAL_DIR / "data.yaml"
        yaml_hash = hashlib.sha256(data_yaml.read_bytes()).hexdigest() if data_yaml.is_file() else None
        configured_classes = sorted(settings.KNOWN_CLASSES)
        if (
            cached.get("weights") != settings.DETECTOR_WEIGHTS
            or cached.get("data_yaml_sha256") != yaml_hash
            or cached.get("known_classes") != configured_classes
        ):
            logger.warning(
                "Cached known metrics do not match the current weights, validation data, or known classes."
            )
            return None
        return {
            key: float(cached[key])
            for key in ("precision", "recall", "map_50", "map_50_95")
        }
    except Exception as e:
        logger.warning("Could not load cached known-detector metrics: %s", e)
        return None


def _current_known_classes(db: Session) -> list[str]:
    classes = {name.strip().casefold() for name in settings.KNOWN_CLASSES if name.strip()}
    classes.update(
        row.name.strip().casefold()
        for row in db.query(db_models.KnownClass).all()
        if row.name.strip()
    )
    return sorted(classes)


def evaluate_ood(db: Session) -> Optional[dict]:
    """Read real ground-truth-matched detector metrics computed from COCO128."""
    cached = _load_ood_cache(db)
    if cached is not None:
        return {key: float(cached[key]) for key in ("auroc", "aupr", "fpr95")}
    return _evaluate_ood_labels(db)


def get_ood_sample_counts(db: Session) -> Optional[dict]:
    cached = _load_ood_cache(db)
    if cached is not None:
        return {
            "ood": int(cached["matched_detections"]),
            "unknown": int(cached["ground_truth_unknown_objects"]),
            "embedding_extraction_available": cached.get("embedding_extraction_available"),
        }
    labels_csv = EVAL_DIR / "ood_labels.csv"
    if labels_csv.is_file():
        try:
            with labels_csv.open(newline="", encoding="utf-8") as source:
                return {"ood": sum(1 for _ in csv.DictReader(source)), "unknown": None}
        except OSError as e:
            logger.warning("Could not count OOD label rows: %s", e)
    return None


def evaluate_unknown_detection(db: Session) -> Optional[dict]:
    """Return unknown-object precision/recall measured on ground-truth-matched COCO128."""
    cached = _load_ood_cache(db)
    if cached is not None:
        return {
            "unknown_recall": float(cached["unknown_recall"]),
            "unknown_precision": float(cached["unknown_precision"]),
        }
    return _evaluate_unknown_labels(db)


def evaluate_ovd(db: Session) -> Optional[dict]:
    """Score OWLv2 predictions against human-confirmed object labels."""
    verified = _verified_ovd_examples(db)
    metrics, total = _score_ovd_examples(verified)
    if total:
        return metrics
    return _evaluate_ovd_labels(db)


def get_ovd_sample_count(db: Session) -> int:
    examples = _verified_ovd_examples(db)
    count = sum(1 for _, predictions in examples if predictions)
    if count:
        return count
    labels_csv = EVAL_DIR / "ovd_labels.csv"
    if not labels_csv.is_file():
        return 0
    try:
        with labels_csv.open(newline="", encoding="utf-8") as source:
            rows = csv.DictReader(source)
            return sum(
                1
                for row in rows
                if row.get("novel_object_id")
                and row.get("true_label")
                and (novel := db.get(db_models.NovelObject, row["novel_object_id"])) is not None
                and novel.ovd_top_predictions
            )
    except (OSError, KeyError) as e:
        logger.warning("Could not count OVD label rows: %s", e)
        return 0


def get_ovd_sample_source(db: Session) -> Optional[str]:
    if any(predictions for _, predictions in _verified_ovd_examples(db)):
        return "human-confirmed crops"
    return "custom folder-labeled photos" if get_ovd_sample_count(db) else None


def _verified_ovd_examples(db: Session) -> list:
    return (
        db.query(db_models.VerifiedObject.confirmed_label, db_models.NovelObject.ovd_top_predictions)
        .join(db_models.NovelObject, db_models.NovelObject.id == db_models.VerifiedObject.novel_object_id)
        .filter(db_models.NovelObject.ovd_top_predictions.is_not(None))
        .all()
    )


def _score_ovd_examples(examples) -> tuple[Optional[dict], int]:
    top1_hits = top5_hits = total = 0
    for true_label, top_predictions in examples:
        if not top_predictions:
            continue
        predictions = [
            _normalize_ovd_label(prediction.get("label", ""))
            for prediction in top_predictions
        ]
        normalized_true_label = _normalize_ovd_label(true_label)
        total += 1
        if predictions and predictions[0] == normalized_true_label:
            top1_hits += 1
        if normalized_true_label in predictions[:5]:
            top5_hits += 1
    if not total:
        return None, 0
    return {
        "top1_accuracy": round(top1_hits / total, 4),
        "top5_accuracy": round(top5_hits / total, 4),
    }, total


def _evaluate_ovd_labels(db: Session) -> Optional[dict]:
    labels_csv = EVAL_DIR / "ovd_labels.csv"
    if not labels_csv.is_file():
        return None
    examples = []
    try:
        with labels_csv.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                novel_object = db.get(db_models.NovelObject, row["novel_object_id"])
                if novel_object is not None and novel_object.ovd_top_predictions:
                    examples.append((row["true_label"], novel_object.ovd_top_predictions))
    except (OSError, KeyError) as e:
        logger.warning("Could not read OVD label rows: %s", e)
        return None
    metrics, _ = _score_ovd_examples(examples)
    return metrics


def _load_ood_cache(db: Session) -> Optional[dict]:
    if not OOD_METRICS_CACHE.is_file():
        return None
    try:
        cached = json.loads(OOD_METRICS_CACHE.read_text(encoding="utf-8"))
        data_yaml = EVAL_DIR / "data.yaml"
        yaml_hash = hashlib.sha256(data_yaml.read_bytes()).hexdigest() if data_yaml.is_file() else None
        if (
            cached.get("weights") != settings.DETECTOR_WEIGHTS
            or cached.get("data_yaml_sha256") != yaml_hash
            or cached.get("known_classes") != _current_known_classes(db)
            or cached.get("ood_method") != settings.OOD_METHOD
            or cached.get("unknown_threshold") != settings.NOVELTY_UNKNOWN_THRESHOLD
            or cached.get("suspicious_threshold") != settings.NOVELTY_SUSPICIOUS_THRESHOLD
            or cached.get("feature_scoring_revision") != 2
        ):
            logger.warning("Cached OOD metrics do not match the current weights, labels, or scoring settings.")
            return None
        required = {
            "auroc", "aupr", "fpr95", "unknown_recall", "unknown_precision",
            "matched_detections", "ground_truth_unknown_objects",
            "embedding_extraction_available",
            "feature_scoring_revision",
        }
        if not required.issubset(cached):
            logger.warning("Cached OOD metrics are missing required fields.")
            return None
        return cached
    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.warning("Could not load cached OOD metrics: %s", e)
        return None


def _evaluate_ood_labels(db: Session) -> Optional[dict]:
    labels_csv = EVAL_DIR / "ood_labels.csv"
    if not labels_csv.is_file():
        return None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    except ImportError:
        logger.warning("scikit-learn not installed; cannot compute OOD metrics.")
        return None

    y_true, y_score = [], []
    with labels_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            detection = db.get(db_models.Detection, row["detection_id"])
            if detection is not None:
                y_true.append(int(row["is_unknown"]))
                y_score.append(detection.novelty_score or 0.0)
    if len(set(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = min(int(np.searchsorted(tpr, 0.95)), len(fpr) - 1)
    return {
        "auroc": round(float(roc_auc_score(y_true, y_score)), 4),
        "aupr": round(float(average_precision_score(y_true, y_score)), 4),
        "fpr95": round(float(fpr[idx]), 4),
    }


def _evaluate_unknown_labels(db: Session) -> Optional[dict]:
    labels_csv = EVAL_DIR / "ood_labels.csv"
    if not labels_csv.is_file():
        return None
    true_positive = false_positive = false_negative = 0
    with labels_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            detection = db.get(db_models.Detection, row["detection_id"])
            if detection is None:
                continue
            predicted_unknown = detection.novelty_status == db_models.NoveltyStatus.UNKNOWN
            true_unknown = bool(int(row["is_unknown"]))
            true_positive += int(predicted_unknown and true_unknown)
            false_positive += int(predicted_unknown and not true_unknown)
            false_negative += int(not predicted_unknown and true_unknown)
    if true_positive + false_negative == 0:
        return None
    return {
        "unknown_recall": round(true_positive / (true_positive + false_negative), 4),
        "unknown_precision": round(
            true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
            4,
        ),
    }


def _normalize_ovd_label(label: str) -> str:
    normalized = " ".join((label or "").strip().casefold().split())
    return {"cell phone": "phone", "mobile phone": "phone"}.get(normalized, normalized)
