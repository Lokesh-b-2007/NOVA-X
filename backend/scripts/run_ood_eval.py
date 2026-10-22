"""Evaluate novelty scores against the annotated COCO128 ground-truth boxes.

Run from backend/ with its virtual environment active:
    python scripts/run_ood_eval.py
"""
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.database import models as db_models  # noqa: E402
from app.database.database import SessionLocal  # noqa: E402
from app.models.detector import DetectorLoadError, get_detector  # noqa: E402
from app.services.novelty_detection import KnownClassEmbeddingBank, NoveltyOutcome, NoveltyScorer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ood_eval")

EVAL_DIR = settings.KNOWN_DATASET_DIR / "eval"
DATA_YAML = EVAL_DIR / "data.yaml"
METRICS_CACHE = EVAL_DIR / "ood_metrics.json"
BACKEND_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BACKEND_DIR / "datasets" / "coco128" / "images" / "train2017"
LABEL_DIR = BACKEND_DIR / "datasets" / "coco128" / "labels" / "train2017"
CLASS_ALIASES = {"cell phone": "phone", "mobile phone": "phone", "mobile telephone": "phone"}


def _normalize_class(name: str) -> str:
    normalized = " ".join(name.strip().casefold().split())
    return CLASS_ALIASES.get(normalized, normalized)


def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _read_ground_truth(label_path: Path, width: int, height: int, names: dict) -> list[dict]:
    objects = []
    if not label_path.is_file():
        return objects
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:5])
        objects.append({
            "class_name": names[class_id],
            "bbox": (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ),
        })
    return objects


def _match_predictions(predictions: list, ground_truth: list, iou_threshold: float = 0.5) -> dict:
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            overlap = _box_iou(prediction.bbox, truth["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, prediction_index, truth_index))
    matches = {}
    used_truth = set()
    for _, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index not in matches and truth_index not in used_truth:
            matches[prediction_index] = truth_index
            used_truth.add(truth_index)
    return matches


def main():
    if not DATA_YAML.is_file():
        raise SystemExit(f"Missing evaluation config: {DATA_YAML}")
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(
            f"COCO128 images/labels are missing. Run `python scripts/run_known_eval.py` first "
            f"to download the supplied dataset (expected {IMAGE_DIR})."
        )

    detector = get_detector()
    try:
        detector._load()  # noqa: SLF001
    except DetectorLoadError as e:
        raise SystemExit(f"Could not load detector: {e}") from e

    model_names = {int(key): value for key, value in detector._model.names.items()}  # noqa: SLF001
    db = SessionLocal()
    try:
        registered_classes = {
            _normalize_class(name): (name, "base")
            for name in settings.KNOWN_CLASSES
            if name.strip()
        }
        cache_known_classes = {name.strip().casefold() for name in settings.KNOWN_CLASSES if name.strip()}
        for row in db.query(db_models.KnownClass).all():
            if row.name.strip():
                registered_classes[_normalize_class(row.name)] = (row.name.strip(), row.source)
                cache_known_classes.add(row.name.strip().casefold())
    finally:
        db.close()
    known_classes = set(registered_classes)
    scorer = NoveltyScorer(method=settings.OOD_METHOD)
    bank = KnownClassEmbeddingBank()
    embedding_extraction_available = None
    true_unknown_labels, novelty_scores = [], []
    ground_truth_unknown_count = 0
    matched_unknown_count = 0
    predicted_unknown_count = 0
    true_positive_unknown_count = 0
    matched_count = 0

    images = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No COCO128 validation images found under {IMAGE_DIR}.")

    for image_index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read validation image: {image_path}")
        height, width = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        ground_truth = _read_ground_truth(label_path, width, height, model_names)
        ground_truth_unknown_count += sum(
            _normalize_class(obj["class_name"]) not in known_classes for obj in ground_truth
        )

        predictions = detector.detect(image)
        matches = _match_predictions(predictions, ground_truth)
        matched_count += len(matches)
        matched_unknown_count += sum(
            _normalize_class(ground_truth[truth_index]["class_name"]) not in known_classes
            for truth_index in matches.values()
        )

        for prediction_index, prediction in enumerate(predictions):
            embedding = None
            if embedding_extraction_available is not False:
                crop = image[
                    max(0, int(prediction.bbox[1])):min(height, int(prediction.bbox[3])),
                    max(0, int(prediction.bbox[0])):min(width, int(prediction.bbox[2])),
                ]
                if crop.size:
                    embedding = detector.extract_embedding(crop)
                    if embedding_extraction_available is None:
                        embedding_extraction_available = embedding is not None
            prediction_class = _normalize_class(prediction.class_name)
            registered = registered_classes.get(prediction_class)
            is_known_prediction = registered is not None
            if is_known_prediction and registered[1] == "continual_learning":
                outcome = NoveltyOutcome(
                    status="KNOWN",
                    novelty_score=round(1.0 - prediction.confidence, 4),
                    known_similarity=None,
                    reason=f"Detector label matches learned known class '{registered[0]}'",
                )
            else:
                outcome = scorer.score(
                    detector_confidence=prediction.confidence if is_known_prediction else 0.0,
                    embedding=embedding,
                    bank=bank if is_known_prediction else None,
                )
            predicted_unknown = outcome.status == "UNKNOWN"
            predicted_unknown_count += int(predicted_unknown)

            truth_index = matches.get(prediction_index)
            true_unknown = (
                _normalize_class(ground_truth[truth_index]["class_name"]) not in known_classes
                if truth_index is not None
                else False
            )
            true_unknown_labels.append(int(true_unknown))
            novelty_scores.append(outcome.novelty_score)
            if true_unknown and predicted_unknown:
                true_positive_unknown_count += 1
            if outcome.status == "KNOWN" and is_known_prediction and embedding is not None:
                bank.update(registered[0], embedding)

        if image_index % 16 == 0 or image_index == len(images):
            logger.info("Evaluated %d/%d COCO128 images", image_index, len(images))

    if not true_unknown_labels or len(set(true_unknown_labels)) < 2:
        raise SystemExit("COCO128 evaluation produced insufficient matched known and unknown detections.")

    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    false_negative_unknown_count = ground_truth_unknown_count - true_positive_unknown_count
    false_positive_unknown_count = predicted_unknown_count - true_positive_unknown_count
    false_positive_unknown_count = max(0, false_positive_unknown_count)
    fpr, tpr, _ = roc_curve(true_unknown_labels, novelty_scores)
    fpr95_index = min(int(next((i for i, value in enumerate(tpr) if value >= 0.95), len(tpr) - 1)), len(fpr) - 1)
    result = {
        "weights": settings.DETECTOR_WEIGHTS,
        "data_yaml_sha256": hashlib.sha256(DATA_YAML.read_bytes()).hexdigest(),
        "known_classes": sorted(cache_known_classes),
        "ood_method": settings.OOD_METHOD,
        "unknown_threshold": settings.NOVELTY_UNKNOWN_THRESHOLD,
        "suspicious_threshold": settings.NOVELTY_SUSPICIOUS_THRESHOLD,
        "embedding_extraction_available": embedding_extraction_available,
        "feature_scoring_revision": 2,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "auroc": round(float(roc_auc_score(true_unknown_labels, novelty_scores)), 4),
        "aupr": round(float(average_precision_score(true_unknown_labels, novelty_scores)), 4),
        "fpr95": round(float(fpr[fpr95_index]), 4),
        "unknown_recall": round(
            true_positive_unknown_count / ground_truth_unknown_count, 4
        ) if ground_truth_unknown_count else 0.0,
        "unknown_precision": round(
            true_positive_unknown_count / predicted_unknown_count, 4
        ) if predicted_unknown_count else 0.0,
        "matched_detections": matched_count,
        "ground_truth_unknown_objects": ground_truth_unknown_count,
        "matched_unknown_objects": matched_unknown_count,
        "predicted_unknown_detections": predicted_unknown_count,
        "true_positive_unknown_detections": true_positive_unknown_count,
        "false_negative_unknown_objects": false_negative_unknown_count,
        "false_positive_unknown_detections": false_positive_unknown_count,
        "matching_iou_threshold": 0.5,
        "dataset": "COCO128 annotated validation boxes",
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CACHE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved OOD metrics to %s", METRICS_CACHE)
    logger.info(
        "AUROC %.4f | AUPR %.4f | FPR95 %.4f | unknown recall %.4f | precision %.4f",
        result["auroc"],
        result["aupr"],
        result["fpr95"],
        result["unknown_recall"],
        result["unknown_precision"],
    )
    logger.warning(
        "COCO128 is not held out and the default detector was trained on COCO. "
        "Unknown-object recall includes annotated objects missed by the detector; "
        "OOD ranking uses detected boxes matched at IoU >= 0.5 plus unmatched predictions as negatives."
    )


if __name__ == "__main__":
    main()
