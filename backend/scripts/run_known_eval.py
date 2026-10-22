"""Run known-class validation once and cache real metrics for the Evaluation page.

Run from backend/ with its virtual environment active:
    python scripts/run_known_eval.py
    python scripts/run_known_eval.py --all-classes
"""
import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.models.detector import DetectorLoadError, get_detector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_known_eval")

EVAL_DIR = settings.KNOWN_DATASET_DIR / "eval"
DATA_YAML = EVAL_DIR / "data.yaml"
CACHE = EVAL_DIR / "known_metrics.json"


def main():
    parser = argparse.ArgumentParser(description="Compute and cache known-detector metrics.")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Evaluate every class instead of only NOVA_KNOWN_CLASSES.",
    )
    args = parser.parse_args()

    if not DATA_YAML.is_file():
        raise SystemExit(f"Missing validation config: {DATA_YAML}")

    detector = get_detector()
    try:
        detector._load()  # noqa: SLF001
    except DetectorLoadError as e:
        raise SystemExit(f"Could not load the detector: {e}") from e

    class_ids = None
    if not args.all_classes:
        class_ids = sorted(
            class_id
            for class_id, name in detector._model.names.items()  # noqa: SLF001
            if name in settings.KNOWN_CLASSES
        )
        if not class_ids:
            raise SystemExit(
                "None of NOVA_KNOWN_CLASSES are present in the detector's labels. "
                "Use --all-classes to evaluate the available model classes."
            )

    kwargs = {
        "data": str(DATA_YAML),
        "device": detector.device,
        "verbose": False,
        "workers": 0,
    }
    if class_ids:
        kwargs["classes"] = class_ids

    logger.info("Validating %s on %s; this may take several minutes on CPU.", settings.DETECTOR_WEIGHTS, DATA_YAML)
    try:
        metrics = detector._model.val(**kwargs)  # noqa: SLF001
    except Exception as e:
        raise SystemExit(f"Ultralytics validation failed: {e}") from e

    result = {
        "weights": settings.DETECTOR_WEIGHTS,
        "data_yaml_sha256": hashlib.sha256(DATA_YAML.read_bytes()).hexdigest(),
        "known_classes": sorted(settings.KNOWN_CLASSES),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map_50": round(float(metrics.box.map50), 4),
        "map_50_95": round(float(metrics.box.map), 4),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info("Saved known metrics to %s", CACHE)
    for key in ("precision", "recall", "map_50", "map_50_95"):
        logger.info("%s: %.4f", key, result[key])
    logger.warning(
        "The supplied COCO128 validation split is not held out and the base weights were trained on COCO; "
        "treat these numbers as a sanity check, not an unbiased benchmark."
    )


if __name__ == "__main__":
    main()
