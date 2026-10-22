"""
NOVA-X Configuration
=====================
Central settings object. Everything that could change between machines
(paths, thresholds, model choice) lives here and is overridable via
environment variables / a .env file. Never hardcode secrets or absolute
paths elsewhere in the codebase.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = BASE_DIR.parent  # NOVA-X/

_DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", REPO_ROOT / "data")).resolve()
_MODELS_DIR = Path(os.getenv("NOVA_MODELS_DIR", REPO_ROOT / "models")).resolve()


class Settings:
    # --- General ---
    APP_NAME: str = "NOVA-X"
    ENV: str = os.getenv("NOVA_ENV", "development")
    DEBUG: bool = os.getenv("NOVA_DEBUG", "true").lower() == "true"
    CORS_ORIGINS: list = os.getenv("NOVA_CORS_ORIGINS", "http://localhost:5173").split(",")

    # --- Database ---
    # SQLite for dev by default. Swap DATABASE_URL to a postgres:// URL to
    # move to Postgres later -- SQLAlchemy models do not change.
    DATABASE_URL: str = os.getenv("NOVA_DATABASE_URL", f"sqlite:///{REPO_ROOT}/nova_x.db")

    # --- Storage paths ---
    DATA_DIR: Path = _DATA_DIR
    KNOWN_DATASET_DIR: Path = _DATA_DIR / "known_dataset"
    UNKNOWN_OBJECTS_DIR: Path = _DATA_DIR / "unknown_objects"
    VERIFIED_OBJECTS_DIR: Path = _DATA_DIR / "verified_objects"
    MODELS_DIR: Path = _MODELS_DIR
    TRAINED_MODELS_DIR: Path = _MODELS_DIR / "trained_models"

    # --- Known-class detector (YOLO) ---
    # Any Ultralytics-compatible weights path or hub name. Defaults to the
    # small pretrained COCO checkpoint (auto-downloaded by ultralytics on
    # first use) so the system runs out of the box.
    DETECTOR_WEIGHTS: str = os.getenv("NOVA_DETECTOR_WEIGHTS", "yolov8n.pt")
    DETECTOR_DEVICE: str = os.getenv("NOVA_DETECTOR_DEVICE", "auto")  # "auto" | "cpu" | "cuda:0"
    DETECTOR_CONF_THRESHOLD: float = float(os.getenv("NOVA_DETECTOR_CONF", 0.35))

    # Which COCO / custom classes count as "known" for this deployment.
    # Anything YOLO detects outside this allow-list is routed to the OOD
    # module rather than displayed as a known detection.
    KNOWN_CLASSES: list = [
        c.strip() for c in os.getenv("NOVA_KNOWN_CLASSES", "person,car,bicycle,dog").split(",") if c.strip()
    ]

    # --- OOD / novelty detection ---
    NOVELTY_UNKNOWN_THRESHOLD: float = float(os.getenv("NOVA_NOVELTY_UNKNOWN_THRESHOLD", 0.70))
    NOVELTY_SUSPICIOUS_THRESHOLD: float = float(os.getenv("NOVA_NOVELTY_SUSPICIOUS_THRESHOLD", 0.40))
    OOD_METHOD: str = os.getenv("NOVA_OOD_METHOD", "hybrid")  # "confidence" | "distance" | "energy" | "hybrid"

    # --- Open-vocabulary model (OWLv2) ---
    OVD_MODEL_ID: str = os.getenv("NOVA_OVD_MODEL_ID", "google/owlv2-base-patch16-ensemble")
    OVD_DEVICE: str = os.getenv("NOVA_OVD_DEVICE", "auto")
    OVD_SCORE_THRESHOLD: float = float(os.getenv("NOVA_OVD_SCORE_THRESHOLD", 0.1))
    OVD_VOCABULARY: list = [
        c.strip()
        for c in os.getenv(
            "NOVA_OVD_VOCABULARY",
            "chair,sofa,couch,table,desk,bed,bench,bookshelf,cabinet,door,"
            "laptop,computer,phone,tablet,keyboard,mouse,monitor,television,"
            "camera,headphones,remote control,printer,charger,"
            "bottle,cup,mug,glass,plate,bowl,fork,knife,spoon,pot,pan,"
            "backpack,handbag,suitcase,umbrella,book,notebook,pen,scissors,"
            "plant,flower,tree,lamp,clock,mirror, vase,shoe,hat,glasses,"
            "box,toy,ball,teddy bear,tool,hammer,wrench,"
            "cat,bird,horse,sheep,cow,elephant,bear,zebra,giraffe,"
            "motorcycle,bus,train,truck,airplane,boat, skateboard,"
            "traffic light,fire hydrant,stop sign,parking meter",
        ).split(",")
        if c.strip()
    ]

    # --- Clustering ---
    CLUSTERING_METHOD: str = os.getenv("NOVA_CLUSTERING_METHOD", "dbscan")  # "dbscan" | "hdbscan"
    DBSCAN_EPS: float = float(os.getenv("NOVA_DBSCAN_EPS", 0.35))
    DBSCAN_MIN_SAMPLES: int = int(os.getenv("NOVA_DBSCAN_MIN_SAMPLES", 3))

    # --- Continual learning ---
    MIN_DATASET_SIZE_FOR_TRAINING: int = int(os.getenv("NOVA_MIN_DATASET_SIZE", 30))

    # --- Webcam / streaming ---
    MAX_FRAME_WIDTH: int = int(os.getenv("NOVA_MAX_FRAME_WIDTH", 640))


settings = Settings()

# Ensure storage directories exist on import so services can write to them
# without every caller needing its own mkdir logic.
for _dir in (
    settings.KNOWN_DATASET_DIR,
    settings.UNKNOWN_OBJECTS_DIR,
    settings.VERIFIED_OBJECTS_DIR,
    settings.TRAINED_MODELS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
