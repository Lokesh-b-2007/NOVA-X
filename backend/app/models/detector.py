"""
Known-Class Object Detector
------------------------------
Thin, testable wrapper around an Ultralytics YOLO model. Loads lazily (so
importing this module never triggers a model download / GPU init) and
raises DetectorLoadError with a clear message if the model can't load —
the API layer converts this into a 503, never a fake result.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("nova_x.detector")


class DetectorLoadError(Exception):
    """Raised when the YOLO model or its weights cannot be loaded."""


@dataclass
class RawDetection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates
    embedding: Optional[np.ndarray] = None  # penultimate-layer feature vector, if extracted


class YoloDetector:
    def __init__(self, weights: str = None, device: str = None, conf_threshold: float = None):
        self.weights = weights or settings.DETECTOR_WEIGHTS
        self.device = device or settings.DETECTOR_DEVICE
        self.conf_threshold = conf_threshold if conf_threshold is not None else settings.DETECTOR_CONF_THRESHOLD
        self._model = None
        self.version_tag = Path(self.weights).stem

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorLoadError(
                "The 'ultralytics' package is not installed. Run "
                "`pip install -r requirements.txt` in the backend/ directory."
            ) from e

        resolved_device = self._resolve_device()
        try:
            logger.info("Loading YOLO weights '%s' on device '%s'...", self.weights, resolved_device)
            self._model = YOLO(self.weights)
            self.device = resolved_device
        except Exception as e:
            raise DetectorLoadError(
                f"Failed to load YOLO weights '{self.weights}': {e}. "
                "If this is a custom NOVA-X checkpoint, verify the path in "
                "NOVA_DETECTOR_WEIGHTS points to a valid .pt file."
            ) from e

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        """Run inference on a single BGR image (numpy array) and return raw detections."""
        self._load()
        try:
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise DetectorLoadError(f"YOLO inference failed: {e}") from e

        detections: list[RawDetection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                RawDetection(
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a feature embedding for a cropped region, for use in OOD
        distance scoring and clustering. Uses the YOLO backbone's pooled
        features when available; returns None (never a fake vector) if
        the underlying model doesn't expose one and no fallback is loaded.
        """
        self._load()
        try:
            import torch
            import cv2

            if crop is None or crop.size == 0:
                return None

            resized = cv2.resize(crop, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device if self.device != "auto" else "cpu")

            model = self._model.model
            features = []
            hook = model.model[-2].register_forward_hook(
                lambda *args: features.append(args[-1])
            )
            try:
                with torch.no_grad():
                    model(tensor)
            finally:
                hook.remove()

            if not features or not torch.is_tensor(features[0]) or features[0].ndim != 4:
                logger.warning("Detector did not expose a spatial penultimate-layer feature tensor.")
                return None

            pooled = torch.nn.functional.adaptive_avg_pool2d(features[0], 1).flatten(1)
            return pooled.cpu().numpy()[0]
        except Exception as e:
            logger.warning("Embedding extraction failed, continuing without it: %s", e)
            return None


_detector_instance: Optional[YoloDetector] = None


def get_detector() -> YoloDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YoloDetector()
    return _detector_instance
