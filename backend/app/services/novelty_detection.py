"""
OOD / Novelty Detection Module
---------------------------------
This module deliberately does NOT rely on a single confidence threshold.
It combines multiple signals into one Novelty Score in [0, 1]:

  1. Confidence-based baseline   — how sure the detector is of its own guess.
  2. Distance-based novelty      — cosine distance from the nearest known-class
                                    embedding centroid (requires an embedding
                                    bank; degrades gracefully without one).
  3. Energy-based OOD (optional) — a logit-energy style score, when raw class
                                    logits are available from the detector.

Each sub-score is independently swappable — see NoveltyScorer.method — so a
more advanced research method (e.g. a learned OOD head, Mahalanobis distance,
KNN in feature space) can replace any one component without touching the
pipeline or the API layer.

Classification thresholds (configurable in app.config):
    score < NOVELTY_SUSPICIOUS_THRESHOLD                      -> KNOWN
    NOVELTY_SUSPICIOUS_THRESHOLD <= score < NOVELTY_UNKNOWN_.. -> SUSPICIOUS
    score >= NOVELTY_UNKNOWN_THRESHOLD                         -> UNKNOWN
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import settings


@dataclass
class NoveltyOutcome:
    status: str  # "KNOWN" | "SUSPICIOUS" | "UNKNOWN"
    novelty_score: float
    known_similarity: Optional[float]
    reason: str


class KnownClassEmbeddingBank:
    """
    Running centroid store of feature embeddings per known class. Used for
    distance-based novelty scoring. In this reference implementation the
    bank starts empty and is built up online from confident known
    detections (see pipeline.py) — a production system would instead seed
    it offline from the training set.
    """

    def __init__(self):
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def update(self, class_name: str, embedding: np.ndarray, momentum: float = 0.9):
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        if class_name not in self._centroids:
            self._centroids[class_name] = embedding
            self._counts[class_name] = 1
        else:
            prev = self._centroids[class_name]
            updated = momentum * prev + (1 - momentum) * embedding
            self._centroids[class_name] = updated / (np.linalg.norm(updated) + 1e-8)
            self._counts[class_name] += 1

    def max_similarity(self, embedding: np.ndarray) -> Optional[float]:
        """Highest cosine similarity to any known-class centroid, or None if the bank is empty."""
        if not self._centroids:
            return None
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        sims = [float(np.dot(embedding, c)) for c in self._centroids.values()]
        return max(sims)

    def is_empty(self) -> bool:
        return not self._centroids


# A single process-wide bank shared across requests. Reset by restarting the
# server (or explicitly, for tests).
_default_bank = KnownClassEmbeddingBank()


def get_default_bank() -> KnownClassEmbeddingBank:
    return _default_bank


class NoveltyScorer:
    """
    Combines confidence, distance and (optionally) energy sub-scores into a
    single novelty score. `method` selects which sub-scores are active:
        "confidence" - baseline only
        "distance"   - embedding distance only (falls back to confidence if no bank/embedding given)
        "energy"     - energy-based only (falls back to confidence if no logits given)
        "hybrid"     - weighted combination of whichever signals are available (default)
    """

    def __init__(self, method: str = "hybrid"):
        self.method = method

    # --- individual sub-scores, each returns a value in [0, 1] where higher = more novel ---

    @staticmethod
    def _confidence_score(confidence: float) -> float:
        # Low detector confidence => high novelty contribution.
        return float(np.clip(1.0 - confidence, 0.0, 1.0))

    @staticmethod
    def _distance_score(embedding: Optional[np.ndarray], bank: Optional[KnownClassEmbeddingBank]) -> Optional[float]:
        if embedding is None or bank is None or bank.is_empty():
            return None
        similarity = bank.max_similarity(embedding)
        # cosine similarity in [-1, 1] -> novelty in [0, 1] (1 - similarity, rescaled)
        return float(np.clip((1.0 - similarity) / 2.0, 0.0, 1.0))

    @staticmethod
    def _energy_score(logits: Optional[np.ndarray]) -> Optional[float]:
        if logits is None:
            return None
        # Free-energy score: E(x) = -logsumexp(logits). Higher energy (less
        # negative / larger) typically indicates OOD. We min-max normalize
        # with a fixed heuristic range since this reference model doesn't
        # ship pre-computed energy statistics; replace with dataset-derived
        # min/max for a calibrated version.
        energy = -float(np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits))
        return float(np.clip((energy + 10.0) / 20.0, 0.0, 1.0))

    def score(
        self,
        detector_confidence: float,
        embedding: Optional[np.ndarray] = None,
        bank: Optional[KnownClassEmbeddingBank] = None,
        logits: Optional[np.ndarray] = None,
    ) -> NoveltyOutcome:
        conf_score = self._confidence_score(detector_confidence)
        dist_score = self._distance_score(embedding, bank)
        energy_score = self._energy_score(logits)

        known_similarity = None
        if embedding is not None and bank is not None and not bank.is_empty():
            known_similarity = bank.max_similarity(embedding)

        active = {"confidence": conf_score}
        if dist_score is not None:
            active["distance"] = dist_score
        if energy_score is not None:
            active["energy"] = energy_score

        if self.method == "confidence":
            final_score = conf_score
        elif self.method == "distance" and dist_score is not None:
            final_score = dist_score
        elif self.method == "energy" and energy_score is not None:
            final_score = energy_score
        else:
            # hybrid (or a requested method whose signal isn't available):
            # weighted average of whatever signals are present, weighted
            # toward distance/energy since they are more discriminative
            # than confidence alone when available.
            weights = {"confidence": 1.0, "distance": 2.0, "energy": 1.5}
            total_weight = sum(weights[k] for k in active)
            final_score = sum(active[k] * weights[k] for k in active) / total_weight

        final_score = float(np.clip(final_score, 0.0, 1.0))

        if final_score >= settings.NOVELTY_UNKNOWN_THRESHOLD:
            status = "UNKNOWN"
            reason = "High novelty score — " + _dominant_reason(active)
        elif final_score >= settings.NOVELTY_SUSPICIOUS_THRESHOLD:
            status = "SUSPICIOUS"
            reason = "Moderate novelty score — " + _dominant_reason(active)
        else:
            status = "KNOWN"
            reason = "Low novelty score — detection matches known-class distribution"

        return NoveltyOutcome(
            status=status,
            novelty_score=round(final_score, 4),
            known_similarity=round(known_similarity, 4) if known_similarity is not None else None,
            reason=reason,
        )


def _dominant_reason(active: dict) -> str:
    dominant = max(active, key=active.get)
    labels = {
        "confidence": "low detector confidence in any known class",
        "distance": "high distance from known class embeddings",
        "energy": "high free-energy relative to known-class logits",
    }
    return labels.get(dominant, "combined signal")
