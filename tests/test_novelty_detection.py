"""
Unit tests for the OOD/novelty scoring module. These run without any
GPU or downloaded model weights — they exercise the pure scoring logic.

Run with:
    cd backend && pip install -r requirements.txt && pytest ../tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.novelty_detection import NoveltyScorer, KnownClassEmbeddingBank  # noqa: E402


def test_high_confidence_is_known():
    scorer = NoveltyScorer(method="hybrid")
    outcome = scorer.score(detector_confidence=0.95)
    assert outcome.status == "KNOWN"


def test_low_confidence_is_unknown():
    scorer = NoveltyScorer(method="hybrid")
    outcome = scorer.score(detector_confidence=0.05)
    assert outcome.status in ("UNKNOWN", "SUSPICIOUS")


def test_non_allowlisted_detection_remains_unknown_without_embedding_bank():
    scorer = NoveltyScorer(method="hybrid")
    bank = KnownClassEmbeddingBank()
    known_embedding = np.ones(32, dtype=np.float32)
    bank.update("person", known_embedding)

    outcome = scorer.score(
        detector_confidence=0.0,
        embedding=-known_embedding,
        bank=None,
    )

    assert outcome.status == "UNKNOWN"
    assert outcome.novelty_score == 1.0


def test_embedding_bank_exact_match_is_known():
    scorer = NoveltyScorer(method="hybrid")
    bank = KnownClassEmbeddingBank()
    emb = np.random.rand(32).astype(np.float32)
    emb /= np.linalg.norm(emb)
    bank.update("person", emb)

    outcome = scorer.score(detector_confidence=0.9, embedding=emb, bank=bank)
    assert outcome.status == "KNOWN"
    assert outcome.known_similarity == pytest.approx(1.0, abs=1e-3)


def test_embedding_bank_opposite_vector_is_more_novel():
    scorer = NoveltyScorer(method="hybrid")
    bank = KnownClassEmbeddingBank()
    emb = np.random.rand(32).astype(np.float32)
    emb /= np.linalg.norm(emb)
    bank.update("person", emb)

    outcome_match = scorer.score(detector_confidence=0.9, embedding=emb, bank=bank)
    outcome_opposite = scorer.score(detector_confidence=0.9, embedding=-emb, bank=bank)
    assert outcome_opposite.novelty_score > outcome_match.novelty_score


def test_score_bounds():
    scorer = NoveltyScorer(method="hybrid")
    for conf in [0.0, 0.5, 1.0]:
        outcome = scorer.score(detector_confidence=conf)
        assert 0.0 <= outcome.novelty_score <= 1.0
