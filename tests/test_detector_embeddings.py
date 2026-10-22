"""Tests for extracting detector features through the model's routed forward pass."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.models.detector import YoloDetector  # noqa: E402


class RoutedFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.Identity(),
            nn.Conv2d(4, 2, kernel_size=1),
        )

    def forward(self, tensor):
        return self.model(tensor)


def test_extract_embedding_runs_full_model_and_pools_penultimate_features():
    detector = YoloDetector(device="cpu")
    detector._model = SimpleNamespace(model=RoutedFeatureModel())

    embedding = detector.extract_embedding(np.zeros((32, 32, 3), dtype=np.uint8))

    assert embedding is not None
    assert embedding.shape == (4,)
    assert np.isfinite(embedding).all()
    assert not detector._model.model.model[-2]._forward_hooks


def test_extract_embedding_returns_none_for_an_empty_crop():
    detector = YoloDetector(device="cpu")
    detector._model = SimpleNamespace(model=RoutedFeatureModel())

    assert detector.extract_embedding(np.zeros((0, 0, 3), dtype=np.uint8)) is None
