"""Common interfaces for comparison models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class ModelSkip(RuntimeError):
    """Raised when a model cannot run in the current environment."""


class RecommendationModel(Protocol):
    name: str
    metadata: dict[str, Any]

    def fit(self, dataset) -> "RecommendationModel":
        ...

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        ...


@dataclass
class FitResult:
    name: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
