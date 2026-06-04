from __future__ import annotations

import importlib.util

import numpy as np
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_build_normalized_adj_shape() -> None:
    from recommender.models.lightgcn import build_normalized_adj

    adj = build_normalized_adj(2, 3, np.array([[0, 1], [1, 2]], dtype=np.int64))
    assert tuple(adj.shape) == (5, 5)
