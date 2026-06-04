from __future__ import annotations

import numpy as np
import pytest

from recommender.models.losses import sample_bpr_triplets


def test_sample_bpr_triplets_never_uses_positive_as_negative() -> None:
    positives = {0: {1, 2}, 1: {0}}
    users, pos, neg = sample_bpr_triplets(positives, num_items=4, num_samples=100, rng=np.random.default_rng(1))

    assert len(users) == len(pos) == len(neg) == 100
    for user, negative in zip(users, neg):
        assert int(negative) not in positives[int(user)]


def test_sample_bpr_triplets_requires_eligible_user() -> None:
    with pytest.raises(ValueError):
        sample_bpr_triplets({0: {0, 1}}, num_items=2, num_samples=1)
