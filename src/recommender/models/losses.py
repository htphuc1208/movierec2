"""Losses and samplers shared by recommender models."""

from __future__ import annotations

import numpy as np


def sample_bpr_triplets(
    user_positive_items: dict[int, set[int]],
    num_items: int,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (user, positive item, negative item) triplets for BPR training."""
    rng = rng or np.random.default_rng()
    eligible_users = [user for user, items in user_positive_items.items() if items and len(items) < num_items]
    if not eligible_users:
        raise ValueError("No users are eligible for BPR sampling")

    users = np.empty(num_samples, dtype=np.int64)
    positives = np.empty(num_samples, dtype=np.int64)
    negatives = np.empty(num_samples, dtype=np.int64)

    for idx in range(num_samples):
        user = int(rng.choice(eligible_users))
        positives_for_user = tuple(user_positive_items[user])
        positive = int(rng.choice(positives_for_user))
        negative = int(rng.integers(0, num_items))
        while negative in user_positive_items[user]:
            negative = int(rng.integers(0, num_items))

        users[idx] = user
        positives[idx] = positive
        negatives[idx] = negative

    return users, positives, negatives


def bpr_loss(pos_scores, neg_scores, reg_loss=None, reg_weight: float = 1e-4):
    """Bayesian Personalized Ranking loss for PyTorch tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("bpr_loss requires torch. Install requirements.txt first.") from exc

    loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
    if reg_loss is not None:
        loss = loss + reg_weight * reg_loss
    return loss
