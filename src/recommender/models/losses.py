"""Losses and samplers shared by recommender models."""

from __future__ import annotations

import numpy as np

# mục tiêu: - sample_bpr_triplets: tạo ra các triplet (user, positive item, negative item) để huấn luyện mô hình với BPR loss
#          - bpr_loss: tính toán loss BPR cho các score của positive và negative
# bpr học cách xếp hạng sao cho score(user, positive_item) > score(user, negative_item)
def sample_bpr_triplets(
    user_positive_items: dict[int, set[int]],
    num_items: int,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (user, positive item, negative item) triplets for BPR training."""
    rng = rng or np.random.default_rng()
    # eligible_users là những user có ít nhất 1 item đã tương tác 
    # và chưa tương tác với tất cả item (để có thể sample được cả positive và negative)
    eligible_users = [user for user, items in user_positive_items.items() if items and len(items) < num_items]
    if not eligible_users:
        raise ValueError("No users are eligible for BPR sampling")

    # mang ket qua, users, positives, negatives, moi mang co do dai bang num_samples
    eligible = np.asarray(eligible_users, dtype=np.int64)
    users = rng.choice(eligible, size=num_samples, replace=True).astype(np.int64, copy=False)
    positives = np.empty(num_samples, dtype=np.int64)
    negatives = rng.integers(0, num_items, size=num_samples, dtype=np.int64)

    positive_arrays = {
        int(user): np.fromiter(items, dtype=np.int64, count=len(items))
        for user, items in user_positive_items.items()
        if items and len(items) < num_items
    }
    for user in np.unique(users):
        positions = np.flatnonzero(users == user)
        positive_items = positive_arrays[int(user)]
        positives[positions] = rng.choice(positive_items, size=len(positions), replace=True)

        bad = np.isin(negatives[positions], positive_items, assume_unique=False)
        while bad.any():
            negatives[positions[bad]] = rng.integers(0, num_items, size=int(bad.sum()), dtype=np.int64)
            bad = np.isin(negatives[positions], positive_items, assume_unique=False)

    return users, positives, negatives


def bpr_loss(pos_scores, neg_scores, reg_loss=None, reg_weight: float = 1e-4):
    """Bayesian Personalized Ranking loss for PyTorch tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("bpr_loss requires torch. Install requirements.txt first.") from exc
    # BPR loss khuyến khích pos_scores > neg_scores, sử dụng logsigmoid để tính loss
    loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
    if reg_loss is not None:
        loss = loss + reg_weight * reg_loss
    return loss
