from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def bpr_loss(pos_scores, neg_scores, l2_penalty=0.0):
    """Bayesian Personalized Ranking pairwise loss."""

    if torch is None:
        raise ImportError("bpr_loss requires torch. Install requirements-ml.txt.")
    ranking_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
    return ranking_loss + l2_penalty


def warp_loss(pos_scores, neg_scores, rank_weights=None, margin: float = 1.0, l2_penalty=0.0):
    """Weighted Approximate-Rank Pairwise loss for top-heavy ranking.

    `rank_weights` should be the log-scaled rank estimate from negative sampling.
    Samples with no violating negative can pass a zero weight.
    """

    if torch is None:
        raise ImportError("warp_loss requires torch. Install requirements-ml.txt.")
    violations = torch.clamp(float(margin) - pos_scores + neg_scores, min=0.0)
    if rank_weights is not None:
        violations = violations * rank_weights.to(device=violations.device, dtype=violations.dtype)
    return violations.mean() + l2_penalty


def sample_negative_items(user_positive_items: dict[int, set[int]], num_items: int, users):
    """Sample one unseen item per user for implicit-feedback training."""

    if torch is None:
        raise ImportError("sample_negative_items requires torch. Install requirements-ml.txt.")
    negatives = []
    for user in users.detach().cpu().tolist():
        positives = user_positive_items.get(int(user), set())
        while True:
            candidate = int(torch.randint(0, num_items, (1,)).item())
            if candidate not in positives:
                negatives.append(candidate)
                break
    return torch.tensor(negatives, dtype=torch.long, device=users.device)
