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
