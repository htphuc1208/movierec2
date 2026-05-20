from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    recs = list(recommended)[:k]
    if not recs:
        return 0.0
    hits = sum(1 for item in recs if item in relevant)
    return hits / len(recs)


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    recs = list(recommended)[:k]
    hits = sum(1 for item in recs if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    recs = list(recommended)[:k]
    dcg = 0.0
    for rank, item in enumerate(recs, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=np.float32)
    pred = np.asarray(list(y_pred), dtype=np.float32)
    if true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((true - pred) ** 2)))
