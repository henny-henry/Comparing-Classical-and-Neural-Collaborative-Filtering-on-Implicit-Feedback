"""Leave-one-out top-K evaluation with sampled negatives (HR@K, NDCG@K).

For every evaluated user the held-out item is ranked against ``num_neg``
items the user has not interacted with in the training data (99 for test,
100 for validation, following He et al., 2017).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import Split, sample_negative

ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sample_candidates(
    split: Split, eval_df: pd.DataFrame, num_neg: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate lists of shape ``(n_users, 1 + num_neg)``; the true item is column 0."""
    rng = np.random.default_rng(seed)
    positives = split.user_positives()
    users = eval_df["user_id"].to_numpy(dtype=np.int32)
    items = np.empty((len(users), 1 + num_neg), dtype=np.int32)
    for row, (u, gt) in enumerate(zip(users, eval_df["item_id"].to_numpy())):
        excluded = positives[u] | {int(gt)}
        items[row, 0] = gt
        negs: set[int] = set()
        while len(negs) < num_neg:
            negs.add(sample_negative(rng, split.num_items, excluded))
        items[row, 1:] = list(negs)
    return np.repeat(users[:, None], 1 + num_neg, axis=1), items


def rank_of_true_item(scores: np.ndarray) -> np.ndarray:
    """0-based rank of column 0 within each row.

    Ties are broken pessimistically (a negative with an equal score is ranked
    above the true item), so a constant scorer cannot obtain a perfect HR.
    """
    return (scores[:, 1:] >= scores[:, :1]).sum(axis=1)


def hr_ndcg(ranks: np.ndarray, k: int) -> tuple[float, float]:
    hits = ranks < k
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 2.0), 0.0)
    return float(hits.mean()), float(ndcg.mean())


def evaluate(
    score_fn: ScoreFn,
    candidates: tuple[np.ndarray, np.ndarray],
    ks: Iterable[int] = (10,),
) -> dict[str, float]:
    """Score all candidates and return ``{"HR@k": .., "NDCG@k": ..}`` for each k."""
    users, items = candidates
    scores = np.asarray(score_fn(users.ravel(), items.ravel()), dtype=np.float64).reshape(items.shape)
    ranks = rank_of_true_item(scores)
    out: dict[str, float] = {}
    for k in ks:
        out[f"HR@{k}"], out[f"NDCG@{k}"] = hr_ndcg(ranks, k)
    return out


def keras_score_fn(model, batch_size: int = 8192) -> ScoreFn:
    """Wrap a two-input Keras scorer ``[user_ids, item_ids] -> score``."""

    def score(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return model.predict([users, items], batch_size=batch_size, verbose=0).reshape(-1)

    return score
