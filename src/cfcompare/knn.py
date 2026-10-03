"""Item-based k-nearest-neighbours on binary implicit feedback (Section 4.1.1).

    s(i, j)  = cosine similarity of the binary item columns of R
    y_hat_ui = sum_{j in N_K(i) ∩ I_u} s(i, j)

where N_K(i) are the K items most similar to i and I_u is the user's history.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Split


class ItemKNN:
    def __init__(self, split: Split):
        self.R = sp.csr_matrix(
            (
                np.ones(len(split.train), dtype=np.float32),
                (split.train["user_id"].to_numpy(), split.train["item_id"].to_numpy()),
            ),
            shape=(split.num_users, split.num_items),
        )
        self.R.data[:] = 1.0  # collapse duplicate (user, item) pairs to 1
        norms = np.sqrt(np.asarray(self.R.sum(axis=0)).ravel())
        inv = sp.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0))
        co = (self.R.T @ self.R).tocsr()
        self.similarity = (inv @ co @ inv).tocsr()
        self.similarity.setdiag(0.0)
        self.similarity.eliminate_zeros()
        self.neighbours: sp.csr_matrix | None = None

    def fit(self, k: int) -> "ItemKNN":
        """Keep the ``k`` most similar neighbours of every item."""
        S = self.similarity
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):
            start, end = S.indptr[i], S.indptr[i + 1]
            if start == end:
                continue
            idx, sim = S.indices[start:end], S.data[start:end]
            if len(sim) > k:
                top = np.argpartition(-sim, k - 1)[:k]
                idx, sim = idx[top], sim[top]
            rows.append(np.full(len(idx), i))
            cols.append(idx)
            vals.append(sim)
        self.neighbours = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=S.shape
        )
        return self

    def score(self, users: np.ndarray, items: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """Score (user, item) pairs; pairs with no neighbour in the history score 0."""
        assert self.neighbours is not None, "call fit(k) first"
        out = np.empty(len(users), dtype=np.float32)
        for start in range(0, len(users), chunk):
            u, i = users[start : start + chunk], items[start : start + chunk]
            overlap = self.neighbours[i].multiply(self.R[u])
            out[start : start + chunk] = np.asarray(overlap.sum(axis=1)).ravel()
        return out
