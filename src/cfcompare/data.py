"""Dataset loading, leave-one-out splitting and negative sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]

DATASET_FILES = {
    "movielens": "ml-1m.ratings.dat",
    "yelp": "yelp.rating",
}


def load_movielens(path: str | Path) -> pd.DataFrame:
    """MovieLens-1M ``ratings.dat`` (``user::item::rating::timestamp``)."""
    return pd.read_csv(path, sep="::", engine="python", names=COLUMNS)


def load_yelp(path: str | Path) -> pd.DataFrame:
    """Yelp ``yelp.rating`` from the ConvNCF authors (whitespace separated)."""
    return pd.read_csv(path, sep=r"\s+", engine="python", header=None, names=COLUMNS)


def load_dataset(name: str, data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / DATASET_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see data/README.md for download instructions.")
    return load_movielens(path) if name == "movielens" else load_yelp(path)


def deduplicate(df: pd.DataFrame, min_items_per_user: int = 10) -> pd.DataFrame:
    """Keep the first (user, item) interaction and users with enough distinct items.

    Applied to ``yelp.rating`` this yields the statistics in Table 2 of the
    thesis. The experiments themselves were run on the raw file.
    """
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").drop_duplicates(["user_id", "item_id"])
    counts = df.groupby("user_id")["item_id"].transform("size")
    return df[counts >= min_items_per_user].reset_index(drop=True)


def leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent interaction.

    Returns ``(rest, held_out)`` where ``held_out`` has exactly one row per user.
    """
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    held_out = df.groupby("user_id").tail(1)
    rest = df.drop(held_out.index)
    return rest.reset_index(drop=True), held_out.reset_index(drop=True)


@dataclass
class Split:
    """Leave-one-out train / validation / test split with contiguous ids.

    ``train`` is the internal training portion (all interactions except the
    two most recent per user), ``val`` holds the second-most-recent and
    ``test`` the most recent interaction of every user.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_items: int

    def user_positives(self) -> list[set[int]]:
        """Items each user interacted with in ``train`` (index = user id)."""
        pos: list[set[int]] = [set() for _ in range(self.num_users)]
        for u, i in zip(self.train["user_id"].to_numpy(), self.train["item_id"].to_numpy()):
            pos[u].add(int(i))
        return pos


def make_split(df: pd.DataFrame) -> Split:
    """Test = last interaction, validation = second-to-last, train = the rest.

    User and item ids are label-encoded to ``0..n-1`` over the full data so
    that every validation/test id also exists in the embedding tables.
    """
    df = df[COLUMNS].copy()
    df["user_id"] = pd.factorize(df["user_id"], sort=True)[0]
    df["item_id"] = pd.factorize(df["item_id"], sort=True)[0]
    train_full, test = leave_last_out(df)
    train, val = leave_last_out(train_full)
    return Split(train, val, test, int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1)


def sample_negative(rng: np.random.Generator, num_items: int, positives: set[int]) -> int:
    """Uniformly sample an item the user has not interacted with."""
    j = int(rng.integers(num_items))
    while j in positives:
        j = int(rng.integers(num_items))
    return j


def pointwise_instances(
    split: Split, num_neg: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(user, item, label) arrays: every positive plus ``num_neg`` sampled negatives."""
    positives = split.user_positives()
    users, items, labels = [], [], []
    for u, i in zip(split.train["user_id"].to_numpy(), split.train["item_id"].to_numpy()):
        users.append(u)
        items.append(i)
        labels.append(1.0)
        for _ in range(num_neg):
            users.append(u)
            items.append(sample_negative(rng, split.num_items, positives[u]))
            labels.append(0.0)
    return (
        np.asarray(users, dtype=np.int32),
        np.asarray(items, dtype=np.int32),
        np.asarray(labels, dtype=np.float32),
    )


def bpr_triplets(
    split: Split, num_neg: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(user, positive item, negative item) triplets, ``num_neg`` per positive."""
    positives = split.user_positives()
    users, pos_items, neg_items = [], [], []
    for u, i in zip(split.train["user_id"].to_numpy(), split.train["item_id"].to_numpy()):
        for _ in range(num_neg):
            users.append(u)
            pos_items.append(i)
            neg_items.append(sample_negative(rng, split.num_items, positives[u]))
    return (
        np.asarray(users, dtype=np.int32),
        np.asarray(pos_items, dtype=np.int32),
        np.asarray(neg_items, dtype=np.int32),
    )


def describe(df: pd.DataFrame) -> dict[str, float]:
    """Interaction statistics as reported in Tables 1 and 2 of the thesis."""
    n_users, n_items, n = df["user_id"].nunique(), df["item_id"].nunique(), len(df)
    density = n / (n_users * n_items)
    return {
        "users": n_users,
        "items": n_items,
        "interactions": n,
        "density_%": 100 * density,
        "sparsity_%": 100 * (1 - density),
        "avg_per_user": n / n_users,
        "avg_per_item": n / n_items,
    }
