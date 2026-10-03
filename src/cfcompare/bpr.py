"""Pair-wise (BPR) models: MF-BPR and ConvNCF (Sections 4.1.6, 4.2).

A *scorer* maps ``[user_ids, item_ids] -> score``. ``build_bpr_model`` wraps
it into a triplet model whose output is ``score(u, i) - score(u, j)``;
training that logit with binary cross-entropy against a target of 1 is
exactly the BPR objective ``-ln sigma(y_ui - y_uj)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers

from .data import Split, bpr_triplets


def _l2(value: float):
    return regularizers.l2(value) if value and value > 0 else None


def build_mf_scorer(num_users: int, num_items: int, emb_size: int = 64) -> Model:
    u_in = layers.Input(shape=(), dtype="int32", name="mf_user_id")
    i_in = layers.Input(shape=(), dtype="int32", name="mf_item_id")
    p = layers.Embedding(num_users, emb_size, name="mf_user_emb")(u_in)
    q = layers.Embedding(num_items, emb_size, name="mf_item_emb")(i_in)
    score = layers.Dot(axes=-1, name="mf_dot")([p, q])
    return Model([u_in, i_in], score, name="MFScorer")


def build_convncf_scorer(
    num_users: int,
    num_items: int,
    emb_size: int = 64,
    conv_filters: Sequence[int] = (32, 64, 64),
    kernel_size: int = 2,
    dense_units: int = 128,
    l2_emb: float = 0.0,
    l2_conv: float = 0.0,
    l2_out: float = 0.0,
) -> Model:
    """Shallow ConvNCF: outer-product interaction map -> Conv2D/MaxPool stack -> score."""
    u_in = layers.Input(shape=(), dtype="int32", name="sc_user_id")
    i_in = layers.Input(shape=(), dtype="int32", name="sc_item_id")
    p = layers.Embedding(num_users, emb_size, embeddings_regularizer=_l2(l2_emb), name="user_embedding")(u_in)
    q = layers.Embedding(num_items, emb_size, embeddings_regularizer=_l2(l2_emb), name="item_embedding")(i_in)

    p_col = layers.Reshape((emb_size, 1), name="u_expand")(p)
    q_row = layers.Reshape((1, emb_size), name="v_expand")(q)
    E = layers.Multiply(name="outer_prod")([p_col, q_row])  # (B, k, k)
    x = layers.Reshape((emb_size, emb_size, 1), name="add_channel")(E)

    for idx, filters in enumerate(conv_filters, 1):
        x = layers.Conv2D(
            filters, kernel_size, activation="relu", padding="same",
            kernel_regularizer=_l2(l2_conv), name=f"conv{idx}",
        )(x)
        x = layers.MaxPooling2D(2, name=f"pool{idx}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=_l2(l2_conv), name="dense")(x)
    score = layers.Dense(1, kernel_regularizer=_l2(l2_out), name="score")(x)
    return Model([u_in, i_in], score, name="ConvNCFScorer")


def build_bpr_model(scorer: Model) -> Model:
    u = layers.Input(shape=(), dtype="int32", name="user_id")
    i = layers.Input(shape=(), dtype="int32", name="pos_item_id")
    j = layers.Input(shape=(), dtype="int32", name="neg_item_id")
    diff = layers.Subtract(name="logit_diff")([scorer([u, i]), scorer([u, j])])
    return Model([u, i, j], diff, name=f"{scorer.name}_BPR")


def pairwise_accuracy(y_true, logit_diff):
    """Fraction of triplets where the positive item outscores the negative one."""
    return tf.reduce_mean(tf.cast(logit_diff > 0.0, tf.float32))


def compile_bpr(model: Model, optimizer) -> Model:
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[pairwise_accuracy],
    )
    return model


class BPRSequence(tf.keras.utils.Sequence):
    """Mini-batches of BPR triplets; negatives are re-sampled every epoch."""

    def __init__(self, split: Split, num_neg: int = 1, batch_size: int = 256, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.split, self.num_neg, self.batch_size = split, num_neg, batch_size
        self.rng = np.random.default_rng(seed)
        self._resample()

    def _resample(self):
        self.u, self.i, self.j = bpr_triplets(self.split, self.num_neg, self.rng)
        self.order = self.rng.permutation(len(self.u))

    def __len__(self):
        return int(np.ceil(len(self.u) / self.batch_size))

    def __getitem__(self, idx):
        b = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = {"user_id": self.u[b], "pos_item_id": self.i[b], "neg_item_id": self.j[b]}
        return x, np.ones((len(b), 1), dtype=np.float32)

    def on_epoch_end(self):
        self._resample()
