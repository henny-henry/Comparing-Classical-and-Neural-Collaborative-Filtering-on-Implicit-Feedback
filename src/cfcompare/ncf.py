"""Neural Collaborative Filtering models (He et al., 2017): GMF, MLP and NeuMF.

Trained point-wise with binary cross-entropy on sampled negatives
(Sections 4.1.3-4.1.5). Layer names match the thesis notebooks so that
pretrained GMF/MLP weights can be transferred into NeuMF.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from tensorflow.keras import Model, layers


def _inputs():
    return (
        layers.Input(shape=(), dtype="int32", name="user_input"),
        layers.Input(shape=(), dtype="int32", name="item_input"),
    )


def build_gmf(num_users: int, num_items: int, emb_size: int = 64) -> Model:
    user_in, item_in = _inputs()
    p = layers.Flatten()(layers.Embedding(num_users, emb_size, name="gmf_user_embedding")(user_in))
    q = layers.Flatten()(layers.Embedding(num_items, emb_size, name="gmf_item_embedding")(item_in))
    out = layers.Dense(1, activation="sigmoid", name="outputs")(layers.Multiply()([p, q]))
    return Model([user_in, item_in], out, name="GMF")


def build_mlp(
    num_users: int, num_items: int, emb_size: int = 64, hidden_units: Sequence[int] = (32, 16, 8)
) -> Model:
    user_in, item_in = _inputs()
    p = layers.Flatten()(layers.Embedding(num_users, emb_size, name="user_embedding")(user_in))
    q = layers.Flatten()(layers.Embedding(num_items, emb_size, name="item_embedding")(item_in))
    x = layers.Concatenate(axis=1)([p, q])
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"hidden_{idx}")(x)
    out = layers.Dense(1, activation="sigmoid", name="outputs")(x)
    return Model([user_in, item_in], out, name="MLP")


def build_neumf(
    num_users: int, num_items: int, emb_size: int = 64, hidden_units: Sequence[int] = (32, 16, 8)
) -> Model:
    user_in, item_in = _inputs()

    # MLP branch
    p_m = layers.Flatten()(layers.Embedding(num_users, emb_size, name="mlp_user_embedding")(user_in))
    q_m = layers.Flatten()(layers.Embedding(num_items, emb_size, name="mlp_item_embedding")(item_in))
    x = layers.Concatenate(axis=1)([p_m, q_m])
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"mlp_hidden_{idx}")(x)

    # GMF branch
    p_g = layers.Flatten()(layers.Embedding(num_users, emb_size, name="gmf_user_embedding")(user_in))
    q_g = layers.Flatten()(layers.Embedding(num_items, emb_size, name="gmf_item_embedding")(item_in))
    gmf_vec = layers.Multiply()([p_g, q_g])

    fused = layers.Concatenate(axis=1)([x, gmf_vec])
    out = layers.Dense(1, activation="sigmoid", name="neumf_outputs")(fused)
    return Model([user_in, item_in], out, name="NeuMF")


def load_pretrained(neumf: Model, gmf: Model, mlp: Model, hidden_units: Sequence[int], alpha: float = 0.5):
    """Initialise NeuMF from trained GMF and MLP models.

    The prediction layer is ``[alpha * h_GMF ; (1 - alpha) * h_MLP]``, ordered
    to match the ``[mlp_vector, gmf_vector]`` concatenation in ``build_neumf``.
    """
    for name in ("gmf_user_embedding", "gmf_item_embedding"):
        neumf.get_layer(name).set_weights(gmf.get_layer(name).get_weights())
    neumf.get_layer("mlp_user_embedding").set_weights(mlp.get_layer("user_embedding").get_weights())
    neumf.get_layer("mlp_item_embedding").set_weights(mlp.get_layer("item_embedding").get_weights())
    for idx in range(len(hidden_units)):
        neumf.get_layer(f"mlp_hidden_{idx}").set_weights(mlp.get_layer(f"hidden_{idx}").get_weights())

    w_g, b_g = gmf.get_layer("outputs").get_weights()
    w_m, b_m = mlp.get_layer("outputs").get_weights()
    neumf.get_layer("neumf_outputs").set_weights(
        [np.concatenate([(1 - alpha) * w_m, alpha * w_g], axis=0), alpha * b_g + (1 - alpha) * b_m]
    )
