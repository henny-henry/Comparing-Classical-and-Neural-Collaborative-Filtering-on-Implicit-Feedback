"""End-to-end experiment runners, one per model family.

Every trainable model follows the protocol of Section 4.5: train for a fixed
number of epochs, evaluate HR@10 / NDCG@10 on the validation split every
``eval_every`` epochs, keep the checkpoint with the best validation NDCG@10
(HR@10 breaks ties) and report HR@K / NDCG@K on the test split for that
checkpoint.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .data import Split, pointwise_instances
from .metrics import evaluate, keras_score_fn, sample_candidates


def _is_better(metrics: dict, best: dict) -> bool:
    ndcg, hr = metrics["NDCG@10"], metrics["HR@10"]
    return ndcg > best["NDCG@10"] or (math.isclose(ndcg, best["NDCG@10"]) and hr > best["HR@10"])


def train_with_validation(
    fit_one_epoch: Callable[[], tuple[float, float]],
    scorer,
    val_candidates,
    epochs: int,
    eval_every: int,
    log: Callable[[str], None] = print,
) -> tuple[dict, dict]:
    """Generic epoch loop; restores the best validation weights into ``scorer``."""
    history = {"loss": [], "accuracy": [], "val_epoch": [], "val_HR@10": [], "val_NDCG@10": []}
    best = {"epoch": 0, "HR@10": -1.0, "NDCG@10": -1.0}
    best_weights = scorer.get_weights()
    for ep in range(1, epochs + 1):
        loss, acc = fit_one_epoch()
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        msg = f"epoch {ep:02d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}"
        if ep % eval_every == 0:
            m = evaluate(keras_score_fn(scorer), val_candidates, ks=(10,))
            history["val_epoch"].append(ep)
            history["val_HR@10"].append(m["HR@10"])
            history["val_NDCG@10"].append(m["NDCG@10"])
            msg += f"  | val HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}"
            if _is_better(m, best):
                best = {"epoch": ep, **m}
                best_weights = scorer.get_weights()
                msg += "  *"
        log(msg)
    scorer.set_weights(best_weights)
    return history, best


def _keras_epoch(model, x, y, batch_size: int, acc_key: str) -> Callable[[], tuple[float, float]]:
    def step():
        h = model.fit(x, y, batch_size=batch_size, epochs=1, shuffle=True, verbose=0).history
        return float(h["loss"][0]), float(h[acc_key][0])

    return step


def _candidates(split: Split, cfg: dict):
    return (
        sample_candidates(split, split.val, cfg["val_neg"], seed=cfg["seed"]),
        sample_candidates(split, split.test, cfg["test_neg"], seed=cfg["seed"] + 1),
    )


# --------------------------------------------------------------------------- ItemKNN
def run_itemknn(split: Split, cfg: dict, log=print) -> dict:
    from .knn import ItemKNN

    val_c, test_c = _candidates(split, cfg)
    knn = ItemKNN(split)
    grid, best = [], {"k": None, "HR@10": -1.0, "NDCG@10": -1.0}
    for k in cfg["k_grid"]:
        m = evaluate(knn.fit(k).score, val_c, ks=(10,))
        grid.append({"k": k, **m})
        log(f"K={k:<4d} val HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}")
        if _is_better(m, best):
            best = {"k": k, **m}
    test = evaluate(knn.fit(best["k"]).score, test_c, ks=cfg["test_ks"])
    return {"best": best, "validation_grid": grid, "test": test}


# --------------------------------------------------------------------------- GMF / MLP / NeuMF
def _pointwise_model(name: str, split: Split, cfg: dict):
    from .ncf import build_gmf, build_mlp, build_neumf

    if name == "gmf":
        return build_gmf(split.num_users, split.num_items, cfg["emb_size"])
    if name == "mlp":
        return build_mlp(split.num_users, split.num_items, cfg["emb_size"], cfg["hidden_units"])
    return build_neumf(split.num_users, split.num_items, cfg["emb_size"], cfg["hidden_units"])


def _compile_pointwise(model, lr: float):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"],
    )
    return model


def run_pointwise(name: str, split: Split, cfg: dict, log=print) -> dict:
    """GMF, MLP, NeuMF (random init) and NeuMF-pre (GMF + MLP pretraining)."""
    import tensorflow as tf

    from .ncf import load_pretrained

    tf.keras.utils.set_random_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    val_c, test_c = _candidates(split, cfg)
    # Negatives are sampled once before training, as in the thesis notebooks.
    users, items, labels = pointwise_instances(split, cfg["num_neg"], rng)
    x, bs = [users, items], cfg["batch_size"]

    if name == "neumf-pre":
        gmf = _compile_pointwise(_pointwise_model("gmf", split, cfg), 1e-3)
        mlp = _compile_pointwise(_pointwise_model("mlp", split, cfg), 1e-3)
        log(f"pretraining GMF for {cfg['gmf_epochs']} and MLP for {cfg['mlp_epochs']} epochs")
        gmf.fit(x, labels, batch_size=bs, epochs=cfg["gmf_epochs"], shuffle=True, verbose=0)
        mlp.fit(x, labels, batch_size=bs, epochs=cfg["mlp_epochs"], shuffle=True, verbose=0)
        model = _pointwise_model("neumf", split, cfg)
        load_pretrained(model, gmf, mlp, cfg["hidden_units"], cfg["alpha"])
    else:
        model = _pointwise_model(name, split, cfg)
    _compile_pointwise(model, cfg["lr"])

    history, best = train_with_validation(
        _keras_epoch(model, x, labels, bs, "binary_accuracy"),
        model, val_c, cfg["epochs"], cfg["eval_every"], log,
    )
    test = evaluate(keras_score_fn(model), test_c, ks=cfg["test_ks"])
    return {"best": best, "history": history, "test": test}


# --------------------------------------------------------------------------- MF-BPR / ConvNCF
def _adagrad(lr: float):
    import tensorflow as tf

    return tf.keras.optimizers.Adagrad(lr)


def _mf_bpr(split: Split, cfg: dict):
    from .bpr import build_bpr_model, build_mf_scorer, compile_bpr

    scorer = build_mf_scorer(split.num_users, split.num_items, cfg["emb_size"])
    model = compile_bpr(build_bpr_model(scorer), _adagrad(cfg["lr"]))
    return scorer, model


def run_mf_bpr(split: Split, cfg: dict, log=print) -> dict:
    import tensorflow as tf

    from .bpr import BPRSequence

    tf.keras.utils.set_random_seed(cfg["seed"])
    val_c, test_c = _candidates(split, cfg)
    seq = BPRSequence(split, cfg["num_neg"], cfg["batch_size"], seed=cfg["seed"])
    scorer, model = _mf_bpr(split, cfg)
    history, best = train_with_validation(
        _keras_epoch(model, seq, None, None, "pairwise_accuracy"),
        scorer, val_c, cfg["epochs"], cfg["eval_every"], log,
    )
    test = evaluate(keras_score_fn(scorer), test_c, ks=cfg["test_ks"])
    return {"best": best, "history": history, "test": test}


def run_convncf(split: Split, cfg: dict, log=print) -> dict:
    import tensorflow as tf

    from .bpr import BPRSequence, build_bpr_model, build_convncf_scorer, compile_bpr

    tf.keras.utils.set_random_seed(cfg["seed"])
    val_c, test_c = _candidates(split, cfg)
    seq = BPRSequence(split, cfg["num_neg"], cfg["batch_size"], seed=cfg["seed"])

    # 1) MF-BPR pretraining of the embeddings
    log(f"pretraining MF-BPR embeddings for {cfg['mf_epochs']} epochs")
    mf_scorer, mf_model = _mf_bpr(split, cfg)
    mf_model.fit(seq, epochs=cfg["mf_epochs"], verbose=0)
    P = mf_scorer.get_layer("mf_user_emb").get_weights()
    Q = mf_scorer.get_layer("mf_item_emb").get_weights()

    arch = {k: cfg[k] for k in ("emb_size", "conv_filters", "kernel_size", "dense_units")}
    # 2) one warm-up epoch without regularisation
    warm = build_convncf_scorer(split.num_users, split.num_items, **arch)
    warm.get_layer("user_embedding").set_weights(P)
    warm.get_layer("item_embedding").set_weights(Q)
    compile_bpr(build_bpr_model(warm), _adagrad(cfg["lr"])).fit(seq, epochs=1, verbose=0)

    # 3) regularised model initialised from the warm-up weights
    scorer = build_convncf_scorer(
        split.num_users, split.num_items, **arch,
        l2_emb=cfg["l2_emb"], l2_conv=cfg["l2_conv"], l2_out=cfg["l2_out"],
    )
    scorer.set_weights(warm.get_weights())
    model = compile_bpr(build_bpr_model(scorer), _adagrad(cfg["lr"]))
    history, best = train_with_validation(
        _keras_epoch(model, seq, None, None, "pairwise_accuracy"),
        scorer, val_c, cfg["epochs"], cfg["eval_every"], log,
    )
    test = evaluate(keras_score_fn(scorer), test_c, ks=cfg["test_ks"])
    return {"best": best, "history": history, "test": test}


def run(model: str, split: Split, cfg: dict, log=print) -> dict:
    if model == "itemknn":
        return run_itemknn(split, cfg, log)
    if model == "mf-bpr":
        return run_mf_bpr(split, cfg, log)
    if model == "convncf":
        return run_convncf(split, cfg, log)
    return run_pointwise(model, split, cfg, log)
