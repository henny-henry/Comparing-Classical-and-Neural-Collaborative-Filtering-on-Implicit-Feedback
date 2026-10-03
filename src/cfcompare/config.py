"""Hyperparameters used in the thesis (Section 4.5), per model and dataset."""

from __future__ import annotations

import copy

DATASETS = ("movielens", "yelp")
MODELS = ("itemknn", "mf-bpr", "gmf", "mlp", "neumf", "neumf-pre", "convncf")

COMMON = {
    "batch_size": 256,
    "emb_size": 64,
    "epochs": 30,
    "eval_every": 3,     # validate every 3 epochs, keep the best checkpoint by NDCG@10
    "val_neg": 100,      # sampled negatives per validation user
    "test_neg": 99,      # 1 positive + 99 negatives per test user
    "test_ks": (5, 10, 20),
    "seed": 42,
}

_NCF = {"lr": 1e-3, "num_neg": 4, "hidden_units": (32, 16, 8)}

_CONFIGS = {
    "itemknn": {
        # Largest neighbourhood in the thesis is 200 (Section 5.1); pass --k-grid to extend it.
        "movielens": {"k_grid": (5, 10, 20, 50, 100, 200)},
        "yelp": {"k_grid": (5, 10, 20, 50, 100, 200)},
    },
    "mf-bpr": {
        "movielens": {"epochs": 50, "lr": 0.05, "num_neg": 4},
        "yelp": {"epochs": 50, "lr": 0.05, "num_neg": 1},
    },
    "gmf": {"movielens": dict(_NCF), "yelp": dict(_NCF)},
    "mlp": {"movielens": dict(_NCF), "yelp": dict(_NCF)},
    "neumf": {"movielens": dict(_NCF), "yelp": dict(_NCF)},
    # GMF / MLP are pretrained for the number of epochs at which they peaked on validation.
    "neumf-pre": {
        "movielens": {**_NCF, "gmf_epochs": 3, "mlp_epochs": 12, "alpha": 0.5},
        "yelp": {**_NCF, "lr": 1e-4, "gmf_epochs": 3, "mlp_epochs": 6, "alpha": 0.5},
    },
    # Shallow ConvNCF, Adagrad(0.05), MF-BPR pretraining + 1 unregularised warm-up epoch.
    # The Yelp L2 value picked by the grid search in Section 5.6.2 is not recorded in the
    # thesis, so the MovieLens optimum is used as the default for both datasets.
    "convncf": {
        "movielens": {
            "lr": 0.05, "num_neg": 4, "mf_epochs": 15, "conv_filters": (32, 64, 64),
            "kernel_size": 2, "dense_units": 128, "l2_emb": 0.1, "l2_conv": 0.1, "l2_out": 1e-3,
        },
        "yelp": {
            "lr": 0.05, "num_neg": 1, "mf_epochs": 5, "conv_filters": (32, 64, 64),
            "kernel_size": 2, "dense_units": 128, "l2_emb": 0.1, "l2_conv": 0.1, "l2_out": 1e-3,
        },
    },
}


def get_config(model: str, dataset: str, **overrides) -> dict:
    cfg = {**COMMON, **copy.deepcopy(_CONFIGS[model][dataset])}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
