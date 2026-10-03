"""Check that every model trains and evaluates end to end on the dataset sample.

    python tests/make_sample.py      # once
    python tests/smoke_test.py       # all models, both datasets, 1 epoch each
    python tests/smoke_test.py --models gmf convncf --datasets movielens

The numbers are meaningless (1 % of the users, 1 epoch). The test only checks
that the pipelines run and return valid metrics. It runs on a laptop CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfcompare.config import DATASETS, MODELS, get_config  # noqa: E402
from cfcompare.data import load_dataset, make_split  # noqa: E402
from cfcompare.experiments import run  # noqa: E402

FAST = {
    "epochs": 1, "eval_every": 1, "k_grid": (5, 20),
    "gmf_epochs": 1, "mlp_epochs": 1, "mf_epochs": 1,
}


def check(result: dict, cfg: dict) -> None:
    for k in cfg["test_ks"]:
        for metric in (f"HR@{k}", f"NDCG@{k}"):
            value = result["test"][metric]
            assert 0.0 <= value <= 1.0, f"{metric}={value} out of range"
    assert result["test"]["NDCG@5"] <= result["test"]["NDCG@20"] + 1e-9
    assert result["test"]["HR@5"] <= result["test"]["HR@20"] + 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--data-dir", default=str(ROOT / "tests" / "data"))
    args = ap.parse_args()

    failures = []
    for dataset in args.datasets:
        split = make_split(load_dataset(dataset, args.data_dir))
        print(f"\n{dataset}: {split.num_users} users, {split.num_items} items, {len(split.train)} train rows")
        for model in args.models:
            cfg = get_config(model, dataset)
            cfg.update({k: v for k, v in FAST.items() if k in cfg or k in ("epochs", "eval_every")})
            start = time.time()
            try:
                result = run(model, split, cfg, log=lambda _: None)
                check(result, cfg)
                t = result["test"]
                print(f"  OK   {model:10s} HR@10={t['HR@10']:.3f} NDCG@10={t['NDCG@10']:.3f} "
                      f"({time.time() - start:.0f}s)")
            except Exception as exc:  # report every failing model, not just the first
                failures.append(f"{dataset}/{model}")
                print(f"  FAIL {model:10s} {type(exc).__name__}: {exc}")
    if failures:
        sys.exit(f"\nfailed: {', '.join(failures)}")
    print("\nall models passed")


if __name__ == "__main__":
    main()
