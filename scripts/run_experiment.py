"""Train and evaluate one model on one dataset with the thesis hyperparameters.

Examples
--------
    python scripts/run_experiment.py --model itemknn --dataset yelp
    python scripts/run_experiment.py --model neumf-pre --dataset movielens
    python scripts/run_experiment.py --model convncf --dataset yelp --l2-emb 1e-3

Results (config, validation curve, best epoch, test HR@K / NDCG@K) are written
to ``results/runs/<dataset>/<model>/`` as ``metrics.json`` plus training plots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfcompare.config import DATASETS, MODELS, get_config  # noqa: E402
from cfcompare.data import load_dataset, make_split  # noqa: E402
from cfcompare.experiments import run  # noqa: E402


def save_plots(history: dict, title: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["loss"], label="train loss")
    ax.plot(epochs, history["accuracy"], label="train accuracy")
    ax.set(xlabel="Epoch", ylabel="Value", title=f"{title}: training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["val_epoch"], history["val_HR@10"], "o-", label="val HR@10")
    ax.plot(history["val_epoch"], history["val_NDCG@10"], "s-", label="val NDCG@10")
    ax.set(xlabel="Epoch", ylabel="Value", title=f"{title}: validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "validation.png", dpi=150)
    plt.close("all")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--epochs", type=int, help="override the number of training epochs")
    ap.add_argument("--lr", type=float, help="override the learning rate")
    ap.add_argument("--num-neg", type=int, help="override negatives per positive")
    ap.add_argument("--l2-emb", type=float, help="ConvNCF: L2 on embeddings")
    ap.add_argument("--l2-conv", type=float, help="ConvNCF: L2 on conv/dense layers")
    ap.add_argument("--l2-out", type=float, help="ConvNCF: L2 on the output layer")
    ap.add_argument("--k-grid", type=int, nargs="+", help="ItemKNN: neighbourhood sizes to try")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = get_config(
        args.model, args.dataset, epochs=args.epochs, lr=args.lr, num_neg=args.num_neg,
        l2_emb=args.l2_emb, l2_conv=args.l2_conv, l2_out=args.l2_out, seed=args.seed,
        k_grid=tuple(args.k_grid) if args.k_grid else None,
    )
    print(f"== {args.model} on {args.dataset} ==\n{json.dumps(cfg)}")

    split = make_split(load_dataset(args.dataset, args.data_dir))
    print(f"users={split.num_users} items={split.num_items} "
          f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    start = time.time()
    result = run(args.model, split, cfg)
    result.update({"model": args.model, "dataset": args.dataset, "config": cfg,
                   "runtime_s": round(time.time() - start, 1)})

    out_dir = Path(args.out_dir) / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    if "history" in result:
        save_plots(result["history"], f"{args.model} ({args.dataset})", out_dir)

    print(f"\nbest validation: {result['best']}")
    for key, value in result["test"].items():
        print(f"test {key:8s} {value:.4f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
