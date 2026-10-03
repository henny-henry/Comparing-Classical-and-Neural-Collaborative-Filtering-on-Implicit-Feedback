"""Plot the reported test NDCG@10 of every model (thesis Tables 18-19).

    python scripts/plot_results.py

Reads results/movielens.csv and results/yelp.csv and writes
results/figures/ndcg10_comparison.png (the figure shown in the README).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL = {"ItemKNN", "MF-BPR"}
COLORS = {"classical": "#eb6834", "neural": "#2a78d6"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main(metric: str = "NDCG@10") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, facecolor=SURFACE)
    for ax, (name, title) in zip(axes, [("movielens", "MovieLens-1M"), ("yelp", "Yelp")]):
        df = pd.read_csv(ROOT / "results" / f"{name}.csv").sort_values(metric)
        colors = [COLORS["classical" if m in CLASSICAL else "neural"] for m in df["model"]]
        bars = ax.barh(df["model"], df[metric], height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, df[metric]):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                    va="center", fontsize=9, color=INK)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.set_xlim(0, 0.56)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel(f"Test {metric}", color=MUTED, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("classical", "neural")]
    fig.legend(handles, ["Classical baseline", "Neural model"], loc="upper right", ncol=2,
               frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "results" / "figures" / "ndcg10_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
