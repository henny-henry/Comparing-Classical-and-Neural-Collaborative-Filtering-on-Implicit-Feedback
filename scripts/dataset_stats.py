"""Reproduce the dataset statistics of Tables 1 and 2 of the thesis.

    python scripts/dataset_stats.py

Table 2 describes Yelp after removing duplicate (user, item) pairs and users
with fewer than 10 distinct items. The experiments used the raw
``yelp.rating`` file, whose statistics are printed as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfcompare.data import deduplicate, describe, leave_last_out, load_dataset  # noqa: E402


def report(title: str, df) -> None:
    train, test = leave_last_out(df)
    full, tr = describe(df), describe(train)
    print(f"\n{title}")
    print(f"  {'':16s}{'all':>12s}{'train':>12s}")
    for key in full:
        fmt = "{:>12,.0f}" if key in ("users", "items", "interactions") else "{:>12.4f}"
        print(f"  {key:16s}" + fmt.format(full[key]) + fmt.format(tr[key]))
    print(f"  {'test':16s}{len(test):>12,d}")


def main() -> None:
    report("MovieLens-1M (Table 1)", load_dataset("movielens", ROOT / "data"))
    yelp = load_dataset("yelp", ROOT / "data")
    report("Yelp, deduplicated, >= 10 items per user (Table 2)", deduplicate(yelp))
    report("Yelp, raw yelp.rating (used by the experiments)", yelp)


if __name__ == "__main__":
    main()
