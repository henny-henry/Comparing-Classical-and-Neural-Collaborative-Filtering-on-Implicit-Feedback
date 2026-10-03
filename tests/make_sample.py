"""Create a small sample of each dataset for quick local tests.

    python tests/make_sample.py            # 1 % of the users (default)
    python tests/make_sample.py --frac 0.05

Keeps every interaction of a random subset of users, so that the
leave-one-out split still works. The files are written to ``tests/data/`` in
the original formats. They are not tracked by git (dataset licences).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"ml-1m.ratings.dat": "::", "yelp.rating": None}  # file -> field separator (None = whitespace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.01, help="fraction of users to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "tests" / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sep in FILES.items():
        lines = (Path(args.data_dir) / name).read_text(encoding="latin-1").splitlines(keepends=True)
        users = sorted({line.split(sep)[0] for line in lines})
        keep = set(random.Random(args.seed).sample(users, max(2, round(args.frac * len(users)))))
        sample = [line for line in lines if line.split(sep)[0] in keep]
        (out_dir / name).write_text("".join(sample), encoding="latin-1")
        print(f"{name}: {len(keep)} of {len(users)} users, {len(sample):,} of {len(lines):,} interactions")


if __name__ == "__main__":
    main()
