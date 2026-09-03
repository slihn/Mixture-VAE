"""Summarise the KMeans feature-set sweep.

    python -m compare.kmeans_report --csv data/kmeans_sweep.csv

KMeans on this benchmark is bimodal -- ~0.69 when it splits on a rolling-mean
feature, ~0.52 when it splits on a signal-free volatility feature, nothing in
between -- so a cell mean is a mixing proportion, not a typical outcome. Every
table here reports `p(bac > 0.6)` as the primary statistic, with the mean beside
it only to show how misleading the mean is.

The three questions the sweep was built to answer:

1. Does restricting to the mean features rescue KMeans everywhere?
   (`means` vs `all`, across the whole tail range.)
2. Is `scales` pinned at chance regardless of how favourable the parameters are?
   That is the negative control: without it, "means beats all" is equally
   consistent with "fewer features are less noisy".
3. Does `clip_factor` matter at `feature_set=all` but NOT at `means`?
   The stated mechanism -- a wide clip makes the volatility axis leptokurtic and
   so a worse thing to bisect -- predicts exactly that interaction, and there is
   no scale axis to lose to once the scale columns are gone.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import numpy as np

TARGET = 0.6


def load(path: str, model: str = 'kmeans') -> List[Dict]:
    with open(path, newline='') as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get('model') == model and not r.get('error')]
    if not rows:
        raise SystemExit(f"no usable {model} rows in {path}")
    return rows


def table(rows: List[Dict], row_key: str, col_key: str, target: float) -> None:
    """p(bac>target) laid out as row_key x col_key, with n and mean underneath."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r[row_key], r[col_key])].append(float(r['bac']))
    rkeys = sorted({k[0] for k in cells}, key=_sortable)
    ckeys = sorted({k[1] for k in cells}, key=_sortable)

    print(f"\np(bac > {target}) by {row_key} x {col_key}")
    print(f"{row_key:>12} | " + " ".join(f"{c:>9}" for c in ckeys))
    print("-" * (14 + 10 * len(ckeys)))
    for rk in rkeys:
        line = []
        for ck in ckeys:
            v = cells.get((rk, ck))
            line.append(f"{np.mean([b > target for b in v]):>9.2f}" if v else f"{'-':>9}")
        print(f"{rk:>12} | " + " ".join(line))
    print(f"{'(mean bac)':>12} | " + " ".join(
        f"{np.mean([b for rk in rkeys for b in cells.get((rk, ck), [])]):>9.3f}"
        for ck in ckeys))


def _sortable(v):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def main():
    p = argparse.ArgumentParser(prog='compare.kmeans_report')
    p.add_argument('--csv', default='data/kmeans_sweep.csv')
    p.add_argument('--model', default='kmeans')
    p.add_argument('--target', type=float, default=TARGET)
    args = p.parse_args()

    rows = load(args.csv, args.model)
    print(f"{len(rows):,} {args.model} rows from {args.csv}")

    # 1. headline: is the feature set the whole story?
    by_fs = defaultdict(list)
    for r in rows:
        by_fs[r['feature_set']].append(float(r['bac']))
    print(f"\nby feature_set (all parameters pooled):")
    print(f"{'feature_set':>12} {'n':>6} {'p(>target)':>11} {'mean':>7} "
          f"{'median':>7} {'range':>15}")
    for fs in sorted(by_fs, key=lambda f: -np.mean([b > args.target for b in by_fs[f]])):
        b = np.array(by_fs[fs])
        print(f"{fs:>12} {len(b):>6} {np.mean(b > args.target):>11.3f} "
              f"{b.mean():>7.3f} {np.median(b):>7.3f} "
              f"{b.min():.3f}-{b.max():.3f}")

    # 2/3. the interactions that identify the mechanism
    for col in ('clip_factor', 'k', 'alpha', 'loc'):
        if col in rows[0]:
            table(rows, 'feature_set', col, args.target)

    # bimodality check: the mean is only meaningful if it is not a mixture
    print("\nbimodality -- share of fits in each mode, by feature_set")
    print(f"{'feature_set':>12} {'<0.55':>8} {'0.55-0.65':>10} {'>0.65':>8}")
    for fs in sorted(by_fs):
        b = np.array(by_fs[fs])
        print(f"{fs:>12} {np.mean(b < 0.55):>8.2f} "
              f"{np.mean((b >= 0.55) & (b <= 0.65)):>10.2f} {np.mean(b > 0.65):>8.2f}")


if __name__ == '__main__':
    main()
