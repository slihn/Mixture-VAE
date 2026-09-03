"""Fit a threshold law to a sweep produced by `compare.sweep`.

The Jump model obeys a two-term law -- the separation it needs, in MAD units,
rises linearly with tail weight:

    cdist_mad_required = 0.339 + 0.0627 * excess_kurtosis      (R^2 = 0.95)

read at bac target 0.67, penalty 100, clip 12. See "CHOOSING --target" below: those
three qualifiers are part of the result, not context.

This recovers that shape from any sweep CSV, for any model. The procedure:

1. Group the rows by every axis except the separation axis (`--x-axis`), so each
   group is one tail setting -- e.g. (alpha, k, clip_factor).
2. Within a group, average balanced accuracy over replicates at each separation,
   and find where that curve first crosses `--target`, interpolating linearly
   between the bracketing points. That crossing, read in `--y` units, is the
   model's *requirement* for that tail setting.
3. Regress the requirement on `--x` (excess kurtosis by default) across groups.

Groups whose curve never crosses the target -- always above it, or always below --
yield no threshold and are reported separately rather than dropped silently. That
distinction is the finding, not a nuisance: a model that is above target at every
separation has no requirement to fit, and one that is never above target has not
been given enough separation to measure.

    python -m compare.threshold_law --csv data/vae_loc_sweep.csv --model vae
    python -m compare.threshold_law --csv data/jump_loc_sweep.csv --model jump --target 0.6

Replicate spread matters here: the VAE's outcome distribution has a fat left tail,
so a mean over few reps can sit below target on one draw. `--min-reps` guards the
obvious case; the printed per-group table shows n and the spread so a thin or
bimodal cell is visible rather than silently regressed on.

CHOOSING `--target`, AND WHY IT IS NOT COSMETIC
-----------------------------------------------
**The intercept is a function of the target; the slope is nearly not.** At jump
`penalty=100, clip=12`: 0.241 + 0.0634k at target 0.58, 0.260 + 0.0629k at 0.60,
0.311 + 0.0600k at 0.65, 0.390 + 0.0524k at 0.70. So a quoted requirement is
meaningless without its target -- `0.339 + 0.0627k` is the target-0.67 reading of the
same data that gives 0.260 at target 0.60.

**Use `--target 0.65` for a law. Keep `p(bac>0.6)` for classification.** The deciding
issue is saturation: a cell already above target at *every* swept separation yields no
crossing and is dropped, and those are exactly the cells where the model does best, so
a low target biases the law toward the hard corners. At a well-tuned jump config
(`penalty=30, clip=8`) target 0.60 discards 6 of 16 cells and fits at R^2 0.877, while
0.66 discards 1 and fits at 0.928. KMeans-on-means loses 41 of 64 cells at 0.60. Watch
the "no crossing / always at/above target" line this script prints -- that count is the
diagnostic, and a large one means the target is too low for that model.

0.6 remains correct as a found/not-found classifier: the bimodality antimode across the
whole grid is 0.562, so 0.6 sits just on the success side of the density trough.

**A law holds within a (penalty, clip) configuration, not across.** Pooled over all
penalties and clips R^2 collapses to ~0.48 versus 0.93-0.98 within one config, so fit
per config and quote the config alongside the target.

`--criterion prob` reads the crossing off `p(bac>target)` instead of the replicate mean.
It was built to test whether the criterion explained the 0.339/0.26 gap -- it does not
(0.2602 vs 0.2647 on the same data). Kept because the question is worth being able to
re-ask on a differently-shaped outcome distribution.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def load(path: str, model: str) -> List[Dict]:
    with open(path, newline='') as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get('model') == model and not r.get('error')]
    if not rows:
        raise SystemExit(f"no usable rows for model={model!r} in {path}")
    return rows


def crossing(xs: np.ndarray, ys: np.ndarray, target: float) -> Optional[float]:
    """First x where y crosses target, linearly interpolated. None if never."""
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if ys.max() < target or ys.min() >= target:
        return None                     # never reaches it / already above throughout
    for i in range(len(xs) - 1):
        lo, hi = ys[i], ys[i + 1]
        if lo < target <= hi:
            if hi == lo:
                return float(xs[i + 1])
            w = (target - lo) / (hi - lo)
            return float(xs[i] + w * (xs[i + 1] - xs[i]))
    return None


def main():
    p = argparse.ArgumentParser(prog='compare.threshold_law')
    p.add_argument('--csv', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--x-axis', default='loc',
                   help='the swept separation axis (grouped over, not grouped by)')
    p.add_argument('--y', default='cdist_mad',
                   help='units to report the threshold in (a per-draw moment column)')
    p.add_argument('--x', default='kurt',
                   help='tail-weight regressor; `kurt` is already excess kurtosis')
    p.add_argument('--target', type=float, default=0.6)
    p.add_argument('--min-reps', type=int, default=3)
    p.add_argument('--criterion', choices=('mean', 'prob'), default='mean',
                   help="how a rung counts as passing. 'mean': replicate-average bac "
                        "reaches --target. 'prob': the SHARE of replicates above "
                        "--target reaches --prob-level. They differ whenever the "
                        "outcome is bimodal -- the prob curve lags the mean curve, so "
                        "it reports a larger requirement.")
    p.add_argument('--prob-level', type=float, default=0.5,
                   help='crossing level for --criterion prob')
    args = p.parse_args()

    rows = load(args.csv, args.model)
    moments = {'sd', 'mad', 'd_med', 'cdist_mad', 'kurt'}
    meta = {'rep', 'model', 'bac', 'secs', 'error'} | moments
    group_axes = [c for c in rows[0] if c not in meta and c != args.x_axis]

    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[a] for a in group_axes)].append(r)

    print(f"model={args.model}  target bac={args.target}  "
          f"threshold in {args.y}  regressor {args.x}")
    print(f"grouping by {group_axes}, sweeping {args.x_axis}\n")

    fitted, no_cross = [], []
    header = "  ".join(f"{a:>12}" for a in group_axes)
    print(f"{header} {'n_x':>4} {'bac range':>15} {'thresh':>8} {args.x:>8}")
    for key, rs in sorted(groups.items()):
        by_x = defaultdict(list)
        for r in rs:
            by_x[float(r[args.x_axis])].append(r)
        by_x = {x: v for x, v in by_x.items() if len(v) >= args.min_reps}
        if len(by_x) < 2:
            continue
        xs = np.array(sorted(by_x))
        if args.criterion == 'prob':
            bac = np.array([np.mean([float(r['bac']) > args.target for r in by_x[x]])
                            for x in xs])
            level = args.prob_level
        else:
            bac = np.array([np.mean([float(r['bac']) for r in by_x[x]]) for x in xs])
            level = args.target
        # Report the threshold in --y units: average the moment at each separation,
        # then interpolate on the same footing as bac.
        yv = np.array([np.mean([float(r[args.y]) for r in by_x[x]]) for x in xs])
        xr = np.array([np.mean([float(r[args.x]) for r in by_x[x]]) for x in xs])

        thr_in_x = crossing(xs, bac, level)
        line = "  ".join(f"{v:>12}" for v in key)
        rng = f"{bac.min():.3f}-{bac.max():.3f}"
        if thr_in_x is None:
            verdict = 'always>=' if bac.min() >= level else 'never'
            no_cross.append((key, verdict, bac.min(), bac.max()))
            print(f"{line} {len(xs):>4} {rng:>15} {verdict:>8} {'':>8}")
            continue
        thr_y = float(np.interp(thr_in_x, xs, yv))
        reg_x = float(np.interp(thr_in_x, xs, xr))
        fitted.append((thr_y, reg_x))
        print(f"{line} {len(xs):>4} {rng:>15} {thr_y:>8.3f} {reg_x:>8.2f}")

    print()
    if no_cross:
        print(f"{len(no_cross)} group(s) with no crossing "
              f"({sum(v == 'always>=' for _, v, _, _ in no_cross)} always at/above target, "
              f"{sum(v == 'never' for _, v, _, _ in no_cross)} never reaching it) "
              f"-- excluded from the regression.")
    if len(fitted) < 3:
        print(f"only {len(fitted)} thresholds; not enough to fit a law.")
        return

    y = np.array([f[0] for f in fitted])
    x = np.array([f[1] for f in fitted])
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"\ntwo-term law over {len(fitted)} groups:")
    print(f"  {args.y}_required = {intercept:.4f} + {slope:.4f} * {args.x}"
          f"   R^2 = {r2:.3f}")
    print(f"  {args.x} spans {x.min():.2f}-{x.max():.2f}; "
          f"threshold spans {y.min():.3f}-{y.max():.3f}")


if __name__ == '__main__':
    main()
