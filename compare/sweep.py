"""Parallel, resumable parameter sweeps over `GAS_SN_Comparator`.

A sweep is the cartesian product of any number of **axes** -- each axis is a
`GAS_SN_Comparator` constructor keyword and the values to try -- crossed with
replicates. Every cell generates one dataset and fits one or more models on it,
emitting a row per (cell, model).

Nothing here is specific to a model or to a parameter. The VAE loc-threshold
sweep and a later `jump_penalty` x `clip_factor` study are the same command with
different `--axis` flags.

    # the VAE two-term law: separation x tail settings
    python -m compare.sweep --out data/vae_loc_sweep.csv --models vae \\
        --axis loc=0.0005,0.0008,0.001,0.0012,0.0014,0.0016,0.0018,0.002 \\
        --axis alpha=0.8,1.1,1.6 --axis k=2.0,2.75,4.0,6.0 \\
        --axis clip_factor=8,12,20 --reps 6 --workers 19

    # a later jump study -- same harness, cheap models, so many more cells
    python -m compare.sweep --out data/jump_penalty.csv --models jump kmeans \\
        --axis jump_penalty=0.03,0.3,3,30 --axis clip_factor=8,10,12,14 \\
        --axis k=2.0,2.75,4.0 --reps 20 --workers 19

Run from the repo root; `gas-impl` and `adp_tf` must be on PYTHONPATH and
`.venv3.11` is the interpreter that has torch.

**Why one draw, many models.** Data generation is shared, and the models then see
*identical* observations, so a cross-model difference in a cell is not confounded
by the draw. It is also much cheaper than sweeping each model separately: a VAE fit
is ~274 s while jump is ~17 s and kmeans ~1 s, so adding the baselines to a VAE
sweep costs a few percent.

**Threading.** Each worker gets ONE thread, in torch and in the BLAS/OpenMP pools
alike (pinned at the top of this module, before any numeric import). Measured here, a 500-epoch VAE
fit takes 274 s at 1 thread, 202 s at 2, 210 s at 3 -- the second thread buys 1.36x
and the third nothing, so per core one thread per worker is ~2.3x more productive
than three. More workers, never more threads.

**Seeding.** `seed=rep` pins the state path, the Kanter draws, torch weight init and
the DataLoader shuffle. `JumpModule` has no `random_state` and its k-means++ init
draws from the global numpy RNG, so numpy is reseeded before every model fit -- that
also makes results independent of the order models are listed in.

**Resumability.** Rows append to `--out` and are flushed per fit; an existing file is
read back at startup and completed (cell, model) pairs are skipped. A killed run
loses at most one fit per worker. Pass `--shuffle SEED` for a long run you may have to
cut short: cells are then fitted in a seeded random order, so whatever completes is an
unbiased sample of the grid instead of the lowest values of the first axis.

**Status file.** `<out>.status` is rewritten every `--status-every` seconds (hourly by
default), at startup, on clean finish, on failure, and on SIGINT/SIGTERM. It is plain
text meant to be read the morning after an overnight run: progress, rate, per-cell
completion so a partial grid can still be analysed, and the verbatim resume command. A
`last fit` far behind `written` is the tell that the job wedged rather than being slow.
"""

import os

# MUST precede any numpy/sklearn/torch import, here or in a forked worker: BLAS and
# OpenMP read their thread caps once, at first use, so setting them later is silently
# ignored. Without this each of N worker processes spawns its own pool of ~n_cores
# OpenMP threads -- sklearn's KMeans is the worst offender -- and the box thrashes.
# Observed: 19 workers driving load average 79 on 24 cores.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import csv
import datetime as dt
import itertools
import signal
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

# Per-fit seconds, for the ETA only. VAE is quoted at 500 epochs and rescaled.
MODEL_COST = {'vae': 274.0, 'jump': 17.0, 'kmeans': 1.0, 'hmm': 8.0}
DATA_COST = 0.6

# Moments of the draw. Shared by every model in the cell, so they describe the
# data, not the fit -- these are the regressors a threshold law is fitted on.
MOMENT_FIELDS = ['sd', 'mad', 'd_med', 'cdist_mad', 'kurt']


def _coerce(v: str):
    """Numeric if it parses as one, else the bare string.

    String values are wanted: `feature_set=means,scales` is as legitimate an axis
    as `clip_factor=8,12,20`, and for KMeans it is the more informative one.
    """
    try:
        return int(v) if v.lstrip('-').isdigit() else float(v)
    except ValueError:
        return v


def parse_axis(spec: str) -> Tuple[str, List]:
    """`name=v1,v2,v3` -> ('name', [v1, v2, v3]); values numeric where possible."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"--axis must be name=v1,v2,... (got {spec!r})")
    name, _, values = spec.partition('=')
    vals = [_coerce(v) for v in values.split(',') if v != '']
    if not vals:
        raise argparse.ArgumentTypeError(f"no values in {spec!r}")
    return name.strip(), vals


def parse_fixed(spec: str) -> Tuple[str, object]:
    """`name=value` for a constant override; int if it looks like one."""
    name, _, raw = spec.partition('=')
    val: object
    try:
        val = int(raw) if raw.lstrip('-').isdigit() else float(raw)
    except ValueError:
        val = raw
    return name.strip(), val


def cell_key(axis_names: Sequence[str], row: Dict) -> Tuple:
    """Resume identity: model, replicate, and the axis coordinates.

    Values are rounded so that CSV float formatting cannot make a resumed run
    disagree with the rows already on disk about what has been done.
    """
    def norm(v):
        try:
            return round(float(v), 10)
        except (TypeError, ValueError):
            return v            # a string axis such as feature_set
    return ((row['model'], int(float(row['rep'])))
            + tuple(norm(row[n]) for n in axis_names))


def run_cell(job) -> List[Dict]:
    """Generate one dataset, fit every requested model on it, return one row each."""
    axis_names, values, rep, models, fixed = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from scipy.stats import kurtosis
    from compare.gassn import GAS_SN_Comparator

    kwargs = dict(zip(axis_names, values))
    kwargs.update(fixed)
    c = GAS_SN_Comparator(seed=rep, **kwargs)

    S, X = c.data()
    x = np.ravel(X)
    med = np.median(x)
    mad = c.MAD_SCALE * np.median(np.abs(x - med))
    states = np.unique(S)
    d_med = float(np.median(x[S == states[-1]]) - np.median(x[S == states[0]]))
    moments = {
        'sd': float(np.std(x)), 'mad': float(mad), 'd_med': d_med,
        'cdist_mad': float(d_med / mad) if mad else float('nan'),
        'kurt': float(kurtosis(x)),
    }

    rows = []
    for model in models:
        # JumpModule seeds its k-means++ init from the global numpy RNG; reseeding
        # per model keeps the cell reproducible AND order-independent.
        np.random.seed(rep)
        t0 = time.time()
        try:
            bac = float(getattr(c, f'fit_{model}')())
            err = ''
        except Exception as exc:                      # one bad cell must not kill the sweep
            bac, err = float('nan'), f"{type(exc).__name__}: {exc}"[:200]
        rows.append({**dict(zip(axis_names, values)), **moments,
                     'model': model, 'rep': rep, 'bac': bac,
                     'secs': round(time.time() - t0, 1), 'error': err})
    return rows


def _resume_command(prog: str) -> str:
    """The literal command to continue this run, PYTHONPATH included.

    `prog` is the wrapper module actually invoked, so a preset run resumes as
    itself rather than as the bare harness (which has no preset grid).
    """
    return (f"PYTHONPATH={os.environ.get('PYTHONPATH', '')} {sys.executable} "
            f"-m {prog} {' '.join(sys.argv[1:])}")


def write_status(path: str, *, grid: int, done_before: int, done_now: int,
                 todo: int, started: float, last_fit: Optional[float],
                 cells: Counter, state: str, prog: str) -> None:
    """Rewrite the human-readable status file. Cheap; hourly and on every exit."""
    now = time.time()
    elapsed = now - started
    rate = done_now / elapsed if done_now and elapsed > 0 else 0.0
    remaining = todo - done_now

    def stamp(t):
        return dt.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

    total_done = done_before + done_now
    lines = [
        f"state        {state}",
        f"written      {stamp(now)}",
        f"last fit     {stamp(last_fit) if last_fit else '(none yet)'}"
        + (f"   [{(now - last_fit) / 60:.0f} min ago]" if last_fit else ""),
        f"pid          {os.getpid()}",
        "",
        f"rows         {grid} total",
        f"complete     {total_done} ({total_done / grid * 100:.1f}%)"
        f"   [{done_before} before this run, {done_now} this run]",
        f"remaining    {remaining}",
        f"elapsed      {elapsed / 3600:.2f} h",
        f"rate         {rate * 3600:.0f} fits/h",
        f"eta          {remaining / rate / 3600:.1f} h" if rate > 0 else "eta          --",
        "",
        "completion by cell -- a threshold needs every loc and rep in the cell:",
    ]
    lines += [f"  {cell}  {n} rows" for cell, n in sorted(cells.items())]
    lines += ["", "resume with:", f"  {_resume_command(prog)}", ""]
    tmp = path + '.tmp'                     # write-then-rename: never a torn read
    with open(tmp, 'w') as fh:
        fh.write('\n'.join(lines))
    os.replace(tmp, path)


def main(preset: Optional[Dict] = None, prog: str = 'compare.sweep',
         description: str = 'Parallel resumable sweeps over GAS_SN_Comparator.'):
    """CLI entry. `preset` supplies the defaults a wrapper module bakes in.

    A preset gives `out`, `axis`, `models`, `reps` and optionally `fixed`. Passing
    `--axis` (or `--fixed`) on the command line REPLACES the preset's list rather
    than appending to it, so a preset grid can be narrowed for a test run without
    editing the module -- `action='append'` on top of a default would otherwise
    silently union the two.
    """
    preset = dict(preset or {})
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument('--out', default=preset.get('out'))
    p.add_argument('--axis', type=parse_axis, action='append', default=None,
                   metavar='NAME=V1,V2', help='a GAS_SN_Comparator kwarg to sweep; repeatable')
    p.add_argument('--fixed', type=parse_fixed, action='append', default=None,
                   metavar='NAME=VALUE', help='a constant kwarg override; repeatable')
    p.add_argument('--models', nargs='+', default=preset.get('models', ['vae']),
                   choices=sorted(MODEL_COST), help='models to fit on each draw')
    p.add_argument('--reps', type=int, default=preset.get('reps', 6))
    p.add_argument('--workers', type=int, default=19)
    p.add_argument('--status-every', type=float, default=3600.0,
                   help='seconds between status-file rewrites (default hourly)')
    p.add_argument('--shuffle', type=int, metavar='SEED', default=None,
                   help='fit cells in a seeded random order, so a run stopped early '
                        'leaves an unbiased sample of the grid rather than a prefix')
    p.add_argument('--dry', action='store_true', help='print the plan and exit')
    args = p.parse_args()

    if args.axis is None:
        args.axis = [parse_axis(a) if isinstance(a, str) else a
                     for a in preset.get('axis', [])]
    if args.fixed is None:
        args.fixed = [parse_fixed(f) if isinstance(f, str) else f
                      for f in preset.get('fixed', [])]
    if not args.out:
        p.error('--out is required (no preset default)')
    if not args.axis:
        p.error('at least one --axis is required (no preset default)')

    axis_names = [n for n, _ in args.axis]
    if len(set(axis_names)) != len(axis_names):
        p.error(f"duplicate axis name in {axis_names}")
    fixed = dict(args.fixed)
    if set(fixed) & set(axis_names):
        p.error(f"{sorted(set(fixed) & set(axis_names))} is both an --axis and --fixed")

    fields = axis_names + ['rep', 'model', 'bac'] + MOMENT_FIELDS + ['secs', 'error']
    combos = list(itertools.product(*[v for _, v in args.axis]))
    n_rows = len(combos) * args.reps * len(args.models)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline='') as fh:
            reader = csv.DictReader(fh)
            missing = set(axis_names) - set(reader.fieldnames or [])
            if missing:
                p.error(f"{args.out} lacks axis column(s) {sorted(missing)}; "
                        f"its header is {reader.fieldnames}. Use a different --out.")
            done = {cell_key(axis_names, r) for r in reader if not r.get('error')}

    todo = []
    for values, rep in itertools.product(combos, range(args.reps)):
        want = [m for m in args.models
                if ((m, rep) + tuple(round(v, 10) if isinstance(v, (int, float))
                                     else v for v in values)) not in done]
        if want:
            todo.append((axis_names, values, rep, want, fixed))

    if args.shuffle is not None:
        # itertools.product walks the first axis outermost, so an interrupted run
        # otherwise completes only the lowest `loc` values -- a prefix, not a sample.
        # A seeded shuffle makes any partial result representative, and resume still
        # works because completion is keyed on the cell, never on position.
        import random
        random.Random(args.shuffle).shuffle(todo)

    epochs = float(fixed.get('vae_epochs', 500))
    per_cell = DATA_COST + sum(MODEL_COST[m] * (epochs / 500 if m == 'vae' else 1)
                               for m in args.models)
    est_h = len(todo) * per_cell / max(args.workers, 1) / 3600
    print(f"axes {dict(args.axis)}\nfixed {fixed or '{}'}\nmodels {args.models}\n"
          f"{len(combos)} combos x {args.reps} reps x {len(args.models)} models "
          f"= {n_rows} rows | {n_rows - len(done) if done else n_rows} to fit "
          f"in {len(todo)} cells\n"
          f"{args.workers} workers x 1 thread | ~{per_cell:.0f} s/cell serial\n"
          f"estimated wall clock {est_h:.1f} h (excludes contention)", flush=True)
    if args.dry or not todo:
        return

    status_path = args.out + '.status'
    started = time.time()
    cells: Counter = Counter()
    progress = {'n': 0, 'last_fit': None}

    def snapshot(state: str) -> None:
        write_status(status_path, grid=n_rows, done_before=len(done),
                     done_now=progress['n'], todo=n_rows - len(done),
                     started=started, last_fit=progress['last_fit'],
                     cells=cells, state=state, prog=prog)

    def _on_signal(signum, _frame):     # a killed job leaves a fresh status, not the last hourly one
        snapshot(f'STOPPED by signal {signal.Signals(signum).name}')
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    snapshot('RUNNING')
    last_status = time.time()
    new_file = not os.path.exists(args.out)
    n_err = 0
    try:
        with open(args.out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                w.writeheader()
            with Pool(args.workers) as pool:
                for done_cells, rows in enumerate(
                        pool.imap_unordered(run_cell, todo), 1):
                    for row in rows:
                        w.writerow(row)
                        progress['n'] += 1
                        n_err += bool(row['error'])
                        cells[tuple(f"{n}={row[n]}" for n in axis_names)] += 1
                    fh.flush()          # a killed run keeps everything already fitted
                    progress['last_fit'] = time.time()
                    if time.time() - last_status >= args.status_every:
                        snapshot('RUNNING')
                        last_status = time.time()
                    if done_cells % 10 == 0 or done_cells == len(todo):
                        rate = done_cells / (time.time() - started)
                        print(f"{done_cells}/{len(todo)} cells | "
                              f"{progress['n']} rows | {n_err} errors | "
                              f"{rate*3600:.0f} cells/h | "
                              f"eta {(len(todo)-done_cells)/rate/3600:.1f} h", flush=True)
    except BaseException as exc:
        snapshot(f'FAILED: {type(exc).__name__}: {exc}')
        raise
    snapshot('FINISHED' + (f' with {n_err} errored rows' if n_err else ''))
    print(f"wrote {args.out} and {status_path}", flush=True)


if __name__ == '__main__':
    main()
