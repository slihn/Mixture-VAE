"""Jump-model sweep: separation required, across tail settings and jump_penalty.

    PYTHONPATH=... python -m compare.jump_sweep --dry
    PYTHONPATH=... python -m compare.jump_sweep
    python -m compare.threshold_law --csv data/jump_sweep.csv --model jump

Re-runnable and enrichable: rows append, completed cells are skipped on restart, and
widening the grid fits only what is new. See `compare/sweep.py` for the harness.

**Supersedes the original 1,760-fit run** (`data/jump_loc_sweep.csv` +
`data/jump_moments.csv`), which gave `cdist_mad = 0.339 + 0.0627 * excess_kurt`
(R^2 = 0.95) over `loc x alpha x k` with clip and penalty both pinned. Two axes are new:

- **`clip_factor`.** The old grid held it at 12, which is the jump model's *worst*
  setting: it estimates per-state means in the m-step, so retained outliers corrupt
  them, and there is a sharp cliff between 10 and 12 (bac 0.700 at clip 8, 0.632 at 10,
  0.509 at 12, 0.505 at 14). A law fitted only at clip 12 describes the model at its
  cliff edge. Expect the threshold to depend on clip as strongly as on alpha or k --
  and note KMeans wants the opposite (>=16), so no single clip is fair to both.
- **`jump_penalty`.** Previously hard-coded at 100.0. It is a dead-zone-only lever: at
  k=2.75 that 100.0 was the *worst* of seven values tried, yet at k>=4 every penalty
  from 0.03 to 30 gives p(bac>0.6)=1.00. So its effect should appear as an interaction
  with k, not a main effect -- which is why k spans the dead zone and beyond here.

Because penalty is expected to matter only inside the dead zone, read the results as a
`penalty x k` interaction first; if the threshold law is flat in penalty at k>=4, that
reproduces the earlier finding and the axis can be dropped from later runs.

**Cost.** ~17 s per jump fit (vs 274 s for the VAE), so this grid is affordable at a
size the VAE could never reach. `kmeans` rides along for ~1 s on the *same* draws, so
the two baselines are compared without the draw confounding them.
"""

from compare import sweep

PRESET = {
    'out': 'data/jump_sweep.csv',
    'models': ['jump', 'kmeans'],
    'reps': 10,
    'axis': [
        # Dense enough that the threshold is interpolated, not guessed.
        'loc=0.0005,0.0008,0.001,0.0012,0.0014,0.0016,0.0018,0.002',
        'alpha=0.8,1.1,1.4,1.6',
        'k=2.0,2.75,4.0,6.0',          # spans the dead zone (<=2.75) and beyond (>=4)
        'clip_factor=8,12,20',         # jump wants <=10, KMeans wants >=16
        'jump_penalty=0.03,0.3,3,30,100',   # 100 = the old hard-coded default
    ],
}


if __name__ == '__main__':
    sweep.main(PRESET, prog='compare.jump_sweep', description=__doc__.splitlines()[0])
