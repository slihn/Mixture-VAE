"""Mixture-VAE sweep: the separation it requires, across tail settings.

    PYTHONPATH=... python -m compare.vae_sweep --dry
    PYTHONPATH=... python -m compare.vae_sweep
    python -m compare.threshold_law --csv data/vae_sweep.csv --model vae

Re-runnable and enrichable; see `compare/sweep.py` for the harness and resume rules.
This is the overnight job -- a VAE fit is ~274 s -- so the status file and chunk-level
resume matter here more than anywhere else.

**What this is testing.** Two claims from the earlier 64-fit sweep, both of which
deserve a harder look:

- *"The VAE has no threshold on any axis."* If true there is no law to fit and the
  interesting result is the absence -- `threshold_law` reports such cells as
  `always>=` rather than dropping them. Note the VAE already scores 0.651 at
  loc=0.001, mid-way up the jump model's old ladder, so the `loc` axis below reaches
  lower than the jump grid ever did.
- *"`clip_factor` makes no difference at all (0.64-0.68 for 8-20)."* That was measured
  at one separation, before the seeding fix revealed the VAE's replicate spread was
  understated (sd 0.013 -> 0.040). A 0.04 band across clip values sits inside that
  noise, so flatness was never really established -- and flat accuracy at one `loc`
  would not imply a flat *threshold* in `loc` anyway.

**The fat left tail is a property of the model, not noise.** Seeded 10-rep runs at the
preset give 0.651 sd 0.040, range 0.587-0.693 -- the low draws survive seeding. So `sd`
is a misleading summary and one outlier in 6 reps can drag a cell's mean below a 0.6
target. Prefer the median or p(bac>0.6) when reading a cell, and treat a single low
cell as a candidate for more reps rather than as a threshold crossing.

`jump` and `kmeans` ride along on the same draws for ~6% extra wall clock, which makes
the three models' requirements directly comparable rather than measured on
different data.
"""

from compare import sweep

PRESET = {
    'out': 'data/vae_sweep.csv',
    'models': ['vae', 'jump', 'kmeans'],   # baselines are ~6% extra, on identical draws
    'reps': 6,
    'axis': [
        # Pilot: bac is ~0.51 at loc<=0.0005 and the 0.6 crossing sits near 0.0011,
        # so the ladder brackets the crossing instead of paying for dead rungs.
        'loc=0.0005,0.0008,0.001,0.0011,0.0012,0.0014,0.0016,0.002',
        'alpha=0.8,1.1,1.6',
        'k=2.0,2.75,4.0,6.0',
        'clip_factor=8,12,20',
    ],
}


if __name__ == '__main__':
    sweep.main(PRESET, prog='compare.vae_sweep', description=__doc__.splitlines()[0])
