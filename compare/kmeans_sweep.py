"""KMeans++ sweep: the feature set is the experiment, not just the GAS-SN parameters.

    PYTHONPATH=... python -m compare.kmeans_sweep --dry
    PYTHONPATH=... python -m compare.kmeans_sweep
    python -m compare.threshold_law --csv data/kmeans_sweep.csv --model kmeans

Re-runnable and enrichable; see `compare/sweep.py` for the harness and resume rules.

**Why this sweep has an axis the others do not.** KMeans is fit on the 15 engineered
columns flattened to one row per timestep, and those columns split into two families:
six rolling MEANS, which carry the +/-loc regime shift (true separation ~0.83), and
nine stds / |diff| columns, which carry *zero* signal -- the two states differ only in
`loc`, never in scale. Every observed failure is KMeans splitting on `centered_std_6`,
`centered_std_14` or `absolute_change`; every success is a split on a mean column. The
two families' best single-split variance-explained are within ~3% of each other, which
is why an unseeded redraw flips the winner.

So "KMeans is at chance on this benchmark" is a statement about *feature engineering*,
not about the model's ability to see regimes. `feature_set` makes that testable:

    all       the 15-column default -- the configuration that fails
    means     the 6 mean columns    -- p(bac>0.6) was 1.00 at every setting tried,
                                       including alpha=0.9, k=1.5 (most Cauchy-like)
    means_x   means plus the raw series
    scales    the 8 signal-free columns -- a NEGATIVE CONTROL that should sit at
                                       chance no matter how favourable the GAS-SN
                                       parameters; if it does not, the premise is wrong
    len6/len14  one rolling length only, to separate window length from family

Including `scales` is the point: a sweep that only shows `means` beating `all` is
consistent with "more features are noisier". Showing `scales` pinned at 0.5 while
`means` is pinned at 0.7, across the whole tail range, is what identifies the mechanism.

**Read the outcome as bimodal.** At `feature_set=all` KMeans lands either ~0.69 or
~0.52, nothing between, so a cell mean is a mixing proportion rather than a typical
outcome. Hence 20 reps, and report p(bac>0.6) alongside the mean. This also makes a
threshold law weaker here than for the jump model: what moves with separation is the
probability of picking the right axis, not a smooth accuracy. Check the per-cell bac
range in `threshold_law`'s table before trusting an interpolated crossing -- a cell
spanning 0.52-0.69 is bimodal and its crossing is an artefact of the mix.

**`clip_factor` runs opposite to the jump model.** KMeans wants tails *kept*: p(bac>0.6)
is 0.00 at clip 10, 0.83 at 16, 1.00 at 20 -- a wide clip makes `centered_std_6`
leptokurtic (a blob plus rare spikes), a worse thing to bisect, so the mean axis wins.
Note that mechanism predicts clip should matter at `feature_set=all` and NOT at
`feature_set=means`, where there is no scale axis to lose to. That interaction is the
sharpest test in this file.

**Cost.** ~1 s per fit, the cheapest sweep here.
"""

from compare import sweep

PRESET = {
    'out': 'data/kmeans_sweep.csv',
    'models': ['kmeans'],
    'reps': 20,          # bimodal outcome: reps buy p(bac>0.6), not a tighter mean
    'axis': [
        'loc=0.0005,0.0008,0.001,0.0012,0.0014,0.0016,0.0018,0.002',
        'alpha=0.8,1.1,1.4,1.6',
        'k=2.0,2.75,4.0,6.0',
        'clip_factor=10,12,16,20',
        'feature_set=all,means,means_x,scales',
    ],
}


if __name__ == '__main__':
    sweep.main(PRESET, prog='compare.kmeans_sweep', description=__doc__.splitlines()[0])
