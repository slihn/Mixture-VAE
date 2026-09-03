# Sweep plan and measured costs

State of the three sweeps built on `compare/sweep.py`, and the numbers that decide
what fits in a given window. Written 2026-09-02.

## Measured throughput

One 500-epoch VAE fit, this box (24 cores, 30 GB):

| torch threads | fit time | fits/h per core |
|---|---|---|
| 1 | 274 s | **13.1** |
| 2 | 202 s | 8.9 |
| 3 | 210 s | 6.2 |

The second thread buys 1.36x and the third buys nothing, so **one thread per worker
and more workers** is ~2.3x more productive per core than the old 4 workers x 3 threads
layout. The harness pins `torch.set_num_threads(1)` in every worker for this reason.

Concurrency does **not** scale linearly — the fit is memory-bandwidth-bound, each
worker holding its own windowed tensor:

| workers | VAE fits/h (500 ep) | in 12 h |
|---|---|---|
| 8 | 62 | 740 |
| 14 | 66 | 793 |
| 19 | **68** | **821** |
| 24 | 63 | 758 |

19 workers is the peak and it is a shallow one; 24 is *worse* than 14. A single fit
takes ~950 s under 19-way load versus 274 s alone — a **3x contention penalty**. Any
estimate that scales per-core throughput linearly will be ~3x too optimistic.

**So: ~70 VAE fits/hour, ~820 in 12 hours, ~1,900 in 28 hours.** Jump is ~17 s/fit and
KMeans ~1 s, so those sweeps are bounded by grid size rather than by the clock.

## Reproducibility notes

- Results are reproducible *given* the harness's call order. `np.random.seed(rep)` must
  be set **after** data generation and immediately before the model fit: `JumpModule`
  has no `random_state` and draws its k-means++ init from the global numpy RNG, so
  seeding before `data()` (which itself consumes the RNG) shifts the draw. `run_cell`
  gets this right; ad-hoc scripts easily do not.
- Iterating a train loader advances its shuffle generator, but jump/kmeans results were
  verified unchanged by a prior pass, so listing several `--models` does not perturb the
  later ones.
- `feature_set='all'` is a verified no-op: the train/val/test tensors are bit-identical
  to the previous code path, which engineered features inside `create_dataloaders`.

## Status

| sweep | grid | fits | nominal | state |
|---|---|---|---|---|
| `compare.kmeans_sweep` | + `feature_set`, + `clip_factor` | 40,960 | ~1 h | ready |
| `compare.jump_sweep` | + `clip_factor`, + `jump_penalty` | 38,400 | ~5 h | ready, not run |
| `compare.vae_sweep` | loc x alpha x k x clip | 4,536 | ~6.5 h nominal / **~20 h real** | **deferred to the weekend** |

## Deferred: the VAE sweep

Deferred because 28 h of wall clock is not available midweek. Nothing about it is
blocked — it is only expensive.

**The pilot already settled the question that would have wasted the run.** 24 fits at
the preset (alpha=1.1, k=2.75, clip=12), 3 reps per rung:

| loc | 0.0002 | 0.0003 | 0.0005 | 0.0007 | 0.001 | 0.0015 | 0.002 |
|---|---|---|---|---|---|---|---|
| bac | 0.506 | 0.510 | 0.520 | 0.535 | 0.585 | **0.691** | 0.776 |
| cdist_mad | 0.106 | 0.160 | 0.267 | 0.371 | 0.519 | 0.734 | 0.909 |

**The VAE has a clean, monotone threshold in separation**, crossing bac 0.6 at
`loc ~ 0.0011`, `cdist_mad ~ 0.56`. This contradicts the standing claim that "the VAE
has no threshold on any axis" — that claim came from sweeping alpha/k/clip at a *fixed*
`loc=0.001`, which is exactly mid-rise on this curve, and says nothing about separation.
So there is a law to fit, and it is directly comparable to the jump model's.

Consequences for the grid when it runs:

- `loc <= 0.0005` is dead (all ~0.51) and can be dropped; the ladder only needs to
  bracket 0.0008-0.002. The current preset still carries 0.0003 — trim it before
  launching to buy back ~15% of the run.
- The preset is 1,512 cells ~ 20 h real. A weekend window covers it; a 12-h window does
  not, so trim `alpha` to two values or `reps` to 5 if it must fit overnight.
- The second claim under test — "`clip_factor` makes no difference at all (0.64-0.68
  for 8-20)" — is still open. It was measured at one separation, before the seeding fix
  showed replicate spread was understated (sd 0.013 -> 0.040), so a 0.04 band across
  clip values sat inside the noise. Flat accuracy at one `loc` would not imply a flat
  *threshold* in `loc` regardless.

## Reading the results

`compare/threshold_law.py` fits `y_required = a + b * excess_kurtosis` per model.
Validated by replaying the original 1,760-fit jump sweep: it recovers
`cdist_mad = 0.2515 + 0.0670 * kurt`, R^2 = 0.987, against the recorded
`0.339 + 0.0627 * kurt`, R^2 = 0.95. **The slope agrees to ~7%; the intercept does
not**, and the R^2 is higher, which points to a difference in how the crossing is read
(mean-bac interpolation here, possibly `p(bac>0.6)` originally). Reconcile that before
quoting a VAE law next to the jump law — the two must be read the same way to be
comparable.
