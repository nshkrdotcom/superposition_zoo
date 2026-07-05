# Experiment Log

A dated lab notebook: every real experiment run against this repo, in the
order it happened, with the actual numbers and what they do and don't mean.
This is the scientific record. `README.md` is user-facing and summarizes
current best-supported conclusions with a pointer back here; it is not
where the raw history lives.

Entries are append-only. Don't rewrite history here — if a finding is later
superseded, add a new entry that says so and cross-references the old one.

---

## 2026-07-04 — Round 1: first zoo sweep

**What:** 5 primitives (`standard_attention`, `linear_attention`, `delta_net`,
`hard_routing`, `ssm`) x 3 seeds, `configs/zoo_starting_six.yaml` (3000
steps, lr=0.003, `default` difficulty: n_features=24, sparsity=0.5,
seq_len=64, n_pointers=6, min_gap=3, d_model=64), on an RTX 5060 Ti.

**Result:** `standard_attention` 99.0% mean recall accuracy; every other
primitive under 4% (`linear_attention` 2.9%, `delta_net` 3.6%,
`hard_routing` 0.09%, `ssm` 0.09%). Seed-to-seed std tiny for every
primitive; direction agreement 1.0 vs. baseline for all four.

**Caveat flagged at the time:** lr/steps were tuned only against
`standard_attention`'s own learning curve. Not yet a fair per-primitive
comparison.

## 2026-07-04 — Round 2: per-primitive tuning + first causal verification

**What:** (a) tested `linear_attention`, `delta_net`, `hard_routing`, `ssm`
on the `easy` config (known solvable) with the same hyperparameters that
worked for `standard_attention` there; (b) extended training to 8000 steps
on `default` for `hard_routing` (3 seeds), `linear_attention` (1 seed),
`delta_net` (1 seed); (c) tried `ssm` at two learning rates (3e-3, 1e-2) on
`easy`; (d) ran `szoo causal-check` (100 checks each) against the trained
`standard_attention` and retuned `hard_routing` (seed 1) checkpoints.

**Result:**
- `hard_routing` on `easy` (3000 steps, same lr as standard attention): **100%** recall accuracy — fully solves it.
- `linear_attention` on `easy`: 60.2% at 3000 steps, climbing to 78.0% by 8000 steps — smooth, not plateaued.
- `delta_net` on `easy`: 43.0% at 3000 steps, climbing to 52.6% by 8000 steps — smooth, noisier, slower than linear_attention.
- `ssm` on `easy`: 4.7% (lr=3e-3) / plateaus at 5.1-5.3% (lr=1e-2) by step 4000, flat through step 8000 at both learning rates.
- `hard_routing` on `default`, 8000 steps, 3 seeds: **85.7% / 91.9% / 86.7%** (mean 88.1%) — up from round 1's 0.09%. Learning curve: flat through step 5000, then 0.6% → 16.6% → 83.5% → 88.4% between steps 5000-8000 — a sharp, delayed phase transition, not a gradual improvement.
- `linear_attention` on `default`, 8000 steps, 1 seed: 13.8% (up from 2.9%).
- `delta_net` on `default`, 8000 steps, 1 seed: 21.1% (up from 3.6%).
- Causal check, `standard_attention` (99.0% recall): 100/100 checks moved toward the substituted content; mean distance to substitute dropped 2.44 → 0.19.
- Causal check, `hard_routing` retuned seed 1 (91.9% recall): 98/100 moved toward substitute; distance dropped 2.29 → 0.36.

**Interpretation:** round 1's near-zero score for `hard_routing` was pure
under-training, not an architectural limit — full reversal of that part of
the round-1 story. `linear_attention`/`delta_net` show real, substantial,
still-improving learning with more steps but no sign yet of a
`hard_routing`-style transition — open whether they'd eventually match it
or have a lower ceiling. `ssm` is the one primitive with actual two-learning-rate
plateau evidence, suggesting (not yet confirming) a real architectural
bottleneck for content-addressed retrieval specifically, consistent with
why the SSM/Mamba literature added explicit selection mechanisms.

**Bugs found and fixed while running this for real (not caught by the
CPU-only unit test suite):** `causal_check_report` crashed with
`StopIteration` against zero-parameter test-double models; a
device-mismatch `RuntimeError` when a `cuda`-loaded model was checked
against a CPU-generated batch; `train()`/`train_phase0()` crashed with
`AttributeError` on a plain-string `run_dir`, only at the very last
artifact-write step — cost ~12 minutes of real GPU time before being
caught. All three fixed, all three now covered by regression tests.

**Revised working hypothesis (not yet confirmed):** the split looks like
it's about *explicit pairwise position selection* (soft, as in standard
attention, or hard, as in `hard_routing`) vs. *compressing history into a
shared, interference-prone state* (`linear_attention`, `delta_net`, `ssm`),
not simply "has a dot product or doesn't" (round 1's framing, which
`hard_routing`'s recovery falsifies).

**Still open going into the next round:** `linear_attention`/`delta_net` at
single-seed; `ssm`'s plateau untested on `default`; no causal verification
yet for any non-attention-family primitive; the packing/superposition
metric (`fraction_well_reconstructed`) has read 1.0 for every primitive so
far regardless of recall ability, meaning it hasn't actually differentiated
anything yet — the core "does architecture affect feature isolation, not
just accuracy" question this whole project exists to ask has not yet been
measured.
