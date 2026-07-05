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

## 2026-07-04 — Infrastructure: vectorized the batch generator

**Why:** every item in the next round of work (more seeds, more steps, a
difficulty grid) is bottlenecked by `recall_task.py`'s per-position Python
candidate-source scan, profiled earlier at ~87ms/batch on the pre-retune
harder settings. Doing this before spending more GPU time on experiments
that would each pay that cost thousands of times.

**What changed:** replaced the O(batch_size x n_pointers x seq_len)
candidate-scanning loop with vectorized tensor ops (topk-based
without-replacement position sampling, a batched causal/non-pointer
validity mask, and a random-priority/argmax trick for uniform-among-valid
source sampling), keeping a small O(batch_size x n_pointers) loop only for
the final O(1)-per-pointer writes. Protected entirely by the existing
15-test suite in `test_recall_task.py` as the correctness oracle — every
statistical/structural property it checks (causality, no-pointer-sources,
exact pointer counts under generous settings, deterministic given a seed,
control-channel key-matching) held unchanged.

**Result, re-profiled at the actual current configs (batch=64):**

| config | before | after | speedup |
|---|---:|---:|---:|
| `default` (current zoo config) | 61.8 ms/batch | 10.0 ms/batch | 6.2x |
| `easy` | 10.9 ms/batch | 4.9 ms/batch | 2.2x |
| original harder (pre-retune) config | 85.5 ms/batch | 19.0 ms/batch | 4.5x |

At 8000 steps on `default`, this is the difference between ~8.2 minutes and
~1.3 minutes of pure data-generation overhead per run — makes the
multi-seed, extended-step, difficulty-grid work below actually affordable
in this session.

## 2026-07-04 — Causal verification completed for all 3 `hard_routing` seeds

**What:** `szoo causal-check`, 100 checks each, against the two remaining
retuned `hard_routing` checkpoints (seeds 0 and 2; seed 1 was checked in
round 2).

**Result:**

| seed | recall accuracy | moved_toward_substitute | moved_away_from_original |
|---|---:|---:|---:|
| 0 | 85.7% | 0.99 | 0.95 |
| 1 | 91.9% | 0.98 | 0.99 |
| 2 | 86.7% | 0.98 | 0.95 |

**Interpretation:** consistent, strong causal effect across all 3 seeds,
in line with each seed's own recall accuracy. `hard_routing`'s recovery in
round 2 is now causally confirmed, not just accuracy-confirmed, across its
full replication set — item 5 from the follow-up checklist is complete.

## 2026-07-04 — Fixed the packing metric (checklist item 6)

**What:** `fraction_well_reconstructed` was computed over every position
(content + pointer) flattened together, so it read `1.0` for every
primitive in round 1 regardless of recall ability — the round-1 sweep's
own numbers show this literally happened for `linear_attention` (2.9%
recall, `fraction_well_reconstructed: 1.0`). Added
`split_packing_summary()` (`metrics/packing.py`), computing
`capacity_summary` separately for content vs. pointer positions, and wired
it into `train()`'s `packing_metrics` (now `{"content": ..., "pointer": ...}`).

**Verified against real checkpoints** (fresh held-out batch, threshold 0.05):

| checkpoint | recall accuracy | content fraction_well_reconstructed | pointer fraction_well_reconstructed |
|---|---:|---:|---:|
| `standard_attention` seed0 | 99.0% | 1.000 | 1.000 |
| `hard_routing` retuned seed1 | 91.9% | 1.000 | 1.000 |
| `linear_attention` seed0 (round-1, 3000 steps) | 2.9% | 1.000 | **0.000** |

**Interpretation:** the fix does exactly what it was meant to. The old flat
metric would have (and, in round 1, literally did) report `1.0` for the
failing `linear_attention` checkpoint; the split version correctly shows
its pointer-position reconstruction is completely broken while its
(trivial) content-position reconstruction is fine. Note `hard_routing` at
91.9% *position-level* recall accuracy still shows `pointer: 1.000` here —
that's not a contradiction: `capacity_summary` aggregates per-feature MSE
*averaged across many instances*, so a feature can average out to "well
reconstructed" even when ~8% of individual positions are wrong. This is a
genuinely different (complementary, not redundant) measurement from
per-position `recall_accuracy`, not a replacement for it — worth keeping
both.

## 2026-07-04 — Literature check (checklist item 14)

**What:** a handful of targeted web searches (not exhaustive) for existing
work combining superposition/feature-isolation analysis with
sequence-mixing-primitive comparison, per the "is this novel" discussion
earlier in this project's history.

**What's directly relevant and worth knowing about:**

- **Zoology / MQAR** (Arora et al., 2023) — confirmed as the real,
  established benchmark this repo's Phase 1 design is modeled on. Its
  headline finding: "Transformers solve this easily; sub-quadratic models
  (Hyena, H3, linear attention) fail dramatically, achieving near-chance
  accuracy" — directly consistent with what round 1/round 2 found
  empirically here (`linear_attention`/`delta_net` far behind
  `standard_attention`/`hard_routing`). **Based** (Arora et al., follow-up)
  is the cited fix: hybrid linear + window attention to recover recall —
  precedent for "you need an explicit pairwise-comparison component,"
  matching this repo's revised working hypothesis.
- **"Understanding Input Selectivity in Mamba"** (arXiv 2506.11891) and
  related Mamba-mechanism papers — report that Mamba solves induction/MQAR
  not via its SSM recurrence itself but via its **short convolution**
  component; a bare selective-recurrence layer without that convolution
  (or comparable gating) is exactly what struggles. This directly explains,
  with real citable grounding, *why* this repo's `ssm` primitive (a bare
  per-channel recurrence, no convolution) plateaus: it's missing the
  specific architectural ingredient the Mamba literature already identified
  as necessary for recall, not a mysterious or repo-specific failure.
- **"When Does Content-Based Routing Work?"** (arXiv 2603.20997) — the
  closest adjacent paper found. 20+ controlled experiments, 200K-1.4B
  params, 15+ routing mechanisms, all on task accuracy/routing precision
  (not superposition or feature-interference geometry). Central finding:
  "every high-performing routing system relies on pairwise token
  comparison, while every mechanism avoiding this becomes ineffective"
  (their recurrent-model baseline: 29% vs. near-100% for pairwise-
  comparison methods) — independent, far larger-scale confirmation of this
  repo's own "explicit pairwise selection vs. compressed state" hypothesis,
  from a completely different task family (routing, not recall).
  **Confirmed via WebFetch that this paper does not measure superposition,
  interference, or representational-geometry metrics** — it is purely
  performance-oriented.
- **"An OV-Coherent Toy Model of Attention Head Superposition"**
  (Anthropic/AlignmentForum) — extends Elhage et al.'s toy-superposition
  framework to attention heads specifically (multiple heads sharing/
  interfering via OV-incoherent skip-trigrams). The closest existing
  extension of *Toy Models of Superposition* into attention mechanisms
  found in this check — but about superposition *within* one architecture
  family (multiple attention heads), not a cross-architecture comparison of
  mixing primitives.

**Honest conclusion:** the *ingredients* (MQAR-style recall, Mamba/SSM
recall limitations, pairwise-comparison-is-necessary-for-routing) are all
established, and one of them (pairwise comparison vs. compressed state)
has now been independently confirmed by a much larger, more rigorous study
than anything run in this repo. The *specific combination* this repo is
attempting — ground-truth superposition/feature-packing metrics as the
outcome variable, not task/routing accuracy, compared across a
structurally diverse zoo including a frozen-binding primitive — was not
found in this pass. This check was a handful of search queries, not a
systematic review; treat "not found" as "not found in this pass," not
"confirmed absent." The `ssm` plateau finding in particular should now be
read as consistent with, and explained by, established Mamba-mechanism
literature (missing convolution/gating), not as a repo-specific surprise.

## 2026-07-04 — First interference measurement: the actual research question (checklist item 8)

**What:** `position_masked_interference` (new, wraps the already-tested
`interference_matrix` from the Phase 0 work) applied to four real trained
checkpoints, on one fresh held-out batch (512 sequences, `default`
difficulty), separately for content vs. pointer positions, summarized as
`mean_absolute_interference`. This is the first time this project has
measured cross-feature interference on a Phase 1 model at all — everything
before this (recall accuracy, packing capacity) was either a task-
performance metric or a per-feature-averaged summary, neither of which can
reveal whether one feature's presence makes another harder to reconstruct.

**Result:**

| checkpoint | recall accuracy | content mean\|interference\| | pointer mean\|interference\| |
|---|---:|---:|---:|
| `standard_attention` seed0 | 99.0% | 0.00001 | **0.00051** |
| `hard_routing` retuned seed1 | 91.9% | 0.00001 | 0.00227 |
| `delta_net` seed0 (round 1, 3000 steps) | 3.6% | 0.00001 | 0.00347 |
| `linear_attention` seed0 (round 1, 3000 steps) | 2.9% | 0.00001 | 0.00355 |

**Interpretation, read carefully — this is one batch, one seed each, not
yet replicated:**

Content-position interference is ~zero for every primitive — expected,
since content reconstruction is a trivial near-identity mapping regardless
of architecture, so there's no cross-feature structure to interfere with
in that regime. Pointer-position interference varies substantially and, at
the extremes, tracks recall accuracy in the obvious direction: the two
primitives that are essentially failing at recall (`linear_attention`,
`delta_net`, ~3% accuracy) both show ~7x more interference at pointer
positions than `standard_attention`. That's not a dissociation — accuracy
and feature-isolation quality move together here, which is itself a
legitimate, useful (if less exciting) finding: this first pass does not
yet show an architecture that's bad at the task but clean about it, or
good at the task but a mess internally.

**But there is a real, more interesting signal in the middle of the
table.** `standard_attention` and `hard_routing` have *similar* recall
accuracy (99.0% vs. 91.9%, a 7-point gap) but a *4.4x* gap in pointer
interference (0.00051 vs. 0.00227) — proportionally much larger than the
accuracy gap. That's the first hint of exactly the kind of dissociation
this whole project exists to look for: two primitives solving the task at
roughly comparable levels of success, but doing so with measurably
different internal cleanliness. One batch, one seed, is nowhere near
enough to call this a finding — but it's the most interesting single
number produced so far, and the clear next thing to replicate before
anything else in this line of measurement.

**Known gap in this pass:** the `linear_attention`/`delta_net` numbers here
are from their *original round-1, 3000-step* checkpoints (2.9%/3.6%
accuracy) because the later extended-training runs (13.8%→24.2% and
21.1%→30.9% respectively) were run as monitoring-only trials that never
persisted a checkpoint. Re-running interference analysis against their
best (24000-step) checkpoints is a clear, cheap follow-up once those are
retrained with `run_dir` set.
