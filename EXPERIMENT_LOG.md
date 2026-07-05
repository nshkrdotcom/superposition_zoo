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

## 2026-07-04 — Extended trials complete: `ssm`'s plateau generalizes to `default` (checklist item 3)

**What:** monitored single-seed trials at 24000 steps for `linear_attention`
and `delta_net` (`default` config, lr=3e-3, up from round 2's 8000 steps),
and `ssm` at 8000 steps on `default` itself (not just `easy`, closing the
gap flagged at the end of round 2).

**Result:**

| primitive | 8000 steps | 24000 steps | shape |
|---|---:|---:|---|
| `linear_attention` | 13.8% | 24.2% | smooth, slowing (asymptotic-looking) |
| `delta_net` | 21.1% | 30.9% | noisier, also slowing |
| `ssm` (on `default`, not `easy`) | — | 1.17% (max 1.24% at step 7000) | flat |

**Interpretation:** `ssm`'s plateau is now confirmed on the actual harder
config, not just `easy` — checklist item 3 is closed, and combined with the
literature check above (Mamba needs its short-convolution component for
recall; this repo's `ssm` has none), this is now a well-grounded, not just
observed, negative result. `linear_attention`/`delta_net` both continued
improving from 8000→24000 steps (roughly +10 points each) but with clearly
diminishing returns and no `hard_routing`-style transition — see the
stopping-rule entry immediately below for what this means for further
step-count escalation.

## 2026-07-04 — Stopping rule for the tuning search (checklist item 15)

Per doc 3's own advice: writing this down now, informed by real trajectory
data, specifically so "still improving, needs more steps" doesn't become an
unfalsifiable excuse to keep extending the search indefinitely.

**The rule, as applied to `linear_attention`/`delta_net` on `default`:**
tripling the step budget (8000→24000) bought roughly +10 percentage points
for each, with a smooth, decelerating (not accelerating) curve and no sign
of a `hard_routing`-style sharp transition anywhere in that range. That
pattern — diminishing returns, no transition — is the stopping condition:
further step-count escalation within this project is deprioritized in
favor of (a) replicating at a *fixed, already-observed-informative* step
count across real seeds, and (b) treating "these two primitives plateau
well below the attention-family primitives at this parameter budget, on
this task" as the current working conclusion, revisable if a future,
specific reason (not just "try more steps and see") comes up to revisit it.

**Concretely:** the next replication round uses **16000 steps** (not
24000) — it captures most of the observed gain (`linear_attention` 19.5% at
16000 vs. 24.2% at 24000; `delta_net` 27.0% at 16000 vs. 30.9% at 24000,
each within ~5 points of the 24000-step value) at meaningfully lower cost,
and persists real checkpoints this time so the interference measurement
above can be redone against each primitive's best available version rather
than its round-1, 3000-step checkpoint.

**This rule does not apply to `ssm`**, whose stopping point was already
decided differently: two learning rates, both flat by step 4000, now
confirmed on both difficulty tiers — that's a plateau-shape signal, not a
still-rising one, and doesn't call for the same "diminishing returns"
reasoning at all.

## 2026-07-04 — Causal-check for all 5 primitives + a real methodology fix (checklist item 4)

**What:** ran `szoo causal-check` against `linear_attention`, `delta_net`,
and `ssm`'s round-1 (weak, 3000-step, ~3%-or-less recall) checkpoints for
the first time — no non-attention-family primitive had been causally
checked before this. Also re-ran `standard_attention` and `hard_routing`
for comparison under the corrected metric described below.

**A real methodology gap, found by actually reading the numbers, not by a
unit test:** all three weak checkpoints showed
`moved_toward_substitute_fraction: 1.0` (or 0.96–1.0) — read naively, that
looks like "100% causal effect, same as the strong primitives." But the
raw distances told a different story: `ssm`'s mean distance to the
substituted value moved from 1.876 to 1.872 — essentially no movement at
all — yet still satisfied the strict "moved closer" boolean by an
infinitesimal margin. The boolean can't distinguish a real effect from a
technically-positive but negligible one. Added
`relative_movement_toward_substitute` to `causal_check_report`
(normalizes movement by the starting distance), with a regression test
(`_WeakEffectModel`) constructing exactly this scenario — a model that
satisfies the boolean on every check while moving a hand-chosen tiny
amount — and asserting the new metric correctly reports it as small.

**Result, all 5 primitives, `relative_movement_toward_substitute`:**

| primitive | recall accuracy | moved_toward (boolean) | relative movement |
|---|---:|---:|---:|
| `standard_attention` | 99.0% | 1.00 | **92.9%** |
| `hard_routing` (retuned) | 91.9% | 0.99 | **82.6%** |
| `delta_net` (round 1) | 3.6% | 1.00 | 5.6% |
| `linear_attention` (round 1) | 2.9% | 1.00 | 4.9% |
| `ssm` (round 1) | 0.09% | 1.00 | **0.2%** |

**Interpretation:** with the corrected metric, the boolean's misleading
uniformity disappears completely — there's a clean, large gap between the
two primitives with genuine, strong causal retrieval (83-93% relative
movement) and the three that don't meaningfully retrieve at all (0.2-5.6%),
matching their recall-accuracy ordering exactly. This is a good outcome
for trusting the tool: once measured correctly, causal verification agrees
with accuracy in the direction you'd expect, rather than the boolean's
false impression that everything shows "100% causal effect." Item 4 from
the checklist is now complete for all 5 primitives, using currently-
available checkpoints; re-running against `linear_attention`/`delta_net`'s
upcoming 16000-step replication checkpoints (in progress, see the stopping-
rule entry above) is a natural, cheap follow-up once those exist.

## 2026-07-04 — Assessed sequential-recurrence performance (checklist item 13); decided not to parallelize

**What:** profiled forward-pass wall time (CPU, 20 iterations, warmed up)
for all 5 primitives at the current `default` config shape, to decide
whether `delta_net`/`ssm`'s sequential-loop recurrence is worth
parallelizing, now that the batch generator itself is no longer the
dominant bottleneck (see Phase 0 entry above).

**Result:**

| primitive | ms/forward (CPU) | vs. `standard_attention` |
|---|---:|---:|
| `standard_attention` | 5.57 | 1.0x |
| `hard_routing` | 4.00 | 0.7x |
| `ssm` | 3.41 | 0.6x |
| `linear_attention` | 10.84 | 1.9x |
| `delta_net` | 20.19 | 3.6x |

**Decision: not attempted this session.** Two real surprises here worth
recording: `ssm` is not a bottleneck at all — its recurrence is a single
elementwise update per step (`h = a*h + c`), cheap enough that it's
actually faster than standard attention. `delta_net` is the real cost
(3.6x attention), from three einsum calls per sequential step rather than
one. At 16000 steps, that's roughly `(20.19-5.57)ms * 16000 ≈ 234s` (~4
minutes) of extra wall clock per run — real, but smaller than the
batch-generator win already banked (Phase 0: ~8 minutes saved per
8000-step run), and a correct chunked/parallel-scan rewrite of the delta
rule is a nontrivial, error-prone undertaking that risks correctness
regressions in a primitive whose core recurrence is currently validated by
a hand-computed closed-form test. Revisit if `delta_net` becomes central to
a specific follow-up question that needs many more long runs of it
specifically; not worth the risk for this session's remaining scope.

## 2026-07-04 — Go/no-go decision: real-language-scale check via `attention_lab` (checklist item 17)

**Decision: no-go for this session, explicitly, not a silent skip.**

Doc 4's own gating logic: a real-language-scale check is warranted only
once a toy-scale primitive shows a clear, replicated, causally-verified
*dissociation* between accuracy and feature-isolation quality — the actual
novel claim this whole program exists to test, not "does an architecture
get a good score."

As of this entry, the interference measurement above (item 8) found
exactly one hint of such a dissociation — `standard_attention` vs.
`hard_routing`, similar accuracy (99.0% vs. 91.9%) but a 4.4x gap in
pointer-position interference — from **one batch, one seed each**. That is
nowhere near "clear, replicated, causally-verified." Attempting a
multi-hour real-language run on the strength of one unreplicated number
would repeat exactly the mistake this whole follow-up round was designed
to avoid (see round 1's own caveat about treating a first-pass result as a
conclusion).

**On feasibility, since it was asked directly:** `attention_lab`'s own
E001-E004 work already ran full 30M-parameter, 3000-step FineWeb-Edu
training on this exact host (RTX 5060 Ti, 16GB VRAM, 192GB system RAM),
taking anywhere from ~2 hours (`standard`) to ~13 hours (`cp_trilinear`)
per run depending on architecture. So yes, technically feasible on this
hardware — but expensive enough that it should never be attempted
speculatively, only once there's a specific, replicated toy-scale result
worth checking at real-language scale.

**What would flip this to a go:** replicate the `standard_attention` vs.
`hard_routing` interference gap across ≥3 seeds each (cheap, minutes, no
new capability needed) and confirm it holds up. If it does, that's exactly
the kind of finding worth spending hours of real-language compute to
check.

## 2026-07-04 — Harder recall variants: scoped, not built (checklist item 10)

**Decision:** designed, not implemented, this session — honestly scoped
rather than silently dropped, per the checklist's own explicit allowance
for this item.

**Two concrete variants worth building next, specified precisely enough to
implement directly when picked back up:**

1. **Multi-hop retrieval.** A pointer's key matches not a content
   position's key directly, but *another pointer's* key — i.e., resolving
   pointer A requires first resolving pointer B (which A points to), which
   points to a real content position. Tests whether a mixing primitive can
   chain retrieval operations, not just perform one in isolation. Minimal
   implementation change: `generate_recall_batch` would need to allow a
   pointer's `source_position` to itself be a pointer for controlled hop
   depths (currently explicitly excluded, "sources are never themselves
   pointers" — see `test_sources_are_never_themselves_pointers`), plus a
   `max_hops` parameter and a corresponding ground-truth field recording
   the full hop chain, not just the immediate source, so causal-check can
   still verify the *final* resolved content is right.
2. **Distractor keys.** Add near-miss keys — vectors deliberately close
   (in cosine similarity) to a pointer's true target key but not identical
   — at other positions, to test whether a primitive's matching mechanism
   is doing exact/robust comparison or something fuzzier that a
   close-but-wrong key could fool. Minimal implementation change: after
   generating the random key tensor, for a controlled fraction of non-
   source positions, set `key[b, s'] = key[b, s] + small_gaussian_noise`
   for a chosen true source `s`, and track which positions are "true
   source" vs. "distractor" so recall accuracy and causal-check can be
   broken down by whether a wrong retrieval came from a random guess or a
   genuine near-miss confusion.

**Why not built now:** both are real, bounded scope, but this session's
remaining time is better spent finishing the replication/interference
follow-through already in progress (items 1/8's redo) and the
documentation pass (README, AGENTS.md) than opening a new benchmark
variant with its own full TDD cycle. Flagged here with enough detail that
picking this up cold later doesn't require re-deriving the design.

## 2026-07-04 — 16000-step replication complete; interference/causal redone on best checkpoints (checklist items 1, 8, 4 redo)

**What:** the stopping-rule-informed replication (3 seeds each,
`linear_attention`/`delta_net`, 16000 steps, `default` config, persisted
checkpoints this time) finished. Redid the interference measurement and
causal-check against the new best checkpoints (seed with highest recall
accuracy each).

**Replication result:**

| primitive | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| `linear_attention` | 23.4% | 23.2% | 24.5% | 23.7% |
| `delta_net` | 31.8% | 35.4% | 33.1% | 33.4% |

Tight seed-to-seed spread for both — this is now a real, replicated number,
not a single-seed spot-check, closing checklist item 1.

**Interference and causal-check, best checkpoint of each (`linear_attention`
seed 0, `delta_net` seed 1), vs. their earlier round-1 (3000-step, ~3%
accuracy) checkpoints:**

| checkpoint | recall accuracy | pointer mean\|interference\| | relative movement (causal) |
|---|---:|---:|---:|
| `linear_attention` round 1 (3000 steps) | 2.9% | 0.00355 | 4.9% |
| `linear_attention` 16k (best seed) | 23.4%→19.0% (fresh batch) | 0.00308 | 24.7% |
| `delta_net` round 1 (3000 steps) | 3.6% | 0.00347 | 5.6% |
| `delta_net` 16k (best seed) | 35.4%→35.8% (fresh batch) | 0.00287 | 30.8% |

**Interpretation:** both interference and causal effect size improved
substantially and consistently alongside accuracy — real, sensible,
internally-consistent movement in the expected direction, not noise. But
both primitives, even at their best available training in this project,
remain far short of `standard_attention` (92.9% relative movement,
0.00051 interference) and `hard_routing` (82.6%, 0.00227). The
accuracy/interference/causal-effect ordering across all 5 real primitives
is now fully consistent: `standard_attention` > `hard_routing` ≫
`delta_net` ≳ `linear_attention` ≫ `ssm`, on every one of the three
measurements this project makes. No further dissociation beyond the one
already noted (`standard_attention` vs. `hard_routing`'s disproportionate
interference gap relative to their similar accuracy) has turned up in this
pass.

## 2026-07-04 — Lean difficulty grid (checklist item 9)

**What:** two quick difficulty variations on `standard_attention` (3000
steps, the config already known to transition quickly at this lr):
higher sparsity (0.8 vs. default 0.5, i.e. fewer active features per
position) and more pointers (10 vs. default 6).

**Result:** 98.2% (higher sparsity) and 99.8% (more pointers) recall
accuracy — `standard_attention` isn't meaningfully stressed by either
change at this scale. **Informative on its own:** the difficulty knobs
tried here aren't actually hard for a working retrieval mechanism to
begin with; they'd need to be pushed further (much higher `n_pointers`,
much longer `seq_len`, or reduced `d_model`/capacity) to matter for an
architecture that already gets the mechanism right. This grid is more
useful for testing *whether a struggling primitive gets relatively worse*
as difficulty increases than for stressing the reference architecture —
a natural next grid to run once a specific comparison question calls for
it, rather than as a general-purpose sweep.

## 2026-07-04 — First depth check: `standard_attention` at `n_layers=2` (checklist item 11)

**What:** trained `standard_attention` with `n_layers=2` (double the
parameters, 103,704 vs. 53,720) on `default`, 3000 steps, same
hyperparameters as the 1-layer reference, then ran `szoo causal-check`
against it.

**Result:** 84.6% recall accuracy (vs. the 1-layer model's 99.0% at the
same step count and hyperparameters) and 73.7% relative movement (vs.
92.9% for 1-layer).

**Interpretation — genuinely inconclusive on the self-repair question,
and worth being precise about why:** more depth did not automatically
help here; the 2-layer model is *behind* the 1-layer model at this same
step budget, not ahead, on both accuracy and causal-effect size. That's
plausibly just "more capacity needs more steps to converge," not evidence
about self-repair specifically — self-repair (doc 3 §4's concern) is
about whether a *component's* causal effect looks artificially small
because downstream layers compensate for an intervention, and testing
that properly requires comparing relative movement at *matched* accuracy
levels, not comparing a well-converged 1-layer model against an
under-converged 2-layer one. The 73.7%-vs-92.9% gap here is consistent
with the accuracy gap (84.6% vs. 99.0%) and doesn't obviously exceed it in
a way that would specifically implicate self-repair. This is a real first
data point, not a real answer — the matched-accuracy comparison (train
the 2-layer model longer, to ~99%, then compare relative movement) is the
actual next step for this question, not yet done.
