# superposition_zoo

A research harness for studying how sequence-mixing architectures — softmax
attention, linear attention, state-space models, delta-rule associative
memory, discrete routing, and frozen vector-symbolic binding — causally
shape superposition and feature isolation in learned representations, using
synthetic benchmarks with **known ground truth** instead of natural-language
probing.

This is not an efficiency project, and it is not trying to prove any one
architecture is "better." It's trying to find out whether the *choice of
mixing primitive* has a measurable, reproducible, causally-verifiable effect
on how cleanly a network packs features under superposition and moves
information across a sequence — using a methodology sound enough that a
positive or negative result actually means something.

**Scientific record-keeping note:** this README is a summary. The full,
dated, append-only history of every real experiment run against this repo
— actual numbers, what they mean, what they don't, methodology bugs found
and fixed along the way — lives in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
Read this file for orientation; read that one for the science.

## Why this exists (short version)

Natural-language probing (does architecture X encode phenomenon Y) runs
into a wall: linear probes find *something* almost everywhere, and you
can't tell "the architecture doesn't have this mechanism" apart from "my
task has a lexical shortcut" or "the model is too small/undertrained to
have learned the real thing" without a lot of statistical machinery to
compensate. This repo takes a different approach: generate the ground
truth yourself (Elhage et al.'s *Toy Models of Superposition*, extended
with a cross-token recall task in the style of the Zoology/MQAR
associative-recall literature), so "does the mixing primitive preserve the
true feature identity while moving it across positions" is a direct
measurement, not an inference.

The full research design, a professional critique of the approach, and the
engineering spec this repo implements are external to this repo (written
before any code here existed) — ask whoever pointed you here if you need
them. `AGENTS.md` has a fuller account of the project's history and
discipline if you're picking this up cold.

## The zoo

Six mixing primitives, five real and TDD-tested, one an honest,
intentionally-unimplemented stub:

| Primitive | What it is | Why it's in the zoo |
|---|---|---|
| `standard_attention` | causal softmax attention | reference; known from the associative-recall literature to be strong at exact recall — this is the positive control |
| `linear_attention` | kernelized causal attention via cumulative sums | mechanically closest to standard attention (identical qkv/out projection shapes, so parameter-matched for free); known to struggle at exact recall without extra machinery — a well-precedented contrast |
| `delta_net` | delta-rule associative memory (Yang et al. 2024) | real overwrite/erase dynamics instead of linear attention's additive-only accumulation |
| `hard_routing` | discrete top-1 causal routing (straight-through Gumbel-softmax) | tests "legible by construction" directly — no continuous blend to be ambiguous about |
| `ssm` | a minimal selective linear recurrence | mechanically the most different primitive from attention in the zoo (per-channel recurrent dynamics vs. pairwise comparison); no convolution/gating, so — per the literature — not expected to be strong at recall |
| `vsa_binding` | frozen random-binding (VSA-style) mixing: bind with a never-trained Rademacher code, bundle via causal cumulative sum, unbind with the same code | the zoo's strongest "legible by construction" test — no learned routing decision anywhere in the mixing step, only a learned content projection in and readout out |

Every primitive implements the same contract (`forward(x: [B, T, D]) -> [B, T, D]`,
strictly causal) and plugs into the same model shell (`SequenceModel`), so
swapping `mixing_primitive_name` in a config is the only thing that changes
between runs. See `src/superposition_zoo/mixing/` for the implementations
and `tests/mixing_helpers.py` for the shared shape/causality/gradient-flow
test contract every primitive is checked against.

## The two benchmarks

**Phase 0 — toy superposition** (`features.py`, `models/toy_autoencoder.py`):
an exact reproduction of Elhage et al. (2022)'s original setup. Sparse
ground-truth features, a linear encoder into a bottleneck, ReLU, a linear
decoder back out. No sequence structure, no mixing primitive — this is the
foundation everything else builds on, and the right place to start if
you're new to this codebase.

**Phase 1 — cross-token superposition/recall** (`recall_task.py`,
`models/sequence_model.py`): extends Phase 0 with a sequence dimension and
"pointer" positions that must retrieve an earlier position's feature
combination. Retrieval is content-addressed (MQAR-style): every position
carries a random key vector, and a pointer's key is set to exactly match
its true source position's key — precisely the kind of content-similarity
matching QK attention is naturally suited to learn. A purely local
processor cannot solve this at all, because the correct output at a
pointer position depends on content that lives at a different position.

Because every field of a generated batch is known ground truth (including
`source_position`), this repo can measure three genuinely different things
about a trained model, not just one:

1. **Task performance** (`metrics/recall.py`) — did it retrieve the right answer?
2. **Causal dependence** (`metrics/causal.py`, `causal_check.py`, `szoo causal-check`)
   — swap the true source's content for something else and check the
   output followed the swap. A direct measurement against known ground
   truth, not an inference requiring null families, matched controls, or
   FDR correction the way natural-language probing does.
3. **Feature-isolation geometry** (`metrics/packing.py`,
   `metrics/interference_report.py`) — does one feature being active make
   another harder to reconstruct? This is the actual "does architecture
   affect superposition, not just accuracy" measurement this project exists
   to make, and it's a genuinely different axis from (1) and (2) — see
   `EXPERIMENT_LOG.md` for the first real numbers and why they matter.

## Quickstart

This repo uses [`uv`](https://docs.astral.sh/uv/) exclusively — no conda,
no manually-managed virtualenv, no separate lockfile tool.

```bash
git clone https://github.com/nshkrdotcom/superposition_zoo.git
cd superposition_zoo
uv sync                     # installs locked dependencies into .venv
uv run pytest               # full test suite (fast; CPU, tiny toy-scale models)
uv run ruff check .         # lint
```

Run Phase 0 first — it's the fastest possible sanity check that the whole
pipeline works, and it's a good onboarding exercise if you're new to
superposition:

```bash
uv run szoo phase0 --config configs/phase0_toy_superposition.yaml --device cuda
```

Then the Phase 1 positive control — standard attention should reach ~100%
recall accuracy on this config. If it doesn't, something in your
environment is broken, not the architecture:

```bash
uv run szoo train --config configs/phase1_easy.yaml --device cuda
```

Then a real cross-primitive comparison:

```bash
uv run szoo sweep --config configs/zoo_starting_six.yaml \
    --primitives standard_attention,linear_attention,delta_net,hard_routing,ssm \
    --seeds 0,1,2,3,4 --device cuda
```

And ground-truth causal verification against any trained checkpoint:

```bash
uv run szoo causal-check --config configs/zoo_starting_six.yaml \
    --checkpoint runs/zoo_starting_six/standard_attention_seed0/checkpoint.pt \
    --n-checks 100 --device cuda
```

Omit `--device cuda` (or pass `--device cpu`) to run on CPU — fine for the
test suite and Phase 0, slow for Phase 1 sweeps.

### CLI reference

```
uv run szoo phase0        --config PATH [--runs-root runs] [--device cpu]
uv run szoo train         --config PATH [--runs-root runs] [--device cpu]
uv run szoo sweep         --config PATH --primitives NAME,NAME,... --seeds N,N,... [--runs-root runs] [--device cpu]
uv run szoo causal-check  --config PATH --checkpoint PATH [--n-checks 50] [--batch-size 256] [--device cpu]
```

Every `train`/`sweep` run writes `metrics.jsonl`, `checkpoint.pt`, and
`summary.json` under `runs/<config-name>/...` (or
`runs/<config-name>/<primitive>_seed<N>/...` for a sweep, plus a top-level
`sweep_results.csv`). `runs/` and `data/` are gitignored — they're
reproducible from a config and a seed, never hand-edited, never committed.
See `.gitignore` for the exact harness-vs-data split.

## Current state (summary — see `EXPERIMENT_LOG.md` for the full history)

Two zoo sweeps plus a full tuning/replication/measurement follow-up have
run so far. The short version:

- **`standard_attention` and `hard_routing`** both converge to strong
  recall accuracy (99.0%, and 85.7-91.9% across 3 replicated seeds) given
  enough training steps, and both have been **causally verified** against
  real checkpoints — patching the true source position moves the model's
  output cleanly toward the substituted content (92.9% and 82.6% relative
  movement, a corrected metric — see the log for why the naive boolean
  version of this check is misleading).
- **`linear_attention` and `delta_net`** are now fully replicated (3 seeds
  each, 16000 steps): 23.7% and 33.4% mean recall accuracy. Both show
  smooth, decelerating improvement with more training and no sign of
  `hard_routing`'s sharp transition; both remain far behind the
  attention-family primitives on every measurement (accuracy, causal
  effect size, feature-isolation quality) even at their best available
  training in this project.
- **`ssm`** is the one primitive with real plateau evidence (two learning
  rates, same ~5% ceiling, confirmed on both difficulty tiers) — consistent
  with, and explained by, published Mamba-mechanism literature (recall
  needs a convolution/gating component this minimal implementation
  doesn't have).
- **Feature-isolation measurement**, now run against every primitive's
  best available checkpoint: the accuracy / causal-effect / interference
  ordering is fully consistent across all three measurements —
  `standard_attention` > `hard_routing` ≫ `delta_net` ≳ `linear_attention`
  ≫ `ssm`. One real dissociation persists: `standard_attention` and
  `hard_routing` have similar accuracy (99.0% vs. 91.9%) but a
  disproportionate 4.4x gap in pointer-position interference — the
  clearest still-unreplicated hint of the accuracy/isolation dissociation
  this whole project exists to look for.
- **A lean difficulty grid and first depth check** ran too: `standard_attention`
  isn't meaningfully stressed by higher sparsity or more pointers at this
  scale (98-100% either way); a 2-layer version of it under-performs the
  1-layer reference at matched step count and hyperparameters (84.6% vs.
  99.0%), which reads as "needs more steps to converge with more capacity,"
  not yet evidence about the self-repair question depth was meant to probe
  — that needs a matched-accuracy comparison, not yet done.
- A **literature check** confirmed the individual ingredients here (MQAR,
  Mamba's recall limitations, pairwise-comparison-vs-compressed-state) are
  established; the specific combination (ground-truth feature-isolation
  metrics as the outcome variable across a structurally diverse zoo) was
  not found in a non-exhaustive search pass.
- A **real-language-scale check** (reusing `attention_lab`'s existing
  pipeline) is an explicit no-go for now — feasible on this hardware, but
  gated on the interference dissociation above actually replicating first,
  per this project's own stated bar for when that's warranted.

Three real bugs were found and fixed by actually running this on GPU
hardware, not by the CPU-only unit test suite: a missing positional
encoding, a recall-task design (offset-based addressing) that turned out
to be much harder to learn than intended, and a `causal_check_report`
device-mismatch/zero-parameter-model crash. All are covered by regression
tests now. See `EXPERIMENT_LOG.md` for the full account of each.

## Repository layout

```text
src/superposition_zoo/
  features.py               # Phase 0: ground-truth sparse feature generator
  recall_task.py            # Phase 1: cross-token recall benchmark generator
  config.py                 # plain dataclasses read from YAML
  train.py                  # single-run trainer (both phases)
  sweep.py                  # zoo sweep driver + cross-seed reliability summary
  causal_check.py           # ground-truth activation patching against real checkpoints
  cli.py                    # `uv run szoo ...`
  mixing/                   # the zoo: one file per primitive + the registry
  models/                   # ToyAutoencoder (phase 0), SequenceModel (phase 1)
  metrics/                  # packing, interference, recall, causal verification, reliability
tests/                      # one test file per module, TDD throughout
configs/                    # phase0/phase1/zoo YAML configs, documented inline
EXPERIMENT_LOG.md           # the dated scientific record -- read this for the actual findings
AGENTS.md                   # onboarding for a new agent/collaborator picking this up cold
```

## Design principles

- **Ground truth over inference.** Every benchmark generates its own
  labels; nothing is probed for a phenomenon whose true answer is unknown.
- **Matched budget by construction.** Primitives that reuse the same
  qkv/out projection shapes (`standard_attention`, `linear_attention`,
  `hard_routing`) are parameter-matched for free; others report their
  actual parameter count so mismatches are visible, not hidden.
- **Replication is not optional.** The sweep API always takes a list of
  seeds; `metrics/reliability.py` reports spread and cross-seed direction
  agreement as first-class outputs, not an afterthought.
- **Positive controls are mandatory, not aspirational.** `configs/phase1_easy.yaml`
  exists specifically so you can confirm the harness works — via standard
  attention's known strength at recall — before trusting any harder-regime
  or cross-primitive result. Multiple real bugs in this repo's history were
  caught by actually running this positive control on real hardware, not by
  the unit test suite.
- **A satisfied boolean is not the same as a real effect.** Discovered the
  hard way: `causal_check_report`'s `moved_toward_substitute_fraction` read
  ~1.0 for primitives with essentially zero actual causal effect, because
  any infinitesimal movement in the right direction satisfies a strict
  inequality. Fixed with a normalized `relative_movement_toward_substitute`
  metric and a regression test that constructs exactly this failure mode.
- **A stopping rule beats an open-ended search.** "Still improving, needs
  more steps" can excuse an indefinitely long tuning search. This project
  writes down, in advance, what pattern of results would count as "stop
  escalating and treat the current result as the working conclusion" —
  see `EXPERIMENT_LOG.md`'s stopping-rule entry.
- **No infrastructure ahead of evidence.** No queue daemon, no ledger, no
  claim-gate/FDR statistical layer. A sweep at this repo's scale finishes
  in minutes to tens of minutes on a single consumer GPU — there's nothing
  here for a scheduler to do yet.
- **Honest incompleteness.** Anything not built gets a concrete spec
  written down (see `EXPERIMENT_LOG.md`'s harder-recall-variants entry),
  not silently dropped.

## Known limitations

- The Phase 1 batch generator (`recall_task.py`) was profiled and
  vectorized once already (a real ~2-6x speedup depending on config,
  logged in `EXPERIMENT_LOG.md`); it may still need further optimization
  if pushed to much larger `batch_size * seq_len * n_pointers` than the
  current configs use.
- `delta_net` is implemented as an explicit sequential recurrence over
  time (~3.6x slower forward pass than `standard_attention`, profiled and
  logged) rather than a parallelized/chunked scan — correct and fine at
  this repo's toy sequence lengths, a real cost at longer ones. `ssm`,
  despite the same sequential-loop implementation pattern, turned out
  *not* to be a bottleneck (0.6x `standard_attention`) — verified by
  profiling rather than assumed.
- The recall task is a single difficulty family (content-addressed
  single-hop retrieval). Two harder variants (multi-hop retrieval,
  distractor keys) are fully specified in `EXPERIMENT_LOG.md` but not yet
  built.
- No cross-primitive parameter-budget auto-matching. `d_model` is held
  fixed across the whole zoo (the primary matching lever); each primitive's
  actual parameter count is reported via `num_parameters()` so any residual
  mismatch is visible, not silently absorbed.
- Depth is currently fixed at `n_layers=1` for every primitive across the
  zoo, specifically to keep causal-verification claims free of the
  self-repair confound that deeper networks can introduce (a component's
  causal effect can look smaller than it really is if downstream layers
  compensate for an intervention). Extending depth is a deliberate future
  step, not an oversight.

## License

MIT © 2026 nshkrdotcom
