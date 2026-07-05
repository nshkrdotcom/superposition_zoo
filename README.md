# superposition_zoo

A research harness for studying how sequence-mixing architectures — softmax
attention, linear attention, state-space models, delta-rule associative
memory, discrete routing, and more — causally shape superposition and
feature isolation in learned representations, using synthetic benchmarks
with **known ground truth** instead of natural-language probing.

This is not an efficiency project, and it is not trying to prove any one
architecture is "better." It's trying to find out whether the *choice of
mixing primitive* has a measurable, reproducible, causally-verifiable effect
on how cleanly a network packs features under superposition and moves
information across a sequence — using a methodology sound enough that a
positive or negative result actually means something.

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

The full research design and the reasoning behind it live in the companion
docs (not in this repo — see "Related documents" below).

## The zoo

Five real, structurally distinct, unit-tested mixing primitives, plus one
honest not-yet-implemented stub:

| Primitive | What it is | Why it's in the zoo |
|---|---|---|
| `standard_attention` | causal softmax attention | reference; known from the associative-recall literature to be strong at exact recall — this is the positive control |
| `linear_attention` | kernelized causal attention via cumulative sums | mechanically closest to standard attention (identical qkv/out projection shapes, so parameter-matched for free); known to struggle at exact recall without extra machinery — a well-precedented contrast |
| `delta_net` | delta-rule associative memory (Yang et al. 2024) | real overwrite/erase dynamics instead of linear attention's additive-only accumulation |
| `hard_routing` | discrete top-1 causal routing (straight-through Gumbel-softmax) | tests "legible by construction" directly — no continuous blend to be ambiguous about |
| `ssm` | a minimal selective linear recurrence | mechanically the most different primitive from attention in the zoo (per-channel recurrent dynamics vs. pairwise comparison) |
| `vsa_binding` | frozen random-binding (VSA-style) mixing | **not implemented** — registered but its constructor raises `NotImplementedError`, same precedent as `attention_lab`'s own `trilinear_cp` placeholder |

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
`source_position`), verifying that a model's retrieval is *causally* driven
by the true source — not a coincidental correlation — is a direct
measurement (swap the source, check the output followed the swap;
`metrics/causal.py`), not an inference requiring null families, matched
controls, or FDR correction.

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
recall accuracy on this config (verified on an RTX 5060 Ti in ~48s; expect
longer on CPU). If it doesn't, something in your environment is broken,
not the architecture:

```bash
uv run szoo train --config configs/phase1_easy.yaml --device cuda
```

Then a real cross-primitive comparison:

```bash
uv run szoo sweep --config configs/zoo_starting_six.yaml \
    --primitives standard_attention,linear_attention,delta_net,hard_routing,ssm \
    --seeds 0,1,2,3,4 --device cuda
```

Omit `--device cuda` (or pass `--device cpu`) to run on CPU — fine for the
test suite and Phase 0, slow for Phase 1 sweeps.

### CLI reference

```
uv run szoo phase0 --config PATH [--runs-root runs] [--device cpu]
uv run szoo train  --config PATH [--runs-root runs] [--device cpu]
uv run szoo sweep  --config PATH --primitives NAME,NAME,... --seeds N,N,... [--runs-root runs] [--device cpu]
```

Every run writes `metrics.jsonl`, `checkpoint.pt`, and `summary.json` under
`runs/<config-name>/...` (or `runs/<config-name>/<primitive>_seed<N>/...`
for a sweep, plus a top-level `sweep_results.csv`). `runs/` and `data/` are
gitignored — they're reproducible from a config and a seed, never
hand-edited, never committed. See `.gitignore` for the exact harness-vs-data
split.

## Zoo sweep results, round 2: per-primitive tuning + causal verification

The first pass (below, "round 1") used hyperparameters tuned only against
`standard_attention`. That's a real confound, not boilerplate — so before
trusting any of it, four underperforming primitives got a follow-up: does
each one solve the identical *easy* config (known solvable, since it's the
positive control) with the same hyperparameters, and does more training
help on the harder config? Both real, running-it-for-real questions, not
hypothetical ones.

**Round 1 (3 seeds, 3000 steps, tuned only for `standard_attention`):**

| primitive | recall accuracy (mean, 3 seeds) | final loss (mean) | params |
|---|---:|---:|---:|
| `standard_attention` | **99.0%** | 0.00045 | 53,720 |
| `linear_attention` | 2.9% | 0.0094 | 53,720 |
| `delta_net` | 3.6% | 0.0091 | 53,980 |
| `hard_routing` | 0.09% | 0.0098 | 53,720 |
| `ssm` | 0.09% | 0.0098 | 49,560 |

**Round 2 (extended training, same lr, on the identical config):**

| primitive | round 1 (3000 steps) | round 2 (8000 steps) | seeds |
|---|---:|---:|---:|
| `hard_routing` | 0.09% | **85.7% / 91.9% / 86.7% (mean 88.1%)** | 3 |
| `linear_attention` | 2.9% | 13.8% | 1 |
| `delta_net` | 3.6% | 21.1% | 1 |
| `ssm` (on the *easier* config, 2 learning rates) | — | 5.3% @ lr=3e-3, 5.1% @ lr=1e-2, both plateaued by step 4000 | 1 each |

This changes the picture substantially, and in a specific, informative way:

- **`hard_routing`'s round-1 near-zero score was pure under-training, full stop.** Its learning curve is a sharp, delayed phase transition — flat through step 5000, then 0.6% → 16.6% → 83.5% → 88.4% between steps 5000 and 8000. Given enough steps, it lands within ~10 points of standard attention, at an identical parameter budget. The straight-through Gumbel-softmax gradient is noisier early on (as the discrete-attention literature would predict) but the primitive is not fundamentally harder to optimize here — it's slower to start.
- **`linear_attention` and `delta_net` both improve substantially with more steps (2.9%→13.8%, 3.6%→21.1%) but show no sign of a `hard_routing`-style transition yet** — their curves are smooth, gradual, still climbing at step 8000, not obviously plateaued. Unresolved: would 20,000+ steps get them to attention-family territory, or do they have a lower ceiling? The associative-recall literature (Zoology/Based) would predict a real, persistent gap here — plain linear attention's shared, additive memory dilutes stored key–value pairs as more accumulate, which is a structural property, not just an optimization speed issue — but this repo hasn't run enough steps to confirm that's what's happening versus "just needs more patience."
- **`ssm` is the one primitive with actual evidence of a hard ceiling, not just under-training**: two different learning rates (3e-3 and 1e-2) on the *easier* config both plateau at essentially the same ~5% by step 4000 and never move from there through step 8000. That's a real, if still single-seed, signal that a per-channel recurrent gate with no explicit pairwise content-comparison operation has a much harder time with content-addressed retrieval specifically — consistent with why the SSM/Mamba literature had to add explicit selection/comparison mechanisms to do well at this kind of task.

**Causal verification, run for real against two trained checkpoints** (`szoo causal-check`, 100 checks each): `standard_attention` (99.0% recall) shows `moved_toward_substitute_fraction=1.0`, mean distance to the substituted content dropping from 2.44 to 0.19. `hard_routing` (retuned, seed 1, 91.9% recall) shows `moved_toward_substitute_fraction=0.98`, distance dropping from 2.29 to 0.36 — consistent with, and a bit more decisive than, its accuracy number. Both are genuine ground-truth causal confirmations that retrieval is driven by the true source position, not a correlate — exactly the thing doc 4's whole methodology exists to make possible.

**What this still doesn't establish:** `linear_attention`/`delta_net` at 8000 steps and `hard_routing`'s retuned result are single-seed (`hard_routing` aside, which has 3), not the full 5-seed replication the project's own bar calls for; nobody has caused-checked a non-attention-family primitive yet to confirm *how* it's retrieving (or failing to); `ssm`'s plateau evidence is on the easy config only, not yet on default. The revised hypothesis — recall performance splits by whether a primitive does *explicit pairwise position selection* (soft, as in standard attention, or hard, as in `hard_routing`) versus *compresses history into a shared, interference-prone state* (`linear_attention`, `delta_net`, `ssm`) — is better supported than the original "has a dot product or doesn't" framing from round 1, but it's still a hypothesis a handful of single-seed runs raised, not a finding five replicated, causally-verified seeds per primitive have confirmed.

**Also caught along the way, while running this for real:** two bugs that only showed up on actual hardware, not in the CPU-only unit test suite — `causal_check_report` crashing on zero-parameter test-double models, and a device-mismatch crash when checking a `cuda`-loaded model against a CPU-generated batch. Both fixed, both now covered by regression tests. And a `train()`/`train_phase0()` ergonomics gap — passing a plain string instead of a `Path` for `run_dir` crashed only at the very last artifact-write step, after a full 8000-step run had already burned real GPU time — fixed by converting up front, so a typo like that fails immediately from now on instead of after the expensive part is done.

## Repository layout

```text
src/superposition_zoo/
  features.py          # Phase 0: ground-truth sparse feature generator
  recall_task.py        # Phase 1: cross-token recall benchmark generator
  config.py              # plain dataclasses read from YAML
  train.py               # single-run trainer (both phases)
  sweep.py                # zoo sweep driver + cross-seed reliability summary
  cli.py                   # `uv run szoo ...`
  mixing/                  # the zoo: one file per primitive + the registry
  models/                   # ToyAutoencoder (phase 0), SequenceModel (phase 1)
  metrics/                   # packing, recall, causal verification, reliability
tests/                        # one test file per module, TDD throughout
configs/                        # phase0/phase1/zoo YAML configs, documented inline
```

## Design principles (see the architecture doc for the full reasoning)

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
  or cross-primitive result. Both of the real bugs this repo shipped with
  (see the git history on `models/sequence_model.py` and `recall_task.py`)
  were caught by actually running this positive control on real hardware,
  not by the unit test suite.
- **No infrastructure ahead of evidence.** No queue daemon, no ledger, no
  claim-gate/FDR statistical layer. A sweep at this repo's scale finishes
  in minutes to tens of minutes on a single consumer GPU — there's nothing
  here for a scheduler to do yet.
- **Honest incompleteness.** `vsa_binding` is registered and documented as
  not yet implemented, rather than a stub that silently looks real.

## Known limitations

- The Phase 1 batch generator (`recall_task.py`) assigns pointers via a
  Python-level loop per batch element; this is correct and fast enough at
  the scales this repo currently targets, but scales poorly if you push
  `batch_size * seq_len * n_pointers` much higher than `configs/phase1_default.yaml`.
  Budget real time (or optimize it) before going substantially larger.
- `delta_net` and `ssm` are implemented as explicit sequential recurrences
  over time, not a parallelized/chunked scan. Correct and fine at this
  repo's toy sequence lengths; a real efficiency concern at longer
  sequences.
- The recall task is a single difficulty family (content-addressed
  single-hop retrieval). Harder variants (multi-hop, larger key spaces,
  adversarial distractors) are natural extensions, not yet built.
- No cross-primitive parameter-budget auto-matching. `d_model` is held
  fixed across the whole zoo (the primary matching lever); each primitive's
  actual parameter count is reported via `num_parameters()` so any residual
  mismatch is visible, not silently absorbed.

## Related documents

The full research design, the critique that motivated this repo's
existence, and the engineering spec this repo implements are external to
this repo (written before any code here existed) — ask whoever pointed you
here if you need them.

## License

MIT © 2026 nshkrdotcom
