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

## First zoo sweep result (preliminary — read the caveat)

Run on an RTX 5060 Ti, `configs/zoo_starting_six.yaml`, 3 seeds, ~35 minutes
total:

| primitive | recall accuracy (mean, 3 seeds) | final loss (mean) | params |
|---|---:|---:|---:|
| `standard_attention` | **99.0%** | 0.00045 | 53,720 |
| `linear_attention` | 2.9% | 0.0094 | 53,720 |
| `delta_net` | 3.6% | 0.0091 | 53,980 |
| `hard_routing` | 0.09% | 0.0098 | 53,720 |
| `ssm` | 0.09% | 0.0098 | 49,560 |

Seed-to-seed spread is tiny for every primitive (std well under 1% of the
mean in every row) and every non-baseline primitive agrees in direction
across all 3 seeds — the *reliability* half of this result is solid.

**The comparison itself is not yet a fair one, and this is a real,
deliberate caveat, not boilerplate.** `lr` and `steps` in this config were
tuned empirically against `standard_attention`'s own learning curve (see
`configs/phase1_default.yaml`'s comment) — verified to sit comfortably past
*its* sharp, grokking-like phase transition. No equivalent per-primitive
tuning pass has been done for the other four. So this table currently shows
"how well each primitive does under hyperparameters chosen for standard
attention," not "how well each primitive can do." Per doc 4's own definition
of a real finding (replicate across seeds ✅, matched parameter budget ✅
mostly, causal verification ❌ not yet run, per-primitive tuning ❌ not yet
done): this clears two of four bars. Treat it as a directional, hypothesis-
generating first pass, not a conclusion.

The directional split is still worth naming as a hypothesis to chase, not a
finding: `standard_attention`, `linear_attention`, and `delta_net` (every
primitive with an explicit pairwise query-key dot-product) show *some*
non-zero recall signal; `hard_routing` (discrete argmax routing, known in
the literature to be hard to optimize via the straight-through estimator)
and `ssm` (a pure per-channel recurrence with no explicit content-matching
operation at all) show exactly zero in 2 of 3 seeds. If that split survives
per-primitive hyperparameter tuning and longer training, it would suggest
recall specifically requires an explicit content-comparison mechanism that
a bare selective-recurrence state update doesn't provide for free — but
that's a hypothesis this sweep raises, not one it has confirmed.

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
