# AGENTS.md

Onboarding for a new agent (or human collaborator) picking this repo up
cold. Read this first, then `README.md` for how to run things, then
`EXPERIMENT_LOG.md` for what's actually been found so far.

## What this repo is, in one paragraph

A research harness comparing sequence-mixing primitives (attention,
linear attention, DeltaNet, hard-routing, a minimal SSM, VSA-style frozen
binding) on synthetic benchmarks with fully known ground truth, to ask
whether the *choice of mixing primitive* measurably changes how a network
packs features under superposition and moves information across a
sequence — not which one gets the best score. This is deliberately not an
efficiency project and deliberately not chasing "architecture X is best."

## The non-negotiable discipline (read this before running anything)

These rules exist because every one of them was learned from a real
mistake, logged in `EXPERIMENT_LOG.md`. Don't relearn them the hard way.

1. **A positive control comes before any harder-regime claim.**
   `configs/phase1_easy.yaml` must show `standard_attention` at ~100%
   recall accuracy. If it doesn't, something in the environment or a
   recent code change is broken — fix that before trusting anything else,
   including your own new results.
2. **Ground truth over inference, always.** Every benchmark here generates
   its own labels. If you're tempted to add a probe-based or inferential
   metric instead of a ground-truth-anchored one, stop and reconsider —
   that's the exact trap this project was built to avoid (see the
   companion research docs on why natural-language probing failed for the
   predecessor project, `attention_lab`).
3. **Replication is not optional.** Never report or trust a single-seed
   number as a finding. Round 1 of this project's own sweep used 3 seeds
   and still turned out to be badly confounded (see below) — seed count
   alone doesn't save you from other mistakes, but skipping it guarantees
   you won't catch seed-to-seed noise.
4. **A boolean "it moved in the right direction" is not the same as a real
   effect.** Always check `relative_movement_toward_substitute` in
   `causal_check_report`, not just `moved_toward_substitute_fraction`. The
   fraction alone was found to read ~1.0 even for primitives with
   essentially zero real causal effect (see `EXPERIMENT_LOG.md`).
5. **Before extending a tuning search, write down what would make you
   stop.** "Still improving, needs more steps" can justify an unboundedly
   long search. This project's actual stopping rule (diminishing returns +
   no phase transition = stop and replicate at a fixed step count instead
   of escalating further) is in `EXPERIMENT_LOG.md` — follow the same
   discipline for any new tuning question, don't just keep doubling steps
   because the number is still going up.
6. **Commit as you go, with real messages.** Every meaningful unit of work
   (a bug fix, a new measurement, a real finding) gets its own commit with
   a message that explains *why*, not just *what*. The git log is part of
   the scientific record here, not just version control housekeeping.
7. **Update `EXPERIMENT_LOG.md`, not just the README, when you find
   something.** README is a summary that gets revised periodically;
   `EXPERIMENT_LOG.md` is the append-only dated record. Don't put new
   findings only in commit messages or only in README — the log is the
   canonical place.

## Repo layout, quick orientation

```text
src/superposition_zoo/
  features.py          # Phase 0 ground-truth sparse feature generator
  recall_task.py        # Phase 1 cross-token recall benchmark (content-addressed, MQAR-style)
  mixing/                # one file per primitive + the registry (mixing/base.py)
  models/                  # ToyAutoencoder (phase 0), SequenceModel (phase 1)
  metrics/                   # packing (+ interference), recall, causal, reliability
  train.py, sweep.py, causal_check.py, cli.py, config.py
tests/                        # one test file per src module; TDD is the norm, not the exception
configs/                        # phase0/phase1/zoo YAML, comments explain the difficulty tiers
EXPERIMENT_LOG.md                 # THE scientific record -- read before assuming something is unknown
README.md                          # user-facing summary + quickstart
```

## Dev workflow

`uv` exclusively. No conda, no manually-managed venv, no pip-tools.

```bash
uv sync                 # install locked deps
uv run pytest            # full suite -- must be green before any commit
uv run ruff check .       # lint -- must be clean before any commit
uv run szoo phase0 --config configs/phase0_toy_superposition.yaml --device cuda
uv run szoo train   --config configs/phase1_easy.yaml --device cuda
uv run szoo sweep   --config configs/zoo_starting_six.yaml --primitives ... --seeds ... --device cuda
uv run szoo causal-check --config PATH --checkpoint PATH --device cuda
```

Use `--device cuda` for anything that's a real experiment (not a unit
test) if a GPU is available — the unit test suite intentionally defaults
to CPU for portability and speed on tiny toy-scale models, but that default
should never leak into a real run you're using to draw a conclusion.

TDD is the actual practice here, not aspirational: every mixing primitive,
every metric, every bug fix in this repo's history has a test written
alongside (often before) the implementation. When you find a bug by
running something for real — and you will; this project's history is full
of exactly that — write the regression test before or immediately after
the fix, not after you've moved on.

## Where things actually stand (as of the last `EXPERIMENT_LOG.md` entry)

Read `EXPERIMENT_LOG.md` in full before assuming you know the current
state — it's dated and append-only, and this section will go stale faster
than that file does. As a pointer, not a substitute: two zoo sweep rounds
plus a tuning/causal-verification follow-up have run; `standard_attention`
and `hard_routing` both show strong, causally-verified recall;
`linear_attention`/`delta_net` are still-improving but unresolved;
`ssm` has genuine plateau evidence explained by published Mamba-mechanism
literature; the first real feature-isolation measurement found one
interesting (unreplicated) hint of the accuracy/isolation dissociation
this whole project is looking for.

## Open questions worth picking up next

(Cross-reference `EXPERIMENT_LOG.md` for the reasoning behind each.)

- Replicate the `standard_attention`/`hard_routing` interference gap across
  ≥3 seeds each — the single most information-dense thing to do next, and
  it's the stated gate for whether a real-language-scale check is even
  warranted.
- Redo the interference measurement against `linear_attention`/`delta_net`'s
  best (16000-step) checkpoints once that replication finishes, rather than
  their weak round-1 checkpoints.
- Depth (`n_layers > 1`) has been deliberately deferred to avoid the
  self-repair confound while single-layer causal claims were still being
  established — revisit once those are solid for more than 2 of 5
  primitives.
- Two harder recall-benchmark variants (multi-hop retrieval, distractor
  keys) are fully specified in `EXPERIMENT_LOG.md` but not implemented.
- `vsa_binding` is real and tested but was predicted (and then confirmed,
  see the log) to structurally lack a content-comparison operation needed
  for this specific recall task — a legitimate, different follow-up
  question is whether it shows anything interesting on the *Phase 0*
  (no-sequence) superposition benchmark instead, where its bind/unbind
  algebra is arguably a more natural fit than on cross-token recall.

## A note on tone and rigor

This project's whole reason for existing is that a predecessor line of
work (natural-language mechanistic-interpretability probing) turned out to
be unfalsifiable in practice — plausible-looking positive results that
couldn't be told apart from confounds without enormous statistical
machinery. The discipline in this repo (ground truth, positive controls,
replication, honest negative results, catching your own boolean-vs-
magnitude mistakes) exists specifically to not repeat that. When in doubt,
err toward more skepticism of your own result, not less — a "boring,"
well-verified negative finding is worth more here than an exciting,
unreplicated positive one.
