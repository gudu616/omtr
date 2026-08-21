# Preregistration: the L0P crossed control (predictability vs memorization)

**Status: registered before any L0P measurement exists.** This document and the frozen
L0P battery file are committed to the public repository BEFORE the first GPU run on
these items; the commit timestamp is the registration timestamp. No model has been run
on any L0P item at registration time, and no analysis choice below may change after
data exist. Results will be published regardless of direction.

## Question

v0.6's surviving claim: within corpus-verified memorized text, memorization dose
(teacher-forced gold log-probability) tracks output sharpness (final-layer entropy of
the model's free continuation), rho −0.70 to −0.82 in five runs. The open
identification gap (v0.6, "What stands", item 1; control gold-range overlap only
13–27%): is this **memorization-specific**, or a byproduct of **generic
predictability** — any highly predictable text producing the same sharpness?

## Design

**L0P**: 16–18 authored English paragraphs (86–91 words), each establishing a
mechanical pattern in the first ~44 words whose continuation completes the pattern.
All content words are invented lexemes, so corpus exposure is ≈ zero by construction;
predictability is titrated across three tiers (strong / medium / weak). Same split
rule, same measurement pipeline (greedy 8-token continuation, full-precision
logit-lens profiles, teacher-forced gold on the expected continuation), same five
models as the existing battery.

**Eligibility gate (novelty)**: the same 11-window probe protocol as L0N, run against
`v4_piletrain_llama` and `v4_olmo-mix-1124_llama`; an item is eligible only if every
window's count ≤ 2. Items failing the gate are excluded before any GPU run.

## Registered analyses

Primary comparison cells: L0 (memorized) vs L0P (absent, predictable), within model.

- **A1 (PRIMARY): sharpness at matched predictability.** Rank-based ANCOVA within
  each model: rank(final_entropy_mean) ~ rank(gold_logprob_per_token) + condition,
  restricted to the gold-overlap region of the two conditions. The condition
  coefficient is the estimand. Combined inference across the five runs by the
  item-level permutation scheme already shipped (one permutation applied jointly to
  the four shared-item Pythia runs; OLMo permuted independently; 20,000 iterations;
  seed 20260822). Two-sided.
- **A2: slope comparison.** Within-L0P Spearman gold~entropy per model, contrasted
  with within-L0; difference assessed by the same permutation scheme.
- **A3 (manipulation check, gates A1/A2):** the L0P gold range must overlap the L0
  gold range over at least 50% of the L0P range in ≥ 4 of 5 models. If not, the
  titration failed.

## Decision rules (written before data)

- **R-generic**: A3 passes, and A1 shows no condition effect (combined p ≥ .05 and no
  consistent sign in 4+ of 5 runs), and A2 slopes are comparable → we will state that
  the sharpness effect is **not shown to be memorization-specific** and reads as
  generic predictability; v0.7 will carry that conclusion in its headline position.
- **R-memo**: A3 passes, and A1 shows memorized items sharper at matched gold
  (combined p < .05, consistent sign in ≥ 4 of 5 runs) → we will state that a
  memorization-specific component **survives its first crossed control** (still
  correlational; causal phase unchanged).
- **R-failed**: A3 fails → we report a failed manipulation, draw no conclusion in
  either direction, and redesign. This outcome will be published too.
- Anything not covered above is exploratory and will be labeled as such.

## What this cannot show

Causation, in any outcome. A1 conditions on gold, which is itself model-derived;
R-memo therefore upgrades plausibility, not mechanism. The causal phase (activation
patching) remains the arbiter and remains preregistered separately before it runs.

*Author: Leo Gudu. Analysis executed by AI agents under the author's direction, per
`DISCLOSURE.md`. 2026-08-21.*
