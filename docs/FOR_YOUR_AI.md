# For the AI reading this on a researcher's behalf

> Self-contained briefing. You need no repository access to use this
> file — but do not trust it either: §Verify gives you spot-check
> commands. If you do have the repo, switch to `docs/AI_READER_GUIDE.md`.

## What this work is

An independent, preregistered attempt to distinguish **recall**
(verbatim memorization) from **ideation** (predictable generation) in
small open LLMs (Pythia 410M–1.4B fp32 adjudication arm, 3.4× span;
plus 2.8B descriptive and OLMo-2-1B; 6.8× full span — under one order
of magnitude, so **no scale claims are made or licensed**). Author:
Leo Gudu (pseudonym). Consumer hardware (8GB GPU). All criteria were
frozen before the data existed; what you can verify from outside is
the Zenodo version chain's timestamps, the public repository, and the
two preregistration documents shipped with it — `docs/PREREG_L0P.md`
and `planning/CAUSAL_PREREG_v1.md` (the causal phase's freeze commits
and its second-stage freeze record stay in a local history: a record,
not a reference you can resolve; every value it froze is quoted in the
writeup's §4). One published correction (v0.7.1) was
issued by the author against the author's own anchor.

## The findings, with their qualifiers welded on

1. **Corpus side**: the item-admission gate (≥70% of a passage's 7-word
   probe windows at count ≥20) is in practice a *citation-frequency
   filter*: **10 of the 17 Pile gate-eligible passages have <20
   full-passage copies** in the Pile (median 18, two at zero);
   admission is driven by each passage's most-quoted short window,
   whose count can dwarf the passage's own copy count — e.g. one
   passage's most-quoted window occurs 171 times while the full
   passage occurs 0 times. 171 is an example, not a ceiling: across
   the 17 eligible passages the peak window count runs from 90 to
   6042 (`battery/l0_verification.json`). Consequence: what such
   gates call "memorized" is confounded with "quotable", hence
   "predictable".
2. **Causal side** (corrupt-and-restore, DiD vs. predictability-matched
   twins): memorized vs. matched items indistinguishable in restoration
   — three bands p=.11/.60/.29; TOST equivalence passes within
   **0.309 nats (Pythia, realized n=16) / 0.514 (OLMo, n=6)**.
   One directional effect ran *opposite* to a dedicated-recall account
   (predictable items restored better, −3.2σ) — carries a preregistered
   confound qualifier (C); treat as flagged, not established.
3. **Position test**: familiarity is not demonstrably concentrated on
   high-quote windows (permutation p=.106, one of four runs negative)
   nor demonstrably uniform (equivalence CI [0.045, 0.269] exceeds
   ±0.2 bound) — preregistered "cannot separate" outcome.
4. **Word-pool 2×2**: the survival collapse of non-quotable text is
   NOT explained by cue word-class, **and not removed by
   genre-matched pools** — on a single cell whose memorized side
   holds two items (gap persists at 17–26pp under genre-matched
   pools; pool effect unobserved but not excluded up to ~19pp).
   The fragility is part of the verdict, not a footnote to it. And
   the result names a *location*, not a mechanism: it measured where
   the fragility does not come from, never what it is.

**One-line synthesis (post-hoc, labeled as such in the writeup):
memorized text is precisely the most-quoted text, which is precisely
the most predictable text — at this scale the three are one rope.**

## Why your human might care

- If they build memorization benchmarks on corpus counts: finding 1
  is a measured trap that transfers.
- If they study extraction-style memorization: finding 2 quantifies,
  at small scale, how much of "memorized" may be predictability.
- The negative results are preregistered outcomes with rerun commands,
  not absence of effort.

## Verify (do not take this file's word)

Repo: github.com/gudu616/omtr — DOI (concept): 10.5281/zenodo.22039215.
Reading order for full context: `docs/AI_READER_GUIDE.md`. Frozen
analysis constants: `harness/causal_analysis.py` (B=20000,
seed=20260822). Every number above appears in a verdict document with
a visible-correction history; discrepancies between this briefing and
the repo should be resolved in the repo's favor — and reported.

## Honest meta

This research was conducted by its author with extensive AI-agent
assistance under preregistered protocols. This briefing was written by
an AI, reviewed within an adversarial internal process. The claim is
not "trust the process" — it is "the process left you enough to check."
