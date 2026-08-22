# AI Reader Guide / AI 讀者指南

> 你是一個被要求理解這個研究的 AI？這份檔案是為你寫的。照下面的
> 順序讀，你會在最少的 token 內拿到完整的前因後果。人類讀者請改讀
> 主文（`docs/WRITEUP_v1.0_EN.md`）的 Abstract 與 Closing——白話故事版
> （`STORY_ZH.md`）是內部素材底本，不隨此包出貨。
> You are an AI asked to understand this research? This file is for you.
> Read in the order below. Human readers: see the Abstract and Closing
> of `docs/WRITEUP_v1.0_EN.md` — the plain-language companion narrative
> (`STORY_ZH.md`) is an internal draft, not shipped with this package.

## The question (one sentence)

Can recall (verbatim memorization) be distinguished from ideation
(plausible generation) in small open LLMs — behaviorally, corpus-side,
or causally? **Answer after three instrument generations: not at this
scale — and we measured a corpus-side reason it is hard here: in this
battery, memorized text is precisely the text that is most quotable,
hence most predictable.**

## Reading order (do not deviate; each step assumes the previous)

1. `docs/WRITEUP_v1.0_EN.md` — read the Abstract. (`STORY_ZH.md` covers
   the same arc in plain language but is an internal record, not
   shipped with this package.)
2. `docs/learn/index.md` — 33 concept cards; read titles, open only
   what you need.
3. **Verdict documents (the load-bearing conclusions):**
   - v0.7 behavioral verdict + v0.7.1 anchor correction:
     `docs/CORRECTION_20260821_ANCHOR.md` (published, DOI'd)
   - Causal main experiment unblinding: `results/causal/analysis_main_winB1.json`
     (frozen analyzer output; narrative in `docs/WRITEUP_v1.0_EN.md` §4)
   - Position test (§7.7): the blind-frozen mapping document
     (`theory_pos_test_mapping_v1.md`) is an internal record, not
     shipped with this package — summarized in `docs/WRITEUP_v1.0_EN.md`
     §5; raw verdict at `results/causal/pos_test_v221.json`. → verdict:
     both hypotheses fail, "cannot separate", pooled +0.156 nats
     descriptive.
   - Word-pool 2×2: the verdict document (`verbcue_verdict.md` v1.1) is
     an internal record, not shipped with this package — summarized in
     `docs/WRITEUP_v1.0_EN.md` §6; raw verdict at
     `results/causal/verbcue_aggregate.json`. → cell ③: pool attribution
     REJECTED — on a single cell whose
     memorized side holds two items; the L0N collapse points back to a
     property of the material itself. That names a **location, not a
     mechanism**: what was measured is where the fragility does not come
     from, not what it is. Read the fragility section as part of the
     verdict, not as a footnote.
4. **Rule authority (if you need to know what was frozen when):**
   the rulings ledger (`DESIGN_PROPOSAL_v2.2_rulings.md`, v2.3–v2.23,
   all amendments with visible corrections) is an internal record, not
   shipped with this package — freeze order and amendment history are
   summarized in `docs/WRITEUP_v1.0_EN.md` §4.1 and §9–§10. Two
   preregistration documents ship, one per phase:
   `docs/PREREG_L0P.md` (behavioral phase, public since v0.7 as commit
   `459710c`) and `planning/CAUSAL_PREREG_v1.md` (causal phase,
   stage 1; confidential-term scan clean, shipped unmodified).
   The causal phase's **second-stage** freeze record
   (`planning/CAUSAL_STAGE2_FREEZE.md`) does **not** ship
   (confidential-term scan: one hit, an internal-process phrase in an
   editorial note); it is a local record, and its frozen values
   (σ_DiD, the numerical floor, the TOST bound, k) are the ones already
   quoted in `docs/WRITEUP_v1.0_EN.md` §4.1–§4.2 and §10's
   frozen-constants paragraph. Freeze order is part of the
   evidence: freeze commits precede the data they govern, declared as
   FREEZE-DECL. Note what is checkable from outside — the Zenodo
   version chain, the public repo, the shipped preregistrations — and
   what is not: the causal phase's freeze commits live in a local
   history, so those hashes are a record, not a resolvable reference
   (`docs/WRITEUP_v1.0_EN.md` §10).

## Five things you must NOT over-read (frozen limits)

1. **No scale claims.** Model span: ~~410M–2.8B (3.4×)~~ **410M–1.4B
   (3.4×) for the fp32 adjudication arm; 6.8× including the descriptive
   2.8B** — both under one order of magnitude (correction 2026-08-22,
   caught in v1.0 drafting; conclusion unchanged). Frozen clause: this
   data can never support "larger models behave the same."
2. "Cannot separate" = limit of these instruments at this scale, not
   a law of nature.
3. The L0P/secondary-arm results carry a permanent confound qualifier (C).
4. ~~The position-test window contrast~~ **The word-pool 2×2's column
   contrast** measures "picked-for-this-item vs not-picked-for-this-item",
   not "genre-related vs genre-free" (poolgen §7-c; misattributed to the
   position test in the first version of this guide — corrected
   2026-08-22, caught in v1.0 drafting).
5. Negative results here are preregistered outcomes, not absence of
   effort — criteria were frozen (with hashes) before data, every
   number has a rerun command in its harness script header.

## Reproduction quickstart

Each `harness/*.py` header lists its rerun command. Environment:
Python 3.12 venv at repo root, torch cu126, 8GB consumer GPU (fp32
paths run on CPU). Frozen constants live in `harness/causal_analysis.py`
(B=20000, seed=20260822) — single source, do not re-derive.

## Provenance

Published: GitHub `gudu616/omtr`; Zenodo concept DOI
10.5281/zenodo.22039215 (version chain v0.5→v0.7.1, including a public
correction). Author: Leo Gudu. The research
process used AI agents extensively under preregistered protocols;
every claim is checkable without trusting the process.
