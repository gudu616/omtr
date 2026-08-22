# OMTR — can recall be told apart from predictable generation in small open LLMs?

> **Disclosure first:** this research was conceived, directed, and decided by the author;
> literature review, engineering, experiments, review, verification and adversarial
> red-teaming, and the papers' text were executed by AI agents (Claude) under the author's
> direction. Full statement: [DISCLOSURE.md](DISCLOSURE.md) (English + Chinese).

A study in two open-corpus model families (Pythia 410M–2.8B, OLMo-2-1B) asking whether
verbatim recall can be distinguished from merely predictable generation — behaviorally,
corpus-side, or causally. Ground truth for "memorized" comes from corpus frequency
queries (infini-gram) against the models' actual training corpora, with
construction-matched and predictability-matched controls.

**Current status — v1.0 (2026-08-22).** Three instrument generations, none of which
separated recall from predictable generation. The first was exploratory and not
preregistered — it was withdrawn by its own self-test; the two that followed were
preregistered, and so was each of the follow-up instruments reported with them.
The causal phase intervened rather than watched: corrupt one prompt token,
restore the clean activation at one of three layer bands, measure what the patch recovers.
On the arm built to remove the predictability confound, recovering the memorized
association and recovering a merely predictable continuation are **indistinguishable in
all three bands**, with the registered equivalence test passing at 0.309 nats (Pythia,
realized n=16) and 0.514 nats (OLMo, n=6). Two qualifiers belong at this level rather
than in a table footnote: OLMo's realized n=6 falls **below the smallest key (8) of the
frozen power table**, so that boundary is extrapolated; and an equivalence result is
only ever as strong as the width it rules out — these bounds are narrower than the
−1.10 to −1.13 nats seen on the secondary arm, which is why the non-separation is
informative rather than vacuous, but they do not exclude effects smaller than
themselves. The secondary arm fired in the direction the
memorization account does not predict — never-memorized predictable text recovered
*better* — and permanently carries a registered, undetachable confound qualifier. A
per-token position test and a 2×2 word-pool control, both preregistered, added a third
non-separation and rejected one instrument-side explanation on a single fragile cell.

Alongside the negative results sits one positive corpus-side measurement that explains why
the question is hard here: the gate that admits an item into the "memorized" condition is
keyed to short-window corpus frequency, so it structurally admits passages containing
often-repeated phrases — and often-quoted text is predictable text. **In this battery, the
memorized text is precisely the most quotable, hence most predictable, text.** Model span
is 410M–2.8B (3.4× for the fp32 adjudication arm, 6.8× including the descriptive model) —
under one order of magnitude, so **no scale claims are made or licensed**.

**v1.0 (2026-08-22) — causal phase.** A corrupt-and-restore intervention, plus two
zero-additional-cost supporting instruments run on the same stored data, now sit
alongside the v0.7 record above: [`docs/WRITEUP_v1.0_EN.md`](docs/WRITEUP_v1.0_EN.md)
is the current paper; two self-contained AI-oriented briefings sit in
[`notes/`](notes/) for readers who want the findings and the reading order without
the full text. The causal preregistration ships at
[`planning/CAUSAL_PREREG_v1.md`](planning/CAUSAL_PREREG_v1.md) — **a note to readers:
this preregistration contains internal cross-references to working records not
shipped with this package; the content that carries the verdict is self-contained,
and is summarized in `docs/WRITEUP_v1.0_EN.md` §4.** (Original Chinese, verbatim:
「本預註冊含指向未隨包工作紀錄的內部交叉引用；判定承載內容自足，摘要見 WRITEUP §4」)

**What v0.7 established (preserved below the v1.0 sections of the paper).** The depth
headline was withdrawn (the v0.6 instrument self-test showed its sign is set by threshold
placement). What stood was a dose–response between memorization strength and output
sharpness (rho −0.70 to −0.82 in all five runs) — and the preregistered L0P crossed control
ran against it: on 18 invented-lexeme paragraphs whose 392 probe windows all return zero in
both training corpora, a dose–response of similar strength appears (rho −0.71 to −0.77), and
at matched predictability the registered test found no memorization-specific sharpening
(condition coefficient positive — the opposite direction — in all five runs, combined
p = .099). Under the preregistration's frozen rules **none of the three registered decision
rules fired**: the paper claimed neither memorization-specificity nor generic
predictability, and published the no-verdict outcome, plus a design error in the
registration itself, as results.

## Version history (one paper, five versions; every reversal kept in the record)

| Version | What happened |
|---|---|
| v0.5 (Zenodo v1) | The adversarially reviewed pilot: memorization strength vs KL-threshold "convergence depth" |
| v0.6 (Zenodo v2) | The paper's own instrument self-test reversed the depth headline the same afternoon; withdrawal on top, v0.5 preserved unrewritten |
| v0.7 | The preregistered L0P crossed control ran; none of its registered decision rules fired; published as-is, including the registration's own design error |
| v0.7.1 | Correction: the "corpus duplication counts" anchor is operationally a median probe-window frequency, dominated by short-phrase repetition, not a passage copy count ([`docs/CORRECTION_20260821_ANCHOR.md`](docs/CORRECTION_20260821_ANCHOR.md)); no registered verdict changes |
| v1.0 | The preregistered causal phase (corrupt-and-restore), plus a per-token position test and a word-pool 2×2 on the same stored data. Primary verdict: recall and predictability are not separable within this design's equivalence bounds. Its own corrections — including four figures corrected in pre-release review — are listed in the paper's §9 |

## Read the paper

| File | What it is |
|---|---|
| [`docs/WRITEUP_v1.0_EN.md`](docs/WRITEUP_v1.0_EN.md) | **The paper** (English, canonical): three instrument generations, the causal phase, the position test, the word-pool 2×2, limitations, corrections history |
| [`docs/WRITEUP_v0.7_EN.md`](docs/WRITEUP_v0.7_EN.md) | v0.7: the preregistered L0P crossed control + the preserved v0.6/v1 record |
| [`docs/WRITEUP_v0.7_ZH.md`](docs/WRITEUP_v0.7_ZH.md) | Chinese version of v0.7, content-equivalent |
| [`docs/WRITEUP_v0.6_EN.md`](docs/WRITEUP_v0.6_EN.md) | v0.6: the instrument self-test that reversed the depth headline |
| [`docs/WRITEUP_v0.6_ZH.md`](docs/WRITEUP_v0.6_ZH.md) | Independent Chinese version of v0.6 (not a translation) |
| [`docs/WRITEUP_v0.3_EN.md`](docs/WRITEUP_v0.3_EN.md) | The frozen version that passed adversarial review |
| [`docs/PREREG_L0P.md`](docs/PREREG_L0P.md) | The L0P preregistration — committed before any L0P measurement existed |
| [`planning/CAUSAL_PREREG_v1.md`](planning/CAUSAL_PREREG_v1.md) **(Chinese)** | The causal-phase preregistration (stage 1), frozen before any GPU measurement. Its content is summarized in English in `docs/WRITEUP_v1.0_EN.md` §4 |
| [`docs/CORRECTION_20260821_ANCHOR.md`](docs/CORRECTION_20260821_ANCHOR.md) **(English + Chinese)** | The published v0.7.1 correction: what the "corpus duplication counts" anchor actually measures |

## Verify the numbers yourself

> **Correction, 2026-08-22 (package v1.0.2).** In the v1.0 package, three of the
> commands below did not run on a fresh clone, and one of them failed *silently*
> in a way that produced a false negative verdict about this very design. All are
> fixed; no reported number changed (1320/1320 values identical on recomputation).
> What was broken, what it would have shown you, and the process defect underneath
> it are in
> [`docs/CORRECTION_20260822_REPRODUCIBILITY.md`](docs/CORRECTION_20260822_REPRODUCIBILITY.md).
> To check the package yourself rather than take our word for it:
> `python harness/verify_shipping.py --bundle .`


Every statistic in the papers' results can be recomputed from the raw per-item outputs
shipped here (the reconciliation gate does it mechanically); the calibration figures and
the independent second-implementation audit regenerate from their own shipped scripts,
with stored outputs alongside:

```
python harness/reconcile.py docs/WRITEUP_v0.7_EN.md   # exit 0 = every number checks out
python harness/reconcile.py --selftest                # prove the gate itself works first
python harness/relative_depth_analysis.py             # the v0.6 self-test, from stored curves
python harness/l0p_analysis.py                        # the v0.7 registered analysis, from raw
python harness/l0p_calibration_schemes.py             # the A1 test-calibration study (synthetic nulls)
python results/l0p/audit_l0p_independent.py           # spec-only second implementation of the L0P analysis
```

The v1.0 causal phase and its two supporting instruments, from the stored patch outputs:

```
python harness/causal_analysis.py --self-test         # prove the frozen analyzer works first
python harness/causal_analysis.py --run-id main_winB1 # the registered causal verdicts
python harness/pos_test_v221.py --run-id main_winB1   # the per-token position test
python harness/verbcue_aggregate.py                   # the word-pool 2x2
python harness/gate_decay_probe.py                    # the corpus-side gate measurement (v0.7.1)
```

Frozen analysis constants (B = 20000, seed = 20260822) live in `harness/causal_analysis.py`
and are imported by everything downstream rather than copied — a single source, not to be
re-derived.

Raw measurement outputs: `results/raw/` (main pilot), `results/night/` (post-review
verification, including the adversarial memo `results/night/R1_verification.md` that
argues against parts of our own interpretation — shipped on purpose), `results/l0p/`
(the preregistered crossed control), `results/causal/` (the causal phase, the position
test and the word-pool 2×2), and `results/gate/` (the corpus-side decay probe behind
the v0.7.1 correction).

## Repository map

- `harness/` — measurement runner, corpus-verification gate, analyses (including the
  frozen causal analyzer, the position test and the word-pool aggregation), figures, the
  reconciliation gate, and the script that assembles this bundle
- `battery/` — the 104-item task battery (v2.1) with per-corpus verification evidence
- `planning/` — the causal-phase preregistration
  ([`planning/CAUSAL_PREREG_v1.md`](planning/CAUSAL_PREREG_v1.md), **Chinese**; summarized
  in English in the paper's §4) and the frozen pairing table used at run time
  (`planning/battery_expansion/`)
- `results/` — raw per-item outputs, analyses, figures; `results/causal/` holds the
  frozen verdict outputs the v1.0 paper quotes
- `docs/` — papers, item-level appendix, per-entry citation verification log
  ([`docs/CITATIONS_VERIFIED.md`](docs/CITATIONS_VERIFIED.md))
- `notes/` — the two AI-oriented briefings named above
- [`PROCESS_LOG.md`](PROCESS_LOG.md) **(Chinese)** — the full process log the papers
  cite, including every mistake we made and fixed. English readers: §2 and §9 of
  `docs/WRITEUP_v1.0_EN.md` summarize what it records
- `RUNS.md` — exact per-model run configurations

Some working records referenced by the papers are internal and do not ship (they are named
as such where they are cited). Every value that carries a verdict is quoted in the paper
itself, so no claim here rests on a file you cannot open. Shipped harness copies redact
a few internal-process comment phrases; execution-identical — the code logic matches the
frozen originals character for character, only a handful of source-code comments were
edited to remove internal workflow terminology. The plain-language concept cards that
accompany this project are likewise withheld from this release pending English
translation.

## Licenses

Code: MIT ([LICENSE-CODE.md](LICENSE-CODE.md)). Produced data and results: CC BY 4.0;
the battery's verbatim excerpts are US public-domain Project Gutenberg text, sources
enumerated in [LICENSE-DATA.md](LICENSE-DATA.md).

## Author

Leo Gudu, independent researcher — guduwangho@gmail.com

Status: the original pilot was exploratory and not preregistered; the v0.7 L0P crossed
control and the v1.0 causal phase were preregistered (frozen before any measurement) and
their frozen decision rules produced, respectively, no verdict and a non-separation —
reported as such. Not yet peer-reviewed by humans. The adversarial review that gated each
release was AI-executed and is fully documented (see DISCLOSURE.md and PROCESS_LOG.md).
