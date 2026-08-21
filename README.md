# OMTR — corpus-verified memorization, output sharpness, and a withdrawn depth metric

> **Disclosure first:** this research was conceived, directed, and decided by the author;
> literature review, engineering, experiments, review, verification and adversarial
> red-teaming, and the papers' text were executed by AI agents (Claude) under the author's
> direction. Full statement: [DISCLOSURE.md](DISCLOSURE.md).

A pilot study in two open-corpus model families (Pythia 410M–2.8B, OLMo-2-1B) asking:
within text a model has verifiably memorized, does memorization strength track how early
the layer-wise prediction settles? Ground truth for "memorized" comes from corpus
frequency queries (infini-gram) against the models' actual training corpora, with a
construction-matched non-memorized control.

**Current status — v0.7 (2026-08-21).** The depth headline is withdrawn (the v0.6
instrument self-test showed its sign is set by threshold placement). What stands is a
dose–response between memorization strength and output sharpness (rho −0.70 to −0.82 in
all five runs) — and the preregistered L0P crossed control has now run against it: on 18
invented-lexeme paragraphs whose 392 probe windows all return zero in both training
corpora, a dose–response of similar strength appears (rho −0.71 to −0.77), and at matched
predictability the registered test found no memorization-specific sharpening (condition
coefficient positive — the opposite direction — in all five runs, combined p = .099).
Under the preregistration's frozen rules **none of the three registered decision rules
fired**: the paper claims neither memorization-specificity nor generic predictability, and
publishes the no-verdict outcome, plus a design error in the registration itself, as
results. Read the v0.7 top section first.

## Version history (one paper, four timestamped corrections in one day)

| Version | What happened |
|---|---|
| v0.5 (Zenodo v1) | The adversarially reviewed pilot: memorization strength vs KL-threshold "convergence depth" |
| v0.6 (Zenodo v2) | The paper's own instrument self-test reversed the depth headline the same afternoon; withdrawal on top, v0.5 preserved unrewritten |
| v0.7 | The preregistered L0P crossed control ran; none of its registered decision rules fired; published as-is, including the registration's own design error |
| v0.7.1 | Correction: the "corpus duplication counts" anchor is operationally a median probe-window frequency, dominated by short-phrase repetition, not a passage copy count ([`docs/CORRECTION_20260821_ANCHOR.md`](docs/CORRECTION_20260821_ANCHOR.md)); no registered verdict changes |

## Read the paper

| File | What it is |
|---|---|
| [`docs/WRITEUP_v0.7_EN.md`](docs/WRITEUP_v0.7_EN.md) | **The paper** (English, canonical): the preregistered L0P crossed control + the preserved v0.6/v1 record |
| [`docs/WRITEUP_v0.7_ZH.md`](docs/WRITEUP_v0.7_ZH.md) | Chinese version, content-equivalent |
| [`docs/WRITEUP_v0.6_EN.md`](docs/WRITEUP_v0.6_EN.md) | v0.6: the instrument self-test that reversed the depth headline |
| [`docs/WRITEUP_v0.6_ZH.md`](docs/WRITEUP_v0.6_ZH.md) | Independent Chinese version of v0.6 (not a translation) |
| [`docs/WRITEUP_v0.3_EN.md`](docs/WRITEUP_v0.3_EN.md) | The frozen version that passed adversarial review |
| [`docs/PREREG_L0P.md`](docs/PREREG_L0P.md) | The L0P preregistration — committed before any L0P measurement existed |

## Verify the numbers yourself

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

Raw measurement outputs: `results/raw/` (main pilot), `results/night/` (post-review
verification, including the adversarial memo `results/night/R1_verification.md` that
argues against parts of our own interpretation — shipped on purpose), and `results/l0p/`
(the preregistered crossed control).

## Repository map

- `harness/` — measurement runner, corpus-verification gate, analysis, figures, the
  reconciliation gate, and the script that assembles this bundle
- `battery/` — the 104-item task battery (v2.1) with per-corpus verification evidence
- `results/` — raw per-item outputs, analyses, figures
- `docs/` — papers, item-level appendix, per-entry citation verification log
  ([`docs/CITATIONS_VERIFIED.md`](docs/CITATIONS_VERIFIED.md)), 24 plain-language
  learning cards (`docs/learn/`, Chinese)
- [`PROCESS_LOG.md`](PROCESS_LOG.md) — the full process log the papers cite, including
  every mistake we made and fixed
- `RUNS.md` — exact per-model run configurations

## Licenses

Code: MIT ([LICENSE-CODE.md](LICENSE-CODE.md)). Produced data and results: CC BY 4.0;
the battery's verbatim excerpts are US public-domain Project Gutenberg text, sources
enumerated in [LICENSE-DATA.md](LICENSE-DATA.md).

## Author

Leo Gudu, independent researcher — guduwangho@gmail.com

Status: correlational; the original pilot was exploratory and not preregistered; the
v0.7 L0P crossed control was preregistered (public commit before any measurement) and
its frozen decision rules produced no verdict — reported as such. Not yet peer-reviewed
by humans. The adversarial review that gated each release was AI-executed and is fully
documented (see DISCLOSURE.md and PROCESS_LOG.md).
