# OMTR — corpus-verified memorization and layer-wise convergence depth

> **Disclosure first:** this research was conceived, directed, and decided by the author;
> literature review, engineering, experiments, review, verification and adversarial
> red-teaming, and the papers' text were executed by AI agents (Claude) under the author's
> direction. Full statement: [DISCLOSURE.md](DISCLOSURE.md).

A pilot study in two open-corpus model families (Pythia 410M–2.8B, OLMo-2-1B) asking:
within text a model has verifiably memorized, does memorization strength track how early
the layer-wise prediction settles? Ground truth for "memorized" comes from corpus
frequency queries (infini-gram) against the models' actual training corpora, with a
construction-matched non-memorized control.

## Read the paper

| File | What it is |
|---|---|
| [`docs/WRITEUP_v0.5_EN.md`](docs/WRITEUP_v0.5_EN.md) | **The paper** (English, canonical) |
| [`docs/WRITEUP_v0.5_ZH.md`](docs/WRITEUP_v0.5_ZH.md) | Independent Chinese version (not a translation) |
| [`docs/WRITEUP_v0.3_EN.md`](docs/WRITEUP_v0.3_EN.md) | The frozen version that passed adversarial review; v0.5 = v0.3 + editorial pass + four approved post-review caveats |

## Verify the numbers yourself

Every statistic in the papers can be recomputed from the raw per-item outputs shipped
here. The reconciliation gate does it mechanically:

```
python harness/reconcile.py docs/WRITEUP_v0.5_EN.md   # exit 0 = every number checks out
python harness/reconcile.py --selftest                # prove the gate itself works first
```

Raw measurement outputs: `results/raw/` (main pilot) and `results/night/` (post-review
verification, including the adversarial memo `results/night/R1_verification.md` that
argues against parts of our own interpretation — shipped on purpose).

## Repository map

- `harness/` — measurement runner, corpus-verification gate, analysis, figures, the
  reconciliation gate, and the script that assembles this bundle
- `battery/` — the 104-item task battery (v2.1) with per-corpus verification evidence
- `results/` — raw per-item outputs, analyses, figures
- `docs/` — papers, item-level appendix, per-entry citation verification log
  ([`docs/CITATIONS_VERIFIED.md`](docs/CITATIONS_VERIFIED.md)), 22 plain-language
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

Status: correlational, exploratory pilot; not preregistered; not yet peer-reviewed by
humans. The adversarial review that gated this release was AI-executed and is fully
documented (see DISCLOSURE.md and PROCESS_LOG.md).
