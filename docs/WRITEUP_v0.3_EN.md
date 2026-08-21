# Corpus-verified memorization tracks how early the layer-wise prediction settles — a pilot in two open-corpus model families (0.4–3B)

*Draft v0.3 (v2.1 matched controls; revised against the Phase-14 adversarial-review ruling). Not yet published.*

**TL;DR**

- **What we built.** A 104-item battery running from verbatim continuation of corpus-verified
  memorized text (L0) through open invention (L5), plus a construction-matched non-memorized
  control (L0N: same ≈88-word extraction, same content-word split rule, drawn from obscure
  Gutenberg books and verified near-absent from each training corpus — 124 of 131 probe windows
  return exactly zero). The full ladder ran on two models, **Pythia-1.4B and OLMo-2-1B**, whose
  training corpora are publicly searchable via infini-gram. Three further Pythia sizes ran the
  bare-continuation family only (L0 / L0N / L1, 53 items each). Every five-run number below lives
  in that family.

- **The finding.** Within corpus-verified memorized text, better-memorized passages settle earlier
  in depth. Spearman of teacher-forced gold log-probability against KL-threshold depth (τ = 0.1),
  within L0: **−0.53, −0.34, −0.69, −0.42, −0.86** (410M / 1B / 1.4B / 2.8B / OLMo-2-1B).
  Negative in all five runs. Individually significant in three. Surviving Benjamini–Hochberg
  across the 15-cell τ = 0.1 family in two: **OLMo-2-1B** (n = 20, p = 1.2e-6, q < .0001) and
  **Pythia-1.4B** (n = 17, p = .002, q = .015). Those two carry the claim. OLMo's coefficient is
  not carried by the three items only it was eligible for: on the 17 items shared with the Pythia
  runs it is **−0.828 (p = 4.1e−5)**.

- **Why we do not call this five replications.** The four Pythia sizes share one 17-item set and
  one training corpus, and per-item memory scores correlate across models at rho 0.76–0.99. Under
  an item-level permutation null, P(all five negative) is **.10–.17**, not the .03 that five
  independent runs would imply. What we have is **two independent corpora, two tokenizers, two
  architectures** — and a scale series within one of them. That is the honest unit, and it is
  still the part we would bet on: the two families agree.

- **Robustness we can claim.** Leave-one-out is tight — no single item moves any of the five
  correlations by more than 0.25 (worst single-deletion shift 0.24, in the weakest Pythia run), and OLMo's −0.86 spans [−0.894, −0.846] across all 20 deletions.
  The gradient is not carried by a high-leverage point.

- **The comparison conditions are flat as a pattern, not as a tested difference.** Matched control:
  −0.05, −0.05, −0.16, +0.35, +0.07 (all n.s.). Single-token fact cloze: −0.14, −0.04, +0.01,
  −0.17, +0.19 (all n.s.). We tested whether these *differ* from the L0 correlation (Fisher z).
  At this n they do not, in four of five models; only OLMo-2-1B separates from both
  (p = .0014 and p < .0001). So we report a consistent ordering of point estimates, not an
  established dissociation. Two mechanical caveats we could not remove at pilot n: the depth
  metric is capped at (L−1)/L by construction and **86–98% of control and cloze positions sit at
  that cap**, leaving those cells 2–5 effective levels against 9–13 in L0; and the cloze null is a
  statement about instrument resolution, not about fact recall: the answer is one token, depth
  averages eight positions, and at the answer position alone the metric is constant in two of five
  models. On single-token answers this instrument does not resolve a gradient in either direction.
  We treat the memorized-text gradient as the result and both nulls as unresolved.

- **External anchor.** Corpus duplication counts from infini-gram — a quantity that exists
  independently of the model — correlate with the gold-log-prob memory scale in all five runs
  (rho **0.64–0.73**, all **p < .006**). Counts also correlate directly with depth in OLMo-2-1B
  (−0.73, p = .0002); the four Pythia runs share the sign but none reach significance
  (−0.16 to −0.36). So the memory scale is anchored to something outside the model's own logits,
  while the depth link is individually resolvable in one run only.

- **Uncertainty ladder.** Mean final-layer entropy, averaged over the eight answer positions,
  orders conditions retrieval-low to invention-high in both families: L0 1.74 / 1.11 nats rising to
  L5 4.25 / 4.51 (Pythia-1.4B / OLMo-2-1B). The matched control lands near the generative end
  (3.58 / 3.32) under the same construction rule — entropy rank-biserial L0 vs L0N −0.63 / −0.85.
  **Two limits on how far the ladder goes.** First, prompt length spans 8× across the ladder
  (53–408 tokens) and predicts entropy *within* the generative family too (rho +0.382, p = .0074 /
  +0.307, p = .034), so "load-bearing contrasts are within-family" does not fully cover the
  ordering. Second, L0 and L0N are matched in words (rank-biserial +0.000, p = 1.0), but the
  control fragments into more subword tokens per word (rank-biserial **−0.775 / −0.818**,
  p = .0005 / .0002) — a lexical-rarity difference, not a length difference. That qualifies the
  within-L0 gradient; it does **not** affect the entropy or depth contrasts, which are unchanged.
  Dropping the longest control item barely moves the contrast (−0.627 → −0.615).

- **A result we want read as negative.** Partialling final entropy out of the L0 gradient leaves
  OLMo-2-1B significant (−0.65, p = .002) but not the Pythia arm, where the four partials scatter
  around zero and **two reverse sign** (+0.10, +0.19, −0.13, −0.18). Prompt length behaves the same
  way; only OLMo survives length and entropy together. In four of five runs we cannot separate this
  gradient from output certainty at this n. Whether that is mediation or confounding is exactly
  what the causal phase is for.

- **Status.** Correlational and exploratory. No preregistration. The within-condition contrast, the
  τ = 0.1 metric and the comparison framing were all chosen after inspecting a pooled analysis;
  the sequence is in the process log. p-values are descriptive except where a q-value is given.
  The causal phase (activation patching between matched L0/L0N pairs; a steerable familiarity
  direction) is designed, and its prereg will be posted with a public timestamp before it runs.
  **Kill condition: if patching moves entropy but not provenance-measured behavior, the depth
  signal is epiphenomenal and we will say so.**

## Why this question

Human cognitive science treats remembering and imagining as one constructive machinery: hippocampal amnesia impairs imagining novel scenes (Hassabis et al. 2007 — though the patient evidence is contested; Squire et al. 2010), and current computational models cast recall as generative reconstruction (Spens & Burgess 2024). For LLMs, the analogous question — is there an internal variable that tracks where an output sits on the memorized↔novel axis? — has been approached behaviorally (Sui et al. 2024; Lee et al. 2022), mechanistically for verbatim memorization circuits (Lasy et al. 2025), and for recall-vs-reasoning dissociations (Fartale et al. 2025); we found no work that puts parametric retrieval and open generation on one graded scale with corpus-verified ground truth and asks what tracks the gradient inside the network. The trap is that in an autoregressive model everything is generation, so "shared machinery?" is trivially yes; the productive version is quantitative, and it requires ground truth for "memorized" — hence models whose training corpora are publicly indexed and searchable at scale (Pythia/The Pile; OLMo-2/olmo-mix, via infini-gram; Biderman et al. 2023; Gao et al. 2020; OLMo Team 2025; Liu et al. 2024). That indexing is complete for the Pythia arm and incomplete for OLMo-2-1B, which we state precisely in Limitations rather than claim away.

## Setup

**Battery (104 items, v2.1).** L0 (n = 20; 17 eligible against the Pile index, 20 against olmo-mix): continuations of famous public-domain passages, wording verbatim from Project Gutenberg. Memorization is verified per-corpus by a probe set of eleven windows — ten 7-*word* windows taken at stride 4 over the ≈44-word continuation, punctuation-trimmed at the edges, plus one window spanning the prompt/continuation boundary — and a passage is eligible for a given model when ≥70% of those eleven windows appear ≥20 times in that model's training corpus (queried against the Llama-tokenizer infini-gram indices `v4_piletrain_llama` and `v4_olmo-mix-1124_llama`). L0N (n = 12 Pile / 11 olmo-mix): the same 88–91-word extraction and the same content-word split rule applied to obscure Gutenberg books, verified near-absent — 124 of 131 probe windows return exactly zero and the maximum count anywhere is 2.

The control condition was rebuilt for v2.1, and the rebuild was larger than "adding a boundary rule." Six of eleven control passages changed: five are identical to v2.0, three are boundary-shifted, three are a new span from the same book, and one is a new book; only **4 of 11** prompts are byte-identical to v2.0, and the content-word boundary rule explains only three of the six changes. Item IDs were reassigned in the rebuild — the text that was `L0N-08` is now `L0N-09` — so any citation of an L0N item ID is version-ambiguous and must name the battery version. The superseded control correlation was **−0.640 (p = .034, n = 11)**; it is now −0.16 in the same model. On ordering: the asymmetry was caught by our own fact-check pass over our learning materials, and the control was rebuilt **before the adversarial review of the results began** (the rebuild script's timestamp precedes the review artifacts by 1h41m). We make no claim about what we had or had not looked at, because no timestamp can support one.

L1: 24 cloze facts. **Nineteen of the 24 answers are a single token on both tokenizers; the five exceptions are listed in Appendix A.7** (Pythia splits " Zeus" and " Poseidon"; OLMo splits " Poseidon", " 100" and " 64"; " thermometer" splits on Pythia). All 24 L1 items also carry a four-sentence instructional preamble (`build_battery_v2.py`'s `L1_PREAMBLE`, prepended at run time) that **no L0 or L0N item carries**. The preamble was added to token-match L1 to L0 (median 50 vs 54 prompt tokens) and it is a sensible choice, but it means the "bare continuation family" contains a prompt-format split, and the split falls exactly between the condition with the gradient and one of the two comparison conditions. L2–L5 (paraphrase, constrained form, forced blends, invention) use four fixed few-shot scaffolds and are reported as a separate prompt family (length/format confounds acknowledged and quantified in Results §1).

**Measures.** No probe is trained anywhere in this pipeline: every quantity below is read directly off the model's own weights, so nothing here can be an artifact of a classifier fitted to the conditions it is meant to distinguish. Per item: greedy-decode 8 answer tokens, one cached forward pass, and at each answer position a full-precision logit-lens profile per layer (entropy, KL-to-final, top-1 agreement, margin; the logit lens is nostalgebraist's 2020 heuristic, and we inherit its known brittleness — see Limitations). Depth: normalized KL-threshold depth (τ ∈ {0.05, 0.1, 0.5}) and KL-curve area. The threshold metric is a **last-crossing** depth — the scan runs from the final layer backwards and returns the deepest layer at which KL-to-final still exceeds τ — divided by `n_layers`. It therefore has a structural ceiling of (L−1)/L: **0.9583** for the 24-layer models (410M, 1.4B), **0.9375** for the 16-layer models (1B, OLMo-2-1B), **0.9688** for 32-layer 2.8B. Across conditions, between 60% and 80% of answer positions sit exactly at that ceiling (86–98% in the control and cloze cells specifically). Memory strength: mean teacher-forced log-prob of the gold span. Note that the two principal variables are measured over **two different spans** — memory strength over the ≈44-word gold continuation, depth over the 8 greedy tokens the model actually produced — which we state here because the draft's earlier versions did not. Tokenization pitfalls (numeric splitting on OLMo, proper-noun shattering on Pythia) were audited and are recorded in the battery file itself; the boundary guard for teacher-forced scoring fired 0 times across **267** model × gold-item pairs (53/53/53/53/55).

**Construct validity of the dependent variable.** We renamed the construct in this revision. What τ-threshold depth measures is **layer-wise convergence**, not "commitment": to 70–97% rank agreement it is a saturation counter — the fraction of answer positions that remain unconverged through the final layer. The correlation between `depth_tau_0.1` and the plain count of at-ceiling positions (`frac_at_cap`) is +0.703 / +0.784 / +0.909 / +0.922 / +0.966 across the five runs, and `frac_at_cap` is itself a *better-behaved* estimator of the memory relationship than the metric we report: rho(gold, `frac_at_cap`) is −0.658 / −0.630 / −0.649 / −0.710 / −0.893, significant at p < .007 in **5/5**, against 3/5 for `depth_tau_0.1`. We deliberately did **not** promote it to primary DV. Two reasons. First, it changes nothing that matters: partialling final entropy leaves exactly one model significant under either operationalization (+0.015 / +0.086 / +0.103 / −0.213 / **−0.639**, OLMo p = .0024 — the same 1/5 that `depth_tau_0.1` gives), so the swap would buy apparent strength and zero robustness. Second, swapping a primary DV after seeing that it scores better is a forking path, and we had already chosen this one. `frac_at_cap` therefore ships as an appendix sensitivity analysis (A.6) rather than as the headline. The off-cap dynamic range is small and worth knowing: in OLMo-2-1B the unconverged positions occupy layers 12–15 of 16 and take three unique values; in Pythia-1.4B 80% of positions sit at layer 23 with a median escape layer of 22.

**Verification discipline.** The harness passed an adversarial review with independent reproduction of every accepted finding (16 defects fixed pre-run, including a TransformerLens/transformers-5 incompatibility that would have killed the Pythia arm). That review, like the later panel whose ruling this revision implements, was executed by AI agents under our direction; the scale and the division of labor are enumerated in `DISCLOSURE.md` and the process log, and we do not describe it as "our own red-team" as though it were unaided. A separate fact-check pass over our own learning materials caught the control-construction mismatch described above. The raw per-item outputs behind every number in this document are in `results/raw/`.

## Results

**1. Uncertainty ladder.** Mean final-layer entropy per condition, averaged over the eight answer positions (flagships): L0 1.74/1.11 < L1 2.61/2.83 ≈ L2 2.78/2.36 < L0N 3.58/3.32 ≈ L3 3.58/3.60 < L4 4.13/4.04 < L5 4.25/4.51 (Pythia-1.4B / OLMo-2-1B). Ladder ends identical across families; L1/L2 swap mid-ladder. **The ordering is confounded with prompt length and we do not claim otherwise.** Prompt length spans 8× across the ladder (53–408 tokens) and correlates with entropy at rho +0.535 / +0.489 over all items; restricting to the generative family L2–L5, where the "within-family contrast" defense would apply, it still correlates at **+0.382 (p = .0074) / +0.307 (p = .034)**. The ladder is a description of our conditions, not a demonstration that condition type rather than prompt length produces the ordering. The one place length is controlled is the L0-vs-L0N contrast, which is matched in words.

**2. Matched memorization contrast (L0 vs L0N).** Rank-biserial: final-layer entropy −0.63 / −0.85; KL-threshold depth (τ = 0.1) −0.37 / −0.69. Honest flags, in both directions:

- The depth effect is threshold-sensitive and small in absolute terms — the L0-vs-L0N depth difference is ≈0.33 layer of 24 in Pythia-1.4B and ≈0.38 of 16 in OLMo-2-1B, and 0.3–1.1 layers across the five models.
- The comparison conditions are the *more* range-restricted cells, not the less: the depth metric has 2–5 distinct values in L0N and the cloze against 9–13 in L0 (standard-deviation ratios 1.88–7.84×), and 86–98% of control and cloze positions sit at the structural ceiling. Whatever "flat" means in those cells, it is measured on a coarser instrument than the one that produced the L0 gradient.
- The KL-area metric reverses sign. The likely mechanism — a sharper final distribution inflates KL at every layer — **would also push the τ-threshold metric later, i.e. toward a positive correlation**. We observe a negative one. So the τ result runs *against* the direction of this artifact rather than with it, which is a point in its favor; we have not quantified the size of the artifact and flag that as owed work. The full five-variant × five-model sign grid is in Appendix A.1.
- Representational separability between conditions peaks mid-stack at 0.44–0.59 and never exceeds 1.0. The relevant baseline, however, is not 1.0: the permutation null is **0.390–0.394**, and the observed values exceed the per-layer 95th percentile of that null at **24 of 24 layers** (Pythia-1.4B) and **16 of 16** (OLMo-2-1B). Separability is above chance everywhere; what we do not claim is that it is *large*, or that a linear decoder trained at this n would generalize.

**3. The core result — a memorization gradient and two flat comparison conditions.**

> Within corpus-verified memorized passages, stronger per-item memorization is associated with
> earlier layer-wise convergence of the logit-lens distribution — concretely, with fewer answer
> positions remaining unconverged through the final layer (Spearman −0.53, −0.34, −0.69, −0.42,
> −0.86 across five runs from two corpus families over a single shared 17-item core, three items
> added for OLMo-2-1B; three of five reach p < .05, and only OLMo-2-1B both survives partialling
> final-distribution entropy and subword fragmentation and differs significantly from the matched
> control and cloze conditions). The association spans roughly one to three layers and is
> concentrated adjacent to the metric's structural ceiling.

Spearman of memory strength vs. layer-wise convergence (τ = 0.1), within condition:

| model | L0 memorized | L0N matched control | L1 fact cloze |
|---|---|---|---|
| Pythia-410M | **−0.53** (p=.029) | −0.05 (n.s.) | −0.14 (n.s.) |
| Pythia-1B | −0.34 (p=.19) | −0.05 (n.s.) | −0.04 (n.s.) |
| Pythia-1.4B | **−0.69** (p=.002) | −0.16 (n.s.) | +0.01 (n.s.) |
| Pythia-2.8B | −0.42 (p=.094) | +0.35 (n.s.) | −0.17 (n.s.) |
| OLMo-2-1B | **−0.86** (p<.0001) | +0.07 (n.s.) | +0.19 (n.s.) |

The five columns are not five independent replications. The four Pythia runs execute a byte-identical 17-item L0 set; OLMo-2-1B runs those same 17 plus `L0-13`, `L0-14` and `L0-18`. Restricted to the shared 17, OLMo is **−0.828 (p = 4.1e−5)**, so the coefficient does not depend on the three extra items. Under an item-level permutation null that respects the shared battery, family-wise P(all five negative) is .10–.17, not the .031 that five independent runs would give.

Within memorized passages — same task, same construction — the better the passage is known, the earlier the layer-wise prediction converges. The same measure on non-memorized text of identical construction shows no resolvable gradient, and neither does single-token fact recall; per the caveats in §2 and the TL;DR, both nulls are measured on cells with 2–5 effective levels, and the cloze null in particular is a statement about instrument resolution rather than about fact recall.

Two of five runs do not reach significance and **we have no satisfying explanation**. We note but do not lean on the fact that 2.8B has the lowest L0 gold-log-prob SD of the five (0.78 vs 1.36 at 410M; ddof=1 throughout): across models, gold SD does not track |rho| (Spearman −0.20), and Spearman is invariant to monotone compression in any case. Range restriction does, however, apply to the **comparison** conditions, where the depth metric has 2–5 effective levels against 9–13 in L0 — we report that above rather than only where it would excuse a weak cell.

**4. Robustness probes (run because the adversarial review demanded them).**

(a) *External dose anchor*: median corpus window count per passage correlates with gold log-prob in all five runs (rho **0.64–0.73**, all **p < .006**) — the memory scale is anchored to a model-independent quantity — and directly with depth in OLMo-2-1B (−0.73, p = .0002); **the four Pythia runs show the same sign but none reach significance (−0.16 to −0.36, all n.s.)**.

(b) *Entropy mediation*: partialling final entropy out of the L0 dose-response leaves OLMo-2 significant (−0.65, p = .002) but not the Pythia arm, where the four partials scatter around zero and **two reverse sign** (+0.10, +0.19, −0.13, −0.18; all n.s.). This is consistent with sharpness acting as a mediator on the path (memory → sharper final distribution → earlier convergence), but pilot data cannot separate mediation from confounding, and in four of five runs it cannot separate the gradient from output certainty at all. The causal phase is designed to.

*(The "shared-logits circularity" probe reported in v0.2 has been withdrawn. Recomputing depth from answer positions 2–8 is an exact-arithmetic no-op in two models — full and pos1+ agree to six decimal places in 410M and 1B — and Pythia-1.4B's position-0 depth is a single constant, so the test could not have failed. It was not a control and we do not present it as one.)*

**5. Side-findings.** On OLMo-2-1B, greedy decoding reused corpus phrasing more than a single temperature-0.8 sample on 11 of 12 invention items (Wilcoxon p = .001); the same comparison on Pythia-1.4B was not significant (p = .21). Two aggravators belong in the same breath: n = 1 sample per item, and that draw is unreproducible from the shipped harness because the sampling predates per-item seeding (released code seeds samples per item; these numbers predate it). Separately, blind-calibrated LLM judging shows OLMo-2-1B genuinely attempts distant blends ~50% of the time vs. ~8% for Pythia-1.4B, so creative-condition internal claims are gated accordingly and are not part of our core claims. **That gate has a hole we should name: the judge scored L2, L3 and L4 only — 72 scores, zero L5.** L5 supplies the top rung of the entropy ladder and the "open invention" end that gives the framing its name, and we have no behavioral-validity data on it at all.

**Statistical framing.** Exploratory pilot: no preregistration, no multiplicity correction except where a q-value is given; p-values are otherwise descriptive. The primary claim is the OLMo-2-1B result together with a five-run point-estimate ordering over two corpus families. **The primary contrast and the metric were chosen after seeing results** — specifically, the decision to analyze within-condition rather than pooled, and the choice of τ = 0.1 among three thresholds, were both made after inspecting a pooled analysis; `PROCESS_LOG.md` records the sequence and the dates. We say this here rather than only in the log, because "exploratory, no preregistration" does not by itself tell a reader which of the forks were taken after looking. The L0 memorization gate sits on a robustness plateau (Pile eligibility 15–18/20 across window-fraction 0.6–0.7 × count 10–40).

## What this does and does not show

Supports: a continuously graded internal correlate of parametric memory strength that tracks where in depth the layer-wise prediction converges — established in OLMo-2-1B, consistent in sign in a four-model Pythia scale series that shares one item set, absent as a resolvable gradient on matched novel text and on atomic facts, and externally anchored to corpus duplication counts in all five runs.

Does not show: causation; a tested dissociation in four of five models; creativity mechanisms (generative-end conditions carry length and format confounds, and L5 has no behavioral-validity data at all); layer-local "memory neurons"; generalization beyond 0.4–3B base models. The Pythia entropy-partial result explicitly limits how much of the gradient is separable from final-distribution sharpness in that family.

Two instrument limitations we cannot discharge at pilot scale. First, **the corpus indexing behind OLMo-2-1B is incomplete.** OLMo-2-0425-1B is trained in two stages, and Stage 2 mid-training (Dolmino-Mix-1124, 50B tokens) is covered by no infini-gram index and is not a subset of Stage 1; "full training corpora publicly indexed" is true for the Pythia arm and false here. The mitigation is real but partial: Stage 1 (olmo-mix-1124) is 95%+ of the total pretraining budget and is indexed, and our L0N controls return zero counts against it. It matters more after this revision than before, because OLMo-2-1B is now the single model carrying the surviving claim, which makes the one incomplete corpus verification and the one significant result the same model. See the OLMo-2-0425-1B model card, which is the authoritative source for the two-stage data (the OLMo 2 paper covers 7B/13B/32B and not this checkpoint). Second, **we ran no tuned-lens agreement check.** The logit lens is a heuristic and is documented as brittle relative to the tuned lens (Belrose et al. 2023); every depth number here inherits that.

## Next: the causal phase

Activation patching between matched L0/L0N pairs; search for a low-dimensional familiarity direction whose amplification/suppression moves outputs along derivative↔novel, quantified by corpus-provenance metrics. Kill condition: if patching moves entropy but not provenance-measured behavior, the depth signal is epiphenomenal and we will say so. The prereg will be posted with a public timestamp (OSF or repo commit) before any causal run; this pilot makes no preregistered claims.

## Appendix A — the full metric grid and the negative results

### A.1 Five depth variants × five runs, within-L0 dose–response

Spearman rho (p) of gold log-prob against each depth variant. The reported headline metric is `tau_0.1`.

| model | `kl_auc_norm` | `tau_0.05` | **`tau_0.1` (reported)** | `tau_0.5` | `argmax` |
|---|---|---|---|---|---|
| Pythia-410M | **+0.713** (.0013) | −0.477 (.053) | **−0.530** (.029) | −0.313 (.221) | −0.380 (.132) |
| Pythia-1B | **+0.797** (.0001) | −0.315 (.218) | −0.337 (.186) | −0.231 (.372) | +0.114 (.663) |
| Pythia-1.4B | **+0.770** (.0003) | −0.773 (.0003) | **−0.693** (.002) | −0.339 (.184) | −0.130 (.619) |
| Pythia-2.8B | **+0.571** (.017) | −0.353 (.165) | −0.419 (.094) | −0.309 (.227) | −0.550 (.022) |
| OLMo-2-1B | **+0.511** (.021) | −0.862 (<.0001) | **−0.860** (<.0001) | −0.781 (.0001) | −0.374 (.104) |

Significant cells by variant: `tau_0.05` 2/5; `tau_0.1` 3/5; `tau_0.5` 1/5; `argmax` 1/5 (and positive in one model); **`kl_auc_norm` 5/5 — all in the opposite direction.** We report this because it is the one variant that is unanimous and significant everywhere, and it points the other way. v0.2 disclosed the KL-area reversal only for the L0-vs-L0N contrast, not for the dose–response, which is the core result; that omission is corrected here. The reversal is mechanically expected (see Results §2): a sharper final distribution inflates KL at every layer, which raises the area *and* delays the last τ-crossing. The τ metric moving negatively is therefore evidence against, not for, the sharpness artifact — but we have not quantified the artifact's magnitude, and the reader is entitled to see all five rulers before deciding.

### A.2 Confidence intervals and leave-one-out, L0 dose–response

10 000 bootstrap resamples. Independent re-runs by two reviewers agreed to within 0.02 on every bound.

| model | rho | bootstrap 95% CI | leave-one-out range | P(rho > 0) |
|---|---|---|---|---|
| Pythia-410M | −0.530 | [−0.879, **−0.031**] | [−0.706, −0.469] | .019 |
| Pythia-1B | −0.337 | [−0.802, **+0.227**] | [−0.577, −0.255] | .120 |
| Pythia-1.4B | −0.693 | [−0.861, −0.356] | [−0.760, −0.657] | .001 |
| Pythia-2.8B | −0.419 | [−0.771, **+0.121**] | [−0.552, −0.316] | .056 |
| OLMo-2-1B | −0.860 | [−0.940, −0.682] | [−0.894, −0.846] | .000 |

**Two of the five CIs cross zero** (1B and 2.8B), and a third has an upper bound of −0.031. "Negative in all five runs" is a statement about point estimates that 12% of Pythia-1B resamples contradict. Against that, leave-one-out is comparatively tight: no single item moves any rho by more than 0.25 (the largest single-deletion shift, in the weakest Pythia run, is 0.24), and OLMo's spans only [−0.894, −0.846] across all 20 deletions, so no high-leverage item manufactures the largest coefficient.

### A.3 Corpus count vs. depth — the four cells v0.2 did not report

Median corpus window count against `depth_tau_0.1`, within L0: OLMo-2-1B −0.73 (p = .0002); Pythia-410M/1B/1.4B/2.8B **−0.296 / −0.276 / −0.362 / −0.159, all n.s.** Same sign as OLMo in all four, individually resolvable in none.

### A.4 Representational separability — the permutation null

Observed between-condition separability peaks mid-stack at 0.44–0.59, which is below 1.0 and reads as weak in isolation. The permutation null, computed by shuffling condition labels, is **0.390–0.394**. Observed values exceed the per-layer 95th percentile of the null at **24/24 layers** in Pythia-1.4B and **16/16** in OLMo-2-1B. This is a pass, not a concession: the conditions are linearly distinguishable above chance at every depth. What it does not license is a claim that the separation is large or that a decoder trained at this n would transfer.

### A.5 Threshold sensitivity of the control null

The claim "the control is flat" is exact only at τ = 0.1. At other thresholds individual control cells reach significance in the *opposite* direction or in the same one: Pythia-1B L0N `depth_tau_0.5` = −0.809 (p = .0014) and `depth_argmax` = −0.713 (p = .0092); Pythia-2.8B L0N `tau_0.05` = +0.638 (p = .0255). We report the τ = 0.1 column because that is the metric we chose (post hoc — see Statistical framing), not because the control is flat under every ruler.

### A.6 `frac_at_cap` sensitivity analysis

`frac_at_cap` is the fraction of answer positions still unconverged at the final layer — the saturation counter that `depth_tau_0.1` is 70–97% rank-equivalent to.

| model | rho(gold, `frac_at_cap`) | p | rho(depth, `frac_at_cap`) | rho(gold, `frac_at_cap`) partialling entropy |
|---|---|---|---|---|
| Pythia-410M | −0.658 | .0041 | +0.703 | +0.015 |
| Pythia-1B | −0.630 | .0067 | +0.784 | +0.086 |
| Pythia-1.4B | −0.649 | .0048 | +0.909 | +0.103 |
| Pythia-2.8B | −0.710 | .0014 | +0.922 | −0.213 |
| OLMo-2-1B | −0.893 | <.0001 | +0.966 | **−0.639** (p = .0024) |

The raw estimator is cleaner than ours — 5/5 significant against 3/5 — and **the controlled result is identical**: exactly one model survives partialling final entropy, either way. That is the whole reason we did not switch: the swap would have moved the headline from "3/5 significant" to "5/5 significant" without moving a single controlled cell, and it would have degraded both comparison conditions further (`frac_at_cap` takes 3/2/3/2/2 distinct values in L0N and 3/4/3/5/3 in the cloze, against 5/3/5/5/2 and 7/7/7/6/5 for depth).

### A.7 The five L1 items whose answer is not a single token on both tokenizers

`L1-Greek_mythology-1` (" Zeus": 2 Pythia tokens, 1 OLMo), `L1-Greek_mythology-2` (" Poseidon": 4 Pythia, 2 OLMo), `L1-weather-2` (" thermometer": 2 Pythia), `L1-cooking-1` (" 100": 2 OLMo), `L1-chess-1` (" 64": 2 OLMo). All five are recorded in `battery/battery.json`'s own `qc_problem` field, two of them flagged HIGH by the tokenization audit; they are one mechanism behind the window-dependence of the cloze null.

## References

Every entry below was opened and verified; `docs/CITATIONS_VERIFIED.md` records the URL and the date checked for each.

**Instruments.**

nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong, 31 August 2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., & Steinhardt, J. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112. https://arxiv.org/abs/2303.08112

Liu, J., Min, S., Zettlemoyer, L., Choi, Y., & Hajishirzi, H. (2024). *Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens.* COLM 2024. arXiv:2401.17377. https://arxiv.org/abs/2401.17377 — indices used here: `v4_piletrain_llama` (Pile-train, 383.3B Llama-2 tokens) and `v4_olmo-mix-1124_llama` (OLMo-mix-1124, 4.58T).

**The prediction-resolves-early lineage.**

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022). *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space.* EMNLP 2022, 30–45. DOI 10.18653/v1/2022.emnlp-main.3. https://aclanthology.org/2022.emnlp-main.3/ — Geva et al. introduce saturation descriptively in §5.1 ("the final token predicted by the model … was promoted to be the top candidate until the last layer"), which is exactly the construct of our `depth_argmax` variant. We should say what follows from that rather than wait to be asked: it makes the saturation-style metric the field-standard operationalization of "the prediction has resolved," and `depth_argmax` is our **weakest** variant — significant in 1 of 5 runs (Appendix A.1). Our reported τ-threshold metric is a graded relative of it, and the honest reading is that the field-standard binary version does not resolve our effect at this n.

Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V., Tay, Y., & Metzler, D. (2022). *Confident Adaptive Language Modeling.* NeurIPS 2022. arXiv:2207.07061. https://arxiv.org/abs/2207.07061 — CALM's founding premise is that generations vary in difficulty and easier continuations resolve with less compute, which is the strongest existing evidence that this quantity is real, and simultaneously the reason it is not a new construct.

Haviv, A., Cohen, I., Gidron, J., Schuster, R., Goldberg, Y., & Geva, M. (2023). *Understanding Transformer Memorization Recall Through Idioms.* EACL 2023. arXiv:2210.03588. https://aclanthology.org/2023.eacl-main.19/ — the closest precursor: the same instrument on a binary memorized-vs-non-memorized idiom contrast, finding two-phase recall. Three differences: binary vs continuous, curated idiom lists vs corpus-count-verified passages, and no open-generation end of the scale. Their §5.2 mechanism — non-memorized predictions are themselves often promoted early via shallow local patterns — is a live alternative explanation for early convergence on prose items, and one our L0N entropy data speaks to.

**Prior work the framing describes.**

Fartale, H., Kattamuri, A., Raja, R., Vats, A., Prasad, I., & Moharir, A. K. (2025). *Disentangling Recall and Reasoning in Transformer Models through Layer-wise Attention and Activation Analysis.* arXiv:2510.03366. https://arxiv.org/abs/2510.03366

Lasy, I., Knees, P., & Woltran, S. (2025). *Understanding Verbatim Memorization in LLMs Through Circuit Discovery.* arXiv:2506.21588. https://arxiv.org/abs/2506.21588

Sui, P., Duede, E., Wu, S., & So, R. J. (2024). *Confabulation: The Surprising Value of Large Language Model Hallucinations.* ACL 2024, 14274–14284. arXiv:2406.04175. https://aclanthology.org/2024.acl-long.770/

Lee, N., Ping, W., Xu, P., Patwary, M., Fung, P., Shoeybi, M., & Catanzaro, B. (2022). *Factuality Enhanced Language Models for Open-Ended Text Generation.* NeurIPS 2022. arXiv:2206.04624. https://arxiv.org/abs/2206.04624 — documents that sampling harms factuality through "uniform randomness"; our §5 decoding side-finding is a measurement of the same decoding→provenance relationship.

**Models and corpora.**

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* arXiv:2304.01373. https://arxiv.org/abs/2304.01373

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., & Leahy, C. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling.* arXiv:2101.00027. https://arxiv.org/abs/2101.00027 — the Pile's 22 subsets include Gutenberg (PG-19), and our L0N controls are Gutenberg texts; 124 of 131 control windows return exactly zero Pile occurrences, which is what rules out the obvious objection.

OLMo Team / Allen Institute for AI (2025). *2 OLMo 2 Furious.* arXiv:2501.00656. https://arxiv.org/abs/2501.00656 — covers the 7B, 13B and 32B models. **OLMo-2-0425-1B is not among them**; for the checkpoint used here, cite the model card, https://huggingface.co/allenai/OLMo-2-0425-1B, which is the authoritative source for the two-stage data (Stage 1 olmo-mix-1124, ~3.9T tokens, "95%+ of total pretraining budget"; Stage 2 Dolmino-Mix-1124, 50B tokens, covered by no infini-gram index). That card carries the Limitations sentence above.

**Cognitive-science framing.**

Hassabis, D., Kumaran, D., Vann, S. D., & Maguire, E. A. (2007). *Patients with hippocampal amnesia cannot imagine new experiences.* PNAS 104(5), 1726–1731. DOI 10.1073/pnas.0610561104. https://pmc.ncbi.nlm.nih.gov/articles/PMC1773058/

Spens, E., & Burgess, N. (2024). *A generative model of memory construction and consolidation.* Nature Human Behaviour 8, 526–543. DOI 10.1038/s41562-023-01799-z. https://www.nature.com/articles/s41562-023-01799-z

Squire, L. R., van der Horst, A. S., McDuff, S. G. R., Frascino, J. C., Hopkins, R. O., & Mauldin, K. N. (2010). *Role of the hippocampus in remembering the past and imagining the future.* PNAS 107(44), 19044–19048. DOI 10.1073/pnas.1014391107 — the patient evidence is contested: Hassabis et al. tested five patients and report that one was apparently unimpaired, and Squire et al. found hippocampal patients who *can* imagine future experiences.

**Recommended, optional.** Biderman, S., Prashanth, U. S. S., Sutawika, L., Schoelkopf, H., Anthony, Q., Purohit, S., & Raff, E. (2023). *Emergent and Predictable Memorization in Large Language Models.* arXiv:2304.11158. https://arxiv.org/abs/2304.11158 — memorization across the Pythia suite; the nearest precedent for corpus-grounded memorization measurement in these exact models.

## Disclosure and provenance

This research was conceived, directed, and decided by the author; literature review, engineering, experiments, **review, verification and adversarial red-teaming**, and this text were executed by AI agents (Claude) under the author's direction. That includes the verification trail this paper leans on: the "16 confirmed defects" fixed pre-run, and the multi-reviewer adversarial panel whose ruling produced this revision, were AI-executed, at the scale enumerated in the process log. Every number was regenerated from raw outputs, and the mistakes we made and fixed are in the public process log. AI-generated text may carry provider watermarks; we disclose AI involvement regardless. Full statement: `DISCLOSURE.md`. Code, battery, raw results, process log: [repo link].

*Author: [name/handle], independent researcher. Correspondence: [contact].*

*Archival note (2026-08-21): v0.3 is the frozen draft that passed the adversarial review; v0.5 is the published successor. The author/contact/repo placeholders above were filled only in v0.5, and the per-fix application map of the review's 29-item ruling lives in the project's internal review records. This file is kept verbatim for provenance.*
