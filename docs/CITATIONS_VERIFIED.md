# Citation verification log

**Provenance.** The reference list in `WRITEUP_v0.5_EN.md` was first verified entry-by-entry
on **2026-08-20** by the adversarial review panel's citation checker (its per-entry verification tables are preserved in the project's internal review records).
This file records a second, independent, full re-verification performed fresh on **2026-08-21** —
every URL below was fetched again on that date, not copied from the prior pass. Where the two
passes could disagree (a link going dead, a title changing), this file reflects only what was
observed on 2026-08-21.

Every reference entry in the writeup was checked. This includes the two URLs embedded inside the
OLMo Team (2025) entry — the arXiv paper and the OLMo-2-0425-1B model card — which are listed as
separate rows below because they are separate claims.

18 distinct reference entries, 19 URLs checked (one entry carries two URLs).

## Verification table

| Entry | URL | Status | Date checked | Note |
|---|---|---|---|---|
| nostalgebraist (2020), *interpreting GPT: the logit lens* | https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens | OK | 2026-08-21 | Title "interpreting GPT: the logit lens", author nostalgebraist, posted 31 Aug 2020 — all match the entry exactly. |
| Belrose et al. 2023 (Tuned Lens) | https://arxiv.org/abs/2303.08112 | OK | 2026-08-21 | Title and all 8 authors (Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, Steinhardt) match. Submitted 2023-03-14, later revised (v6, 2025-11-11) — same paper, no title/author drift. |
| Liu et al. 2024 (Infini-gram) | https://arxiv.org/abs/2401.17377 | OK | 2026-08-21 | Title "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens" and all 5 authors (Liu, Min, Zettlemoyer, Choi, Hajishirzi) match. |
| Geva et al. 2022 (EMNLP) | https://aclanthology.org/2022.emnlp-main.3/ | OK | 2026-08-21 | Title and authors (Geva, Caciularu, Wang, Goldberg) match; "Kevin Wang" matches the entry's "Wang, K. R." |
| Schuster et al. 2022 (CALM) | https://arxiv.org/abs/2207.07061 | OK | 2026-08-21 | Title "Confident Adaptive Language Modeling" and all 8 authors match. |
| Haviv et al. 2023 (EACL, idioms) | https://aclanthology.org/2023.eacl-main.19/ | OK | 2026-08-21 | Title "Understanding Transformer Memorization Recall Through Idioms" and all 6 authors match; venue confirmed as EACL 2023. |
| **Chen, Han & Miyao 2026** | https://arxiv.org/abs/2603.21658 | OK | 2026-08-21 | Title "A Comparative Analysis of LLM Memorization at Statistical and Internal Levels: Cross-Model Commonalities and Model-Specific Signatures" and authors Bowen Chen, Namgi Han, Yusuke Miyao match exactly. Submitted 2026-03-23. Abstract confirms Pythia/OLMo among the model families studied and middle-layer decoding as a method, matching the entry's characterization. Full-text deep read logged separately below. |
| Fartale et al. 2025 | https://arxiv.org/abs/2510.03366 | OK | 2026-08-21 | Title "Disentangling Recall and Reasoning in Transformer Models through Layer-wise Attention and Activation Analysis" and all 6 authors match. |
| Lasy, Knees & Woltran 2025 | https://arxiv.org/abs/2506.21588 | OK | 2026-08-21 | Title "Understanding Verbatim Memorization in LLMs Through Circuit Discovery" and all 3 authors match. |
| Sui et al. 2024 (ACL, Confabulation) | https://aclanthology.org/2024.acl-long.770/ | OK | 2026-08-21 | Title and all 4 authors (Sui, Duede, Wu, So) match. |
| Lee et al. 2022 (Factuality) | https://arxiv.org/abs/2206.04624 | OK | 2026-08-21 | Title "Factuality Enhanced Language Models for Open-Ended Text Generation" and all 7 authors match. |
| Biderman et al. 2023 (Pythia) | https://arxiv.org/abs/2304.01373 | OK | 2026-08-21 | Title and all 13 authors match, in the entry's order. |
| Gao et al. 2020 (The Pile) | https://arxiv.org/abs/2101.00027 | OK | 2026-08-21 | Title "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" and all 12 authors match. |
| OLMo Team 2025, *2 OLMo 2 Furious* (arXiv) | https://arxiv.org/abs/2501.00656 | OK | 2026-08-21 | Title matches. Abstract confirms coverage is "7B, 13B and 32B scales" — matches the entry's explicit claim that OLMo-2-0425-1B (the 1B checkpoint used in this study) is *not* covered by this paper. |
| OLMo-2-0425-1B model card (embedded in the OLMo Team entry) | https://huggingface.co/allenai/OLMo-2-0425-1B | OK | 2026-08-21 | Card confirms the two-stage training the entry attributes to it: Stage 1 = olmo-mix-1124, ~3.9T tokens, "95%+ of total pretraining budget"; Stage 2 = Dolmino-Mix-1124, 50B tokens. Matches the writeup's "What this does and does not show" section verbatim in substance. |
| Hassabis et al. 2007 (PNAS, hippocampal amnesia) | https://pmc.ncbi.nlm.nih.gov/articles/PMC1773058/ | OK | 2026-08-21 | Title "Patients with hippocampal amnesia cannot imagine new experiences", authors Hassabis, Kumaran, Vann, Maguire, PNAS 104(5), 1726-1731 — all match. |
| Spens & Burgess 2024 (Nature Human Behaviour) | https://www.nature.com/articles/s41562-023-01799-z | OK | 2026-08-21 | First fetch hit Nature's cookie-auth redirect (303 → idp.nature.com → 302 back with an error param); the retried URL resolved and confirmed title "A generative model of memory construction and consolidation", authors Eleanor Spens and Neil Burgess, Nature Human Behaviour vol. 8, pp. 526-543. |
| Squire et al. 2010 (PNAS, contested patient evidence) | https://doi.org/10.1073/pnas.1014391107 | UNREACHABLE (bot-blocked, content cross-verified) | 2026-08-21 | The entry gives only a DOI, no plain URL. The DOI resolves to `pnas.org/doi/full/10.1073/pnas.1014391107`, which WebFetch could not retrieve — both that URL and `pnas.org/doi/10.1073/pnas.1014391107` returned HTTP 403 (PNAS's bot wall), on repeated tries. Direct automated fetch failed. As a cross-check, WebSearch independently confirmed the same DOI resolves to a PNAS paper titled "Role of the hippocampus in remembering the past and imagining the future," authors Squire, van der Horst, McDuff, Frascino, Hopkins, Mauldin, PNAS 107(44), 19044-19048 — matching the entry exactly. Recorded as unreachable rather than OK because I did not myself load the page content; the match is corroborated, not directly observed. |
| Biderman et al. 2023 ("Recommended, optional": Emergent and Predictable Memorization) | https://arxiv.org/abs/2304.11158 | OK | 2026-08-21 | Title and all 7 authors match. |

## Chen, Han & Miyao (2026) deep-read notes

Fetched https://arxiv.org/abs/2603.21658 (abstract/metadata) and the full text at
https://arxiv.org/html/2603.21658 (rendered directly, no `v1` suffix needed) on 2026-08-21.

- **(a) Infini-gram use — confirmed.** Section 4.3: "We show the average token frequency
  distribution for memorized and unmemorized sequences using Infini-gram." Appendix A.1.3: "In
  this study, we have used the Infini-gram to get the token frequency of memorized and unmemorized
  sequences." They flag that Infini-gram "mainly uses the Llama-2 tokenizer for its n-gram
  database" rather than each model's native tokenizer, and that they "only queried the frequency
  for fully memorized, half-memorized, and unmemorized sequences" — i.e., three discrete buckets,
  not the corpus-count-verified item-by-item eligibility gate this writeup uses.

- **(b) Our differentiation sentence — accurate.** Their memorization score is
  `M_i(X,Y) = Σ 𝟙(x_{i,k}=y_{i,k}) / n` (Section 3.2): a fraction in [0,1] over 32 continuation
  tokens, with M=1 "fully memorized" and M=0 "unmemorized." Their internal-level analysis (logit
  lens / middle-layer decoding, Section 4.5) is reported as a group contrast: "For both memorized
  and unmemorized sequences, decoding probabilities increase gradually in early layers before
  exhibiting a sharp burst in later layers," with Figure 4 plotting "Solid/dotted line ... for
  memorized/unmemorized sequences" and Section 4.4 reporting "memorized sequences exhibit lower
  similarity recovery compared to unmemorized sequences." This is a memorized-vs-non-memorized
  population/group comparison at the model level, which is exactly what our "Why this question"
  section states their design is. No mismatch found.

- **(c) Within-memorized-set continuous dose-response regression — not found; no scooping risk
  identified.** Despite `M_i` being a continuous score in principle, the paper's reported analyses
  bucket it into discrete categories (fully / half / unmemorized) rather than regressing an
  internal-depth measure on it continuously within the memorized subset. Explicitly re-queried on
  this point with an exhaustive-quote prompt; the answer both times was that no such correlation,
  regression, or within-group trend is reported anywhere in the text — comparisons throughout
  Section 4 are memorized-group vs. unmemorized-group, not a within-memorized dose-response. **No
  quote of this kind exists to report**, and therefore no scooping risk against our core claim
  (within-L0 Spearman of memorization strength against layer-wise convergence depth) was found.

## Method note

All checks used WebFetch (primary) plus WebSearch as a cross-check where WebFetch was blocked
(Squire et al. only). Every "OK" row reflects content actually retrieved and read on 2026-08-21 by
this pass, not a copy of the 2026-08-20 panel result.

---

## Full-text verification pass (2026-08-21, third sweep)

Owner rule: sources that the paper cites or characterizes must be read in FULL —
title/author/URL matching is not verification. Nine readers fetched the complete
text of all 18 remaining sources (the 19th, Chen 2026, was full-read earlier the
same day). Access: 18/18 full text obtained (incl. PMC mirrors for PNAS/Nature-adjacent
sources). Claim verdicts: 44 HOLDS · 9 QUALIFIED · 2 CONTRADICTED.

**Corrections applied to the draft (both languages):**
- **OLMo 2 paper (arXiv:2501.00656) — CONTRADICTED, fixed.** The draft said the paper
  "covers 7B/13B/32B, not this checkpoint." The current revision (v3, 2025-10-08)
  contains Appendix B "OLMo 2 1B" incl. "B.1 Difficulties with OLMo 2 1B", with the
  1B training recipe and results tables. Two earlier title-level checks missed this;
  only full-text reading caught it. Both mentions corrected.
- QUALIFIED wording tightened for: infini-gram index-count provenance (live API docs,
  not the arXiv text), Geva §5.1 (saturation is formally defined and quantified, not
  "descriptive"; their operationalization is FFN-mechanistic), Haviv (same instrument
  *family*), Hassabis 2007 ("new experiences" is the paper's own vocabulary), Sui 2024
  (adjacent behavioral axis), Pythia-indexing attribution (indexing is infini-gram's
  contribution), Biderman 2023 emergent (k-extraction vs graded log-probability).

Per-source reports with quotes: workflow journal `wf_c76f554d-814` (internal record).
