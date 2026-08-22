# Recall is not separable from predictability in small open LLMs

### Three instrument generations — the first exploratory, the later two preregistered — three failures to separate, and a corpus-side measurement of a reason it is hard here

**Leo Gudu** · 2026-08-22 · v1.0

*Disclosure: research conducted by the author with extensive AI-agent assistance
under preregistered protocols; every number carries a rerun command.*

---

## Abstract

Can verbatim recall be told apart from plausible generation inside a small
open-corpus language model? This document reports the end state of a project
that asked that question three times with three different instruments and got a
separation out of none of them. The first generation was exploratory and not
preregistered; from the second onward, the decision rules were frozen before
each look at the data.

The first instrument — a threshold-crossing "convergence depth" read off the
logit lens — produced a significant, externally anchored, five-run correlation,
and then failed its own self-test: the sign of the effect is set by where the
threshold sits on the KL curve, not by how well an item is memorized. It was
withdrawn in the next version of the document that had reported it.

The second instrument — a preregistered crossed control that puts memorized
text against never-memorized text matched on predictability — returned no
verdict at all. None of its three registered decision rules fired. A
dose–response of similar strength appeared on text that is corpus-absent by
construction.

The third instrument — a preregistered corrupt-and-restore activation-patching
experiment — moved the question from watching to intervening. On the arm built
to remove the predictability confound, restoring the memorized association and
restoring a merely predictable continuation are **indistinguishable in all
three layer bands**, and the registered equivalence test passes at boundaries
of 0.309 nats (Pythia family) and 0.514 nats (OLMo). On the secondary arm the
registered contrast fired in the direction the memorization account does not
predict: never-memorized predictable text recovers *better* than memorized
text — a contrast registered as carrying an undetachable battery confound (§4.4),
and reported as a registered cell rather than a mechanism. A follow-up per-token
position test added a third non-separation, and a preregistered 2×2 word-pool
control rejected one instrument-side account of an earlier failure — on a
single cell whose memorized side holds two items, and the rejection names where
the effect does not come from, not what it is.

Alongside the negative results sits one positive corpus-side measurement that
explains why the question is hard here. The gate that admits an item into the
memorized condition is keyed to short-window corpus frequency, so it
structurally admits passages containing often-repeated phrases — and
often-quoted text is predictable text. In this battery, memorized text is
precisely the text that is most quotable, hence most predictable. The two
things this project set out to separate are entangled by how the material was
selected, not only by the limits of the instruments.

Model span is 410M–2.8B parameters. Nothing here supports any claim about
larger models, and the design says so in a clause frozen before the data
existed.

---

## 1. The question

An autoregressive language model produces every token the same way, so "does it
recall or does it invent?" is trivially answered "it generates" and
uninformatively so. The productive version is quantitative: is there an
internal quantity, or an intervention, that separates output the model has seen
verbatim from output that is merely a good guess?

The practical stake is that a model's errors are easier to trust or distrust if
you can tell which kind they are. The project's own goal is a cheap, locally
hostable AI companion; understanding when a small model is reciting and when it
is confabulating is a prerequisite for relying on one.

Answering the question at all requires ground truth for "memorized," which
requires models whose training corpora are public and searchable. This work
uses the Pythia family (trained on The Pile) and OLMo-2-1B (trained on
olmo-mix), both queried through infini-gram. Related work and the reasoning
behind the design are set out in the published v0.7 document under "Why this
question"; every citation there was checked against full text and the check is
recorded in `docs/CITATIONS_VERIFIED.md`.

**The answer, stated once and then defended:** at this scale, with these
instruments, recall and predictable generation cannot be separated — and the
reason is measurable rather than merely asserted.

---

## 2. Three instrument generations and their verdicts

### 2.1 Generation one: convergence depth (v0.5 → v0.6) — withdrawn by self-test

**This generation was exploratory and not preregistered** — the within-condition
contrast, the threshold, and the comparison framing were all chosen after a
pooled analysis had been inspected, and the sequence is in the process log.
Preregistration begins with generation two.

The first instrument scored, for each item, the layer at which the model's
next-token distribution stopped changing by more than a fixed 0.1-nat KL
threshold, and correlated that depth against a memorization score. Within
corpus-verified memorized text the correlation was negative in all five runs
(−0.53, −0.34, −0.69, −0.42, −0.86), two of them surviving multiple-comparison
correction, and it was anchored to a quantity outside the model.

The instrument was disqualified by a control run the same day the result was
published. Recomputing depth under relative criteria (last crossing of α times
the layer-0 KL) and entropy-scaled criteria flips the sign: across the 80
criterion × model cells, 12 are significantly positive against 5 significantly
negative. Pythia-410M runs −0.53 on the frozen metric, **+0.59** at α = 0.5 and
**+0.91** under the entropy-scaled criterion at τ = 0.25 of final entropy. Two
independent implementations, written blind to each other, agree on all 165
within-L0 cells to within 0.0005.

**Verdict: the claim that memorization strength tracks earlier layer-wise
convergence was withdrawn.** The cells that stay negative are not separable
from final-distribution sharpness, and the family's sign is an artifact of
threshold placement. Full tables ship in `results/relative_depth/`; the control
is `harness/relative_depth_analysis.py`.

What survived the reversal was a weaker and more robust claim: within memorized
text, how well a passage is memorized tracks how confidently the model
continues it (teacher-forced gold log-probability against final-layer entropy
of the model's own free continuation, rho −0.70 to −0.82, p < .001 in all five
runs).

That surviving claim had one disclosed gap: its control was matched on
construction, not on predictability. The data could not say whether the
relationship is specific to memorization or is generic predictability.

### 2.2 Generation two: the preregistered crossed control (v0.7) — no verdict arrived

The L0P battery was built to cross that gap: 18 authored English paragraphs
whose referring nouns are invented lexemes carried by an ordinary English
frame, so corpus exposure is ≈ 0 by construction and verified by the same
infini-gram protocol — **all 392 probe windows return exactly zero in both
training corpora**. Predictability is titrated across three tiers. The
preregistration (`docs/PREREG_L0P.md`, public commit `459710c`) was posted
before any L0P measurement existed.

The registered manipulation check passed in all five models: the memorized
items' gold range covers 74.3–99.0% of the L0P gold range, against a registered
threshold of 50% in at least four of five.

**Result one — the dose–response reappears on text that is corpus-absent by
construction.** Within L0P, gold log-probability tracks final-layer entropy at
rho −0.71 to −0.77 across all five runs, against −0.70 to −0.82 within
memorized text. The registered slope comparison finds no difference it can
resolve: mean Fisher-z difference +0.050, shared-item permutation p = .809.

**Result two — no memorization-specific sharpening at matched
predictability.** The registered primary analysis puts the condition
coefficient positive in all five runs (+0.026 to +0.169; combined studentized
statistic +1.31), meaning memorized text reads as very slightly *less* sharp
than never-seen predictable text at the same gold level. The combined
permutation p is .099 — not significant under the registered rule, and inside
the fragile zone (p between .02 and .10) declared in advance for calibration
reasons.

**Verdict: none of the three preregistered decision rules fired.** The
memorization rule fails on both its clauses. The generic-predictability rule
fails because the registration required "no consistent sign in 4+ of 5 runs"
and the five signs are consistent — in the direction the memorization account
does not predict. Under its own frozen rules the document drew **no registered
conclusion in either direction**.

**A preregistration design error was published as such.** Adversarial internal
review of the analysis code, completed before anyone saw a result, showed by
simulation that the generic-predictability exit was substantially harder to
reach than the registration assumed: the four Pythia runs share one item set,
their coefficients move together, and "no consistent sign in 4+ of 5" is
disfavored whatever the truth. The registration was not amended, because an
amendment issued between data and unblinding is indistinguishable, to a reader,
from one fitted to the data. The registration also carried no prospective power
or minimum-detectable-effect analysis. Both omissions were fixed in the next
registration, which is generation three.

### 2.3 Generation three: causal intervention (this document) — R-entangled

Generation three is reported in full in §4. Its registered primary verdict is
**R-entangled**: within the resolution this design can afford, recovery of the
memorized association and recovery of the predictable continuation are
indistinguishable.

### 2.4 What the three generations have in common

Each generation was a stricter instrument than the last, and each returned a
weaker claim than the last. That is the shape of the result. The project did
not run out of ideas; it ran into a property of the material, which §3
measures directly.

---

## 3. The entanglement finding

This is the one positive measurement in the document, and it is corpus-side:
it requires no model at all.

The gate that admits a passage into the memorized condition asks whether at
least 70% of the passage's eleven probe windows (ten 7-word windows at stride 4
over the ≈44-word continuation, plus one window spanning the prompt/continuation
boundary) appear at least 20 times in the model's training corpus. The headline
name for this quantity in v0.5–v0.7 was "corpus duplication counts." New probes
in 2026-08-21 showed that name to be wrong, and the correction was published
(`docs/CORRECTION_20260821_ANCHOR.md`).

**What the counts actually measure (measured):**

- **Within a single passage, window counts are far from uniform.** The ratio
  between a passage's highest- and lowest-count window has median **16.2×**
  (Pile, 17 gate-eligible items) and **42.3×** (olmo-mix, 20 items). A true
  copy count would make every window of the same passage count about equally.
- **Complete copies are much rarer than the gate's threshold suggests.**
  Extending the probe window stepwise from 7 to 44 words at the same start, the
  count at full passage length — the actual copy count — has median **18** over
  the 17 Pile-eligible items, which is *below the gate's own threshold of 20*
  (10 of 17 below 20; 2 items at 0). The olmo-mix figures are median **64**,
  6 of 20 below 20, 3 at 0. The shortfall against the threshold is a Pile-side
  finding; the gap between peak window count and copy count holds in both
  corpora.
- **A concrete case.** The *Pride and Prejudice* item's first probed 7-gram —
  "well fixed in the minds of the", a mid-sentence fragment of the gold
  continuation, not the famous opening line — occurs **171 times** in Pile,
  while the complete 44-word passage occurs **0 times** at this probe's
  tokenization.

**What follows for the anchor (measured).** The model-external anchor should be
read as "median probe-window frequency — in practice dominated by the passage's
most-repeated short phrases," not as a count of passage duplicates. The
correlations themselves are unchanged (rho 0.64–0.73 against the memory scale,
all five runs); what they anchor the memory scale to is short-phrase repetition
frequency. The dose–response survives — a phrase repeated 171 times was seen
171 times — but the dose attaches to **short phrases** while gold
log-probability is measured over the **whole ≈44-word passage**. The published
unit of dose was overstated. This is a precision problem, not an existence
problem.

**What follows for the project (interpretation, flagged as such).** A gate
keyed to short-window frequency structurally favours passages containing
often-repeated phrases, and often-quoted text tends to be predictable text. The
memorized condition is therefore not a random sample of memorized text; it is a
sample of *quotable* text. This corpus-side observation is consistent with the
predictability confound that the preregistered control had already surfaced
model-side. It is labeled as interpretation because the registered analyses
that would separate this account from localized exposure are underpowered —
neither account is established. §5 reports the strongest attempt to separate
them, and it also fails.

---

## 4. The causal phase

### 4.1 What was done and what was frozen first

v0.5 committed to a *direction* — that the next phase would move from observing
to intervening, by activation patching — and named a kill condition for it. It
did not specify the operation. Everything operational below was defined and
frozen afterwards, in the causal preregistration and its second stage: corrupt a
single real-word token in the prompt, restore the clean run's residual state at
one of three layer bands (early / mid / late) at the cue neighbourhood, and
measure how much of the model's teacher-forced gold log-probability over the
first two continuation tokens (ΔY, in nats) the patch recovers. Corruption strength is calibrated per
item at the q = 0.75 quantile of surviving candidates, with a top-50 rank
guard.

The preregistration was frozen in two stages: stage 1 before any GPU
measurement (commit `d04816a`), stage 2 after a registered smoke run backfilled
five whitelisted values (`e32fca6` / `510e87c` / `a3bd168`). The analysis code
was written under a schema-only discipline — its author read key names and
types, never values — and frozen before unblinding (`16d062a`). Every
specification repair between the two stages carries a visible correction note
and its own declared commit. These hashes belong to the project's local working
history and are given as a record of freeze order, not as references a reader
can resolve; §10 says which parts of the sequence are independently checkable
and which are not. The stage-1 preregistration ships with the package
(`planning/CAUSAL_PREREG_v1.md`); the second-stage freeze record does not, and
is likewise a record rather than a reference — the values it froze (σ_DiD, the
numerical floor, the equivalence bound, k) are quoted in §4.2, §4.3 and §4.7,
so nothing this document relies on sits only in the unshipped file. The verdicts below are the frozen machinery's
output, unedited, from `results/causal/analysis_main_winB1.json`.

### 4.2 The primary arm, and why it is primary despite being weaker

**The load-bearing arm is L0M.** For each of seven memorized passages there is
a twin that differs by exactly one prompt word, with continuation text that is
verbatim identical. Continuation predictability is therefore matched by
construction, and the contrast isolates the memorized key → content association
itself. The construction was checked by two registered gates before unblinding.
The pre-screen gate on twin clean-gold correlation passed in all five models —
ρ̂ = **0.9967 to 0.9999** against a registered threshold of **0.928**
(tanh(1.645/√(n−3)) at n = 4 pairs), with 90% CI lower bounds of 0.914–0.998
(`results/causal/smoke_eval_winA20260822.json`); the frozen note attached to
this gate states that it is an upper bound on the effect-level correlation and
a pre-screen only. The matching gate that the primary verdict actually depends
on also passed: median |Δgold| is **0.017** over the whole passage and **0.056**
over the launch window, against thresholds of 0.5 and 1.0
(`results/causal/analysis_main_winB1.json`, `smoke_quoted.l0m_pass_rule`).

Its cost was stated before unblinding. The primary arm is *weaker* than the
secondary: simulated declaration rate at delta = 0.5 is **.388–.544** for the
primary arm (internal record, not shipped with this package —
`theory_v3_l0m_opchar.json`; the two rows bracketing the unknown family
correlation, value quoted above) against **.704–.905** for the
secondary arm (internal record, not shipped with this package —
`theory_v3_final.md` §4.2; value quoted above). **It is
primary because it removes the confound, not because it is strong.**

### 4.3 Registered primary result — R-entangled in all three layer bands

The L0M-vs-L0 recovery contrast is **−0.17σ / +0.09σ / −0.03σ** (early / mid /
late), equivalently −0.059 / +0.030 / −0.009 nats. Those nats figures are the
**unweighted average of the two family means**, not a pooled average over all
22 pairs; the analyzer stores the family means, and pooling the pairs instead
gives −0.047 in the early band, because the Pythia family carries 16 pairs
against OLMo's 6. None fired: combined two-sided p = **.110 / .601 / .287**. The registered TOST equivalence test
passed in all three bands.

**Equivalence boundaries, per family at its realized n** (both tracks, since
neither alone is the claim):

| Family | Realized n (pooled pairs) | Boundary | Result |
|---|---|---|---|
| Pythia (410M / 1B / 1.4B) | 16 | **0.309 nats** | passes in all three bands |
| OLMo-2-1B | 6 | **0.514 nats** | passes; n falls below the frozen table's smallest key (8), flagged as such |

The frozen table maps realized n to a multiple of σ_DiD = 0.343 nats and the
analyzer takes the smallest table key that is ≥ n, which is the conservative
direction. Both boundaries are wider than the design-time projection, because
exclusions reduced n below the planned counts (21 planned Pythia pairs, 7
planned OLMo pairs). **The equivalence claim extends no further than the
boundaries in the table above**, and quoting the narrower track alone would
overstate it. Reporting both tracks is a standing requirement in this project's
writing rules, adopted before this document was drafted and after an earlier
draft was found to have under-reported one of them; it is not a choice made
here.

The analyzer's frozen narrative for this cell is stored in Chinese and is
identical in all three bands (`results/causal/analysis_main_winB1.json`, field
`frozen_narrative`). Verbatim:

> §11-B：在本設計的等價邊界內分不開記憶專屬與一般可預測性。等價邊界由
> σ_DiD 與各家族實際 n 決定，非「沒有效應」。

In translation: *§11-B: within this design's equivalence boundaries,
memorization-specific and general predictability cannot be separated. The
boundaries are set by σ_DiD and each family's realized n; they do not mean
"no effect."*

One gloss, which is this document's own wording and forms no part of the frozen
text: swapping out the memorized key changed nothing this instrument could see.

**Mandatory disclosure attached to this verdict by the frozen joint rule:** the
R-entangled declaration was reached **without provenance evidence**, because
the provenance track returned no computable family statistic (§4.6).

### 4.4 Registered secondary result — a fire in the wrong direction (C)

The L0-vs-L0P contrast — memorized text against invented-lexeme predictable
text, gold-calipered — fired in all three bands in the direction the
memorization account does not predict. Mean **−1.10 to −1.13 nats**
(−3.20σ to −3.30σ), combined p = **.005 / .009 / .005**: **the
never-memorized predictable text recovers better than the memorized text.**
The registered equivalence test fails on this arm in every band.

This arm permanently carries its frozen limitation **(C)**: the L0↔L0P axis
confounds "memorized or not" with "which battery the item came from," and the
confound was registered as undetachable on this arm. The direction is reported
as a registered cell, not as a mechanism.

The parallel sensitivity regime (dual-calipered) is reported beside it, as
registered: mean −1.35 nats (−3.9σ) in all three bands, no rule fired in early
(p = .064) or mid (p = .061), wrong-direction fire in late (p = .021),
equivalence failing throughout. It carries the same (C) qualifier.

### 4.5 The descriptive model

Pythia-2.8B does not fit the fp32 path on this hardware and ran in fp16. A rule
frozen before unblinding removed it from adjudication and reported it
descriptively throughout, because its measured dtype bias of **0.056 nats**
exceeds the frozen gate of 0.0274 nats. Its primary-arm means are
−0.018 / −0.037 / +0.088 nats (n = 7 pairs) and its secondary-arm means
−0.606 / −0.508 / +0.147 nats (n = 9 pairs). These numbers enter no verdict.

### 4.6 The provenance track ran and was emptied by its own quality gates

All **24** (model, pair) provenance cells defined by the frozen pairing table
entered the run — 7 / 6 / 5 / 2 / 4 across the five models, over 8 distinct
non-memorized items. The authority for those counts is an internal record, not shipped with this
package (`theory_v3_l0n_pairs.json`), which carries them as a
frozen expectation alongside the matching rule that produced them (caliper 0.30
on |Δgold|, maximum bipartite matching); the pairing table used at run time
(`planning/battery_expansion/adjudication_pairs_v218.json`, shipped) is derived
from it and reproduces the same counts.
In the early band **17 of the 24 were excluded by frozen instrument guards** —
9 rank-guard zero-survivor, 5 pool-too-small-final, 3 with no provenance row —
alongside 4 further item-level "unusable" notices, leaving **7 usable cells**:
two in each Pythia model and one in OLMo. The late band is identical in count
and the mid band leaves 6. Because a family statistic needs at least two pairs
in a run, none could be computed for OLMo in any band, nor for Pythia in the
mid band, so the track's status is **INCOMPLETE in all three bands**.

The declaration that requires this track, R-specific, was therefore
**structurally unreachable — a statement about the instrument, not about
memorization.** It is worth being exact about what that did and did not cost,
because the obvious reading overstates it. On this data the positive branch was
never open in the first place: the frozen joint rule lets provenance constrain
only the declaration of R-specific, and the gold track measured *equivalence*,
so R-specific was off the table on the gold result alone, whatever the
provenance track had returned (`analysis_main_winB1.json`,
`joint_verdict_primary.why`). **What the emptied track actually removed is the
contradiction check** — the registered cell for "gold equivalent but provenance
fires," which is how this design was supposed to catch its own inconsistency.
**It is not a lost positive result.**

The equivalence verdict in §4.3 is declared, per the frozen
joint table, with the mandatory disclosure that it was reached without
provenance evidence.

The rank-guard exclusions are an observation under this run's hand-written
corruption word pools, not a demonstrated property of the non-memorized text.
Survival collapses specifically for proper-noun cues (71% → 18% between the two
conditions) while verb cues survive about equally well in both (~95%, small n),
and the non-memorized items happen to draw proper-noun cues more often. That
word-pool attribution was left unresolved at unblinding and was put to a
dedicated test; §6 reports the outcome.

### 4.7 Sensitivity line (frozen)

With σ_DiD = 0.343 nats measured on the holdout twins and each family's
realized n, the design's 80%-power minimum detectable effect is **0.302 nats**
(fp32-only arm). The observed primary-arm effects are far below this line.
**The non-firing of the difference exits must not be read as "no effect,"** and
the equivalence claim extends no further than the boundaries quoted in §4.3.

### 4.8 What the instrument work found on the way

fp16 forward passes carry kernel-selection noise of the same order as the
quantities under test: the same item in the same model differs by up to **0.229
nats** depending only on which other items share its batch, established by
three independent diagnostics: constructive exclusion of a masking explanation,
an fp32 comparison in which the same spread is 675–4919× smaller, and a kernel
signature in which batch sizes 4 and 8 return bit-identical values. The four
models that fit in memory therefore ran the decisive arm in fp32.

Measured dtype bias across three models is **non-monotone in size**: 0.011 nats
at 1B, **0.240** at 1.4B, 0.056 at 2.8B — the middle model is the worst, not the
largest. That killed single-point extrapolation and produced the
phase's standing rule, adopted mid-run from its own failure: **a parameter
extrapolated across models must be measured on at least two models an order of
magnitude apart before it may carry a decision.** That rule is what forbids
every scale claim in this document (§8, limit 1).

---

## 5. The per-token position test

### 5.1 The question and the discipline

The main experiment stored per-token data, and a criterion frozen in the v0.7
era was waiting for it, so this test cost no GPU time. It asks a sharper
version of §3's open question: is the model's familiarity with a passage
**concentrated at the positions covered by its most-quoted windows** (localized
exposure), or **spread evenly over the passage** (quotability as a property of
the whole text)?

The procedure was run under the strictest available discipline. The mapping
from the frozen criterion to the actual data files was itself **frozen blind** —
its author was forbidden to read any per-token value while writing it — and an
independent reviewer re-derived the two discretionary decisions without seeing
the reasoning, converging on the same answers. Nine points from that review
were then corrected, and two more followed from later checks — eleven in all,
each listed with what changed and where. Two of the eleven are withdrawals by
the reviewing side, and they are separate events that should not be read as
one: in the first, it had over-escalated an error report — claiming that an
un-applied gate had contaminated real data — and withdrew the premise after
being asked to quantify the damage and recounting (the contamination had never
happened); in the second, described in §9, it withdrew a false-positive-rate
range after rewriting its own simulation at a finer granularity. Two tokenizer traps were caught by
synthetic self-test before any real per-token value was read: BOS prepending and the
leading-space offset convention are both per-model-family settings, and the
frozen specification had written both as global facts. Both were corrected by
measurement, on chain, before unblinding.

Per the frozen mapping, the analysis uses only the clean run's per-token gold
log-probabilities — never the corrupted or patched runs — so that the
conclusion does not depend on any decision made inside the causal phase. Window
labels come from each model's own training corpus. The main adjudication is
four fp32 runs and the "≥4/5 runs" clause therefore becomes **4/4**;
Pythia-2.8B is descriptive.

### 5.2 Results (n = 17 items, 59 fp32 cells; `results/causal/pos_test_v221.json`)

**Support test (i) — fails.** The criterion requires the effect to be positive
in 4/4 runs with p < .05. It is positive in **3 of 4** (Pythia-1B runs −0.021)
and the permutation p is **.106**. The mandatory gate version — which removes
the entire first window from both the in-window and out-of-window sets, to keep
a launch-position confound out of the comparison — gives **p = .299**, also 3
of 4. The two are a conjunction, so (i) fails twice over.

**Equivalence test (ii) — reportable, and fails.** The frozen rule requires the
standard deviation and CI half-width to be reported *before* the pass/fail, and
declares the test unexecutable if the per-item SD exceeds 0.501 nats at n = 17.
The measured per-item SD is **0.386 nats**, below that line, so the test is
executable and its outcome is informative. The cluster bootstrap (resampling
items, not cells) gives a 90% CI of **[0.045, 0.269]** around a pooled mean of
**+0.156 nats**, a half-width of 0.112. The upper endpoint **0.269 exceeds the
±0.2 nats equivalence boundary**, so equivalence does not hold.

**Verdict: both branches fail; the honest report is "cannot separate."** This
is the third time this project has measured that answer with a rule frozen in
advance.

**Descriptive note, as required.** The pooled +0.156 nats is positive and its
CI excludes zero: a positive descriptive tendency exists. But under the
permutation test that controls for *where the high-count windows sit*, it is
not distinguishable from randomly assigning the same labels to the same window
geometry. Sensitivity runs at other quantile thresholds move the pooled mean as
expected without changing the picture (q = 0.60: +0.110; q = 0.90: +0.197;
adjudication stays at q = 0.75). The descriptive model gives +0.123 nats
(n = 12). A cross-check through the previously validated permutation routine
(p = .052) is reported for arithmetic verification only — it is not a validity
check, because that routine cannot absorb the position confound either.

### 5.3 What this test cannot say, stated with the result

- **Four runs are not four independent witnesses.** Three of them are Pythia
  models sharing one tokenizer, one training corpus and one set of window
  labels, with nested training data. The effective number of independent units
  is about **two** (Pythia family 1, OLMo 1), not four.
- **The clusters are unequal, and two items have a single witness.** Item
  cluster sizes run 1 to 4; L0-10 appears only in Pythia-410M and L0-18 only in
  OLMo. They are named here, and no conclusion is read off them, because
  nothing in this data can separate an item property from a model property when
  there is only one witness.
- **The gate narrowed the construct.** Excluding the whole first window means
  this test says nothing about the launch region — which is exactly where
  localized exposure should be most visible, if a model that is reciting is
  most obviously reciting at the start. Construct width was traded for
  calibration, knowingly.
- **The registered prior was right about the failure mode and wrong about the
  culprit.** Before unblinding, the mapping document predicted the most likely
  outcome was "both branches fail," and named "any one model reverses sign" as
  the most likely mechanism, expecting OLMo to be that model. The event
  happened; the culprit was Pythia-1B (−0.021), while OLMo was the *most*
  positive run (+0.398). The prior is recorded here because it was recorded
  before the data, and it carried no weight in the verdict either way.

---

## 6. The word-pool 2×2

### 6.1 Why this test exists

§4.6 left a suspicion pointed at the instrument rather than at the material:
the provenance track's items died at the rank guard, and the deaths were
concentrated on proper-noun cues. Two explanations were on the table. Either
the collapse is specific to the **word class** of the corrupted cue, or the
**replacement pool** was a bad fit for those items' genre — replacing "sauce"
in a recipe with the name "Thomas" and then complaining that the item is
fragile.

A 2×2 puts both on trial at once: cue word class (proper noun / verb) crossed
with replacement pool (generic / tailored). The original single-arm design was
killed during planning because both hypotheses predicted the same result from
it — one search saved one GPU night. Genre matching in the tailored pools is
guaranteed by construction: words are drawn mechanically by frequency from the
item's own source book, with zero human judgement. The experiment itself took
about eight minutes.

### 6.2 Preconditions (all checked before the verdict)

Corruption adequacy holds in all four cells (median drop 0.21–1.40 nats against
an adequacy floor of 0.001434 nats, which is 10× the measured numerical floor).
Neither of the two honest exits is triggered. The "both sides collapse" exit
needs the memorized side to fall, and its survival rate is .88–.97; the "both
sides survive" exit needs the non-memorized side near ceiling, and its survival
rate is .63–.79, against the ≈95% that would trigger it. (These are survival
rates, not p-values.) The stop-rule flag proportions run 9.7%–34.7%, all below
the 40% trigger — the 34.7% is close to the line and is recorded as such.

### 6.3 The verdict, with the fragility that travels with it

Across all four cells, the memorized-minus-non-memorized survival difference is
descriptively positive, **+17.2 to +26.5 percentage points** (verb × generic
26.54, verb × tailored 17.16, proper × generic 19.62, proper × tailored 25.90;
`results/causal/verbcue_aggregate.json`). One of the four, proper × generic,
has a confidence interval that crosses zero ([−4.39, 45.59]), and that is
stated here rather than left to the reader to notice.

Against the frozen decision rule ("reject the pool account if the CI lower
bound exceeds 15.7 pp, or the CI covers 30.6 pp while excluding 0"):

- **verb × tailored pool: Δ = 17.16 pp, CI [9.19, 71.70] → rejects the pool
  account.** This is the cell the verdict rests on.
- **verb × generic pool: Δ = 26.54 pp, CI [9.02, 52.08] → ruled to have no
  discriminating power and not mapped to any cell.** On review this cell has
  only a single item on the memorized side, so resampling returns the same item
  every time, the memorized-side variance is zero, and the interval is not a
  statement about the uncertainty of Δ at all. The bias runs *toward* the
  verdict — understated variance narrows the CI, which is what makes "excludes
  0" easier to satisfy — so it was withdrawn rather than counted.

**Verdict, in the frozen wording of the cell it maps to (translated from the
Chinese original in the v1.1 verdict document, an internal record not shipped
with this package):** *the pool
attribution is rejected. Even when the cue is a verb, the non-memorized items
remain more fragile than the memorized ones — which pushes the account back
onto a property of the non-memorized material itself.*

**What "a property of the material itself" does and does not say.** It names a
location, not a mechanism. What was measured is where the fragility does *not*
come from — not what it is. Nothing here identifies the property, and the
phrase must not be read as though it did.

**Fragility statement, which may not be demoted to a footnote.** The verdict is
**carried by a single cell**, whose memorized side has n = 2, at the floor the
method allows, with a CI upper bound of 71.70 pp. Cluster counts are in single
digits and entry rates are low: 14 and 15 of 31 attempts entered for the two
verb cells, and 31–55% of attempts were lost to the "no valid corruption
candidate" category. **No inference beyond the sentence quoted above is
permitted by this result.**

**On the second axis (the pool effect itself):** verb class −5.88 pp, CI
[−33.0, 18.69]; proper-noun class +2.86 pp, CI [−14.38, 18.80]. **No pool
effect was observed, and a pool effect of up to roughly 19 pp is not
excluded.** The central prediction of the pool hypothesis is therefore
**unsupported, not falsified** — the earlier phrasing "this is not a property
of the replacement pool" was withdrawn on review because it reports "we did not
ask" as "we asked and the answer was no."

### 6.4 Two honest records that the verdict document requires be carried

**(a) The strongest cell is outside the deciding rule's jurisdiction.** The
proper-noun × tailored-pool cell rests on far more items than the cell that
carries the verdict (n = 6 and n = 3, against n = 2 and n = 4) and its CI
[12.85, 50.07] likewise covers 30.6 pp while
excluding 0 — but the frozen threshold was written for the verb arm, so it
formally does not adjudicate the proper-noun cells, and it was not used to. The
strongest support for the conclusion comes from a cell the frozen rule does not
reach. That is written down rather than quietly used.

**(b) Why one cell had a single memorized item, and the gap that let it
happen.** Two items produced no valid corruption candidate in *every* model
under the verb × generic condition, a silent token-alignment loss that the
pool-generation document had warned about, taking the memorized side from 3
items to 1. The gate that should have caught this checks whether the two
columns have *equal* usable counts rather than whether each has *enough*, so
0 = 0 passes. A minimum-usable-clusters-per-side clause is owed to the next
round. For the avoidance of a different confusion: the adequacy gate's
denominator-emptying trap did **not** fire here (zero items were excluded for
adequacy); the emptying came from the mechanism just described, and the two
must not be described as one.

### 6.5 A construct limitation on the pool axis

The column contrast measures **"picked-for-this-item versus
not-picked-for-this-item," not "genre-related versus genre-free."** The generic
pool is not genre-neutral: it was found to contain a word that happens to
appear at the cue position of one of the recipe items. Any reading of the pool
axis as a clean genre manipulation overstates what was built.

---

## 7. Post-hoc discussion

*Everything in this section is interpretation formed after unblinding. It has
no registered standing and appears nowhere in the results above.*

Three independent lines now point the same way. The model-side result from
generation two: the dose–response appears on never-memorized predictable text.
The corpus-side correction: the gate that admits items structurally favours
often-quoted, hence predictable, passages. The causal result: predictability
recovers at least as well as memorization under intervention — the "at least"
half resting on the confound-free arm, the "better" half on the secondary arm
that carries (C) — and the memorized key adds nothing detectable where the
confound is removed.

None of these three is individually decisive, and their convergence is a
synthesis made after the fact rather than a test that was run. Stated as the
hypothesis it is: at this scale, in this material, "memorized" and "predictable"
may not be two conditions that a better instrument would separate — they may be
close to the same condition, because the world produced the text that way.
Quotable text is repeated text is predictable text.

The 2×2 result in §6 fits the same shape from the other side. Once the
instrument was cleared of blame, what remained was located in the material —
obscure text is more fragile inside these models than famous text — though the
2×2 measured only where the fragility does not come from, never what it is.
Well-learned text is stubborn text, and this data does not say why.

None of that is established here. It is what this data would suggest to someone
designing the next experiment, and it is written down so that a reader can tell
it apart from what was measured.

---

## 8. Limitations

Five limits were frozen before the data existed and are reproduced here in full,
because they bound every claim above.

**1. No scale claims — this data can never support "larger models behave the
same."** The fp32 arm that carries the adjudication spans 410M → 1.4B, a factor
of **3.4×**. Even including the descriptive 2.8B model, the span is 410M → 2.8B,
a factor of **6.8×**. Both are **under one order of magnitude**, and the
project's standing rule requires two models an order of magnitude apart before
any cross-model parameter may carry a decision (§4.8). Any sentence of the form
"as models get larger, …" is out of bounds for this data set, whatever the
direction.

**2. "Cannot separate" is a limit of these instruments at this scale, not a law
of nature.** Every non-separation reported here is bounded by a stated
resolution: 0.302 nats minimum detectable effect for the causal phase, and
equivalence boundaries of 0.309 / 0.514 nats. A larger battery or a sharper
instrument may separate what this one could not.

**3. The secondary causal arm permanently carries confound (C).** The L0↔L0P
axis confounds "memorized or not" with "which battery the item came from." The
confound was registered as undetachable on that arm before the data existed, so
the wrong-direction fire in §4.4 is a registered cell and not a mechanism. It
appears with the (C) qualifier everywhere it is cited.

**4. The word-pool 2×2's column contrast measures "picked-for-this-item versus
not-picked-for-this-item," not "genre-related versus genre-free"** (§6.5), and
its verdict is carried by a single cell whose fragility statement (§6.3) must
travel with any citation of it.

*The position test has a different construct limitation of its own, and the two
should not be run together: what narrows the position test is the gate
described in §5.3, which means it says nothing about the launch region.*

**5. Negative results here are preregistered outcomes, not an absence of
effort.** Criteria were frozen, with hashes and commits, before the data
existed; every number has a rerun command in its harness script header. A
verdict of "none" or "cannot separate" is this design reporting what its own
rules produced, which is the only thing that makes such a verdict worth
reading.

Two further limits, specific to individual tests, are stated where they arise
and repeated here so that no reader has to find them: the four position-test
runs are approximately two independent units, not four (§5.3); and the
R-entangled verdict was declared without provenance evidence, because that
track was emptied by its own quality gates (§4.6).

---

## 9. Corrections history

*Letter suffixes below (b, c, d) denote pre-release drafting iterations of
this document; the released version is v1.0.*

The project's public record includes its reversals. Each was published with the
original text struck through rather than rewritten.

**v0.6 — a headline claim withdrawn by its own self-test.** The convergence-depth
result was disqualified the same afternoon its first version was published, by
a control the paper itself had named as decisive (§2.1). The instrument, not
just the number, was retired.

**v0.7.1 — the anchor correction (published, DOI'd).** The model-external
anchor named "corpus duplication counts" was renamed to "median probe-window
frequency, dominated by short-phrase repetition," with per-corpus numbers, the
probe script and the raw counts shipped alongside (§3). No registered verdict
changed. The correction draft itself was found, in adversarial internal review,
to have inflated its own headline number by counting three items that never
passed the gate; the recomputed 17-item version is what was published.

**Corrections inside the causal and follow-up phases.** These were internal but
are recorded on the same terms:

- A frozen specification stated BOS prepending as a global fact when it is
  per-model (Pythia 0, OLMo 1), and stated the leading-space offset convention
  the same way. Both were corrected by measurement before any real per-token
  value was read — the line the blind period actually draws; item text and
  schema-level counts were open, and the fix used them. The specification's own
  hard assertions would have aborted the run rather than silently corrupting
  it — which made the error cheap, not absent.
- A mapping document claimed that two 17-item sets were the same set, as an
  independent consistency check. They have the same size and different members
  (symmetric difference of two items). The claim was load-bearing where it
  appeared, and was withdrawn in full.
- **Both** simulators' false-positive rates, quoted on either side of a
  disagreement about a new gate, were downgraded — not one side's. The side
  that produced the first range (.10–.16) withdrew it after rewriting its own
  simulation at token granularity. The other side then opened its own code,
  confirmed the rewrite was right, and found that its figures (.16–.58) were a
  window-level granularity overestimate — the larger error of the two, and the
  one that ran against its own position. Both ranges are preserved in the
  frozen record beside the corrected ones. The root cause is that the penalty
  under test was modelled on whole windows while the estimator averages over
  tokens, so the window-level simulation was answering the wrong question. The
  gate was kept on a reason that depends on neither simulator: it is the only
  variant that stays safe under every candidate value of the granularity
  parameter, which nobody can measure.
- The 2×2 verdict document was revised on review in four places, including the
  withdrawal of a cell from adjudication and the demotion of "not a property of
  the pool" to "no pool effect observed" (§6.3).
- The repository's reader guide misattributed the construct limitation of §6.5
  to the position test's in-window / out-of-window contrast, and compressed the
  model span into a figure that did not match the range it was attached to.
  Both were corrected on chain (`d984cab`) while this document was being
  drafted, and §6.5 and §8 state the corrected versions.

**Four figures carried in an earlier internal draft are corrected here.** Each
was found by reopening the source file rather than trusting the draft, and each
is listed with what the file actually says.

- **Equivalence boundaries.** The draft gave the primary arm's boundaries as
  0.240 / 0.514 nats at realized n. The frozen analyzer's output for the primary
  arm is **0.309 / 0.514** nats at realized n = 16 / 6; the 0.240 boundary
  belongs to the *secondary* arm at n = 24. The error ran in the direction that
  flatters the result, because a narrower equivalence boundary is a stronger
  claim. §4.3 states the corrected values.
- **The twin-construction gate.** The draft described the clean-gold twin
  correlation as "above 0.999 in all five models against a 0.499 threshold."
  The five-model output gives ρ̂ = **0.9967–0.9999** against a threshold of
  **0.928**. The gate passes either way; the quoted threshold was simply not
  the one the gate used, and OLMo's value is 0.997, not "0.999+."
- **Provenance-track accounting.** The draft read as though 21 of 24 pairs were
  excluded, which would leave 3 survivors; 7 cells in fact survived in the early
  band. The skipped list contains 17 pair-level exclusions plus 4 item-level
  notices. §4.6 gives the full accounting.
- **Dtype-bias ordering.** The draft listed the three measured biases as
  "0.011 / 0.056 / 0.240," an ascending order that reads as if it were ordered
  by model size and would make the sequence monotone — which is the opposite of
  the point being made. Ordered by size the values are 0.011 (1B), **0.240**
  (1.4B), 0.056 (2.8B).

**Corrections made to this document before release.** It was reviewed against
its own sources before it went out, and four of its own claims did not survive:

- **A "verbatim" quotation that was not verbatim.** §4.3 introduced the
  analyzer's frozen narrative with the word "verbatim" and then gave an English
  rendering of it that also contained a clause the frozen text does not have.
  The frozen narrative is stored in Chinese; §4.3 now quotes it in the original
  with a translation beside it, and the added clause is marked as this
  document's own gloss.
- **A missing confound qualifier.** The Abstract described the secondary arm's
  wrong-direction result without the (C) qualifier that the registration
  attached to it permanently. The qualifier now travels with that sentence too.
- **An internal count that contradicted itself.** The Abstract counted the
  word-pool 2×2 as a fourth "non-separation" while the subtitle and the closing
  section counted three. The 2×2 is a rejection, not a non-separation; all
  three places now say three.
- **A verifiability claim wider than the published artifacts support.** §10
  said the freeze sequence is "checkable in git history." Part of it is — the
  Zenodo version chain, the public repository, the shipped preregistrations —
  but the causal phase's freeze commits live in a local history that is not
  pushed. §4.1 and §10 now separate the two, and the local hashes are presented
  as a record rather than as something a reader can resolve.

Two further figures were tightened rather than corrected: the primary arm's
nats values are now labeled as family averages (a reader pooling all pairs gets
−0.047 instead of −0.059), and the sensitivity arm's effect is quoted as −3.9σ
rather than −3.93σ, which was the least extreme of its three bands.

**One further correction, from a later pass of the same review.** The
account of the simulator disagreement was one-sided. An earlier draft of this
section recorded that the
side which produced the first false-positive-rate range withdrew it, and said
nothing about the fact that **the other side's numbers were downgraded too, and
by more** — the record shows the larger error belonged to the side reporting it,
which is the direction that makes the record worth keeping. The mechanism was
also misdescribed: the first withdrawal followed that side's own token-level
rewrite, and it is a different event from the separate over-escalation that was
withdrawn after a recount. §5.1 and §9 now keep the two events apart and give
both downgrades.

Two claims were also narrowed in the same round, after the people whose work
each section describes checked it line by line against what they had actually
done. "Before any real data was read" became "before any real per-token value
was read," because item text and schema-level counts were open during the blind
period and the fix used them; per-token values are the line that period
actually draws. And §4.6's account of the emptied provenance track now says
what it cost precisely: the positive branch was already closed by the gold
result, so what was lost is the contradiction check, not a route to a positive
finding.

**A shipping-scope calibration, part of the packaging cleanup before archival
release.** One document that had been expected
to ship was withdrawn from the package late — the causal phase's second-stage
freeze record — so every sentence describing what a reader receives was checked
against the package manifest rather than against the plan. Two preregistration
documents ship, one per phase, and they are named in §10; the withdrawn record
is treated the same way as the local commit hashes, as a record rather than a
reference, and each value it froze is quoted in §4 so that nothing load-bearing
lives only in a file the reader cannot open.

---

## 10. Reproducibility

**Everything below recomputes from the shipped repository.** No number in this
document requires trusting the process that produced it.

- **Rerun commands live in each harness script's header.** The causal main
  experiment: `.venv/Scripts/python.exe harness/causal_main.py --stage patch
  --run-id winB1`. Its analysis: `.venv/Scripts/python.exe
  harness/causal_analysis.py --run-id main_winB1` (and `--self-test`). The
  position test: `.venv/Scripts/python.exe harness/pos_test_v221.py --run-id
  main_winB1` (and `--self-test`). The word-pool aggregation:
  `.venv/Scripts/python.exe harness/verbcue_aggregate.py`. The corpus-side
  decay probe: `.venv/Scripts/python.exe harness/gate_decay_probe.py`.
- **Frozen constants have a single source.** B = 20000, seed = 20260822,
  α = 0.05 are defined in `harness/causal_analysis.py` and imported by
  everything downstream rather than copied. σ_DiD = 0.343 nats and the minimum
  detectable effect of 0.302 nats come from the same frozen block. The list of
  descriptive-only models is likewise defined once and imported; a mislabeled
  `backend_dtype` field elsewhere in the outputs is a known instrument bug,
  documented so that a future reader does not filter on it.
- **Where the numbers live.** Causal verdicts:
  `results/causal/analysis_main_winB1.json`. Position test:
  `results/causal/pos_test_v221.json`. Word-pool 2×2:
  `results/causal/verbcue_aggregate.json`. Corpus-side counts:
  `results/gate/gate_decay.json` and `battery/l0_verification.json`.
  Depth-instrument reversal: `results/relative_depth/`.
- **Environment.** Python 3.12 virtual environment at the repository root,
  torch cu124, one 8 GB consumer GPU; the fp32 paths that carry the
  adjudication run on CPU where they must.
- **Freeze order is part of the evidence, and here is exactly how much of it
  you can check.** What is independently verifiable from published artifacts is
  the Zenodo version chain and its timestamps (v0.5 → v0.7.1, each version
  deposited before the work that followed it), the public repository's own
  history, and the two preregistration documents shipped inside the package —
  `docs/PREREG_L0P.md` for the behavioral phase, public commit `459710c`, posted
  before any L0P measurement existed, and `planning/CAUSAL_PREREG_v1.md` for the
  causal phase's first freeze stage. What is **not** independently verifiable is
  the causal phase's internal freeze chain: the commit hashes quoted in §4.1 and
  §9 identify commits in a local working history that is not pushed, so they are
  offered as a record of the order in which things were frozen, not as something
  a reader can resolve. The second-stage freeze record does not ship either, and
  is a record in the same sense; every value it froze is quoted in §4, so no
  claim here rests on a document you cannot see. Where the two can be told
  apart, prefer the published artifact.

**Published artifacts.** GitHub `gudu616/omtr`; Zenodo concept DOI
**10.5281/zenodo.22039215**, covering the version chain v0.5 → v0.6 → v0.7 →
v0.7.1 (the last of which is the public anchor correction, DOI
10.5281/zenodo.22046829).

**Author.** Leo Gudu.

---

## Closing

This document reports one positive corpus-side measurement; a first-generation
instrument — exploratory, not preregistered — withdrawn by its own self-test;
three preregistered tests that each failed to separate recall from
predictability — the crossed control of generation two, the causal phase of
generation three, and the position test run inside that same phase; and one
hypothesis rejected on a single fragile cell. (The subtitle's count is of
*generations*, three, of which the first was not preregistered; the count here
is of *preregistered tests*, also three, because generation three contributed
two of them. The two counts are of different things.) The
negative results are the product, not the residue.

The question "can recall be told apart from ideation in a small open model?"
now has a specific answer rather than a vague one: not with these three
instruments, not at 410M–2.8B parameters, not within 0.309 nats (Pythia) or
0.514 nats (OLMo) on the arm built to be clean — **and the reason is not only instrumental.** Here, in this
material: the text that got memorized is the text that gets quoted, and the
text that gets quoted is the text that is easy to predict. Whoever builds the
next instrument for this
question should assume the two conditions arrive entangled, and design the
material, not just the measurement, to pull them apart.
