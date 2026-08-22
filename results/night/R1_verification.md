# R1 teacher-forced depth — adversarial verification

Reviewer: night-shift adversarial verification pass. Date: 2026-08-20.
Inputs: `results/night/tf_depth_*.json` (R1), `results/raw/pilot_*.json` (frozen greedy
pilot), `results/analysis_*.json`, `archive/followup_report.json` (relocated since this
pass was written; historical, not current canonical), `battery/battery.json`,
`battery/l0_verification.json`. Scripts: `results/night/R1_verify_part1.py`, `R1_verify_part2.py`.
All numbers below were recomputed from source; none are quoted from the R1 summary.

---

## Verdict up front

**(b), and the memory-specific component is not distinguishable from zero.**

R1 did not resolve the two-span confound — it replaced it with a tighter one. Teacher
forcing put the memory measure (mean gold logprob) and the nuisance variable (mean final
entropy) on the same tokens in the same forward pass, and in doing so drove their rank
correlation from about −0.75 to **−0.98 within L0**. Mean gold logprob on a teacher-forced
span *is* the model's negative cross-entropy on that span; when the model is near-calibrated
there, it equals the negative mean entropy up to a small offset. Empirically it does:
`mean(gold + H) = +0.11 to +0.34` with SD 0.18–0.33, OLS slope of gold on −H = 0.80–1.13,
R² = 0.90–0.98. The R1 dose-response is therefore a **confidence → convergence-depth**
relation restated, not a memory → depth relation.

Four independent tests agree that nothing memory-specific survives:

| test | result | models where memory-specific effect survives |
|---|---|---|
| condition beyond gold logprob (pooled ΔR², rank space) | +0.002 to +0.014 | 0/5 |
| partial rho(condition, depth \| gold) | −0.235 to +0.160, all p>.20, sign flips | 0/5 |
| matched-predictability pairs (Wilcoxon on ΔDepth) | p = .156–.312, median ΔDepth ≈ 0, sign flips | 0/5 |
| corpus duplication count → depth, controlling gold | −0.31 to +0.07, all p>.23 | 0/5 |
| Fisher-z L0 vs L0N, bootstrap 95% CI on Δz | CI excludes 0 only for OLMo | 1/5 |

And a smoking gun for the metric itself: on the *same layer curves*, `kl_auc_norm`
correlates **positively** with gold logprob within L0 in 5/5 models (+0.51 to +0.80,
p<.03 all), i.e. better-memorized spans have *more* total KL area — the opposite reading
of "converges earlier" that `depth_tau_0.1` gives. Both cannot be true descriptions of
retrieval depth. The reconciliation is that a fixed absolute threshold of 0.1 nats is not
entropy-normalized: a peaked (memorized) final distribution can be matched to within
0.1 nats by the penultimate layer, a diffuse one cannot, regardless of when the computation
"finished". R1 stored only `depth_tau_0.1` — the one metric that had shown the effect in
the greedy run — so this cross-check could not be run on the teacher-forced curves at all.

Option (a) is false: the confound is tighter, not resolved, and the dissociation did not
uniformly strengthen (Δz grew for pythia-410m and pythia-1b, **shrank** for pythia-1.4b
and pythia-2.8b, unchanged for OLMo). Option (c)'s diagnosis is right but its prescription
is wrong: the greedy version carries the same confound, only more loosely, and its own
frozen entropy-partial (`T3`) was already null in 4/5 models (+0.10, +0.19, −0.13, −0.18;
only OLMo −0.65). Reverting to greedy would preserve the effect for the wrong reason.

**What may honestly be claimed from R1:** a descriptive group contrast — under teacher
forcing, memorized spans reach the 0.1-nat convergence threshold earlier than
corpus-absent control spans (rank-biserial −0.63 to −0.85, stronger than greedy in 4/5
models) — together with the disclosure that this contrast is no larger than the
condition's difference in final-layer entropy (rank-biserial −0.67 to −0.93), does not
survive matching on predictability, and reverses sign under an alternative convergence
metric. The dose-response inside L0 (rho −0.69 to −0.93) is real and robust to
leave-one-out and to dropping the boundary position, but it is a predictability
dose-response, not a memory dose-response: L0N shows the same relation attenuated
(−0.28 to −0.68), and the L0/L0N gap does not exceed what range restriction plus sampling
noise produce.

### What would have to change to make a memory-specific claim

1. Re-run teacher-forced depth storing **all** depth variants (`kl_auc_norm`,
   `depth_argmax`, τ ∈ {0.05, 0.1, 0.5}) plus the full per-layer KL curve. A claim resting
   on one of five metrics, where a second metric of the same construct points the other
   way, is not reportable.
2. Replace the absolute τ with an **entropy-normalized or relative** criterion (e.g. the
   layer at which KL falls to a fixed fraction of the layer-0 KL). If the effect is real it
   should survive; if it is threshold-scaling it will not.
3. Fix the **overlap problem** in the design. L0 gold logprob spans −3.6 to −0.02; L0N
   spans −5.0 to −2.07. The two conditions barely overlap, so "memorized vs not" and
   "predictable vs not" cannot be separated by any statistical adjustment. The battery
   needs corpus-absent items that the model nevertheless predicts well (formulaic,
   templated, or highly compositional text with duplication count 0).
4. Treat the five models as roughly **one to two independent replications**, not five. The
   four Pythia models were run on an identical 29-item set (`--eligible-as pythia-1.4b`),
   and per-item tf_depth correlates +0.68 to +0.89 across all model pairs, per-item gold
   logprob +0.84 to +0.98. "5/5 models" is one item sample measured five times.

---


## Full test output


Generated by results/night/R1_verify_part1.py against results/night/tf_depth_*.json,
results/raw/pilot_*.json (greedy baseline), results/analysis_*.json, battery/battery.json.

## 0. Reproduction of the headline numbers

| model | n L0 | n L0N | within-L0 rho (p) | within-L0N rho (p) |
|---|---|---|---|---|
| pythia-410m | 17 | 12 | -0.877 (0.0000) | -0.550 (0.0640) |
| pythia-1b | 17 | 12 | -0.780 (0.0002) | -0.277 (0.3839) |
| pythia-1.4b | 17 | 12 | -0.843 (0.0000) | -0.677 (0.0156) |
| pythia-2.8b | 17 | 12 | -0.689 (0.0022) | -0.469 (0.1245) |
| OLMo-2-0425-1B | 20 | 11 | -0.932 (0.0000) | -0.344 (0.3001) |

Cross-check that the x-variable is unchanged between the greedy pilot and R1
(only the y-variable was supposed to change):

- pythia-410m: max |gold_tf - gold_greedy| = 0.00e+00 over 29 items
- pythia-1b: max |gold_tf - gold_greedy| = 0.00e+00 over 29 items
- pythia-1.4b: max |gold_tf - gold_greedy| = 0.00e+00 over 29 items
- pythia-2.8b: max |gold_tf - gold_greedy| = 0.00e+00 over 29 items
- OLMo-2-0425-1B: max |gold_tf - gold_greedy| = 0.00e+00 over 31 items

## 1. Fisher-z difference L0 vs L0N (Spearman SE x1.06)

| model | rho L0 (n) | rho L0N (n) | z | p | one-sided p (L0 more negative) |
|---|---|---|---|---|---|
| pythia-410m | -0.877 (17) | -0.550 (12) | -1.70 | 0.090 | 0.045 |
| pythia-1b | -0.780 (17) | -0.277 (12) | -1.73 | 0.084 | 0.042 |
| pythia-1.4b | -0.843 (17) | -0.677 (12) | -0.93 | 0.354 | 0.177 |
| pythia-2.8b | -0.689 (17) | -0.469 (12) | -0.77 | 0.443 | 0.222 |
| OLMo-2-0425-1B | -0.932 (20) | -0.344 (11) | -2.98 | 0.003 | 0.001 |

Two-sided p<.05 in 1/5 models: ['OLMo-2-0425-1B']
One-sided p<.05 in 3/5 models: ['pythia-410m', 'pythia-1b', 'OLMo-2-0425-1B']

Same test on the reviewed greedy numbers (from archive/followup_report.json T1):

| model | greedy rho L0 | greedy rho L0N | z | p |
|---|---|---|---|---|
| pythia-410m | -0.530 | -0.055 | -1.22 | 0.224 |
| pythia-1b | -0.337 | -0.054 | -0.67 | 0.500 |
| pythia-1.4b | -0.693 | -0.160 | -1.57 | 0.116 |
| pythia-2.8b | -0.419 | +0.350 | -1.85 | 0.065 |
| OLMo-2-0425-1B | -0.860 | +0.075 | -3.10 | 0.002 |

## 2. Partial Spearman controlling tf_final_entropy_mean (rank-residual)

| model | cond | n | depth~gold raw | depth~gold \| H | cap~gold raw | cap~gold \| H | gold~H |
|---|---|---|---|---|---|---|---|
| pythia-410m | L0 | 17 | -0.877 (0.000) | +0.167 (0.523) | -0.917 (0.000) | +0.208 (0.422) | -0.983 (0.000) |
| pythia-410m | L0N | 12 | -0.550 (0.064) | -0.007 (0.983) | -0.564 (0.056) | -0.063 (0.846) | -0.930 (0.000) |
| pythia-1b | L0 | 17 | -0.780 (0.000) | +0.103 (0.694) | -0.828 (0.000) | +0.130 (0.619) | -0.980 (0.000) |
| pythia-1b | L0N | 12 | -0.277 (0.384) | +0.182 (0.572) | -0.155 (0.630) | +0.021 (0.948) | -0.902 (0.000) |
| pythia-1.4b | L0 | 17 | -0.843 (0.000) | +0.125 (0.633) | -0.912 (0.000) | +0.223 (0.390) | -0.983 (0.000) |
| pythia-1.4b | L0N | 12 | -0.677 (0.016) | -0.531 (0.075) | -0.824 (0.001) | -0.755 (0.005) | -0.811 (0.001) |
| pythia-2.8b | L0 | 17 | -0.689 (0.002) | -0.147 (0.573) | -0.879 (0.000) | +0.056 (0.830) | -0.971 (0.000) |
| pythia-2.8b | L0N | 12 | -0.469 (0.124) | -0.182 (0.572) | -0.365 (0.243) | -0.098 (0.762) | -0.811 (0.001) |
| OLMo-2-0425-1B | L0 | 20 | -0.932 (0.000) | -0.501 (0.025) | -0.941 (0.000) | -0.335 (0.148) | -0.982 (0.000) |
| OLMo-2-0425-1B | L0N | 11 | -0.344 (0.300) | +0.455 (0.160) | -0.366 (0.268) | +0.309 (0.355) | -0.845 (0.001) |

## 3. Mechanical-coupling hypothesis

### 3a. Within L0N only (text verifiably absent from the training corpus index)

| model | n | gold~entropy | gold~depth | gold~frac_at_cap | entropy~depth |
|---|---|---|---|---|---|
| pythia-410m | 12 | -0.930 (0.000) | -0.550 (0.064) | -0.564 (0.056) | +0.567 (0.054) |
| pythia-1b | 12 | -0.902 (0.000) | -0.277 (0.384) | -0.155 (0.630) | +0.361 (0.249) |
| pythia-1.4b | 12 | -0.811 (0.001) | -0.677 (0.016) | -0.824 (0.001) | +0.561 (0.058) |
| pythia-2.8b | 12 | -0.811 (0.001) | -0.469 (0.124) | -0.365 (0.243) | +0.420 (0.175) |
| OLMo-2-0425-1B | 11 | -0.845 (0.001) | -0.344 (0.300) | -0.366 (0.268) | +0.683 (0.020) |

### 3b. Pooled L0+L0N: does condition add anything beyond gold_logprob?

| model | n | pooled gold~depth | gold~depth \| cond | cond~depth raw | cond~depth \| gold |
|---|---|---|---|---|---|
| pythia-410m | 29 | -0.838 (0.000) | -0.779 (0.000) | -0.594 (0.001) | -0.153 (0.429) |
| pythia-1b | 29 | -0.726 (0.000) | -0.608 (0.000) | -0.540 (0.003) | -0.054 (0.782) |
| pythia-1.4b | 29 | -0.869 (0.000) | -0.796 (0.000) | -0.594 (0.001) | +0.160 (0.407) |
| pythia-2.8b | 29 | -0.828 (0.000) | -0.616 (0.000) | -0.644 (0.000) | +0.151 (0.435) |
| OLMo-2-0425-1B | 31 | -0.888 (0.000) | -0.771 (0.000) | -0.703 (0.000) | -0.235 (0.202) |

### 3c. Rank-space variance decomposition (depth ranks ~ gold ranks + condition dummy)

| model | R2 gold only | R2 gold+cond | dR2 cond | R2 cond only | dR2 gold |
|---|---|---|---|---|---|
| pythia-410m | 0.702 | 0.708 | +0.006 | 0.353 | +0.355 |
| pythia-1b | 0.527 | 0.530 | +0.002 | 0.292 | +0.238 |
| pythia-1.4b | 0.755 | 0.768 | +0.014 | 0.353 | +0.415 |
| pythia-2.8b | 0.685 | 0.690 | +0.005 | 0.415 | +0.275 |
| OLMo-2-0425-1B | 0.788 | 0.799 | +0.011 | 0.494 | +0.305 |

## 4. L0 vs L0N contrast (rank-biserial), tf vs greedy

| model | tf depth | tf frac_at_cap | greedy depth_tau_0.1 | tf entropy | greedy entropy |
|---|---|---|---|---|---|
| pythia-410m | -0.696 | -0.672 | -0.377 | -0.667 | -0.598 |
| pythia-1b | -0.632 | -0.647 | -0.691 | -0.716 | -0.647 |
| pythia-1.4b | -0.696 | -0.755 | -0.368 | -0.814 | -0.627 |
| pythia-2.8b | -0.755 | -0.843 | -0.627 | -0.941 | -0.931 |
| OLMo-2-0425-1B | -0.845 | -0.891 | -0.686 | -0.927 | -0.845 |

Group medians (tf):

| model | cap value | L0 depth med | L0N depth med | L0 cap-frac med | L0N cap-frac med |
|---|---|---|---|---|---|
| pythia-410m | 0.9568 | 0.9333 | 0.9524 | 0.745 | 0.908 |
| pythia-1b | 0.9364 | 0.9125 | 0.9341 | 0.764 | 0.946 |
| pythia-1.4b | 0.9576 | 0.9427 | 0.9560 | 0.854 | 0.954 |
| pythia-2.8b | 0.9671 | 0.9460 | 0.9604 | 0.679 | 0.863 |
| OLMo-2-0425-1B | 0.9375 | 0.9138 | 0.9364 | 0.688 | 0.982 |

## 5. ADVERSARIAL: is the L0/L0N difference just range restriction?

| model | cond | gold min | gold max | gold SD | gold IQR | depth SD | cap-frac SD |
|---|---|---|---|---|---|---|---|
| pythia-410m | L0 | -3.586 | -0.056 | 1.357 | 2.461 | 0.0269 | 0.1802 |
| pythia-410m | L0N | -4.380 | -2.420 | 0.547 | 0.646 | 0.0075 | 0.0622 |
| pythia-1b | L0 | -3.276 | -0.054 | 1.303 | 2.440 | 0.0307 | 0.1607 |
| pythia-1b | L0N | -4.042 | -2.236 | 0.510 | 0.728 | 0.0068 | 0.0355 |
| pythia-1.4b | L0 | -2.966 | -0.047 | 1.164 | 2.256 | 0.0274 | 0.1187 |
| pythia-1.4b | L0N | -3.869 | -2.143 | 0.524 | 0.811 | 0.0053 | 0.0608 |
| pythia-2.8b | L0 | -2.556 | -0.020 | 0.784 | 0.856 | 0.0186 | 0.1554 |
| pythia-2.8b | L0N | -3.761 | -2.072 | 0.488 | 0.599 | 0.0063 | 0.0875 |
| OLMo-2-0425-1B | L0 | -3.280 | -0.032 | 1.084 | 1.083 | 0.0176 | 0.2045 |
| OLMo-2-0425-1B | L0N | -4.958 | -2.352 | 0.759 | 0.826 | 0.0031 | 0.0206 |

### 5a. Restrict L0 to the L0N gold range, recompute within-L0 rho

| model | L0N gold range | n L0 kept | rho (p) | full-L0 rho |
|---|---|---|---|---|
| pythia-410m | [-4.38, -2.42] | 7 | -0.250 (0.589) | -0.877 |
| pythia-1b | [-4.04, -2.24] | 6 | +0.657 (0.156) | -0.780 |
| pythia-1.4b | [-3.87, -2.14] | 5 | -0.700 (0.188) | -0.843 |
| pythia-2.8b | [-3.76, -2.07] | 2 | n<5 | -0.689 |
| OLMo-2-0425-1B | [-4.96, -2.35] | 4 | n<5 | -0.932 |

### 5b. Symmetric check: L0N is a rank-restricted slice. What does an L0 slice of the
same *rank width* give? (bootstrap: draw n_L0N contiguous-in-gold L0 items, all windows)

| model | n windows | median window rho | frac windows with p<.05 | L0N rho |
|---|---|---|---|---|
| pythia-410m | 6 | -0.846 | 1.00 | -0.550 |
| pythia-1b | 6 | -0.762 | 1.00 | -0.277 |
| pythia-1.4b | 6 | -0.745 | 1.00 | -0.677 |
| pythia-2.8b | 6 | -0.542 | 0.33 | -0.469 |
| OLMo-2-0425-1B | 10 | -0.882 | 1.00 | -0.344 |

## 6. ADVERSARIAL: robustness of the within-L0 tf correlation

### 6a. Leave-one-out range of within-L0 rho

| model | full rho | LOO min | LOO max | worst-case p | most influential item |
|---|---|---|---|---|---|
| pythia-410m | -0.877 | -0.921 | -0.853 | 0.0000 | L0-08 |
| pythia-1b | -0.780 | -0.799 | -0.741 | 0.0010 | L0-16 |
| pythia-1.4b | -0.843 | -0.897 | -0.818 | 0.0001 | L0-15 |
| pythia-2.8b | -0.689 | -0.753 | -0.626 | 0.0094 | L0-15 |
| OLMo-2-0425-1B | -0.932 | -0.972 | -0.921 | 0.0000 | L0-04 |

### 6b. Ceiling saturation: how much of the depth signal is just 'never converged'?

| model | rho(depth, frac_at_cap) all | items with cap-frac=1.0 (L0/L0N) | depth unique values L0 |
|---|---|---|---|
| pythia-410m | +0.917 (p=2.9e-12) | 0/17 and 0/12 | 17 |
| pythia-1b | +0.959 (p=2.3e-16) | 0/17 and 0/12 | 16 |
| pythia-1.4b | +0.967 (p=1.3e-17) | 0/17 and 0/12 | 17 |
| pythia-2.8b | +0.865 (p=1.5e-09) | 0/17 and 0/12 | 17 |
| OLMo-2-0425-1B | +0.986 (p=4.9e-24) | 0/20 and 5/11 | 20 |

### 6c. Do the tf and greedy depth measures even agree per item? (within L0)

| model | n | rho(tf_depth, greedy_depth) | tf mean | greedy mean |
|---|---|---|---|---|
| pythia-410m | 17 | +0.630 (p=0.007) | 0.9282 | 0.9243 |
| pythia-1b | 17 | +0.544 (p=0.024) | 0.9060 | 0.8824 |
| pythia-1.4b | 17 | +0.652 (p=0.005) | 0.9351 | 0.9335 |
| pythia-2.8b | 17 | +0.579 (p=0.015) | 0.9418 | 0.9249 |
| OLMo-2-0425-1B | 20 | +0.933 (p=0.000) | 0.9127 | 0.9125 |

### 6d. Item-level non-independence across the 5 'replications'

Items shared by all 5 models: 28

Pairwise Spearman of per-item tf_depth across models (all L0+L0N shared items):

| | pythia-410m | pythia-1b | pythia-1.4b | pythia-2.8b | OLMo-2-0425-1B |
|---|---|---|---|---|---|
| pythia-410m | +1.00 | +0.85 | +0.89 | +0.84 | +0.81 |
| pythia-1b | +0.85 | +1.00 | +0.68 | +0.75 | +0.86 |
| pythia-1.4b | +0.89 | +0.68 | +1.00 | +0.82 | +0.77 |
| pythia-2.8b | +0.84 | +0.75 | +0.82 | +1.00 | +0.79 |
| OLMo-2-0425-1B | +0.81 | +0.86 | +0.77 | +0.79 | +1.00 |

Pairwise Spearman of per-item gold_logprob across models:

| | pythia-410m | pythia-1b | pythia-1.4b | pythia-2.8b | OLMo-2-0425-1B |
|---|---|---|---|---|---|
| pythia-410m | +1.00 | +0.98 | +0.96 | +0.84 | +0.90 |
| pythia-1b | +0.98 | +1.00 | +0.98 | +0.90 | +0.92 |
| pythia-1.4b | +0.96 | +0.98 | +1.00 | +0.94 | +0.96 |
| pythia-2.8b | +0.84 | +0.90 | +0.94 | +1.00 | +0.97 |
| OLMo-2-0425-1B | +0.90 | +0.92 | +0.96 | +0.97 | +1.00 |

## 7. Corpus duplication count (memory measure independent of the forward pass)

| model | n | dupcount~tf_depth | dupcount~gold | dup~tf_depth \| gold |
|---|---|---|---|---|
| pythia-410m | 17 | -0.588 (0.013) | +0.691 (0.002) | -0.130 (0.619) |
| pythia-1b | 17 | -0.538 (0.026) | +0.730 (0.001) | -0.042 (0.874) |
| pythia-1.4b | 17 | -0.532 (0.028) | +0.679 (0.003) | +0.069 (0.794) |
| pythia-2.8b | 17 | -0.600 (0.011) | +0.645 (0.005) | -0.306 (0.232) |
| OLMo-2-0425-1B | 20 | -0.690 (0.001) | +0.711 (0.000) | -0.059 (0.806) |

## 8. Nuisance: gold span length

| model | cond | n_gold_pos~depth | n_gold_pos~gold | depth~gold \| n_pos |
|---|---|---|---|---|
| pythia-410m | L0 | +0.164 (0.530) | -0.316 (0.216) | -0.868 (0.000) |
| pythia-410m | L0N | +0.544 (0.067) | -0.195 (0.543) | -0.564 (0.056) |
| pythia-1b | L0 | +0.061 (0.815) | -0.326 (0.201) | -0.826 (0.000) |
| pythia-1b | L0N | -0.137 (0.671) | -0.227 (0.479) | -0.280 (0.378) |
| pythia-1.4b | L0 | +0.113 (0.667) | -0.304 (0.236) | -0.821 (0.000) |
| pythia-1.4b | L0N | +0.310 (0.327) | -0.289 (0.362) | -0.646 (0.023) |
| pythia-2.8b | L0 | +0.205 (0.430) | -0.288 (0.263) | -0.674 (0.003) |
| pythia-2.8b | L0N | +0.340 (0.280) | -0.156 (0.628) | -0.483 (0.112) |
| OLMo-2-0425-1B | L0 | +0.082 (0.732) | -0.083 (0.729) | -0.937 (0.000) |
| OLMo-2-0425-1B | L0N | -0.067 (0.846) | -0.396 (0.228) | -0.423 (0.195) |

## A. Is 'memory strength' distinguishable from 'final-layer confidence'?

Teacher-forced mean gold logprob is the negative cross-entropy of the model on the
gold span; tf_final_entropy_mean is the mean entropy of the same predictive
distributions. For a model that is accurate on those tokens the two are the same
number up to the calibration gap, so they cannot be treated as independent variables.

| model | cond | rho(gold, -H) | mean(gold + H) | sd(gold + H) | OLS slope gold on -H | R2 |
|---|---|---|---|---|---|---|
| pythia-410m | L0 | +0.983 | +0.302 | 0.237 | +1.132 | 0.983 |
| pythia-410m | L0N | +0.930 | -0.151 | 0.248 | +1.424 | 0.872 |
| pythia-1b | L0 | +0.980 | +0.292 | 0.180 | +0.973 | 0.982 |
| pythia-1b | L0N | +0.902 | -0.056 | 0.262 | +1.181 | 0.753 |
| pythia-1.4b | L0 | +0.983 | +0.311 | 0.191 | +0.922 | 0.980 |
| pythia-1.4b | L0N | +0.811 | -0.049 | 0.280 | +1.210 | 0.737 |
| pythia-2.8b | L0 | +0.971 | +0.341 | 0.315 | +0.796 | 0.897 |
| pythia-2.8b | L0N | +0.811 | -0.088 | 0.285 | +1.014 | 0.658 |
| OLMo-2-0425-1B | L0 | +0.982 | +0.109 | 0.326 | +1.052 | 0.912 |
| OLMo-2-0425-1B | L0N | +0.845 | -0.028 | 0.354 | +1.101 | 0.789 |

### A1. Swap the predictor: does entropy predict depth as well as gold logprob does?

| model | cond | rho(gold, depth) | rho(H, depth) | rho(gold,depth \| H) | rho(H,depth \| gold) |
|---|---|---|---|---|---|
| pythia-410m | L0 | -0.877 (0.000) | +0.897 (0.000) | +0.167 (0.523) | +0.275 (0.286) |
| pythia-410m | L0N | -0.550 (0.064) | +0.567 (0.054) | -0.007 (0.983) | +0.063 (0.846) |
| pythia-1b | L0 | -0.780 (0.000) | +0.806 (0.000) | +0.103 (0.694) | +0.199 (0.445) |
| pythia-1b | L0N | -0.277 (0.384) | +0.361 (0.249) | +0.182 (0.572) | +0.399 (0.199) |
| pythia-1.4b | L0 | -0.843 (0.000) | +0.863 (0.000) | +0.125 (0.633) | +0.331 (0.195) |
| pythia-1.4b | L0N | -0.677 (0.016) | +0.561 (0.058) | -0.531 (0.075) | +0.196 (0.542) |
| pythia-2.8b | L0 | -0.689 (0.002) | +0.701 (0.002) | -0.147 (0.573) | +0.238 (0.358) |
| pythia-2.8b | L0N | -0.469 (0.124) | +0.420 (0.175) | -0.182 (0.572) | +0.098 (0.762) |
| OLMo-2-0425-1B | L0 | -0.932 (0.000) | +0.908 (0.000) | -0.501 (0.025) | +0.075 (0.753) |
| OLMo-2-0425-1B | L0N | -0.344 (0.300) | +0.683 (0.020) | +0.455 (0.160) | +0.745 (0.008) |

## B. Matched-predictability test: at equal gold logprob, does memorized text
converge earlier?

Each L0N item is matched to its nearest L0 item in gold logprob (with replacement);
pairs with |dgold| > 0.5 nats are dropped as unmatchable. Positive dDepth means the
memorized item converged LATER, i.e. against the hypothesis.

| model | n pairs | mean \|dgold\| | median dDepth (L0 - L0N) | Wilcoxon p | median dCap |
|---|---|---|---|---|---|
| pythia-410m | 10 | 0.069 | -0.0033 | 0.232 | +0.012 |
| pythia-1b | 11 | 0.153 | -0.0088 | 0.206 | -0.049 |
| pythia-1.4b | 8 | 0.130 | +0.0013 | 0.195 | +0.004 |
| pythia-2.8b | 6 | 0.170 | +0.0033 | 0.156 | -0.027 |
| OLMo-2-0425-1B | 9 | 0.138 | -0.0023 | 0.312 | -0.019 |

## C. Range restriction, matched on gold SPREAD (nats) rather than item count

For every contiguous-in-gold window of L0 items whose gold spread is within 25% of
the L0N spread for that model, report the within-window rho.

| model | L0N spread | L0N rho | n matched windows | median window rho | window rho range |
|---|---|---|---|---|---|
| pythia-410m | 1.96 | -0.550 | 24 | -0.785 | [-1.000, -0.536] |
| pythia-1b | 1.81 | -0.277 | 18 | -0.707 | [-1.000, -0.400] |
| pythia-1.4b | 1.73 | -0.677 | 24 | -0.646 | [-0.833, +0.300] |
| pythia-2.8b | 1.69 | -0.469 | 20 | -0.360 | [-0.700, +0.029] |
| OLMo-2-0425-1B | 2.61 | -0.344 | 55 | -0.921 | [-0.939, -0.371] |

## D. Bootstrap CI on the Fisher-z difference (asymptotic SE is optimistic at n=11-17)

| model | dz observed | bootstrap 95% CI | frac bootstrap dz < 0 |
|---|---|---|---|
| pythia-410m | -0.746 | [-1.807, +0.212] | 0.935 |
| pythia-1b | -0.761 | [-1.625, +0.288] | 0.924 |
| pythia-1.4b | -0.408 | [-1.296, +0.563] | 0.805 |
| pythia-2.8b | -0.337 | [-1.330, +1.373] | 0.703 |
| OLMo-2-0425-1B | -1.317 | [-2.699, -0.317] | 0.994 |

## E. Convention check: does my rank-biserial reproduce the frozen greedy contrasts?

| model | my greedy depth rb | frozen analysis value | my greedy entropy rb | frozen |
|---|---|---|---|---|
| pythia-410m | -0.377 | -0.377 | -0.598 | -0.598 |
| pythia-1b | -0.691 | -0.691 | -0.647 | -0.647 |
| pythia-1.4b | -0.368 | -0.368 | -0.627 | -0.627 |
| pythia-2.8b | -0.627 | -0.627 | -0.931 | -0.931 |
| OLMo-2-0425-1B | -0.686 | -0.686 | -0.845 | -0.845 |

## F. Effect size of the memory-specific residual (what is left after predictability)

Difference of Fisher-z (L0 minus L0N) is the memory-specific increment in the
dose-response slope; dR2 of condition in the pooled rank model is the memory-specific
increment in explained variance. Both, side by side, tf vs greedy.

| model | tf dz (L0-L0N) | greedy dz | tf dR2 cond | tf partial rho cond\|gold |
|---|---|---|---|---|
| pythia-410m | -0.746 | -0.535 | +0.0063 | -0.153 (p=0.429) |
| pythia-1b | -0.761 | -0.297 | +0.0024 | -0.054 (p=0.782) |
| pythia-1.4b | -0.408 | -0.692 | +0.0136 | +0.160 (p=0.407) |
| pythia-2.8b | -0.337 | -0.812 | +0.0054 | +0.151 (p=0.435) |
| OLMo-2-0425-1B | -1.317 | -1.368 | +0.0109 | -0.235 (p=0.202) |

## G. Is the dose-response specific to the tau=0.1 metric? (greedy data, where all
depth variants were stored; the R1 script saved only depth_tau_0.1)

Within L0, rho(gold_logprob, metric). A genuine 'memory retrieves earlier in the
stack' effect should show up in every convergence metric with a consistent sign.

| model | cond | n | tau=0.05 | tau=0.1 | tau=0.5 | argmax | kl_auc_norm |
|---|---|---|---|---|---|---|---|
| pythia-410m | L0 | 17 | -0.477 (0.053) | -0.530 (0.029) | -0.313 (0.221) | -0.380 (0.132) | +0.713 (0.001) |
| pythia-410m | L0N | 12 | -0.008 (0.979) | -0.055 (0.866) | -0.182 (0.572) | -0.147 (0.649) | +0.287 (0.366) |
| pythia-1b | L0 | 17 | -0.315 (0.218) | -0.337 (0.186) | -0.231 (0.372) | +0.114 (0.663) | +0.797 (0.000) |
| pythia-1b | L0N | 12 | -0.131 (0.685) | -0.054 (0.868) | -0.809 (0.001) | -0.713 (0.009) | +0.049 (0.880) |
| pythia-1.4b | L0 | 17 | -0.773 (0.000) | -0.693 (0.002) | -0.339 (0.184) | -0.130 (0.619) | +0.770 (0.000) |
| pythia-1.4b | L0N | 12 | +0.037 (0.910) | -0.160 (0.619) | -0.007 (0.983) | -0.368 (0.240) | +0.063 (0.846) |
| pythia-2.8b | L0 | 17 | -0.353 (0.165) | -0.419 (0.094) | -0.309 (0.227) | -0.550 (0.022) | +0.571 (0.017) |
| pythia-2.8b | L0N | 12 | +0.638 (0.025) | +0.350 (0.265) | +0.208 (0.517) | -0.386 (0.215) | +0.280 (0.379) |
| OLMo-2-0425-1B | L0 | 20 | -0.862 (0.000) | -0.860 (0.000) | -0.781 (0.000) | -0.374 (0.104) | +0.511 (0.021) |
| OLMo-2-0425-1B | L0N | 11 | const | +0.075 (0.828) | -0.550 (0.079) | -0.366 (0.268) | -0.055 (0.873) |

Same for the L0-vs-L0N rank-biserial contrast (frozen analysis values):

| model | tau=0.05 | tau=0.1 | tau=0.5 | argmax | kl_auc_norm | final_entropy |
|---|---|---|---|---|---|---|
| pythia-410m | -0.397 | -0.377 | -0.255 | -0.348 | +0.225 | -0.598 |
| pythia-1b | -0.662 | -0.691 | -0.623 | -0.265 | +0.127 | -0.647 |
| pythia-1.4b | -0.221 | -0.368 | -0.167 | -0.044 | +0.186 | -0.627 |
| pythia-2.8b | -0.583 | -0.627 | -0.598 | -0.186 | +0.608 | -0.931 |
| OLMo-2-0425-1B | -0.600 | -0.686 | -0.655 | -0.168 | +0.355 | -0.845 |

## H. Did teacher forcing tighten or loosen the gold/entropy collinearity?

In the greedy design the entropy came from a different token sequence than the gold
logprob, so the two were only loosely coupled. Teacher forcing put them on the same
tokens in the same forward pass.

| model | cond | greedy rho(gold, H) | tf rho(gold, H) |
|---|---|---|---|
| pythia-410m | L0 | -0.745 | -0.983 |
| pythia-410m | L0N | -0.601 | -0.930 |
| pythia-1b | L0 | -0.799 | -0.980 |
| pythia-1b | L0N | -0.692 | -0.902 |
| pythia-1.4b | L0 | -0.787 | -0.983 |
| pythia-1.4b | L0N | -0.336 | -0.811 |
| pythia-2.8b | L0 | -0.821 | -0.971 |
| pythia-2.8b | L0N | -0.231 | -0.811 |
| OLMo-2-0425-1B | L0 | -0.699 | -0.982 |
| OLMo-2-0425-1B | L0N | -0.409 | -0.845 |

## I. Robustness: drop the first gold position (the prompt boundary)

| model | cond | n | rho all positions | rho excluding position 0 |
|---|---|---|---|---|
| pythia-410m | L0 | 17 | -0.877 (0.000) | -0.877 (0.000) |
| pythia-410m | L0N | 12 | -0.550 (0.064) | -0.580 (0.048) |
| pythia-1b | L0 | 17 | -0.780 (0.000) | -0.780 (0.000) |
| pythia-1b | L0N | 12 | -0.277 (0.384) | -0.277 (0.384) |
| pythia-1.4b | L0 | 17 | -0.843 (0.000) | -0.843 (0.000) |
| pythia-1.4b | L0N | 12 | -0.677 (0.016) | -0.651 (0.022) |
| pythia-2.8b | L0 | 17 | -0.689 (0.002) | -0.689 (0.002) |
| pythia-2.8b | L0N | 12 | -0.469 (0.124) | -0.469 (0.124) |
| OLMo-2-0425-1B | L0 | 20 | -0.932 (0.000) | -0.944 (0.000) |
| OLMo-2-0425-1B | L0N | 11 | -0.344 (0.300) | -0.344 (0.300) |

---

## Reading of each test

### Test 1 — Fisher-z, L0 vs L0N

The R1 headline is that within-L0 rho became much more negative under teacher forcing
(−0.69 to −0.93, all p≤.0022) while within-L0N is no longer flat. The correct
comparison is the *difference*, and it is weak. Asymptotic two-sided Fisher-z with the
1.06 Spearman SE correction clears p<.05 in **1/5 models** (OLMo, p=.003); one-sided in
3/5. The 20,000-draw bootstrap of Δz, which is the honest test at n=11–17, gives 95% CIs
that exclude zero only for OLMo. The greedy version cleared two-sided p<.05 in 1/5 as
well (OLMo, p=.002). So on this test R1 changed nothing: the memory-specific claim rested
on one model before and rests on the same one model now.

Worse, the direction of change is inconsistent. Δz went from −0.535 to −0.746 (410m) and
−0.297 to −0.761 (1b), but from −0.692 to −0.408 (1.4b) and −0.812 to −0.337 (2.8b). The
statement "the effect got stronger" is not supported for the dissociation; it is supported
only for the within-L0 slope, which is the part that is confounded.

### Test 2 — Partial controlling final entropy

Within L0 the partial collapses to +0.17, +0.10, +0.13, −0.15 for the four Pythias
(all p>.5) and −0.50 (p=.025) for OLMo. Within L0N it is −0.01, +0.18, −0.53, −0.18, +0.46,
all non-significant except nothing. `tf_frac_at_cap` behaves the same way.

This must be read carefully in both directions. The collinearity is so severe
(rho(gold, H) = −0.97 to −0.98 within L0) that the partial is near-degenerate — it is
subtracting the predictor from itself, and its near-zero value is not by itself proof of
no effect. But that degeneracy *is* the finding: at this level of collinearity, the design
cannot separate the two constructs, and any claim that distinguishes them is unfalsifiable
with this data. Test A1 makes the symmetry explicit: entropy predicts depth exactly as
well as gold logprob does (+0.70 to +0.91 vs −0.69 to −0.93), and each partials the other
to null. There is one variable here, not two.

The only cell that behaves differently is OLMo L0, where the partial stays at −0.50
(p=.025) — the same model that was the sole survivor in the greedy T3 (−0.654). Whatever
is going on with OLMo, it is a single-model result on 20 items.

### Test 3 — Mechanical coupling

Within L0N — text with a verified duplication count of 0 in the training corpus index, so
nothing to retrieve — gold logprob still predicts final entropy at −0.81 to −0.93 and
predicts depth at −0.28 to −0.68, same sign as L0 in 5/5, reaching p<.05 in one model
(1.4b, −0.677) and p=.064 in another. Generic predictability does couple to convergence
depth in non-memorized text. The coupling is weaker than in L0, but L0N's gold logprob has
half the spread of L0's (SD 0.49–0.76 vs 0.78–1.36) and its depth has a quarter to a sixth
the spread (SD 0.003–0.008 vs 0.018–0.031), so attenuation is expected.

Pooling L0+L0N settles the apportionment. Gold logprob alone explains 53–79% of depth rank
variance. Adding the condition dummy adds **0.2% to 1.4%**. Reversing the order, condition
alone explains 29–49% and gold adds 24–42% on top. The partial rho of condition given gold
is −0.235 to +0.160, non-significant in 5/5, and flips sign across models. A generic
predictability account explains essentially all of it; memorization status is a proxy for
predictability here and contributes nothing detectable of its own.

The matched-pair test (section B) is the cleanest statement of the same result: pairing
each L0N item with the nearest L0 item in gold logprob (mean gap 0.07–0.17 nats) gives
median depth differences of −0.009 to +0.003 against within-condition SDs of 0.02–0.03,
Wilcoxon p = .156–.312, sign inconsistent across models. At matched predictability,
memorized and non-memorized spans converge at the same layer.

Section 7 adds an independent memory measure that is not a forward-pass quantity at all:
median corpus duplication count of the item's probes. It correlates with tf_depth at −0.53
to −0.69 (p<.03 in 5/5), which looks supportive — until you control for gold logprob, at
which point it is −0.31 to +0.07, p>.23 in 5/5. Training-set duplication predicts
convergence depth only through the model's confidence.

### Test 4 — Contrast, tf vs greedy

The contrast did strengthen, in 4/5 models: rank-biserial on depth went −0.377→−0.696
(410m), −0.368→−0.696 (1.4b), −0.627→−0.755 (2.8b), −0.686→−0.845 (OLMo), and weakened
only for pythia-1b (−0.691→−0.632). `tf_frac_at_cap` gives a similar or slightly larger
contrast (−0.65 to −0.89). My rank-biserial implementation reproduces the frozen greedy
values to three decimals (section E), so this comparison is apples-to-apples.

But the entropy contrast strengthened in lockstep and is as large or larger in every model
(tf: −0.667, −0.716, −0.814, −0.941, −0.927; greedy: −0.598, −0.647, −0.627, −0.931,
−0.845). The depth contrast never exceeds the entropy contrast. Combined with the null
matched-pair test, the honest reading is that teacher forcing sharpened the *confidence*
separation between the two conditions and the depth contrast followed it.

### Sections 5 and C — range restriction

L0N is a restricted slice: gold spread 1.69–2.61 nats against L0's 2.5–3.5, and depth SD
3–6× smaller. Restricting L0 to the L0N gold range destroys the within-L0 correlation
(−0.25, +0.66, −0.70, and n<5 for two models) — but n falls to 2–7, so this proves
nothing either way.

The better-powered version (section C) takes every contiguous-in-gold L0 window whose gold
*spread in nats* is within 25% of L0N's, and finds median window rho −0.785, −0.707,
−0.646, −0.360, −0.921. For pythia-1.4b (L0N −0.677 vs window median −0.646) and
pythia-2.8b (−0.469 vs −0.360), L0N sits squarely inside the range-matched L0 distribution:
those two models show **no dissociation at all** once spread is matched. For 410m, 1b and
OLMo, L0N is weaker than the range-matched L0 windows, so range restriction is not the
whole story there. Range restriction is a genuine partial explanation, not a complete one —
which is consistent with Test 1's finding that the dissociation is real only in OLMo.

### Section 6 — measure quality and robustness

The within-L0 correlation is not fragile: leave-one-out keeps rho between −0.63 and −0.97
with worst-case p=.0094, and dropping the first gold position changes nothing (section I).
The problem with R1 is not noise, it is interpretation.

`tf_depth` and `tf_frac_at_cap` are near-redundant (rho +0.87 to +0.99), so the depth
measure is largely "what fraction of positions never converged before the last layer" — a
censored quantity. Five of eleven OLMo L0N items sit at frac=1.0 exactly, which is a
partial ceiling on the control group specifically.

The tf and greedy depth measures agree only moderately per item within L0 (+0.54 to +0.65
for the Pythias, +0.93 for OLMo). They are not interchangeable measurements of one
construct, so "the same effect, measured better" overstates the relationship between the
two runs.

Section 8 rules out gold-span length as the driver: `n_gold_positions` correlates with
depth at +0.06 to +0.21 within L0 (all p>.4), and the depth~gold partial controlling
length is −0.67 to −0.94, essentially unchanged.

### Section H — what teacher forcing actually did

| | greedy rho(gold, H) | tf rho(gold, H) |
|---|---|---|
| L0, across models | −0.70 to −0.82 | **−0.97 to −0.98** |
| L0N, across models | −0.23 to −0.69 | **−0.81 to −0.93** |

The reviewed pilot's two-span confound was also, incidentally, what kept the predictor and
the nuisance variable partially independent. R1 closed the two-span gap and in the same
move made the predictor and the nuisance variable the same measurement. This is the single
most important structural fact about R1 and it should lead the write-up of the result.

---

# R5 — range matching on the reviewed (greedy) measure

Follow-up scoped to the frozen draft rather than to R1. Script:
`results/night/R5_range_matching.py`; raw output reproduced below.

## Verdict for the draft

**Yes — the "matched control stayed flat" sentence needs a caveat, and it needs two of
them, not one.** The control condition is handicapped on both axes at once: its gold
logprob spans only 13–27% of the two conditions' combined range, and its depth values
are so heavily tied that in two models the correlation it is being compared against is
not attainable in it.

Scoring the five models on the requested test — is the observed L0N rho flatter than a
gold-spread-matched contiguous slice of L0?

| model | L0N rho | median matched-window rho | share of matched windows as flat as L0N | survives range matching? |
|---|---|---|---|---|
| OLMo-2-0425-1B | +0.075 | −0.778 | 0/55 | **yes**, but see the tie ceiling below |
| pythia-1.4b | −0.160 | −0.615 | 2/24 (8%) | marginal |
| pythia-1b | −0.054 | −0.379 | 2/18 (11%) | marginal, and the dissociation was never significant (Fisher-z p=.50) |
| pythia-410m | −0.055 | −0.418 | 5/24 (21%) | **no** |
| pythia-2.8b | +0.350 | −0.095 | 5/20 (25%) | **no** — the range-matched L0 slice is itself flat |

So the greedy dissociation survives range matching in **one model cleanly (OLMo), one
marginally (pythia-1.4b), and fails in three**. This tracks the Fisher-z result exactly:
OLMo was the only model where the L0-vs-L0N difference was significant to begin with
(p=.002 greedy, p=.003 tf, and the only bootstrap CI excluding zero). Range matching does
not overturn the paper's strongest case; it removes the impression that four weaker cases
point the same way.

For pythia-2.8b the finding is stronger than a caveat. Restrict L0 to a slice of the
predictability range the size of L0N's and the L0 correlation goes to −0.095 — the
memorized condition stops showing a dose-response too. Nothing distinguishes the
conditions in that model once the comparison is fair.

### The second caveat, which is the more serious one

`depth_tau_0.1` is a per-item mean of a variable that takes n_layers discrete values, and
in the control condition almost every position sits at the cap. The per-item means
therefore collapse onto very few distinct values:

| model | L0N distinct depth values | largest tie group | max attainable \|rho\| in L0N | observed \|rho\| in L0 |
|---|---|---|---|---|
| OLMo-2-0425-1B | 2 | 9/11 | **0.671** | **0.860** |
| pythia-1b | 3 | 10/12 | **0.650** | 0.337 |
| pythia-410m | 5 | 7/12 | 0.895 | 0.530 |
| pythia-1.4b | 5 | 7/12 | 0.895 | 0.693 |
| pythia-2.8b | 5 | 6/12 | 0.929 | 0.419 |

"Max attainable" is the Spearman obtained by arranging that condition's own observed depth
values in perfect monotone order against gold — the ceiling its tie structure imposes.

For OLMo, the flagship result, **the L0 correlation of −0.860 could not have been produced
in L0N under any arrangement of the control's data** (ceiling 0.671). The two coefficients
are not on the same scale, so contrasting them — including via Fisher-z, which assumes
they are — overstates the dissociation by an unknown amount. Reporting −0.860 against
+0.075 without saying that the second number was capped near ±0.67 is the kind of omission
a reviewer will find. The same applies to pythia-1b, though there the ceiling does not bind
(observed 0.337 < ceiling 0.650), so its comparison is fair even if its effect is null.

### What does *not* need a caveat

Depth granularity alone does not manufacture the flat control. Snapping the L0 depths onto
the control's own coarse depth grid, leaving gold untouched, leaves the L0 correlation
essentially intact (−0.569, −0.328, −0.661, −0.429, −0.631; still p<.05 in 3/5). A real
relation of L0's strength would have shown through the control's grid. The flatness is
about the control's restricted predictability range and its tie ceiling, not about the
measure being too coarse to see anything anywhere.

Applying both handicaps at once (control grid plus control gold range) leaves n=2–7 and
the estimates become undefined or meaningless (+0.045, −0.239, +0.527, and two undefined).
That is a statement about power, not evidence of absence, and should not be cited either
way.

### Suggested wording for the draft

Replace "the matched control stayed flat" with something that survives scrutiny, e.g.:
the control condition shows no dose-response (rho −0.16 to +0.35, none significant), but
it occupies only the low-predictability tail of the gold-logprob range (13–27% overlap
with the memorized condition) and its depth values are tied enough to cap the attainable
correlation at 0.65–0.93. A gold-spread-matched slice of the memorized condition is
similarly flat in pythia-410m and pythia-2.8b; the dissociation is robust to range
matching only in OLMo-2-0425-1B, where the tie ceiling in the control (0.671) is below the
correlation observed in the memorized condition (0.860) and the two coefficients are
therefore not directly comparable.

---

## R5 full test output


Scope: the frozen pilot values only. y = `depth.depth_tau_0.1` from
`results/raw/pilot_*.json` (the per-position mean as shipped), x =
`gold_logprob_per_token`. These are the numbers behind `followup_report.json` T1 and
the draft's "matched control stayed flat" sentence. Item sets are identical to the
R1 teacher-forced run (verified id-by-id), so nothing below is a sampling difference.

### R5.0 Baseline (reproduces followup_report.json T1)

| model | n L0 | rho L0 (p) | n L0N | rho L0N (p) |
|---|---|---|---|---|
| pythia-410m | 17 | -0.530 (0.0286) | 12 | -0.055 (0.8659) |
| pythia-1b | 17 | -0.337 (0.1864) | 12 | -0.054 (0.8682) |
| pythia-1.4b | 17 | -0.693 (0.0020) | 12 | -0.160 (0.6189) |
| pythia-2.8b | 17 | -0.419 (0.0937) | 12 | +0.350 (0.2649) |
| OLMo-2-0425-1B | 20 | -0.860 (0.0000) | 11 | +0.075 (0.8276) |

### R5.a Gold-logprob spread and overlap, L0 vs L0N

| model | L0 range | L0 spread | L0 SD | L0N range | L0N spread | L0N SD | overlap | L0 items in L0N range |
|---|---|---|---|---|---|---|---|---|
| pythia-410m | [-3.59, -0.06] | 3.53 | 1.36 | [-4.38, -2.42] | 1.96 | 0.55 | 1.17 nats (27% of union) | 7/17 |
| pythia-1b | [-3.28, -0.05] | 3.22 | 1.30 | [-4.04, -2.24] | 1.81 | 0.51 | 1.04 nats (26% of union) | 6/17 |
| pythia-1.4b | [-2.97, -0.05] | 2.92 | 1.16 | [-3.87, -2.14] | 1.73 | 0.52 | 0.82 nats (22% of union) | 5/17 |
| pythia-2.8b | [-2.56, -0.02] | 2.54 | 0.78 | [-3.76, -2.07] | 1.69 | 0.49 | 0.48 nats (13% of union) | 2/17 |
| OLMo-2-0425-1B | [-3.28, -0.03] | 3.25 | 1.08 | [-4.96, -2.35] | 2.61 | 0.76 | 0.93 nats (19% of union) | 4/20 |

The gold logprob values are byte-identical between the greedy pilot and R1 (the
teacher-forced run recomputed the same quantity and reproduced it to 0.00e+00), so
the non-overlap is a property of the battery, not of either measurement. L0N occupies
the low-predictability tail; only a handful of L0 items ever reach into it.

### R5.b Range-matched L0 windows vs the L0N correlation

Every contiguous-in-gold window of L0 items (min 5 items) whose gold spread falls
within 25% of that model's L0N spread. `pct <= L0N` is the share of matched windows
whose rho is at least as close to zero as the observed L0N rho, i.e. an empirical
(descriptive, overlapping-window) p-value for "the control is flatter than a
range-matched slice of the memorized condition".

| model | L0N spread | L0N rho | n windows | median window rho | window rho range | pct <= L0N |
|---|---|---|---|---|---|---|
| pythia-410m | 1.96 | -0.055 | 24 | -0.418 | [-0.975, +0.432] | 0.21 |
| pythia-1b | 1.81 | -0.054 | 18 | -0.379 | [-0.975, +0.051] | 0.11 |
| pythia-1.4b | 1.73 | -0.160 | 24 | -0.615 | [-0.928, +0.154] | 0.08 |
| pythia-2.8b | 1.69 | +0.350 | 20 | -0.095 | [-0.470, +0.647] | 0.25 |
| OLMo-2-0425-1B | 2.61 | +0.075 | 55 | -0.778 | [-0.860, -0.060] | 0.00 |

Random (non-contiguous) spread-matched subsets of L0, n = n_L0N, 20000 draws kept if
their gold spread is within 25% of the L0N spread:

| model | n kept | median rho | 5th-95th pct | pct >= L0N rho |
|---|---|---|---|---|
| pythia-410m | 0 | too few matched draws | - | - |
| pythia-1b | 0 | too few matched draws | - | - |
| pythia-1.4b | 2 | too few matched draws | - | - |
| pythia-2.8b | 5702 | -0.305 | [-0.535, -0.060] | 0.000 |
| OLMo-2-0425-1B | 19488 | -0.841 | [-0.933, -0.745] | 0.000 |

### R5.c The other reason a correlation goes flat: tied depth values

`depth_tau_0.1` is a mean over positions of a quantity taking n_layers discrete
values, and in the control condition nearly every position sits at the cap, so the
per-item means collapse onto a handful of values. Max attainable |rho| is the
Spearman you would get if the observed depth values were arranged in perfect
monotone order against gold — the ceiling the tie structure imposes.

| model | cond | n | distinct depth values | largest tie group | depth SD | max attainable \|rho\| | observed rho |
|---|---|---|---|---|---|---|---|
| pythia-410m | L0 | 17 | 10 | 6/17 | 0.0429 | 0.977 | -0.530 |
| pythia-410m | L0N | 12 | 5 | 7/12 | 0.0171 | 0.895 | -0.055 |
| pythia-1b | L0 | 17 | 10 | 4/17 | 0.0723 | 0.990 | -0.337 |
| pythia-1b | L0N | 12 | 3 | 10/12 | 0.0135 | 0.650 | -0.054 |
| pythia-1.4b | L0 | 17 | 9 | 4/17 | 0.0458 | 0.986 | -0.693 |
| pythia-1.4b | L0N | 12 | 5 | 7/12 | 0.0244 | 0.895 | -0.160 |
| pythia-2.8b | L0 | 17 | 13 | 3/17 | 0.0489 | 0.996 | -0.419 |
| pythia-2.8b | L0N | 12 | 5 | 6/12 | 0.0160 | 0.929 | +0.350 |
| OLMo-2-0425-1B | L0 | 20 | 10 | 5/20 | 0.0248 | 0.988 | -0.860 |
| OLMo-2-0425-1B | L0N | 11 | 2 | 9/11 | 0.0032 | 0.671 | +0.075 |

### R5.d Discretisation test: put L0 on the control's depth grid

Take the L0 items — where the effect is claimed — and round each item's depth to the
nearest value actually observed in that model's L0N depth support, leaving gold
untouched. If the L0 correlation dies, the control's flatness is explained by the
granularity of the measure in that range rather than by the absence of a relation.
Second column additionally restricts L0 to the L0N gold range (both handicaps at once).

| model | L0 rho | L0 rho on L0N depth grid | L0 rho, grid + L0N gold range (n) | L0N rho |
|---|---|---|---|---|
| pythia-410m | -0.530 | -0.569 (p=0.017) | +0.045 (p=0.924, n=7) | -0.055 |
| pythia-1b | -0.337 | -0.328 (p=0.198) | -0.239 (p=0.648, n=6) | -0.054 |
| pythia-1.4b | -0.693 | -0.661 (p=0.004) | +0.527 (p=0.361, n=5) | -0.160 |
| pythia-2.8b | -0.419 | -0.429 (p=0.086) | undefined (n=2, 2 distinct depths) | +0.350 |
| OLMo-2-0425-1B | -0.860 | -0.631 (p=0.003) | undefined (n=4, 2 distinct depths) | +0.075 |

