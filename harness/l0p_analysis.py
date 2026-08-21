"""Registered analysis for the L0P crossed control (docs/PREREG_L0P.md).

WRITTEN BEFORE THE L0P DATA EXISTED for four of five models (pythia-410m had
finished when this file was drafted; its numbers were not looked at). The
prereg (public commit 459710c) is the spec; nothing below may change after
results are seen except bug fixes, which must be logged in PROCESS_LOG.md.

Inputs
    results/raw/pilot_{tag}.json   L0 (memorized) records, existing battery
    results/l0p/pilot_{tag}.json   L0P (absent, predictable) records, new run
    x = gold_logprob_per_token, y = final_entropy_mean, per record.
    OK records only (no 'error' key, gold present) — same filter as
    pooled_anchor_analysis.load_records.

Registered analyses (prereg wording -> implementation)
    A3 (manipulation check, gates A1/A2)
        "the L0P gold range must overlap the L0 gold range over at least 50%
        of the L0P range in >= 4 of 5 models"
        -> per model: frac = max(0, min(hiP,hi0) - max(loP,lo0)) / (hiP - loP)
           where [loP,hiP] = min/max gold over L0P, [lo0,hi0] over L0.
           Pass if frac >= 0.5; gate passes if >= 4 of 5 models pass.
    A1 (PRIMARY)
        "Rank-based ANCOVA within each model: rank(final_entropy_mean) ~
        rank(gold_logprob_per_token) + condition, restricted to the
        gold-overlap region of the two conditions."
        -> restrict FIRST (records of both conditions whose gold lies in the
           closed observed overlap interval), THEN rank (rankdata, average
           ties, normalized by n so the coefficient is scale-free across
           models), then OLS y_rank ~ 1 + x_rank + c with c = 1 for L0
           (memorized), 0 for L0P. Estimand: beta_c per model.
           Negative beta_c = memorized items sharper (lower entropy) at
           matched gold, the direction R-memo names.
        "Combined inference across the five runs by the item-level
        permutation scheme already shipped (one permutation applied jointly
        to the four shared-item Pythia runs; OLMo permuted independently;
        20,000 iterations; seed 20260822). Two-sided."
        -> combined statistic = unweighted mean over the five runs of the
           STUDENTIZED condition coefficient t = beta_c / se(beta_c) (the
           shipped scheme's "mean over runs" reduction; Fisher z does not
           apply to a regression coefficient). The reported estimand stays
           beta_c per the prereg; only the permutation comparison is
           studentized. WHY (decided before any real L0P number was seen,
           calibration scripts in the session scratchpad,
           diagnose_a1_schemes.py): with condition-dependent gold ranges —
           the expected real structure — permuting labels against the raw
           beta is anticonservative (measured FPR .080-.115 at nominal .05
           over 150-200 synthetic nulls); the studentized version measured
           .047/.053/.053 on the same nulls, .053-.063 under harsh nulls
           with as few as 0-4 L0 items inside the overlap, with no power
           loss. Studentizing a permutation statistic is standard practice
           and changes none of the frozen elements (estimand, unit=item,
           sharing, iterations, seed, sidedness).
           Null: the condition LABEL is permuted across items. One
           permutation of the shared Pythia label vector (union of the four
           runs' L0+L0P item ids, canonical sorted order, sharing asserted)
           is applied to all four Pythia runs per iteration; OLMo's label
           vector is permuted independently. Each model's analysis sample
           (the observed overlap restriction) is FROZEN; only labels of the
           retained items change. p = (1 + #{|perm| >= |obs|}) / (N + 1).
           Guard: a permuted restricted sample with a constant condition
           column contributes statistic 0 and is counted in the output
           (expected ~never; a CONSTANT OBSERVED sample would abort with a
           clear message instead of silently producing a number).
    A2
        "Within-L0P Spearman gold~entropy per model, contrasted with
        within-L0; difference assessed by the same permutation scheme."
        -> d = arctanh(rho_L0P) - arctanh(rho_L0) per model (Fisher z
           difference, matching the shipped z-scale reduction); combined =
           unweighted mean of the five d; same label-permutation scheme on
           the FULL (unrestricted) samples; two-sided p (sidedness not
           stated in the prereg for A2; two-sided chosen to match A1 —
           implementation decision, recorded here).
           CAVEAT (structural, known before data): a wider gold range in one
           condition mechanically inflates |rho| there (restriction of
           range), so A2 can flag a "slope difference" that is really a
           range difference. The registered A2 is computed exactly as
           registered; an EXPLORATORY range-matched companion (same rho
           contrast inside the A1 overlap region) is reported alongside,
           labeled exploratory per the prereg's final clause, so the reader
           can see whether any A2 difference survives range matching.

Decision rules are printed by verdict(); "comparable" in R-generic is
operationalized as A2 combined two-sided p >= .05 (implementation decision,
recorded here). Patterns not covered by R-generic / R-memo / R-failed are
reported as "no registered rule fired" with the observed pattern, no verdict.

Usage
    .venv/Scripts/python.exe harness/l0p_analysis.py
Outputs
    results/l0p/l0p_analysis.json
    results/l0p/L0P_ANALYSIS.md   (section appended, not clobbered)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "results" / "l0p"

MAIN_TAGS = [
    "EleutherAI_pythia-410m",
    "EleutherAI_pythia-1b",
    "EleutherAI_pythia-1.4b",
    "EleutherAI_pythia-2.8b",
    "allenai_OLMo-2-0425-1B",
]
N_ITER = 20000
SEED = 20260822          # prereg §Registered analyses, A1
XKEY = "gold_logprob_per_token"
YKEY = "final_entropy_mean"


def load_ok(path, level):
    data = json.load(open(path, encoding="utf-8"))
    recs = [r for r in data["records"]
            if "error" not in r and r.get(XKEY) is not None
            and r.get("level") == level and r.get(YKEY) is not None]
    return recs, data.get("meta", {})


def fisher_z(rho):
    eps = 1e-12
    return float(np.arctanh(np.clip(rho, -1 + eps, 1 - eps)))


def load_all():
    runs = []
    for tag in MAIN_TAGS:
        l0, meta0 = load_ok(PROJ / "results" / "raw" / f"pilot_{tag}.json", "L0")
        l0p, metap = load_ok(PROJ / "results" / "l0p" / f"pilot_{tag}.json", "L0P")
        ids = [r["id"] for r in l0] + [r["id"] for r in l0p]
        gold = np.array([r[XKEY] for r in l0] + [r[XKEY] for r in l0p], float)
        ent = np.array([r[YKEY] for r in l0] + [r[YKEY] for r in l0p], float)
        cond = np.array([1] * len(l0) + [0] * len(l0p), int)   # 1 = L0 memorized
        order = np.argsort(ids)                                # canonical order
        runs.append({
            "tag": tag,
            "model": metap.get("model", tag.replace("_", "/", 1)),
            "family": "pythia" if "pythia" in tag.lower() else "olmo",
            "ids": [ids[i] for i in order],
            "gold": gold[order], "ent": ent[order], "cond": cond[order],
            "n_l0": len(l0), "n_l0p": len(l0p),
            "battery_version_l0p": metap.get("battery_version"),
        })
    return runs


# ---------------------------------------------------------------------- A3 --
def a3_overlap(runs):
    out, n_pass = [], 0
    for r in runs:
        g0 = r["gold"][r["cond"] == 1]
        gp = r["gold"][r["cond"] == 0]
        lo0, hi0 = float(g0.min()), float(g0.max())
        lop, hip = float(gp.min()), float(gp.max())
        ov = max(0.0, min(hi0, hip) - max(lo0, lop))
        frac = ov / (hip - lop) if hip > lop else 0.0
        ok = frac >= 0.5
        n_pass += ok
        # refuter finding M2: frac <= (L0 width)/(L0P width) algebraically, so
        # a narrow L0 range caps the overlap no matter how well the titration
        # is centred. The ratio is printed so an A3 failure can be attributed.
        width_ratio = ((hi0 - lo0) / (hip - lop)) if hip > lop else None
        out.append({"tag": r["tag"], "model": r["model"],
                    "l0_gold_range": [round(lo0, 4), round(hi0, 4)],
                    "l0p_gold_range": [round(lop, 4), round(hip, 4)],
                    "overlap_fraction_of_l0p_range": round(frac, 4),
                    "l0_width_over_l0p_width": (None if width_ratio is None
                                                else round(width_ratio, 4)),
                    "passes_0.5": bool(ok)})
    return {"per_model": out, "n_models_passing": int(n_pass),
            "gate_passes": bool(n_pass >= 4)}


# ---------------------------------------------------------------------- A1 --
def restrict_overlap(r):
    """Indices (into the canonical arrays) inside the closed observed
    gold-overlap interval of the two conditions. Frozen before permutation."""
    g0 = r["gold"][r["cond"] == 1]
    gp = r["gold"][r["cond"] == 0]
    lo = max(g0.min(), gp.min())
    hi = min(g0.max(), gp.max())
    return np.where((r["gold"] >= lo) & (r["gold"] <= hi))[0], (float(lo), float(hi))


def fit_ancova(gold, ent, cond):
    """Rank ANCOVA rank(ent) ~ 1 + rank(gold) + cond, ranks normalized by n.
    Returns (beta_c, t_c) or None if the sample is degenerate (n < 4 or a
    constant condition column). pinv used for BOTH the coefficients and the
    standard error so the two stay consistent on an ill-conditioned X
    (refuter finding S7)."""
    n = len(gold)
    cond = np.asarray(cond)
    if n < 4 or cond.min() == cond.max():
        return None
    yr = rankdata(ent) / n
    xr = rankdata(gold) / n
    X = np.column_stack([np.ones(n), xr, cond.astype(float)])
    XtX_pinv = np.linalg.pinv(X.T @ X)
    coef = XtX_pinv @ (X.T @ yr)
    resid = yr - X @ coef
    s2 = float(resid @ resid) / (n - 3)
    se = float(np.sqrt(s2 * XtX_pinv[2, 2]))
    if not np.isfinite(se) or se <= 0:
        return None
    return float(coef[2]), float(coef[2] / se)


def a1_primary(runs, rng):
    """FROZEN EXCLUSION RULE (refuter finding M1, decided before unblinding):
    A3's gate is >= 4 of 5, so a model with essentially no gold overlap can
    legally reach A1 with an overlap sample that cannot support a condition
    contrast. A model enters the A1 combined statistic only if its observed
    overlap sample has >= 2 L0 items AND >= 2 L0P items. An excluded model is
    reported with its reason, contributes NO sign to the sign counts, and the
    prereg's ">= 4 of 5 runs" thresholds stay literal (out of 5, not out of
    the included count). The same exclusion applies identically inside every
    permutation iteration (the model is simply absent from the mean)."""
    samples = []
    for r in runs:
        idx, interval = restrict_overlap(r)
        c = r["cond"][idx]
        n0, np_ = int(c.sum()), int((1 - c).sum())
        samples.append({"idx": idx, "interval": interval, "n": len(idx),
                        "n_l0_in": n0, "n_l0p_in": np_,
                        "included": bool(n0 >= 2 and np_ >= 2)})

    included = [i for i, s in enumerate(samples) if s["included"]]
    assert included, "A1: no model has a usable overlap sample"

    obs = [fit_ancova(runs[i]["gold"][samples[i]["idx"]],
                      runs[i]["ent"][samples[i]["idx"]],
                      runs[i]["cond"][samples[i]["idx"]])
           if samples[i]["included"] else None
           for i in range(len(runs))]
    for i in included:
        assert obs[i] is not None, (
            f"A1 degenerate for {runs[i]['tag']} despite passing the "
            "inclusion rule — investigate before unblinding anything else.")
    obs_beta = [obs[i][0] for i in included]
    obs_t = [obs[i][1] for i in included]
    combined_obs = float(np.mean(obs_t))          # studentized reduction

    pythia = [i for i, r in enumerate(runs) if r["family"] == "pythia"]
    olmo = [i for i, r in enumerate(runs) if r["family"] == "olmo"]

    # shared Pythia label universe: identical id lists asserted, so the
    # canonical arrays line up index-for-index across the four runs
    ref = runs[pythia[0]]
    for i in pythia[1:]:
        assert runs[i]["ids"] == ref["ids"], \
            f"{runs[i]['tag']} item set differs from {ref['tag']}"
        assert np.array_equal(runs[i]["cond"], ref["cond"])
    n_shared = len(ref["ids"])

    degenerate = 0
    perm_combined = np.empty(N_ITER)
    perm_per_run = np.full((N_ITER, len(runs)), np.nan)
    for it in range(N_ITER):
        perm = rng.permutation(n_shared)          # ONE draw, all four Pythia
        ts = []
        for i, (r, s) in enumerate(zip(runs, samples)):
            if r["family"] == "pythia":
                c_full = r["cond"][perm]
            else:
                c_full = r["cond"][rng.permutation(len(r["cond"]))]
            if not s["included"]:
                continue                          # absent from the mean, as observed
            fit = fit_ancova(r["gold"][s["idx"]], r["ent"][s["idx"]],
                             c_full[s["idx"]])
            if fit is None:
                degenerate += 1
                t = 0.0
            else:
                t = fit[1]
            ts.append(t)
            perm_per_run[it, i] = t
        perm_combined[it] = np.mean(ts)

    p_two = float((1 + np.sum(np.abs(perm_combined) >= abs(combined_obs)))
                  / (N_ITER + 1))
    n_neg = int(sum(b < 0 for b in obs_beta))
    n_pos = int(sum(b > 0 for b in obs_beta))

    per_run = []
    for i, (r, s) in enumerate(zip(runs, samples)):
        row = {
            "tag": r["tag"], "model": r["model"],
            "overlap_interval_gold": [round(s["interval"][0], 4),
                                      round(s["interval"][1], 4)],
            "n_in_overlap": s["n"], "n_l0_in": s["n_l0_in"],
            "n_l0p_in": s["n_l0p_in"],
            "included": s["included"],
        }
        if s["included"]:
            b, t = obs[i]
            row.update({
                "beta_condition": round(b, 4),
                "t_condition": round(t, 3),
                "p_permutation_two_sided_marginal": round(
                    float((1 + np.nansum(np.abs(perm_per_run[:, i]) >= abs(t)))
                          / (N_ITER + 1)), 4),
            })
        else:
            row["excluded_reason"] = ("fewer than 2 items of a condition "
                                      "inside the gold-overlap region")
        per_run.append(row)
    return {
        "estimand": "condition coefficient (L0=1) of rank ANCOVA, "
                    "normalized ranks, within gold-overlap region; "
                    "negative = memorized sharper at matched gold; "
                    "permutation comparison uses the studentized coefficient",
        "per_run": per_run,
        "n_models_included": len(included),
        "combined_mean_beta_observed": round(float(np.mean(obs_beta)), 4),
        "combined_mean_t_observed": round(combined_obs, 4),
        "permutation": {"n_iterations": N_ITER, "seed": SEED,
                        "unit": "item (condition label permuted)",
                        "statistic": "mean over runs of studentized "
                                     "condition coefficient",
                        "sharing": "one permutation of the shared "
                                   f"{n_shared}-item Pythia label vector per "
                                   "iteration; OLMo independent",
                        "null_mean": round(float(perm_combined.mean()), 4),
                        "null_sd": round(float(perm_combined.std(ddof=1)), 4),
                        "degenerate_samples_stat_set_0": int(degenerate)},
        "p_two_sided": p_two,
        "p_mc_se": round(float(np.sqrt(p_two * (1 - p_two) / N_ITER)), 5),
        "sign_pattern": {"n_negative": n_neg, "n_positive": n_pos,
                         "consistent_4_of_5": bool(max(n_neg, n_pos) >= 4)},
    }


# ---------------------------------------------------------------------- A2 --
def rho_diff(gold, ent, cond):
    m0, mp = cond == 1, cond == 0
    if m0.sum() < 6 or mp.sum() < 6:
        return None, None, None
    r0 = spearmanr(gold[m0], ent[m0]).statistic
    rp = spearmanr(gold[mp], ent[mp]).statistic
    return float(rp), float(r0), fisher_z(rp) - fisher_z(r0)


def a2_slopes(runs, rng):
    obs = []
    for r in runs:
        rp, r0, d = rho_diff(r["gold"], r["ent"], r["cond"])
        assert d is not None, f"too few items in a condition for {r['tag']}"
        obs.append({"rho_l0p": rp, "rho_l0": r0, "dz": d})
    combined_obs = float(np.mean([o["dz"] for o in obs]))

    pythia_idx = [i for i, r in enumerate(runs) if r["family"] == "pythia"]
    ref = runs[pythia_idx[0]]
    n_shared = len(ref["ids"])

    skipped = 0
    perm_combined = np.full(N_ITER, np.nan)
    for it in range(N_ITER):
        perm = rng.permutation(n_shared)
        ds = []
        for r in runs:
            if r["family"] == "pythia":
                c = r["cond"][perm]
            else:
                c = r["cond"][rng.permutation(len(r["cond"]))]
            _, _, d = rho_diff(r["gold"], r["ent"], c)
            if d is None:            # cannot happen: permutation preserves counts
                skipped += 1
                d = 0.0
            ds.append(d)
        perm_combined[it] = np.mean(ds)

    p_two = float((1 + np.sum(np.abs(perm_combined) >= abs(combined_obs)))
                  / (N_ITER + 1))

    # EXPLORATORY companion: same rho contrast inside the A1 overlap region,
    # so a slope difference that is only a range difference shows itself.
    # No permutation p — labeled exploratory per the prereg's final clause.
    explor = []
    for r in runs:
        idx, _ = restrict_overlap(r)
        rp, r0, d = rho_diff(r["gold"][idx], r["ent"][idx], r["cond"][idx])
        explor.append({"tag": r["tag"],
                       "rho_l0p": None if rp is None else round(rp, 3),
                       "rho_l0": None if r0 is None else round(r0, 3),
                       "dz": None if d is None else round(d, 4),
                       "note": None if d is not None else
                       "fewer than 6 items in a condition inside the overlap"})
    valid = [e["dz"] for e in explor if e["dz"] is not None]
    return {
        "statistic": "mean over runs of arctanh(rho_L0P) - arctanh(rho_L0), "
                     "gold vs final_entropy_mean, full samples",
        "exploratory_range_matched": {
            "label": "EXPLORATORY (not registered): same contrast inside the "
                     "A1 gold-overlap region, to separate range restriction "
                     "from a real slope difference",
            "per_run": explor,
            "mean_dz_over_valid_runs": (round(float(np.mean(valid)), 4)
                                        if valid else None),
            "n_valid_runs": len(valid)},
        "per_run": [{"tag": r["tag"], "model": r["model"],
                     "rho_l0p": round(o["rho_l0p"], 3),
                     "rho_l0": round(o["rho_l0"], 3),
                     "dz": round(o["dz"], 4)}
                    for r, o in zip(runs, obs)],
        "combined_mean_dz_observed": round(combined_obs, 4),
        "null_sd": round(float(np.nanstd(perm_combined, ddof=1)), 4),
        "p_two_sided": p_two,
        "p_mc_se": round(float(np.sqrt(p_two * (1 - p_two) / N_ITER)), 5),
        "skipped_degenerate": int(skipped),
    }


# ------------------------------------------------------- exploratory: LOO --
def a1_loo_sensitivity(runs, ss):
    """EXPLORATORY (pre-committed before unblinding, refuter suggestion #7):
    the A1 overlap interval's endpoints are defined by single items (closed
    interval min/max), so one extreme item can move every model's analysis
    sample. For each model, the (<= 2) items whose gold value binds lo or hi
    of its overlap interval are collected; for each such item id, the full
    registered A1 is recomputed with that item dropped from every run that
    contains it, on an independent stream spawned from the registered seed.
    Reported as a range of combined mean t and p. No verdict role."""
    binding = []
    for r in runs:
        g0 = r["gold"][r["cond"] == 1]
        gp = r["gold"][r["cond"] == 0]
        lo, hi = max(g0.min(), gp.min()), min(g0.max(), gp.max())
        for bound in (lo, hi):
            for j, g in enumerate(r["gold"]):
                if g == bound:
                    binding.append(r["ids"][j])
    reruns = []
    ids_to_drop = sorted(set(binding))
    children = ss.spawn(len(ids_to_drop))
    for item_id, child in zip(ids_to_drop, children):
        mod_runs = []
        for r in runs:
            keep = [k for k, i in enumerate(r["ids"]) if i != item_id]
            mod_runs.append({**r,
                             "ids": [r["ids"][k] for k in keep],
                             "gold": r["gold"][keep], "ent": r["ent"][keep],
                             "cond": r["cond"][keep]})
        a1 = a1_primary(mod_runs, np.random.default_rng(child))
        reruns.append({"dropped_item": item_id,
                       "combined_mean_t": a1["combined_mean_t_observed"],
                       "p_two_sided": a1["p_two_sided"]})
    ts = [x["combined_mean_t"] for x in reruns]
    ps = [x["p_two_sided"] for x in reruns]
    return {"label": "EXPLORATORY leave-one-out over boundary-defining items; "
                     "no verdict role",
            "n_reruns": len(reruns), "per_rerun": reruns,
            "combined_t_range": ([round(min(ts), 4), round(max(ts), 4)]
                                 if ts else None),
            "p_range": ([round(min(ps), 5), round(max(ps), 5)] if ps else None)}


# ----------------------------------------------------------------- verdict --
def verdict(a3, a1, a2):
    if not a3["gate_passes"]:
        return {"rule": "R-failed",
                "statement": "A3 failed: the titration did not place L0P gold "
                             "in the L0 range in enough models. No conclusion "
                             "in either direction; redesign. Published anyway."}
    # Literal prereg operationalization (refuter findings S1/S2): R-memo is
    # exactly "combined p < .05, consistent sign in >= 4 of 5 runs" in the
    # sharper (negative) direction — no extra mean-beta condition. Per-run
    # sign(beta) == sign(t) within a model, so the count is the same either
    # way; both means are still reported for the reader.
    p1 = a1["p_two_sided"]
    consistent = a1["sign_pattern"]["consistent_4_of_5"]
    negative = a1["sign_pattern"]["n_negative"] >= 4
    comparable = a2["p_two_sided"] >= 0.05
    if p1 < 0.05 and negative:
        return {"rule": "R-memo",
                "statement": "Memorized items are sharper at matched gold "
                             "(combined p < .05, consistent negative sign in "
                             ">= 4 of 5 runs): a memorization-specific "
                             "component survives its first crossed control. "
                             "Still correlational; causal phase unchanged."}
    if p1 >= 0.05 and not consistent and comparable:
        return {"rule": "R-generic",
                "statement": "No condition effect at matched gold and "
                             "comparable slopes: the sharpness effect is not "
                             "shown to be memorization-specific and reads as "
                             "generic predictability. Headline of v0.7."}
    return {"rule": "none",
            "statement": "No registered decision rule fired (pattern falls "
                         "between R-generic and R-memo). Report the observed "
                         "pattern as exploratory; draw no registered "
                         "conclusion."}


def render_md(a3, a1, a2, ver, runs, loo=None):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"\n## Registered L0P analysis ({ts})\n",
         "Generated by `harness/l0p_analysis.py` per `docs/PREREG_L0P.md` "
         "(frozen; seed 20260822, 20000 iterations).\n"]
    L.append("### A3 manipulation check (gates A1/A2)\n")
    L.append("| model | L0 gold range | L0P gold range | overlap / L0P range "
             "| L0 width / L0P width | pass >= 0.5 |")
    L.append("|---|---|---|---|---|---|")
    for m in a3["per_model"]:
        L.append(f"| {m['model']} | [{m['l0_gold_range'][0]}, {m['l0_gold_range'][1]}] "
                 f"| [{m['l0p_gold_range'][0]}, {m['l0p_gold_range'][1]}] "
                 f"| {m['overlap_fraction_of_l0p_range']:.3f} "
                 f"| {m['l0_width_over_l0p_width']} | {m['passes_0.5']} |")
    L.append(f"\n**Gate: {a3['n_models_passing']}/5 models pass -> "
             f"{'PASS' if a3['gate_passes'] else 'FAIL'}**")
    L.append("(The width ratio is an algebraic cap on the overlap fraction: "
             "a model can only fail A3 'because of titration' if its ratio "
             "is >= 0.5; below that the failure is structural.)\n")

    L.append("### A1 primary: condition effect at matched gold\n")
    L.append("Models that fail A3 still enter the combined statistic (the "
             "prereg does not exclude them); models with fewer than 2 items "
             "of a condition inside the overlap are excluded by the frozen "
             "M1 rule and marked below.\n")
    L.append("| model | overlap n (L0/L0P) | beta_condition | t | marginal p (perm, 2s) |")
    L.append("|---|---|---|---|---|")
    for r in a1["per_run"]:
        if r["included"]:
            L.append(f"| {r['model']} | {r['n_in_overlap']} ({r['n_l0_in']}/{r['n_l0p_in']}) "
                     f"| {r['beta_condition']:+.4f} | {r['t_condition']:+.3f} "
                     f"| {r['p_permutation_two_sided_marginal']:.4f} |")
        else:
            L.append(f"| {r['model']} | {r['n_in_overlap']} ({r['n_l0_in']}/{r['n_l0p_in']}) "
                     f"| EXCLUDED | — | — |")
    L.append(f"\n- Combined mean beta = **{a1['combined_mean_beta_observed']:+.4f}** "
             f"(estimand; negative = memorized sharper). Permutation compares "
             f"the studentized version: mean t = "
             f"{a1['combined_mean_t_observed']:+.4f}, null sd "
             f"{a1['permutation']['null_sd']:.4f}.")
    L.append(f"- **Combined permutation p (two-sided) = {a1['p_two_sided']:.5f}** "
             f"(MC se {a1['p_mc_se']:.5f}).")
    L.append(f"- Sign pattern: {a1['sign_pattern']['n_negative']} negative / "
             f"{a1['sign_pattern']['n_positive']} positive; consistent in >=4/5: "
             f"{a1['sign_pattern']['consistent_4_of_5']}.\n")

    L.append("### A2 slope comparison\n")
    L.append("| model | rho L0P | rho L0 | dz |")
    L.append("|---|---|---|---|")
    for r in a2["per_run"]:
        L.append(f"| {r['model']} | {r['rho_l0p']:+.3f} | {r['rho_l0']:+.3f} "
                 f"| {r['dz']:+.4f} |")
    L.append(f"\n- Combined mean dz = {a2['combined_mean_dz_observed']:+.4f}; "
             f"p (two-sided) = {a2['p_two_sided']:.5f} "
             f"(MC se {a2['p_mc_se']:.5f}).")
    ex = a2["exploratory_range_matched"]
    L.append(f"- {ex['label']}: mean dz = {ex['mean_dz_over_valid_runs']} "
             f"over {ex['n_valid_runs']} valid runs.")
    L.append("- Known structural bias (measured on synthetic nulls before "
             "unblinding): with equal true slopes but a narrower gold range "
             "in one condition, the registered A2 flags 'not comparable' far "
             "above its nominal rate (~34% at spread ratio 0.5). If the "
             "registered and range-matched readings disagree, the verdict "
             "still follows the registered rule, but the write-up must carry "
             "both numbers and say the disagreement is consistent with a "
             "range artifact.\n")

    if loo is not None:
        L.append(f"### {loo['label']}\n")
        L.append(f"- {loo['n_reruns']} reruns (one boundary-defining item "
                 f"dropped each time): combined mean t in "
                 f"[{loo['combined_t_range'][0]}, {loo['combined_t_range'][1]}], "
                 f"p in [{loo['p_range'][0]}, {loo['p_range'][1]}].\n")
    L.append(f"### Verdict\n\n**{ver['rule']}** — {ver['statement']}\n")
    return "\n".join(L)


def main():
    runs = load_all()
    for r in runs:
        print(f"{r['tag']}: L0 n={r['n_l0']}  L0P n={r['n_l0p']}  "
              f"battery={r['battery_version_l0p']}")
    # refuter finding S5: a1 and a2 get INDEPENDENT streams derived from the
    # registered seed via SeedSequence.spawn, so a fix that changes a1's draw
    # count can never silently move a2's p. Recorded implementation decision.
    ss = np.random.SeedSequence(SEED)
    rng_a1, rng_a2 = (np.random.default_rng(c) for c in ss.spawn(2))
    a3 = a3_overlap(runs)
    a1 = a1_primary(runs, rng_a1)
    a2 = a2_slopes(runs, rng_a2)
    ver = verdict(a3, a1, a2)
    loo = a1_loo_sensitivity(runs, np.random.SeedSequence(SEED + 1))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "harness/l0p_analysis.py",
        "prereg": "docs/PREREG_L0P.md (public commit 459710c)",
        "runs": [{"tag": r["tag"], "n_l0": r["n_l0"], "n_l0p": r["n_l0p"]}
                 for r in runs],
        "A3": a3, "A1": a1, "A2": a2, "verdict": ver,
        "exploratory_loo": loo,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / "l0p_analysis.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    mpath = OUT_DIR / "L0P_ANALYSIS.md"
    header = "" if mpath.exists() else "# L0P registered analysis\n"
    with open(mpath, "a", encoding="utf-8") as f:
        f.write(header + render_md(a3, a1, a2, ver, runs, loo))

    print(f"\njson -> {jpath}\nmd   -> {mpath}\n")
    print(f"A3 gate: {a3['n_models_passing']}/5 pass -> "
          f"{'PASS' if a3['gate_passes'] else 'FAIL'}")
    for m in a3["per_model"]:
        print(f"  {m['model']:28s} overlap {m['overlap_fraction_of_l0p_range']:.3f}")
    print(f"A1 combined beta {a1['combined_mean_beta_observed']:+.4f} "
          f"(mean t {a1['combined_mean_t_observed']:+.3f})  "
          f"p={a1['p_two_sided']:.5f}  "
          f"signs -{a1['sign_pattern']['n_negative']}/+{a1['sign_pattern']['n_positive']}")
    for r in a1["per_run"]:
        if r["included"]:
            print(f"  {r['model']:28s} n={r['n_in_overlap']:2d} "
                  f"({r['n_l0_in']}/{r['n_l0p_in']}) beta {r['beta_condition']:+.4f} "
                  f"t {r['t_condition']:+.3f} "
                  f"p={r['p_permutation_two_sided_marginal']:.4f}")
        else:
            print(f"  {r['model']:28s} n={r['n_in_overlap']:2d} "
                  f"({r['n_l0_in']}/{r['n_l0p_in']}) EXCLUDED (M1 rule)")
    print(f"A2 combined dz {a2['combined_mean_dz_observed']:+.4f}  "
          f"p={a2['p_two_sided']:.5f}")
    for r in a2["per_run"]:
        print(f"  {r['model']:28s} rho_L0P {r['rho_l0p']:+.3f}  "
              f"rho_L0 {r['rho_l0']:+.3f}")
    print(f"\nVERDICT: {ver['rule']} — {ver['statement']}")


if __name__ == "__main__":
    main()
