#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entropy-normalized / relative convergence-depth analysis (OMTR).

Answers the R1 critique (results/night/R1_verification.md, "Replace the absolute tau
with an entropy-normalized or relative criterion"): the frozen depth metric
`depth_tau_0.1` = last layer whose KL-to-final exceeds a FIXED 0.1 nats, normalized by
n_layers.  That threshold is arithmetically confounded with the sharpness of the final
distribution: a peaked final distribution is easy to match to within 0.1 nats early, a
diffuse one is not, independent of when the computation actually finished.

This script recomputes depth under criteria that remove (or at least rescale) that
confound, and re-runs the within-condition dose-response tests:

  A. RELATIVE threshold   depth_rel_a   : last layer with kl >= alpha * kl[layer 0]
                                          alpha in {0.5, 0.25, 0.1, 0.05}
  B. ENTROPY-SCALED       depth_ent_t   : last layer with kl >= tau * final_entropy_mean
                                          tau in {0.05, 0.1, 0.25}
  C. FROZEN ABSOLUTE      depth_abs_0.1 : last layer with kl >= 0.1   (sanity column)

All depths are normalized as (i + 1) / n_layers, exactly like the frozen metric
(harness/run_pilot.py :: depth_metrics).  n_layers is inferred from len(layer_profile).

Two curve sources are used for every criterion:
  * "mean"  -- record["layer_profile"], the per-layer values AVERAGED over the 8 answer
               positions (this is a curve of averages, not the average of curves);
  * "first" -- record["first_pos_profile"], the exact curve for answer position 0.

Outputs
  results/relative_depth/relative_depth.json
  results/relative_depth/RELATIVE_DEPTH.md

Run:  D:/ai/research/omtr/.venv/Scripts/python.exe harness/relative_depth_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GLOB = os.path.join(PROJ, "results", "raw", "pilot_*.json")
OUT_DIR = os.path.join(PROJ, "results", "relative_depth")

REL_ALPHAS = [0.5, 0.25, 0.1, 0.05]
ENT_TAUS = [0.05, 0.1, 0.25]
ABS_TAU = 0.1
LEVELS = ["L0", "L0N", "L1"]
CURVES = ["mean", "first"]


# ------------------------------------------------------------------ depth core

def last_crossing_depth(kls, thresh):
    """Normalized index of the LAST layer whose KL >= thresh (0.0 if none).

    Mirrors harness/run_pilot.py :: depth_metrics -- scan backwards, depth = (i+1)/n.
    """
    n = len(kls)
    for i in range(n - 1, -1, -1):
        if kls[i] >= thresh:
            return (i + 1) / n
    return 0.0


def criteria_for_curve(kls, final_entropy_mean):
    """All depth criteria for one KL curve. Returns OrderedDict name -> depth (or None)."""
    out = OrderedDict()
    kl0 = kls[0]
    for a in REL_ALPHAS:
        key = "rel_a%g" % a
        # kl0 <= 0 would make the relative criterion meaningless; never happens here,
        # but guard rather than silently emit 0.
        out[key] = last_crossing_depth(kls, a * kl0) if kl0 > 0 else None
    for t in ENT_TAUS:
        key = "ent_t%g" % t
        out[key] = (last_crossing_depth(kls, t * final_entropy_mean)
                    if final_entropy_mean is not None and final_entropy_mean > 0 else None)
    out["abs_t0.1"] = last_crossing_depth(kls, ABS_TAU)
    return out


CRITERIA = (["rel_a%g" % a for a in REL_ALPHAS]
            + ["ent_t%g" % t for t in ENT_TAUS]
            + ["abs_t0.1"])

CRIT_LABEL = {
    "rel_a0.5": "relative alpha=0.50 (KL >= 0.50*KL_L0)",
    "rel_a0.25": "relative alpha=0.25 (KL >= 0.25*KL_L0)",
    "rel_a0.1": "relative alpha=0.10 (KL >= 0.10*KL_L0)",
    "rel_a0.05": "relative alpha=0.05 (KL >= 0.05*KL_L0)",
    "ent_t0.05": "entropy-scaled tau=0.05 (KL >= 0.05*H_final)",
    "ent_t0.1": "entropy-scaled tau=0.10 (KL >= 0.10*H_final)",
    "ent_t0.25": "entropy-scaled tau=0.25 (KL >= 0.25*H_final)",
    "abs_t0.1": "frozen absolute tau=0.1 (recomputed)",
}


# ------------------------------------------------------------------ loading

def load_items():
    """model -> list of per-item dicts with gold, entropy, depths under every criterion."""
    models = OrderedDict()
    for path in sorted(glob.glob(RAW_GLOB)):
        blob = json.load(open(path, encoding="utf-8"))
        model = blob["meta"]["model"]
        items = []
        n_layers_seen = set()
        for rec in blob["records"]:
            if "error" in rec:
                continue
            gold = rec.get("gold_logprob_per_token")
            if gold is None:
                continue
            lp = rec.get("layer_profile")
            fp = rec.get("first_pos_profile")
            if not lp:
                continue
            n_layers_seen.add(len(lp))
            fem = rec.get("final_entropy_mean")
            item = {
                "id": rec.get("id"),
                "level": rec.get("level"),
                "gold": float(gold),
                "final_entropy_mean": None if fem is None else float(fem),
                "n_layers": len(lp),
                "depths": {},
            }
            curves = {"mean": lp, "first": fp}
            for cname, prof in curves.items():
                if not prof:
                    item["depths"][cname] = {k: None for k in CRITERIA}
                    continue
                kls = [float(l["kl_to_final"]) for l in prof]
                item["depths"][cname] = criteria_for_curve(kls, item["final_entropy_mean"])
            item["stored_depth_tau_0.1"] = (rec.get("depth") or {}).get("depth_tau_0.1")
            ppd = rec.get("per_position_depths") or []
            item["pos0_stored_depth_tau_0.1"] = (ppd[0].get("depth_tau_0.1")
                                                 if ppd else None)
            # exact implementation check: the frozen metric recomputed on the position-0
            # curve must reproduce per_position_depths[0] bit-for-bit.
            if fp:
                kls0 = [float(l["kl_to_final"]) for l in fp]
                item["_pos0_recomp"] = {
                    "0.05": last_crossing_depth(kls0, 0.05),
                    "0.1": last_crossing_depth(kls0, 0.1),
                    "0.5": last_crossing_depth(kls0, 0.5),
                }
                item["_pos0_stored"] = {t: (ppd[0].get("depth_tau_%s" % t) if ppd else None)
                                        for t in ("0.05", "0.1", "0.5")}
                item["last_layer_kl_first"] = kls0[-1]
            item["last_layer_kl_mean"] = float(lp[-1]["kl_to_final"])
            items.append(item)
        models[model] = {
            "path": os.path.basename(path),
            "n_layers": sorted(n_layers_seen),
            "items": items,
        }
    return models


# ------------------------------------------------------------------ stats

def rho(x, y):
    """Spearman with tie/degeneracy bookkeeping. Returns a dict (never raises)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    dx, dy = int(np.unique(x).size), int(np.unique(y).size)
    out = {"n": n, "n_distinct_x": dx, "n_distinct_y": dy,
           "rho": None, "p": None, "degenerate": False, "note": ""}
    if n < 3 or dx < 2 or dy < 2:
        out["degenerate"] = True
        out["note"] = ("all-tied y (%d distinct)" % dy if dy < 2 else
                       "all-tied x (%d distinct)" % dx if dx < 2 else
                       "n<3")
        return out
    r, p = spearmanr(x, y)
    if not np.isfinite(r):
        out["degenerate"] = True
        out["note"] = "spearman returned nan"
        return out
    out["rho"], out["p"] = float(r), float(p)
    if dy <= 3:
        out["note"] = "near-degenerate: only %d distinct depth values" % dy
    return out


def analyze(models):
    res = OrderedDict()
    for model, blob in models.items():
        items = blob["items"]
        by_level = {lv: [it for it in items if it["level"] == lv] for lv in LEVELS}
        mres = {
            "source_file": blob["path"],
            "n_layers": blob["n_layers"],
            "n_items_used": len(items),
            "n_by_level": {lv: len(v) for lv, v in by_level.items()},
            "validity": {},
            "cells": {},
            "depth_distributions": {},
        }

        # exact implementation check: position-0 recomputation vs stored per-position depth
        exact = {}
        for t in ("0.05", "0.1", "0.5"):
            diffs, n_ok, n_cmp = [], 0, 0
            for it in items:
                a = it.get("_pos0_recomp", {}).get(t)
                b = it.get("_pos0_stored", {}).get(t)
                if a is None or b is None:
                    continue
                n_cmp += 1
                d = abs(a - b)
                diffs.append(d)
                if d < 1e-12:
                    n_ok += 1
            exact["tau_%s" % t] = {"n_compared": n_cmp, "n_exact_match": n_ok,
                                   "max_abs_diff": (max(diffs) if diffs else None)}
        mres["implementation_exact_check"] = exact

        # ceiling diagnostic: KL at the final layer is ~0 by construction (the lens on
        # resid_post[n-1] reproduces the final logits), so no criterion can ever cross
        # there and every depth is capped at (n-1)/n.
        mres["final_layer_kl"] = {
            "mean_curve_max_abs": float(np.max(np.abs([it["last_layer_kl_mean"]
                                                       for it in items]))),
            "first_curve_max_abs": float(np.max(np.abs([it["last_layer_kl_first"]
                                                        for it in items
                                                        if "last_layer_kl_first" in it]))),
            "depth_ceiling": (items[0]["n_layers"] - 1) / items[0]["n_layers"] if items else None,
        }

        # validity: recomputed frozen abs-0.1 vs the stored per-position-averaged metric
        for curve in CURVES:
            for scope, subset in [("all_gold", items)] + [(lv, by_level[lv]) for lv in LEVELS]:
                a = [it["depths"][curve]["abs_t0.1"] for it in subset]
                b = [it["stored_depth_tau_0.1"] for it in subset]
                a = [np.nan if v is None else v for v in a]
                b = [np.nan if v is None else v for v in b]
                mres["validity"]["%s|%s" % (curve, scope)] = rho(a, b)

        # baseline: the paper's own stored metrics, same items, same tests
        mres["baseline"] = {}
        for bname, getter in [
                ("stored_depth_tau_0.1_mean8",
                 lambda it: it.get("stored_depth_tau_0.1")),
                ("stored_depth_tau_0.1_pos0",
                 lambda it: it.get("pos0_stored_depth_tau_0.1"))]:
            for lv in LEVELS:
                sub = by_level[lv]
                d = [np.nan if getter(it) is None else float(getter(it)) for it in sub]
                g = [it["gold"] for it in sub]
                h = [np.nan if it["final_entropy_mean"] is None
                     else it["final_entropy_mean"] for it in sub]
                mres["baseline"]["%s|%s" % (bname, lv)] = {
                    "gold_vs_depth": rho(g, d),
                    "entropy_vs_depth": rho(h, d),
                }

        for curve in CURVES:
            for crit in CRITERIA:
                for lv in LEVELS:
                    sub = by_level[lv]
                    d = [it["depths"][curve][crit] for it in sub]
                    d = [np.nan if v is None else v for v in d]
                    g = [it["gold"] for it in sub]
                    h = [np.nan if it["final_entropy_mean"] is None
                         else it["final_entropy_mean"] for it in sub]
                    cell = {
                        "gold_vs_depth": rho(g, d),
                        "entropy_vs_depth": rho(h, d),
                        "gold_vs_entropy": rho(g, h),
                        "depth_values": sorted(set(round(v, 10) for v in d
                                                   if np.isfinite(v))),
                        "depth_mean": (float(np.nanmean(d)) if np.any(np.isfinite(d))
                                       else None),
                        "depth_sd": (float(np.nanstd(d, ddof=1))
                                     if np.sum(np.isfinite(d)) > 1 else None),
                    }
                    mres["cells"]["%s|%s|%s" % (curve, crit, lv)] = cell
        res[model] = mres
    return res


# ------------------------------------------------------------------ reporting

def fmt_rho(c):
    if c["degenerate"]:
        return "n/a (%s)" % c["note"]
    return "%+.3f (p=%.4f)" % (c["rho"], c["p"])


def fmt_rho_short(c):
    if c["degenerate"]:
        return "n/a"
    star = "*" if c["p"] < 0.05 else ""
    return "%+.2f%s" % (c["rho"], star)


def write_markdown(res, path):
    L = []
    A = L.append
    A("# Entropy-normalized / relative convergence depth")
    A("")
    A("Generated by `harness/relative_depth_analysis.py`. Every number recomputed from")
    A("`results/raw/pilot_*.json`; nothing quoted from earlier reports.")
    A("")
    A("Depth = normalized index `(i+1)/n_layers` of the LAST layer whose `kl_to_final`")
    A("meets the criterion, scanned backwards -- the same shape as the frozen")
    A("`depth_tau_0.1`, only the threshold changes. `n_layers` inferred from")
    A("`len(layer_profile)`. Two curve sources: **mean** = `layer_profile` (averaged over")
    A("the 8 answer positions), **first** = `first_pos_profile` (answer position 0 exactly).")
    A("")
    A("Criteria:")
    A("")
    for k in CRITERIA:
        A("- `%s` -- %s" % (k, CRIT_LABEL[k]))
    A("")
    A("## Bottom line (auto-generated from the cells below)")
    A("")
    A("For each model: sign and significance of the within-L0 `gold ~ depth` rank")
    A("correlation under the paper's own stored metric, then under each new criterion")
    A("(position-averaged curve). `k` = distinct depth values; k=1 means the criterion is")
    A("constant across every L0 item and the test cannot be run at all.")
    A("")
    for m, r in res.items():
        b = r["baseline"]["stored_depth_tau_0.1_mean8|L0"]["gold_vs_depth"]
        neg_sig, pos_sig, null_, degen = [], [], [], []
        for crit in CRITERIA:
            c = r["cells"]["mean|%s|L0" % crit]["gold_vs_depth"]
            if c["degenerate"]:
                degen.append(crit)
            elif c["p"] < 0.05 and c["rho"] < 0:
                neg_sig.append(crit)
            elif c["p"] < 0.05 and c["rho"] > 0:
                pos_sig.append(crit)
            else:
                null_.append(crit)
        A("- **%s** -- baseline %s. New criteria: %d significant NEGATIVE (%s); "
          "%d significant POSITIVE, i.e. sign-reversed (%s); %d non-significant (%s); "
          "%d degenerate/untestable (%s)." % (
              m, fmt_rho(b),
              len(neg_sig), ", ".join(neg_sig) or "none",
              len(pos_sig), ", ".join(pos_sig) or "none",
              len(null_), ", ".join(null_) or "none",
              len(degen), ", ".join(degen) or "none"))
    A("")
    A("## Models and counts")
    A("")
    A("| model | file | n_layers | items with gold | L0 | L0N | L1 |")
    A("|---|---|---|---|---|---|---|")
    for m, r in res.items():
        A("| %s | %s | %s | %d | %d | %d | %d |" % (
            m, r["source_file"], ",".join(str(x) for x in r["n_layers"]),
            r["n_items_used"], r["n_by_level"]["L0"], r["n_by_level"]["L0N"],
            r["n_by_level"]["L1"]))
    A("")

    A("## Ceiling diagnostic (read this before any table below)")
    A("")
    A("`kl_to_final` at the LAST layer is 0 by construction -- the logit lens applied to")
    A("`resid_post[n_layers-1]` reproduces the final logits exactly. No threshold, however")
    A("small, can ever be crossed at that layer, so every backwards-scanned depth is capped")
    A("at `(n_layers-1)/n_layers` and a criterion whose threshold is small relative to the")
    A("penultimate-layer KL is pinned to that ceiling for every item.")
    A("")
    A("| model | max abs KL at last layer (mean curve) | (first-pos curve) | depth ceiling |")
    A("|---|---|---|---|")
    for m, r in res.items():
        f = r["final_layer_kl"]
        A("| %s | %.2e | %.2e | %.4f |" % (m, f["mean_curve_max_abs"],
                                           f["first_curve_max_abs"], f["depth_ceiling"]))
    A("")

    A("## Implementation check (exact): position-0 recomputation vs `per_position_depths[0]`")
    A("")
    A("`first_pos_profile` and `per_position_depths[0]` describe the same position, so the")
    A("frozen criterion recomputed here must reproduce the stored value bit-for-bit.")
    A("")
    A("| model | tau | n compared | n exact match | max abs diff |")
    A("|---|---|---|---|---|")
    for m, r in res.items():
        for t, c in r["implementation_exact_check"].items():
            A("| %s | %s | %d | %d | %s |" % (
                m, t, c["n_compared"], c["n_exact_match"],
                "n/a" if c["max_abs_diff"] is None else "%.3e" % c["max_abs_diff"]))
    A("")

    A("## Validity check: recomputed frozen tau=0.1 vs stored `depth.depth_tau_0.1`")
    A("")
    A("The stored metric is the MEAN over the 8 per-position depths; the recomputation")
    A("here thresholds a single curve (either the position-averaged curve or position 0).")
    A("They are therefore different estimators of the same construct and are expected to")
    A("rank-correlate, not to match. How strongly they actually do is reported below --")
    A("this is a validity result, not a formality.")
    A("")
    A("| model | curve | scope | rho (p) | n | distinct recomputed | distinct stored |")
    A("|---|---|---|---|---|---|---|")
    for m, r in res.items():
        for key, c in r["validity"].items():
            curve, scope = key.split("|")
            A("| %s | %s | %s | %s | %d | %d | %d |" % (
                m, curve, scope, fmt_rho(c), c["n"], c["n_distinct_x"], c["n_distinct_y"]))
    A("")

    A("## Baseline: the paper's own stored frozen metric on these same items")
    A("")
    A("`stored_depth_tau_0.1_mean8` = `record.depth.depth_tau_0.1`, the mean of the eight")
    A("per-position frozen depths -- the metric the dose-response claim rests on.")
    A("`stored_depth_tau_0.1_pos0` = `per_position_depths[0].depth_tau_0.1`, the same")
    A("criterion on position 0 alone (apples-to-apples with the `first` curve tables).")
    A("")
    A("| model | baseline | L0 gold~depth | L0 H~depth | L0N gold~depth | L1 gold~depth |")
    A("|---|---|---|---|---|---|")
    for m, r in res.items():
        for bname in ("stored_depth_tau_0.1_mean8", "stored_depth_tau_0.1_pos0"):
            g0 = r["baseline"]["%s|L0" % bname]["gold_vs_depth"]
            h0 = r["baseline"]["%s|L0" % bname]["entropy_vs_depth"]
            gn = r["baseline"]["%s|L0N" % bname]["gold_vs_depth"]
            g1 = r["baseline"]["%s|L1" % bname]["gold_vs_depth"]
            A("| %s | %s | %s (k=%d) | %s | %s | %s |" % (
                m, bname, fmt_rho(g0), g0["n_distinct_y"], fmt_rho(h0),
                fmt_rho(gn), fmt_rho(g1)))
    A("")

    for curve in CURVES:
        A("## Curve source: `%s`" % ("layer_profile (position-averaged)" if curve == "mean"
                                     else "first_pos_profile (position 0)"))
        A("")
        A("### Within-L0 dose-response: gold_logprob_per_token vs depth")
        A("")
        A("`k` = number of distinct depth values in the cell (tie ceiling). A cell with")
        A("k<=2 cannot express a dose-response no matter what rho says.")
        A("")
        header = "| model | " + " | ".join(CRITERIA) + " |"
        A(header)
        A("|---" * (len(CRITERIA) + 1) + "|")
        for m, r in res.items():
            cells = []
            for crit in CRITERIA:
                c = r["cells"]["%s|%s|L0" % (curve, crit)]["gold_vs_depth"]
                if c["degenerate"]:
                    cells.append("n/a (k=%d)" % c["n_distinct_y"])
                else:
                    cells.append("%+.3f (p=%.3f, k=%d)" % (c["rho"], c["p"], c["n_distinct_y"]))
            A("| %s | %s |" % (m, " | ".join(cells)))
        A("")

        A("### Within-L0 entanglement: final_entropy_mean vs depth")
        A("")
        A(header)
        A("|---" * (len(CRITERIA) + 1) + "|")
        for m, r in res.items():
            cells = []
            for crit in CRITERIA:
                c = r["cells"]["%s|%s|L0" % (curve, crit)]["entropy_vs_depth"]
                cells.append("n/a" if c["degenerate"]
                             else "%+.3f (p=%.3f)" % (c["rho"], c["p"]))
            A("| %s | %s |" % (m, " | ".join(cells)))
        A("")

        for lv in ["L0N", "L1"]:
            A("### Within-%s: gold vs depth" % lv)
            A("")
            A(header)
            A("|---" * (len(CRITERIA) + 1) + "|")
            for m, r in res.items():
                cells = []
                for crit in CRITERIA:
                    c = r["cells"]["%s|%s|%s" % (curve, crit, lv)]["gold_vs_depth"]
                    if c["degenerate"]:
                        cells.append("n/a (k=%d)" % c["n_distinct_y"])
                    else:
                        cells.append("%+.3f (p=%.3f, k=%d)" % (c["rho"], c["p"],
                                                               c["n_distinct_y"]))
                A("| %s | %s |" % (m, " | ".join(cells)))
            A("")

        A("### Depth value spread (within L0): distinct values / mean / sd")
        A("")
        A(header)
        A("|---" * (len(CRITERIA) + 1) + "|")
        for m, r in res.items():
            cells = []
            for crit in CRITERIA:
                cell = r["cells"]["%s|%s|L0" % (curve, crit)]
                mu = "n/a" if cell["depth_mean"] is None else "%.3f" % cell["depth_mean"]
                sd = "n/a" if cell["depth_sd"] is None else "%.3f" % cell["depth_sd"]
                cells.append("k=%d, %s+-%s" % (len(cell["depth_values"]), mu, sd))
            A("| %s | %s |" % (m, " | ".join(cells)))
        A("")

    A("## Reference: gold vs final_entropy_mean within each level (the confound itself)")
    A("")
    A("| model | L0 | L0N | L1 |")
    A("|---|---|---|---|")
    for m, r in res.items():
        row = []
        for lv in LEVELS:
            c = r["cells"]["mean|abs_t0.1|%s" % lv]["gold_vs_entropy"]
            row.append(fmt_rho(c))
        A("| %s | %s |" % (m, " | ".join(row)))
    A("")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models = load_items()
    res = analyze(models)
    payload = {
        "generated_by": "harness/relative_depth_analysis.py",
        "inputs": sorted(os.path.basename(p) for p in glob.glob(RAW_GLOB)),
        "criteria": {k: CRIT_LABEL[k] for k in CRITERIA},
        "rel_alphas": REL_ALPHAS,
        "ent_taus": ENT_TAUS,
        "levels": LEVELS,
        "curve_sources": {"mean": "layer_profile (averaged over 8 answer positions)",
                          "first": "first_pos_profile (answer position 0)"},
        "results": res,
        "per_item": {m: b["items"] for m, b in models.items()},
    }
    jpath = os.path.join(OUT_DIR, "relative_depth.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    mpath = os.path.join(OUT_DIR, "RELATIVE_DEPTH.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(write_markdown(res, mpath))
    print("wrote", jpath)
    print("wrote", mpath)

    # console digest: the decisive cell
    print("\n=== BASELINE within-L0 gold vs stored frozen depth ===")
    for m, r in res.items():
        a = r["baseline"]["stored_depth_tau_0.1_mean8|L0"]["gold_vs_depth"]
        b = r["baseline"]["stored_depth_tau_0.1_pos0|L0"]["gold_vs_depth"]
        ha = r["baseline"]["stored_depth_tau_0.1_mean8|L0"]["entropy_vs_depth"]
        print(" %-24s mean8 %s (k=%d)  pos0 %s (k=%d)  | H~depth mean8 %s" % (
            m.split("/")[-1], fmt_rho(a), a["n_distinct_y"], fmt_rho(b),
            b["n_distinct_y"], fmt_rho(ha)))
    print("\n=== within-L0 gold vs depth (curve=mean) ===")
    for m, r in res.items():
        parts = []
        for crit in CRITERIA:
            c = r["cells"]["mean|%s|L0" % crit]["gold_vs_depth"]
            parts.append("%s=%s(k=%d)" % (crit, fmt_rho_short(c), c["n_distinct_y"]))
        print(" %-24s %s" % (m.split("/")[-1], "  ".join(parts)))
    print("\n=== within-L0 gold vs depth (curve=first) ===")
    for m, r in res.items():
        parts = []
        for crit in CRITERIA:
            c = r["cells"]["first|%s|L0" % crit]["gold_vs_depth"]
            parts.append("%s=%s(k=%d)" % (crit, fmt_rho_short(c), c["n_distinct_y"]))
        print(" %-24s %s" % (m.split("/")[-1], "  ".join(parts)))
    print("\n=== within-L0 entropy vs depth (curve=mean) ===")
    for m, r in res.items():
        parts = []
        for crit in CRITERIA:
            c = r["cells"]["mean|%s|L0" % crit]["entropy_vs_depth"]
            parts.append("%s=%s" % (crit, fmt_rho_short(c)))
        print(" %-24s %s" % (m.split("/")[-1], "  ".join(parts)))
    print("\n=== validity (recomputed abs0.1 vs stored mean-of-positions, all_gold) ===")
    for m, r in res.items():
        for curve in CURVES:
            c = r["validity"]["%s|all_gold" % curve]
            print(" %-24s %-6s %s n=%d" % (m.split("/")[-1], curve, fmt_rho(c), c["n"]))
    print("\n=== implementation exact check (pos0 recomp vs per_position_depths[0]) ===")
    for m, r in res.items():
        s = " ".join("tau%s:%d/%d(max%.1e)" % (t, c["n_exact_match"], c["n_compared"],
                                               c["max_abs_diff"] or 0.0)
                     for t, c in r["implementation_exact_check"].items())
        print(" %-24s %s" % (m.split("/")[-1], s))
    print("\n=== final-layer KL (ceiling) ===")
    for m, r in res.items():
        f = r["final_layer_kl"]
        print(" %-24s max|KL_last| mean=%.2e first=%.2e ceiling=%.4f" % (
            m.split("/")[-1], f["mean_curve_max_abs"], f["first_curve_max_abs"],
            f["depth_ceiling"]))


if __name__ == "__main__":
    main()
