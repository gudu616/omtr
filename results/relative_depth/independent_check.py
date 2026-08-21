"""Independent adversarial re-implementation of relative / entropy-normalized
convergence-depth criteria for the OMTR pilot.

Written from the written spec only. The other agent's script
(harness/relative_depth_analysis.py) and its outputs were deliberately NOT read.

Spec implemented here
---------------------
depth(curve, threshold) = (last layer index i with kl_to_final[i] > threshold) + 1
                          normalized by n_layers (= len(curve));
                          0.0 if no layer ever exceeds the threshold.

This convention is validated against the stored per_position_depths[0]
depth_tau_{0.05,0.1,0.5} using the first_pos_profile curve (exact match test).

Criteria:
  abs   tau      : threshold = tau                      (the paper's frozen metric)
  rel   alpha    : threshold = alpha * kl_to_final[0]   (relative to layer-0 KL)
  ent   tau      : threshold = tau * H_final            (entropy-normalized)
                   two readings of "H_final" are computed because the phrase is
                   ambiguous:
                     ent   -> H_final = record['final_entropy_mean']
                     entc  -> H_final = curve[-1]['entropy']  (curve-internal)

Curves:
  mean  -> record['layer_profile']     (averaged over the 8 answer positions)
  first -> record['first_pos_profile'] (answer position 0 exactly)

Correlations: Spearman rho of the depth against
  (a) gold_logprob_per_token   -- the dose-response of interest
  (b) final_entropy_mean       -- the sharpness confound
computed WITHIN each level (L0, L0N, L1) separately, per model.
"""

import io
import json
import os
from itertools import combinations

from scipy.stats import spearmanr

RAW = r"D:/ai/research/omtr/results/raw"
OUT_DIR = r"D:/ai/research/omtr/results/relative_depth"
OUT_JSON = os.path.join(OUT_DIR, "independent_check.json")

MODELS = [
    ("EleutherAI/pythia-410m", "pilot_EleutherAI_pythia-410m.json"),
    ("EleutherAI/pythia-1b", "pilot_EleutherAI_pythia-1b.json"),
    ("EleutherAI/pythia-1.4b", "pilot_EleutherAI_pythia-1.4b.json"),
    ("EleutherAI/pythia-2.8b", "pilot_EleutherAI_pythia-2.8b.json"),
    ("allenai/OLMo-2-0425-1B", "pilot_allenai_OLMo-2-0425-1B.json"),
]

ALPHAS = [0.5, 0.25, 0.1, 0.05]
ENT_TAUS = [0.05, 0.1, 0.25]
LEVELS = ["L0", "L0N", "L1"]


# ---------------------------------------------------------------- primitives
def depth_last_crossing(kl, threshold):
    """Normalized last layer whose kl_to_final strictly exceeds threshold."""
    n = len(kl)
    last = -1
    for i, v in enumerate(kl):
        if v > threshold:
            last = i
    return (last + 1) / n


def kl_curve(profile):
    return [float(layer["kl_to_final"]) for layer in profile]


def spear(xs, ys):
    """Spearman with explicit degeneracy reporting."""
    # exact float distinctness: rounding here would under-count near-ties that
    # are genuinely distinct means-of-8 (verified on pythia-410m L0).
    k = len(set(ys))
    if k < 2 or len(set(xs)) < 2:
        return {"rho": None, "p": None, "k_distinct_depth": k, "n": len(ys)}
    rho, p = spearmanr(xs, ys)
    return {
        "rho": round(float(rho), 6),
        "p": float(p),
        "k_distinct_depth": k,
        "n": len(ys),
    }


# ---------------------------------------------------------------- data load
def load(path):
    with io.open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    recs = blob["records"] if isinstance(blob, dict) else blob
    keep = []
    for r in recs:
        if "error" in r and r["error"]:
            continue
        if r.get("gold_logprob_per_token") is None:
            continue
        if not r.get("layer_profile"):
            continue
        keep.append(r)
    return blob.get("meta", {}), keep


def criteria_for(record, curve_key):
    """Return {criterion_name: depth} for one record and one curve source."""
    profile = record[curve_key]
    kl = kl_curve(profile)
    kl0 = kl[0]
    h_rec = float(record["final_entropy_mean"])
    h_curve = float(profile[-1]["entropy"])

    out = {"abs_t0.1": depth_last_crossing(kl, 0.1)}
    for a in ALPHAS:
        out["rel_a%g" % a] = depth_last_crossing(kl, a * kl0)
    for t in ENT_TAUS:
        out["ent_t%g" % t] = depth_last_crossing(kl, t * h_rec)
        out["entc_t%g" % t] = depth_last_crossing(kl, t * h_curve)
    return out


# ---------------------------------------------------------------- validation
def validate_position0(records):
    """Recompute frozen taus on first_pos_profile; compare to stored
    per_position_depths[0]. Returns (n_checked, n_exact, max_abs_diff)."""
    n = 0
    exact = 0
    worst = 0.0
    for r in records:
        ppd = r.get("per_position_depths")
        if not ppd:
            continue
        stored = ppd[0]
        kl = kl_curve(r["first_pos_profile"])
        for tau in (0.05, 0.1, 0.5):
            key = "depth_tau_%g" % tau
            if key not in stored:
                continue
            mine = depth_last_crossing(kl, tau)
            diff = abs(mine - float(stored[key]))
            worst = max(worst, diff)
            n += 1
            exact += diff == 0.0
    return n, exact, worst


def last_layer_kl_stats(records):
    vals = []
    for r in records:
        for key in ("layer_profile", "first_pos_profile"):
            vals.append(abs(float(r[key][-1]["kl_to_final"])))
    return {"max_abs_last_layer_kl": max(vals), "n": len(vals)}


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    result = {"spec": __doc__.strip(), "models": {}}

    for model, fname in MODELS:
        meta, recs = load(os.path.join(RAW, fname))
        n_layers = len(recs[0]["layer_profile"])
        assert all(len(r["layer_profile"]) == n_layers for r in recs)
        assert all(len(r["first_pos_profile"]) == n_layers for r in recs)

        n_chk, n_exact, worst = validate_position0(recs)

        entry = {
            "n_layers": n_layers,
            "n_records_used": len(recs),
            "validation_pos0_exactness": {
                "checks": n_chk,
                "exact": n_exact,
                "max_abs_diff": worst,
            },
            "last_layer_kl": last_layer_kl_stats(recs),
            "depth_cap": (n_layers - 1) / n_layers,
            "all_items_recomputed_abs_t0.1_vs_stored": {
                label: spear(
                    [float(r["depth"]["depth_tau_0.1"]) for r in recs],
                    [
                        depth_last_crossing(kl_curve(r[curve_key]), 0.1)
                        for r in recs
                    ],
                )
                for curve_key, label in (
                    ("layer_profile", "mean_curve"),
                    ("first_pos_profile", "first_curve"),
                )
            },
            "levels": {},
        }

        for level in LEVELS:
            sub = [r for r in recs if r["level"] == level]
            if len(sub) < 4:
                continue
            gold = [float(r["gold_logprob_per_token"]) for r in sub]
            ent = [float(r["final_entropy_mean"]) for r in sub]
            stored = [float(r["depth"]["depth_tau_0.1"]) for r in sub]

            lvl = {
                "n": len(sub),
                "baseline_stored_tau0.1_mean8": {
                    "gold_vs_depth": spear(gold, stored),
                    "entropy_vs_depth": spear(ent, stored),
                },
                "gold_vs_final_entropy": spear(gold, ent),
                "curves": {},
            }

            for curve_key, label in (
                ("layer_profile", "mean_curve"),
                ("first_pos_profile", "first_curve"),
            ):
                per_crit = {}
                depths_by_crit = {}
                for r in sub:
                    for name, val in criteria_for(r, curve_key).items():
                        depths_by_crit.setdefault(name, []).append(val)
                for name, depths in depths_by_crit.items():
                    per_crit[name] = {
                        "gold_vs_depth": spear(gold, depths),
                        "entropy_vs_depth": spear(ent, depths),
                        "depth_min": min(depths),
                        "depth_max": max(depths),
                        "depth_at_cap_frac": sum(
                            1 for d in depths if abs(d - (n_layers - 1) / n_layers) < 1e-12
                        )
                        / len(depths),
                    }
                # agreement of recomputed frozen tau=0.1 with the stored mean-of-8
                per_crit["_recomputed_abs_t0.1_vs_stored"] = spear(
                    stored, depths_by_crit["abs_t0.1"]
                )
                lvl["curves"][label] = per_crit

            entry["levels"][level] = lvl
        result["models"][model] = entry

    with io.open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    # ------------------------------------------------------------ console
    for model, entry in result["models"].items():
        v = entry["validation_pos0_exactness"]
        print(
            "%-26s L=%2d  pos0 exact %d/%d (max diff %.3e)  last-layer |KL| max %.2e"
            % (
                model,
                entry["n_layers"],
                v["exact"],
                v["checks"],
                v["max_abs_diff"],
                entry["last_layer_kl"]["max_abs_last_layer_kl"],
            )
        )
    print()
    order = ["rel_a0.5", "rel_a0.25", "rel_a0.1", "rel_a0.05",
             "ent_t0.05", "ent_t0.1", "ent_t0.25",
             "entc_t0.05", "entc_t0.1", "entc_t0.25", "abs_t0.1"]
    for level in LEVELS:
        for curve in ("mean_curve", "first_curve"):
            print("=== level %s / %s : gold vs depth ===" % (level, curve))
            hdr = "%-12s" % "criterion"
            for m, _ in MODELS:
                hdr += "%-24s" % m.split("/")[-1]
            print(hdr)
            row = "%-12s" % "BASE(stored)"
            for m, _ in MODELS:
                lv = result["models"][m]["levels"].get(level)
                s = lv["baseline_stored_tau0.1_mean8"]["gold_vs_depth"] if lv else None
                row += "%-24s" % (
                    "n/a" if not s or s["rho"] is None
                    else "%+.3f p=%.4g k=%d" % (s["rho"], s["p"], s["k_distinct_depth"])
                )
            print(row)
            for crit in order:
                row = "%-12s" % crit
                for m, _ in MODELS:
                    lv = result["models"][m]["levels"].get(level)
                    s = lv["curves"][curve][crit]["gold_vs_depth"] if lv else None
                    row += "%-24s" % (
                        "n/a k=%d" % s["k_distinct_depth"] if s["rho"] is None
                        else "%+.3f p=%.4g k=%d" % (s["rho"], s["p"], s["k_distinct_depth"])
                    )
                print(row)
            print()

    print("=== L0 mean_curve : final_entropy_mean vs depth (confound) ===")
    for crit in order:
        row = "%-12s" % crit
        for m, _ in MODELS:
            s = result["models"][m]["levels"]["L0"]["curves"]["mean_curve"][crit]["entropy_vs_depth"]
            row += "%-24s" % (
                "n/a" if s["rho"] is None else "%+.3f p=%.4g" % (s["rho"], s["p"])
            )
        print(row)
    row = "%-12s" % "BASE(stored)"
    for m, _ in MODELS:
        s = result["models"][m]["levels"]["L0"]["baseline_stored_tau0.1_mean8"]["entropy_vs_depth"]
        row += "%-24s" % ("n/a" if s["rho"] is None else "%+.3f p=%.4g" % (s["rho"], s["p"]))
    print(row)

    print()
    print("=== L0 gold vs final_entropy_mean (the confound itself) ===")
    for m, _ in MODELS:
        s = result["models"][m]["levels"]["L0"]["gold_vs_final_entropy"]
        print("  %-26s %+.3f p=%.3g" % (m, s["rho"], s["p"]))

    print()
    print("=== recomputed frozen tau=0.1 vs stored mean-of-8 (L0 only) ===")
    for m, _ in MODELS:
        for curve in ("mean_curve", "first_curve"):
            s = result["models"][m]["levels"]["L0"]["curves"][curve]["_recomputed_abs_t0.1_vs_stored"]
            print("  %-26s %-11s %s" % (
                m, curve,
                "n/a k=%d" % s["k_distinct_depth"] if s["rho"] is None
                else "%+.3f p=%.4g" % (s["rho"], s["p"])))

    print("\nwrote %s" % OUT_JSON)


if __name__ == "__main__":
    main()
