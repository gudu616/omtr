"""Pooled model-external anchor + deduped replication check (OMTR).

Two questions, both from already-collected data. Nothing is re-run on a GPU here.

TASK A — deduped replication
    results/night/pilot_EleutherAI_pythia-{410m,1.4b}-deduped.json were run on
    Pythia checkpoints trained on the DEDUPLICATED Pile. Same battery, same
    record schema. Within-L0 (and within-L0N) Spearman of
    gold_logprob_per_token vs depth.depth_tau_0.1 — does the memorisation
    dose-response survive deduplication?

TASK B — pooled model-external anchor
    The per-item corpus window count (median over the "ok" probes for that
    item's corpus, exactly as harness/followup_analysis.py's T2_dupcount does
    it) is a MODEL-EXTERNAL dose variable: it is measured on the training
    corpus, not on the model. Within-L0, count vs depth_tau_0.1 is negative in
    all five main runs, but no single run is significant except OLMo.

    Pooling by Stouffer / Fisher meta-combination would be INVALID: the four
    Pythia runs are scored on the SAME 17 items with the SAME count vector, so
    their five p-values are far from independent — a chance alignment of counts
    with item difficulty is shared by all four runs at once and would be
    counted four times.

    Instead: an item-level permutation that respects the sharing.
      * one permutation of the 17 Pythia items is drawn per iteration and
        applied to ALL FOUR Pythia runs (so the shared-item dependence is
        reproduced under the null exactly as it exists in the data),
      * OLMo's 20 items are permuted independently (different corpus, different
        item set, independent run),
      * the five rhos are recomputed and reduced to one combined statistic
        (mean Fisher z),
      * 20000 iterations; p = (1 + #{perm <= observed}) / (iters + 1), one-sided
        negative, which is the direction the memorisation account predicts.

Usage:
    .venv/Scripts/python.exe harness/pooled_anchor_analysis.py
Outputs:
    results/relative_depth/pooled_anchor.json
    results/relative_depth/POOLED_ANCHOR.md   (section appended, not clobbered)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, norm

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "results" / "relative_depth"

MAIN_TAGS = [
    "EleutherAI_pythia-410m",
    "EleutherAI_pythia-1b",
    "EleutherAI_pythia-1.4b",
    "EleutherAI_pythia-2.8b",
    "allenai_OLMo-2-0425-1B",
]
DEDUPED_TAGS = [
    "EleutherAI_pythia-410m-deduped",
    "EleutherAI_pythia-1.4b-deduped",
]
N_ITER = 20000
SEED = 20260821


def corpus_for(tag):
    """Which pretraining corpus the item counts must come from."""
    return "pile" if "pythia" in tag.lower() else "olmo_mix"


def load_records(path):
    """OK records only: an 'error' record or a missing gold is not scoreable."""
    data = json.load(open(path, encoding="utf-8"))
    recs = [r for r in data["records"]
            if "error" not in r and r.get("gold_logprob_per_token") is not None]
    n_layers = None
    for r in data["records"]:
        if r.get("layer_profile"):
            n_layers = len(r["layer_profile"])   # inferred, never hardcoded
            break
    return recs, n_layers, data.get("meta", {})


def within_level(recs, level, xkey="gold_logprob_per_token"):
    rs = [r for r in recs if r["level"] == level and r.get("depth")]
    if len(rs) < 6:
        return None
    x = [r[xkey] for r in rs]
    y = [r["depth"]["depth_tau_0.1"] for r in rs]
    rho, p = spearmanr(x, y)
    return {"n": len(rs), "rho": float(rho), "p": float(p)}


def fisher_z(rho, n=None):
    """arctanh with a guard: a perfect permutation would otherwise give inf."""
    eps = 1e-12
    return float(np.arctanh(np.clip(rho, -1 + eps, 1 - eps)))


# ---------------------------------------------------------------- TASK A ----
def task_a():
    out = {}
    for tag in DEDUPED_TAGS:
        recs, n_layers, meta = load_records(PROJ / "results" / "night" / f"pilot_{tag}.json")
        out[tag] = {
            "model": meta.get("model", tag.replace("_", "/", 1)),
            "n_layers": n_layers,
            "n_ok_records": len(recs),
            "L0": within_level(recs, "L0"),
            "L0N": within_level(recs, "L0N"),
            "L1": within_level(recs, "L1"),
        }
    # the non-deduped counterparts, recomputed here rather than quoted
    for tag in ["EleutherAI_pythia-410m", "EleutherAI_pythia-1.4b"]:
        recs, n_layers, meta = load_records(PROJ / "results" / "raw" / f"pilot_{tag}.json")
        out[tag + " (non-deduped reference)"] = {
            "model": meta.get("model", tag.replace("_", "/", 1)),
            "n_layers": n_layers,
            "n_ok_records": len(recs),
            "L0": within_level(recs, "L0"),
            "L0N": within_level(recs, "L0N"),
            "L1": within_level(recs, "L1"),
        }
    return out


# ---------------------------------------------------------------- TASK B ----
def item_counts(battery, ver, recs, corpus):
    """Per-item (median over ok probes) corpus window count, L0 only.

    Replicates followup_analysis.py T2_dupcount exactly: record id -> battery
    title -> l0_verification entry -> probes[corpus] -> median of the counts
    whose status == 'ok'. Items with no ok probe in that corpus are dropped.
    """
    ids, counts, depths = [], [], []
    for r in recs:
        if r["level"] != "L0" or not r.get("depth"):
            continue
        title = battery.get(r["id"], {}).get("title")
        v = ver.get(title)
        if not v:
            continue
        ok = [p["count"] for p in v["probes"].get(corpus, []) if p["status"] == "ok"]
        if not ok:
            continue
        ids.append(r["id"])
        counts.append(float(np.median(ok)))
        depths.append(float(r["depth"]["depth_tau_0.1"]))
    order = np.argsort(ids)          # canonical item order, so the shared
    ids = [ids[i] for i in order]    # permutation lines up across runs
    return ids, np.array(counts)[order], np.array(depths)[order]


def task_b():
    battery = {it["id"]: it for it in
               json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))["items"]}
    ver = {r["title"]: r for r in
           json.load(open(PROJ / "battery" / "l0_verification.json", encoding="utf-8"))}

    runs = []
    for tag in MAIN_TAGS:
        recs, n_layers, meta = load_records(PROJ / "results" / "raw" / f"pilot_{tag}.json")
        corpus = corpus_for(tag)
        ids, counts, depths = item_counts(battery, ver, recs, corpus)
        rho, p = spearmanr(counts, depths)
        runs.append({
            "tag": tag, "model": meta.get("model", tag), "corpus": corpus,
            "n_layers": n_layers, "family": "pythia" if "pythia" in tag else "olmo",
            "item_ids": ids, "counts": counts, "depths": depths,
            "n": len(ids), "rho": float(rho), "p_marginal": float(p),
        })

    pythia = [r for r in runs if r["family"] == "pythia"]
    olmo = [r for r in runs if r["family"] == "olmo"]

    # the sharing this design has to respect: identical item set AND identical
    # count vector across the four Pythia runs. Assert it, do not assume it.
    ref_ids, ref_counts = pythia[0]["item_ids"], pythia[0]["counts"]
    for r in pythia[1:]:
        assert r["item_ids"] == ref_ids, f"{r['tag']} item set differs from {pythia[0]['tag']}"
        assert np.allclose(r["counts"], ref_counts), f"{r['tag']} count vector differs"

    z_obs = [fisher_z(r["rho"]) for r in runs]
    combined_obs = float(np.mean(z_obs))
    # sensitivity: precision-weighted (sqrt(n-3)) version of the same reduction
    zw_obs = [fisher_z(r["rho"]) * np.sqrt(r["n"] - 3) for r in runs]
    combined_w_obs = float(np.mean(zw_obs))

    rng = np.random.default_rng(SEED)
    n_py = len(ref_ids)
    perm_combined = np.empty(N_ITER)
    perm_combined_w = np.empty(N_ITER)
    perm_combined_py = np.empty(N_ITER)     # Pythia-only, same shared permutation
    perm_combined_py_indep = np.empty(N_ITER)   # WRONG null, kept to size the inflation
    perm_rhos = np.empty((N_ITER, len(runs)))
    for it in range(N_ITER):
        perm_py = rng.permutation(n_py)          # ONE permutation, all 4 Pythia
        zs, zws, rhos = [], [], []
        for r in runs:
            if r["family"] == "pythia":
                c = r["counts"][perm_py]
            else:
                c = r["counts"][rng.permutation(r["n"])]   # OLMo independent
            rho = spearmanr(c, r["depths"]).statistic
            rhos.append(float(rho))
            zs.append(fisher_z(rho))
            zws.append(fisher_z(rho) * np.sqrt(r["n"] - 3))
        perm_rhos[it] = rhos
        perm_combined[it] = np.mean(zs)
        perm_combined_w[it] = np.mean(zws)
        perm_combined_py[it] = np.mean([z for z, r in zip(zs, runs)
                                        if r["family"] == "pythia"])
        # the null you would get if you pretended the four Pythia runs were
        # independent replications on independent items — this is what Stouffer
        # implicitly assumes, and its sd is visibly too small
        perm_combined_py_indep[it] = np.mean(
            [fisher_z(spearmanr(r["counts"][rng.permutation(r["n"])], r["depths"]).statistic)
             for r in pythia])

    def perm_p(obs, dist):
        one = float((1 + np.sum(dist <= obs)) / (len(dist) + 1))          # negative direction
        two = float((1 + np.sum(np.abs(dist) >= abs(obs))) / (len(dist) + 1))
        # a permutation p is itself an estimate; report its Monte Carlo se so the
        # reader can see whether N_ITER is enough to separate it from 0.05
        mc_se = float(np.sqrt(one * (1 - one) / len(dist)))
        return {"p_one_sided_negative": one, "p_two_sided": two,
                "p_one_sided_mc_se": mc_se}

    # per-run marginal permutation p under the SAME machinery (for context only)
    per_run = []
    for i, r in enumerate(runs):
        per_run.append({
            "tag": r["tag"], "model": r["model"], "corpus": r["corpus"],
            "n_layers": r["n_layers"], "n_items": r["n"],
            "rho_count_vs_depth_tau_0.1": round(r["rho"], 3),
            "p_asymptotic": round(r["p_marginal"], 4),
            "p_permutation_one_sided": round(
                float((1 + np.sum(perm_rhos[:, i] <= r["rho"])) / (N_ITER + 1)), 4),
            "fisher_z": round(fisher_z(r["rho"]), 4),
        })

    # the INVALID comparison, computed only to show how much it overstates
    # signed one-sided z per run: negative rho -> negative z
    stouffer_z = float(np.sum([-np.sign(r["rho"]) * norm.ppf(r["p_marginal"] / 2)
                               for r in runs]) / np.sqrt(len(runs)))
    stouffer_p = float(norm.cdf(stouffer_z))

    return {
        "design": {
            "x": "per-item median corpus window count (median of ok probes, "
                 "battery/l0_verification.json probes[corpus]) — model-external",
            "y": "depth.depth_tau_0.1 (frozen absolute-threshold metric)",
            "scope": "within L0 only",
            "n_iterations": N_ITER,
            "seed": SEED,
            "permutation_unit": "item",
            "sharing_handled": "one permutation of the 17 shared Pile items is applied "
                               "to all four Pythia runs per iteration; OLMo's 20 "
                               "olmo_mix items are permuted independently",
            "combined_statistic": "mean Fisher z (arctanh rho) over the five runs",
            "pythia_shared_items": ref_ids,
            "olmo_items": olmo[0]["item_ids"],
        },
        "per_run": per_run,
        "pooled": {
            "combined_mean_fisher_z_observed": round(combined_obs, 4),
            "permutation_null_mean": round(float(perm_combined.mean()), 4),
            "permutation_null_sd": round(float(perm_combined.std(ddof=1)), 4),
            **{k: round(v, 5) for k, v in perm_p(combined_obs, perm_combined).items()},
            "implied_pooled_rho": round(float(np.tanh(combined_obs)), 3),
        },
        "pooled_precision_weighted_sensitivity": {
            "combined_mean_weighted_z_observed": round(combined_w_obs, 4),
            "permutation_null_sd": round(float(perm_combined_w.std(ddof=1)), 4),
            **{k: round(v, 5) for k, v in perm_p(combined_w_obs, perm_combined_w).items()},
        },
        "pythia_only_leave_out_olmo": {
            "note": "The five-run pooled test is checked against the four Pythia runs "
                    "alone, because OLMo is on its own significant and could carry the "
                    "pool by itself. Same shared-item permutation, four runs, mean "
                    "Fisher z. This is the honest question: does the shared-item Pile "
                    "evidence stand without OLMo?",
            "n_runs": len(pythia),
            "combined_mean_fisher_z_observed": round(
                float(np.mean([fisher_z(r["rho"]) for r in pythia])), 4),
            "permutation_null_sd": round(float(perm_combined_py.std(ddof=1)), 4),
            **{k: round(v, 5) for k, v in perm_p(
                float(np.mean([fisher_z(r["rho"]) for r in pythia])), perm_combined_py).items()},
            "olmo_alone_rho": round(olmo[0]["rho"], 3),
            "olmo_alone_p_permutation_one_sided": round(
                float((1 + np.sum(perm_rhos[:, MAIN_TAGS.index(olmo[0]["tag"])]
                                  <= olmo[0]["rho"])) / (N_ITER + 1)), 5),
            "wrong_independent_null_sd": round(float(perm_combined_py_indep.std(ddof=1)), 4),
            "wrong_independent_null_p": round(float(perm_p(
                float(np.mean([fisher_z(r["rho"]) for r in pythia])),
                perm_combined_py_indep)["p_one_sided_negative"]), 5),
            "inflation_note": "Pretending the four Pythia runs are independent shrinks the "
                              "null sd and turns the same observed statistic into a much "
                              "smaller p. That gap is the size of the error Stouffer makes.",
        },
        "invalid_for_contrast_only": {
            "note": "Stouffer over the five marginal p-values. INVALID here — the four "
                    "Pythia runs share one 17-item set and one count vector, so this "
                    "treats one shared coincidence as four independent ones.",
            "stouffer_z": round(stouffer_z, 3),
            "stouffer_p_one_sided": round(stouffer_p, 5),
        },
    }


def render_md(a, b):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append(f"\n## Pooled model-external anchor & deduped replication ({ts})\n")
    L.append("Generated by `harness/pooled_anchor_analysis.py`; data "
             "`results/raw/pilot_*.json` + `results/night/pilot_*-deduped.json`.\n")

    L.append("\n### A. Deduped replication (within-condition, gold_logprob vs depth_tau_0.1)\n")
    L.append("| run | layers | L0 n | L0 rho | L0 p | L0N n | L0N rho | L0N p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in a.items():
        l0, l0n = v["L0"], v["L0N"]
        label = v["model"] + (" — non-deduped reference" if "non-deduped" in k else "")
        L.append(f"| {label} "
                 f"| {v['n_layers']} | {l0['n']} | {l0['rho']:.3f} | {l0['p']:.4f} "
                 f"| {l0n['n']} | {l0n['rho']:.3f} | {l0n['p']:.4f} |")
    L.append("")

    L.append("\n### B1. Corpus count vs depth within L0, per run\n")
    L.append("| run | corpus | layers | n items | rho | p (asymptotic) | p (permutation, 1-sided) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in b["per_run"]:
        L.append(f"| {r['model']} | {r['corpus']} | {r['n_layers']} | {r['n_items']} "
                 f"| {r['rho_count_vs_depth_tau_0.1']:.3f} | {r['p_asymptotic']:.4f} "
                 f"| {r['p_permutation_one_sided']:.4f} |")

    p = b["pooled"]
    w = b["pooled_precision_weighted_sensitivity"]
    inv = b["invalid_for_contrast_only"]
    L.append(f"\n### B2. Pooled item-level permutation ({b['design']['n_iterations']} iterations, "
             f"seed {b['design']['seed']})\n")
    L.append(f"- Permutation unit: {b['design']['permutation_unit']}. "
             f"{b['design']['sharing_handled']}.")
    L.append(f"- Combined statistic: {b['design']['combined_statistic']}.")
    L.append(f"- **Observed combined mean Fisher z = {p['combined_mean_fisher_z_observed']:.4f}** "
             f"(implied pooled rho = {p['implied_pooled_rho']:.3f}).")
    L.append(f"- Permutation null: mean {p['permutation_null_mean']:.4f}, "
             f"sd {p['permutation_null_sd']:.4f}.")
    L.append(f"- **Permutation p = {p['p_one_sided_negative']:.5f}** (one-sided, negative "
             f"direction; Monte Carlo se {p['p_one_sided_mc_se']:.5f}); "
             f"two-sided {p['p_two_sided']:.5f}.")
    L.append(f"- Precision-weighted sensitivity: observed "
             f"{w['combined_mean_weighted_z_observed']:.4f}, "
             f"p = {w['p_one_sided_negative']:.5f} (one-sided).")
    py = b["pythia_only_leave_out_olmo"]
    L.append(f"- **Pythia-only (drop OLMo, {py['n_runs']} runs, same shared permutation): "
             f"combined mean Fisher z = {py['combined_mean_fisher_z_observed']:.4f}, "
             f"p = {py['p_one_sided_negative']:.5f}** (one-sided). "
             f"OLMo alone: rho = {py['olmo_alone_rho']:.3f}, "
             f"p = {py['olmo_alone_p_permutation_one_sided']:.5f}. {py['note']}")
    L.append(f"- Size of the dependence error: the correct shared-item null for the four "
             f"Pythia runs has sd {py['permutation_null_sd']:.4f}; a null that wrongly "
             f"treats them as independent has sd {py['wrong_independent_null_sd']:.4f} and "
             f"would report p = {py['wrong_independent_null_p']:.5f} for the same observed "
             f"statistic (vs the correct {py['p_one_sided_negative']:.5f}).")
    L.append(f"- For contrast only, the invalid Stouffer combination gives "
             f"z = {inv['stouffer_z']:.3f}, p = {inv['stouffer_p_one_sided']:.5f}. "
             f"{inv['note']}")
    L.append("")
    return "\n".join(L)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = task_a()
    b = task_b()
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "harness/pooled_anchor_analysis.py",
        "task_a_deduped_replication": a,
        "task_b_pooled_anchor": b,
    }
    jpath = OUT_DIR / "pooled_anchor.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    mpath = OUT_DIR / "POOLED_ANCHOR.md"
    header = "" if mpath.exists() else "# Pooled anchor / relative-depth appendix\n"
    with open(mpath, "a", encoding="utf-8") as f:   # append a section, never clobber
        f.write(header + render_md(a, b))

    print(f"json -> {jpath}")
    print(f"md   -> {mpath}")
    print("\n=== TASK A: deduped within-L0 (gold vs depth_tau_0.1) ===")
    for k, v in a.items():
        print(f"  {k:52s} L0 n={v['L0']['n']} rho={v['L0']['rho']:+.3f} p={v['L0']['p']:.4f}"
              f"   L0N n={v['L0N']['n']} rho={v['L0N']['rho']:+.3f} p={v['L0N']['p']:.4f}")
    print("\n=== TASK B: corpus count vs depth, within L0 ===")
    for r in b["per_run"]:
        print(f"  {r['model']:28s} {r['corpus']:9s} n={r['n_items']} "
              f"rho={r['rho_count_vs_depth_tau_0.1']:+.3f} p_asym={r['p_asymptotic']:.4f} "
              f"p_perm={r['p_permutation_one_sided']:.4f}")
    p = b["pooled"]
    print(f"\n  POOLED mean Fisher z = {p['combined_mean_fisher_z_observed']:+.4f} "
          f"(rho ~ {p['implied_pooled_rho']:+.3f})")
    print(f"  null mean {p['permutation_null_mean']:+.4f} sd {p['permutation_null_sd']:.4f}")
    print(f"  permutation p (1-sided neg) = {p['p_one_sided_negative']:.5f}   "
          f"two-sided = {p['p_two_sided']:.5f}")
    w = b["pooled_precision_weighted_sensitivity"]
    print(f"  weighted sensitivity: {w['combined_mean_weighted_z_observed']:+.4f} "
          f"p={w['p_one_sided_negative']:.5f}")
    py = b["pythia_only_leave_out_olmo"]
    print(f"  PYTHIA-ONLY (drop OLMo): z={py['combined_mean_fisher_z_observed']:+.4f} "
          f"p={py['p_one_sided_negative']:.5f}   "
          f"(OLMo alone rho={py['olmo_alone_rho']:+.3f} p={py['olmo_alone_p_permutation_one_sided']:.5f})")
    inv = b["invalid_for_contrast_only"]
    print(f"  [INVALID contrast] Stouffer z={inv['stouffer_z']:+.3f} p={inv['stouffer_p_one_sided']:.5f}")


if __name__ == "__main__":
    main()
