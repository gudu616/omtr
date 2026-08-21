"""Type-I calibration of the A1 label-permutation test in l0p_analysis.py.

Known risk: simple label permutation in ANCOVA is miscalibrated when the
covariate (gold) distribution differs by condition. The synthetic null here
deliberately gives L0 a WIDER gold range than L0P (the expected real
structure), so this measures the miscalibration the real data could hit.

200 replicate null datasets; per replicate, run the a1_primary machinery at
N_ITER=400 and record the combined two-sided p. Report the fraction with
p < .05 (target ~.05) and the p distribution deciles.
"""
import importlib.util
from pathlib import Path
import numpy as np

HARNESS = Path(__file__).resolve().parent / "l0p_analysis.py"
spec = importlib.util.spec_from_file_location("l0p_analysis", HARNESS)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.N_ITER = 400

TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b",
        "EleutherAI_pythia-1.4b", "EleutherAI_pythia-2.8b",
        "allenai_OLMo-2-0425-1B"]


def synth_runs(rep_rng):
    """Null: same entropy~gold relation both conditions. L0 range wider.
    Pythia items shared (same golds/entropies across the four runs would be
    unrealistically strong sharing; instead share item IDS but draw each
    model's values with a shared item effect + model noise, like reality)."""
    l0_item = rep_rng.uniform(-3.0, -0.2, 17)     # shared item-level gold core
    l0p_item = rep_rng.uniform(-2.2, -0.8, 18)
    item_eff0 = rep_rng.normal(0, 0.05, 17)       # shared item quirks
    item_effp = rep_rng.normal(0, 0.05, 18)
    runs = []
    for tag in TAGS:
        is_olmo = "OLMo" in tag
        if is_olmo:
            g0 = rep_rng.uniform(-3.0, -0.2, 20)
            e0 = 1.5 + 0.9 * g0 + rep_rng.normal(0, 0.08, 20)
            ids0 = [f"L0-{k:02d}" for k in range(1, 21)]
        else:
            g0 = l0_item + rep_rng.normal(0, 0.1, 17)
            e0 = 1.5 + 0.9 * g0 + item_eff0 + rep_rng.normal(0, 0.06, 17)
            ids0 = [f"L0-{k:02d}" for k in range(1, 18)]
        gp = (rep_rng.uniform(-2.2, -0.8, 18) if is_olmo
              else l0p_item + rep_rng.normal(0, 0.1, 18))
        ep = 1.5 + 0.9 * gp + (0 if is_olmo else item_effp) \
            + rep_rng.normal(0, 0.06 if not is_olmo else 0.08, 18)
        ids = ids0 + [f"L0P-{k:02d}" for k in range(1, 19)]
        gold = np.concatenate([g0, gp])
        ent = np.concatenate([e0, ep])
        cond = np.array([1] * len(g0) + [0] * 18)
        order = np.argsort(ids)
        runs.append({"tag": tag, "model": tag,
                     "family": "olmo" if is_olmo else "pythia",
                     "ids": [ids[i] for i in order],
                     "gold": gold[order], "ent": ent[order],
                     "cond": cond[order]})
    return runs


def main():
    n_rep = 200
    ps, betas = [], []
    for rep in range(n_rep):
        rep_rng = np.random.default_rng(1_000_000 + rep)
        runs = synth_runs(rep_rng)
        a1 = mod.a1_primary(runs, np.random.default_rng(rep))
        ps.append(a1["p_two_sided"])
        betas.append(a1["combined_mean_beta_observed"])
        if (rep + 1) % 50 == 0:
            print(f"  {rep+1}/{n_rep} done")
    ps = np.array(ps)
    print(f"\nfraction p < .05 : {np.mean(ps < .05):.3f}  "
          f"(binomial se {np.sqrt(.05*.95/n_rep):.3f}, target .05)")
    print(f"fraction p < .10 : {np.mean(ps < .10):.3f}  (target .10)")
    print(f"p deciles: {np.round(np.percentile(ps, np.arange(10, 100, 10)), 3)}")
    print(f"mean |beta| under null: {np.mean(np.abs(betas)):.4f}")


if __name__ == "__main__":
    main()
