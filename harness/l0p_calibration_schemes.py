"""Which A1 permutation scheme is calibrated?

Schemes (all: one shared item-level draw for the 4 Pythia runs, OLMo
independent, combined = mean over 5 runs, two-sided p):
  naive-beta : permute condition labels, statistic = beta_c (current impl)
  naive-t    : permute condition labels, statistic = t(beta_c)
  FL-beta    : Freedman-Lane — fit reduced model y~1+x on the frozen restricted
               sample, permute residual POSITIONS (shared draw = random total
               order of the 35 shared items, applied within each model's
               retained subset), y* = fitted + permuted resid, refit full
               model with OBSERVED labels, statistic = beta_c*.

Conditions:
  null-imbal   entropy = f(gold)+noise both conditions, L0 gold range wider
  null-bal     same, equal gold ranges (isolates the imbalance cause)
  null-weak    imbalanced ranges, noise big enough that |rho| ~ .75 (realistic)
  effect       imbalanced ranges + true condition shift (power check)

Output: FPR (p<.05) or power per scheme x condition. 150 reps, 300 perms.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

TAGS = ["p410", "p1b", "p14", "p28", "olmo"]
N_REP, N_PERM = 150, 300


def synth(rep, cond_name):
    rng = np.random.default_rng(5_000_000 + rep)
    noise = 0.30 if cond_name == "null-weak" else 0.08
    shift = -0.35 if cond_name == "effect" else 0.0
    if cond_name == "null-bal":
        r0lo, r0hi, rplo, rphi = -2.2, -0.8, -2.2, -0.8
    else:
        r0lo, r0hi, rplo, rphi = -3.0, -0.2, -2.2, -0.8
    l0_core = rng.uniform(r0lo, r0hi, 17)
    l0p_core = rng.uniform(rplo, rphi, 18)
    runs = []
    for tag in TAGS:
        olmo = tag == "olmo"
        if olmo:
            g0 = rng.uniform(r0lo, r0hi, 20)
        else:
            g0 = l0_core + rng.normal(0, 0.1, 17)
        gp = rng.uniform(rplo, rphi, 18) if olmo else l0p_core + rng.normal(0, 0.1, 18)
        e0 = 1.5 + 0.9 * g0 + rng.normal(0, noise, len(g0)) + shift
        ep = 1.5 + 0.9 * gp + rng.normal(0, noise, 18)
        ids = ([f"L0-{k:02d}" for k in range(1, len(g0) + 1)]
               + [f"L0P-{k:02d}" for k in range(1, 19)])
        gold, ent = np.concatenate([g0, gp]), np.concatenate([e0, ep])
        cond = np.array([1] * len(g0) + [0] * 18)
        o = np.argsort(ids)
        runs.append({"tag": tag, "family": "olmo" if olmo else "pythia",
                     "ids": [ids[i] for i in o],
                     "gold": gold[o], "ent": ent[o], "cond": cond[o]})
    return runs


def restrict(r):
    g0, gp = r["gold"][r["cond"] == 1], r["gold"][r["cond"] == 0]
    lo, hi = max(g0.min(), gp.min()), min(g0.max(), gp.max())
    return np.where((r["gold"] >= lo) & (r["gold"] <= hi))[0]


def fit_full(gold, ent, cond, want_t=False):
    n = len(gold)
    yr, xr = rankdata(ent) / n, rankdata(gold) / n
    X = np.column_stack([np.ones(n), xr, cond.astype(float)])
    coef, res, *_ = np.linalg.lstsq(X, yr, rcond=None)
    if not want_t:
        return coef[2]
    dof = n - 3
    resid = yr - X @ coef
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(s2 * XtX_inv[2, 2])
    return coef[2] / se if se > 0 else 0.0


def reduced_fit(gold, ent):
    n = len(gold)
    yr, xr = rankdata(ent) / n, rankdata(gold) / n
    X = np.column_stack([np.ones(n), xr])
    coef, *_ = np.linalg.lstsq(X, yr, rcond=None)
    fitted = X @ coef
    return yr, xr, fitted, yr - fitted


def one_rep(runs, scheme, rng):
    idxs = [restrict(r) for r in runs]
    n_shared = len(runs[0]["ids"])
    if scheme in ("naive-beta", "naive-t"):
        want_t = scheme == "naive-t"
        obs = [fit_full(r["gold"][i], r["ent"][i], r["cond"][i], want_t)
               for r, i in zip(runs, idxs)]
        comb = np.mean(obs)
        perm = np.empty(N_PERM)
        for it in range(N_PERM):
            pi = rng.permutation(n_shared)
            vals = []
            for r, i in zip(runs, idxs):
                c = (r["cond"][pi] if r["family"] == "pythia"
                     else r["cond"][rng.permutation(len(r["cond"]))])
                cc = c[i]
                if cc.min() == cc.max():
                    vals.append(0.0)
                else:
                    vals.append(fit_full(r["gold"][i], r["ent"][i], cc, want_t))
            perm[it] = np.mean(vals)
        return float((1 + np.sum(np.abs(perm) >= abs(comb))) / (N_PERM + 1))
    # Freedman-Lane
    pre = []
    for r, i in zip(runs, idxs):
        yr, xr, fitted, resid = reduced_fit(r["gold"][i], r["ent"][i])
        n = len(i)
        X = np.column_stack([np.ones(n), xr, r["cond"][i].astype(float)])
        pre.append({"X": X, "fitted": fitted, "resid": resid, "idx": i})
        # observed statistic = full-model beta on the real y
    obs = [fit_full(r["gold"][i], r["ent"][i], r["cond"][i])
           for r, i in zip(runs, idxs)]
    comb = np.mean(obs)
    perm = np.empty(N_PERM)
    for it in range(N_PERM):
        order = rng.permutation(n_shared)      # shared random total order
        rank_of = np.empty(n_shared, int)
        rank_of[order] = np.arange(n_shared)
        vals = []
        for r, p in zip(runs, pre):
            if r["family"] == "pythia":
                sub = p["idx"]
                loc = np.argsort(rank_of[sub])     # induced permutation of subset
            else:
                loc = rng.permutation(len(p["idx"]))
            ystar = p["fitted"] + p["resid"][loc]
            coef, *_ = np.linalg.lstsq(p["X"], ystar, rcond=None)
            vals.append(coef[2])
        perm[it] = np.mean(vals)
    return float((1 + np.sum(np.abs(perm) >= abs(comb))) / (N_PERM + 1))


def main():
    for cond_name in ["null-imbal", "null-bal", "null-weak", "effect"]:
        line = f"{cond_name:11s}"
        for scheme in ["naive-beta", "naive-t", "FL-beta"]:
            ps = []
            for rep in range(N_REP):
                runs = synth(rep, cond_name)
                ps.append(one_rep(runs, scheme, np.random.default_rng(rep)))
            frac = np.mean(np.array(ps) < 0.05)
            line += f"  {scheme}: {frac:.3f}"
        print(line, flush=True)
    print(f"\n(reps={N_REP}, perms={N_PERM}; null target .05, "
          f"binomial se {np.sqrt(.05*.95/N_REP):.3f})")


if __name__ == "__main__":
    main()
