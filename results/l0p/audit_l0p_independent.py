"""
獨立稽核腳本 —— 完全不讀 l0p_analysis.py 的原始碼,只依照文字規格重新實作。
"""
import json
import numpy as np
from scipy.stats import rankdata, spearmanr

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
TAGS = [
    "EleutherAI_pythia-410m",
    "EleutherAI_pythia-1b",
    "EleutherAI_pythia-1.4b",
    "EleutherAI_pythia-2.8b",
    "allenai_OLMo-2-0425-1B",
]

def load_group(tag, subdir, level):
    path = rf"{BASE}\results\{subdir}\pilot_{tag}.json"
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for r in d["records"]:
        if r.get("level") != level:
            continue
        if "error" in r:
            continue
        x = r.get("gold_logprob_per_token")
        y = r.get("final_entropy_mean")
        if x is None or y is None:
            continue
        out.append({"id": r["id"], "x": float(x), "y": float(y)})
    return out

def ols_c_coef(rx, ry, c):
    """OLS: ry ~ 1 + rx + c.  Return beta_c, t_c, n, df."""
    n = len(rx)
    X = np.column_stack([np.ones(n), rx, c])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    resid = ry - X @ beta
    df = n - X.shape[1]  # n - 3
    sigma2 = (resid @ resid) / df
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    beta_c = beta[2]
    se_c = se[2]
    t_c = beta_c / se_c
    return beta_c, t_c, n, df

def restricted_sample(l0, l0p, lo0, hi0, loP, hiP):
    lo = max(lo0, loP)
    hi = min(hi0, hiP)
    rl0 = [r for r in l0 if lo <= r["x"] <= hi]
    rl0p = [r for r in l0p if lo <= r["x"] <= hi]
    return rl0, rl0p, lo, hi

def a1_from_restricted(rl0, rl0p):
    xs = np.array([r["x"] for r in rl0] + [r["x"] for r in rl0p])
    ys = np.array([r["y"] for r in rl0] + [r["y"] for r in rl0p])
    c = np.array([1] * len(rl0) + [0] * len(rl0p))
    n = len(xs)
    rx = rankdata(xs) / n
    ry = rankdata(ys) / n
    beta_c, t_c, n_, df = ols_c_coef(rx, ry, c)
    return beta_c, t_c, n_, df

def a2_full(l0, l0p):
    x0 = np.array([r["x"] for r in l0]); y0 = np.array([r["y"] for r in l0])
    xP = np.array([r["x"] for r in l0p]); yP = np.array([r["y"] for r in l0p])
    rho0 = spearmanr(x0, y0).statistic
    rhoP = spearmanr(xP, yP).statistic
    dz = np.arctanh(rhoP) - np.arctanh(rho0)
    return rho0, rhoP, dz

results = {}
for tag in TAGS:
    l0 = load_group(tag, "raw", "L0")
    l0p = load_group(tag, "l0p", "L0P")

    x0s = [r["x"] for r in l0]; xPs = [r["x"] for r in l0p]
    lo0, hi0 = min(x0s), max(x0s)
    loP, hiP = min(xPs), max(xPs)
    overlap = max(0.0, min(hi0, hiP) - max(lo0, loP))
    fraction = overlap / (hiP - loP)

    rl0, rl0p, lo, hi = restricted_sample(l0, l0p, lo0, hi0, loP, hiP)
    beta_c, t_c, n_restricted, df = a1_from_restricted(rl0, rl0p)

    rho0, rhoP, dz = a2_full(l0, l0p)

    # range-aligned dz: spearman recomputed within the restricted sample per group
    rx0 = np.array([r["x"] for r in rl0]); ry0 = np.array([r["y"] for r in rl0])
    rxP = np.array([r["x"] for r in rl0p]); ryP = np.array([r["y"] for r in rl0p])
    rho0_r = spearmanr(rx0, ry0).statistic
    rhoP_r = spearmanr(rxP, ryP).statistic
    dz_aligned = np.arctanh(rhoP_r) - np.arctanh(rho0_r)

    results[tag] = dict(
        lo0=lo0, hi0=hi0, loP=loP, hiP=hiP, overlap=overlap, fraction=fraction,
        beta_c=beta_c, t_c=t_c, n_restricted=n_restricted, n_l0_r=len(rl0), n_l0p_r=len(rl0p), df=df,
        rho0=rho0, rhoP=rhoP, dz=dz,
        rho0_r=rho0_r, rhoP_r=rhoP_r, dz_aligned=dz_aligned,
        l0_ids=[r["id"] for r in l0], l0p_ids=[r["id"] for r in l0p],
        rl0_ids=[r["id"] for r in rl0], rl0p_ids=[r["id"] for r in rl0p],
    )

print("=== per-model results (my independent reimplementation) ===")
for tag in TAGS:
    r = results[tag]
    print(f"{tag}: fraction={r['fraction']:.4f} beta={r['beta_c']:+.4f} t={r['t_c']:+.4f} "
          f"n_restricted={r['n_restricted']}({r['n_l0_r']}/{r['n_l0p_r']}) "
          f"rhoP={r['rhoP']:+.4f} rho0={r['rho0']:+.4f} dz={r['dz']:+.4f} dz_aligned={r['dz_aligned']:+.4f}")

mean_beta = np.mean([results[t]["beta_c"] for t in TAGS])
mean_t = np.mean([results[t]["t_c"] for t in TAGS])
mean_dz = np.mean([results[t]["dz"] for t in TAGS])
mean_dz_aligned = np.mean([results[t]["dz_aligned"] for t in TAGS])
print()
print(f"mean beta = {mean_beta:+.4f}")
print(f"mean t    = {mean_t:+.4f}")
print(f"mean dz   = {mean_dz:+.4f}")
print(f"mean dz_aligned = {mean_dz_aligned:+.4f}")

# save results for reuse in permutation test
import pickle
with open(r"C:\Users\USER\AppData\Local\Temp\claude\D--ai----\791f0911-cf6e-4aee-aed2-337be19acbbc\scratchpad\audit_results.pkl", "wb") as f:
    pickle.dump(results, f)

