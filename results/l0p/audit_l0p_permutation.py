"""
獨立稽核 —— 置換檢定(不讀原始 l0p_analysis.py)
種子刻意選一個不是 20260822 的值,確認量級,不追求逐位相同。
"""
import json
import pickle
import numpy as np
from scipy.stats import rankdata, spearmanr

SEED = 314159265
rng = np.random.default_rng(SEED)
N_PERM = 5000

with open(r"C:\Users\USER\AppData\Local\Temp\claude\D--ai----\791f0911-cf6e-4aee-aed2-337be19acbbc\scratchpad\audit_results.pkl", "rb") as f:
    R = pickle.load(f)

TAGS = list(R.keys())
PYTHIA = [t for t in TAGS if t.startswith("EleutherAI")]
OLMO = [t for t in TAGS if t not in PYTHIA]
assert len(PYTHIA) == 4 and len(OLMO) == 1

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])

def load_group(tag, subdir, level):
    path = rf"{BASE}\results\{subdir}\pilot_{tag}.json"
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in d["records"]:
        if r.get("level") != level:
            continue
        if "error" in r:
            continue
        x = r.get("gold_logprob_per_token")
        y = r.get("final_entropy_mean")
        if x is None or y is None:
            continue
        out[r["id"]] = (float(x), float(y))
    return out

# xy_by_id[tag] = {id: (x,y)} for both L0 and L0P pooled
xy_by_id = {}
for tag in TAGS:
    l0 = load_group(tag, "raw", "L0")
    l0p = load_group(tag, "l0p", "L0P")
    merged = dict(l0)
    merged.update(l0p)
    xy_by_id[tag] = merged

# sanity: pythia models share identical id pools
pythia_l0_ids = R[PYTHIA[0]]["l0_ids"]
pythia_l0p_ids = R[PYTHIA[0]]["l0p_ids"]
for t in PYTHIA:
    assert sorted(R[t]["l0_ids"]) == sorted(pythia_l0_ids)
    assert sorted(R[t]["l0p_ids"]) == sorted(pythia_l0p_ids)

def ols_c_t(rx, ry, c):
    n = len(rx)
    X = np.column_stack([np.ones(n), rx, c])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    resid = ry - X @ beta
    df = n - X.shape[1]
    sigma2 = (resid @ resid) / df
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    return beta[2] / se[2]

# ---------- observed statistics (recomputed from R, cross-check) ----------
obs_mean_t = np.mean([R[t]["t_c"] for t in TAGS])
obs_mean_dz = np.mean([R[t]["dz"] for t in TAGS])
print(f"observed mean t (A1)  = {obs_mean_t:+.4f}")
print(f"observed mean dz (A2) = {obs_mean_dz:+.4f}")

pythia_pool_ids = pythia_l0_ids + pythia_l0p_ids       # 35 ids, shared across 4 pythia models
n_pythia_l0 = len(pythia_l0_ids)                        # 17

olmo_tag = OLMO[0]
olmo_pool_ids = R[olmo_tag]["l0_ids"] + R[olmo_tag]["l0p_ids"]  # 38 ids
n_olmo_l0 = len(R[olmo_tag]["l0_ids"])                   # 20

perm_stats_a1 = np.empty(N_PERM)
perm_stats_a2 = np.empty(N_PERM)

for it in range(N_PERM):
    # --- one shared draw for the 4 pythia models ---
    shuffled = rng.permutation(pythia_pool_ids)
    fake_l0_set_py = set(shuffled[:n_pythia_l0].tolist())
    # fake_l0p_set_py = remainder

    # --- independent draw for OLMo ---
    shuffled_o = rng.permutation(olmo_pool_ids)
    fake_l0_set_o = set(shuffled_o[:n_olmo_l0].tolist())

    t_vals = []
    dz_vals = []
    for tag in TAGS:
        fake_l0_set = fake_l0_set_py if tag in PYTHIA else fake_l0_set_o

        # ---- A1: restricted sample frozen, only label (fake_l0 membership) moves ----
        rl0_ids = R[tag]["rl0_ids"]
        rl0p_ids = R[tag]["rl0p_ids"]
        restricted_ids = rl0_ids + rl0p_ids
        xs = np.array([xy_by_id[tag][i][0] for i in restricted_ids])
        ys = np.array([xy_by_id[tag][i][1] for i in restricted_ids])
        c = np.array([1 if i in fake_l0_set else 0 for i in restricted_ids])
        n = len(xs)
        rx = rankdata(xs) / n
        ry = rankdata(ys) / n
        t_vals.append(ols_c_t(rx, ry, c))

        # ---- A2: full sample, fake labels ----
        all_ids = R[tag]["l0_ids"] + R[tag]["l0p_ids"]
        fake_l0_ids = [i for i in all_ids if i in fake_l0_set]
        fake_l0p_ids = [i for i in all_ids if i not in fake_l0_set]
        x0 = np.array([xy_by_id[tag][i][0] for i in fake_l0_ids])
        y0 = np.array([xy_by_id[tag][i][1] for i in fake_l0_ids])
        xP = np.array([xy_by_id[tag][i][0] for i in fake_l0p_ids])
        yP = np.array([xy_by_id[tag][i][1] for i in fake_l0p_ids])
        rho0 = spearmanr(x0, y0).statistic
        rhoP = spearmanr(xP, yP).statistic
        dz_vals.append(np.arctanh(rhoP) - np.arctanh(rho0))

    perm_stats_a1[it] = np.mean(t_vals)
    perm_stats_a2[it] = np.mean(dz_vals)

p_a1 = np.mean(np.abs(perm_stats_a1) >= np.abs(obs_mean_t))
p_a2 = np.mean(np.abs(perm_stats_a2) >= np.abs(obs_mean_dz))

print(f"seed = {SEED}, n_perm = {N_PERM}")
print(f"A1 permutation p (two-tailed) = {p_a1:.4f}   (claimed 0.0986)")
print(f"A2 permutation p (two-tailed) = {p_a2:.4f}   (claimed 0.809)")
print(f"perm stat A1 mean={perm_stats_a1.mean():+.4f} sd={perm_stats_a1.std():.4f}")
print(f"perm stat A2 mean={perm_stats_a2.mean():+.4f} sd={perm_stats_a2.std():.4f}")

