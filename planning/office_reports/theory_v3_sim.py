"""N-g / N1 / N-a: prereg-grade simulation at B=20000, SE <= .0025.

Speed comes from an EXACT identity, not an approximation. For a one-sample t on
v_i = s_i*y_i under sign flips, sum(v^2) = sum(y^2) is invariant, so t depends
on the flip vector only through A = sum(s_i*y_i):

    t(A) = A*sqrt(n-1) / ( sqrt(n) * sqrt(Q - A^2/n) ),   Q = sum(y^2)

so a whole permutation distribution is one matrix product Y @ S.T. Verified
against the direct computation to 9 decimals.

The flip matrix S is drawn ONCE and reused across simulation replicates. Each
replicate's p-value is still a valid Monte Carlo permutation p; reuse only
correlates replicates slightly, and it is what makes B=20000 affordable.

Masks are the REAL per-run matched-item sets from theory_v3_regimes.json, not
random subsets.

Run: .venv/Scripts/python.exe planning/office_reports/theory_v3_sim.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

PROJ = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHIA = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b",
          "EleutherAI_pythia-1.4b", "EleutherAI_pythia-2.8b"]
OLMO = "allenai_OLMo-2-0425-1B"
B = 20000
BLK = 2500
ALPHA = 0.05


def t_from_A(A, n, Q):
    return A * np.sqrt(n - 1) / (np.sqrt(n) * np.sqrt(np.maximum(Q - A * A / n, 1e-30)))


class Family:
    """one evidence family: R runs over a shared item union, each run using its
    own subset (M4 union reading)."""

    def __init__(self, union, per_run_sets, rng):
        self.m = len(union)
        idx = dict((k, i) for i, k in enumerate(union))
        self.M = np.zeros((len(per_run_sets), self.m), np.float32)
        for r, st in enumerate(per_run_sets):
            for k in st:
                self.M[r, idx[k]] = 1.0
        self.n = self.M.sum(1)
        self.S = rng.choice([-1.0, 1.0], size=(B, self.m)).astype(np.float32)

    def draw(self, C, delta, rho_fam, rng):
        """y (C, R, m); shared item effect a_i gives corr(run,run') = rho_fam"""
        R = self.M.shape[0]
        a = rng.standard_normal((C, 1, self.m)).astype(np.float32)
        e = rng.standard_normal((C, R, self.m)).astype(np.float32)
        y = delta + np.sqrt(rho_fam) * a + np.sqrt(1 - rho_fam) * e
        return y * self.M

    def p_and_T(self, y):
        """two-sided sign-flip p and observed family statistic, per replicate.
        Blocked over the permutation dimension: an unblocked (C,R,B) temporary
        at B=20000 is 160 MB and the expression needs three of them, which is
        what killed the first run on this machine. Blocking caps it at ~16 MB
        and is also friendlier to cache."""
        C, R, m = y.shape
        Q = (y * y).sum(-1)                                   # (C, R)
        A_obs = y.sum(-1)
        T_obs = t_from_A(A_obs, self.n, Q).mean(-1)            # (C,)
        absT = np.abs(T_obs)[:, None]
        yr = y.reshape(C * R, m)
        nn = self.n[None, :, None]
        c1 = (np.sqrt(self.n - 1) / np.sqrt(self.n))[None, :, None]
        Qc = Q[:, :, None]
        cnt = np.zeros(C, np.int64)
        for b0 in range(0, B, BLK):
            Sb = self.S[b0:b0 + BLK]
            A = (yr @ Sb.T).reshape(C, R, -1)
            tmp = A * A
            tmp /= nn
            np.subtract(Qc, tmp, out=tmp)
            np.maximum(tmp, 1e-30, out=tmp)
            np.sqrt(tmp, out=tmp)
            np.divide(A, tmp, out=tmp)
            tmp *= c1
            T = tmp.mean(1)
            cnt += (np.abs(T) >= absT).sum(1)
        return (1 + cnt) / (B + 1), T_obs


def tost_ok(y, bound):
    """TOST per family on the pooled items; intersection-union across families
    is exactly level alpha with no correction."""
    C = y.shape[0]
    v = y.reshape(C, -1)
    w = (v != 0)
    n = w.sum(1)
    mean = v.sum(1) / n
    var = (((v - mean[:, None]) ** 2) * w).sum(1) / (n - 1)
    se = np.sqrt(var / n)
    return (tdist.sf((mean + bound) / se, n - 1) < ALPHA) &            (tdist.cdf((mean - bound) / se, n - 1) < ALPHA)


def stouffer(p1, T1, p2, T2):
    z = (np.sign(T1) * norm.isf(p1 / 2) + np.sign(T2) * norm.isf(p2 / 2)) / np.sqrt(2)
    return 2 * norm.sf(np.abs(z)), z


def track(fam_py, fam_ol, delta, rho_fam, eq_bound, n_sim, rng, chunk=500):
    """Run one evidence track. Returns PER-REPLICATE indicator arrays so that
    two tracks can be combined exactly, rather than by multiplying rates."""
    fp, wd, eq = [], [], []
    done = 0
    while done < n_sim:
        C = min(chunk, n_sim - done)
        y1 = fam_py.draw(C, delta, rho_fam, rng)
        y2 = fam_ol.draw(C, delta, 0.0, rng)
        p1, T1 = fam_py.p_and_T(y1)
        p2, T2 = fam_ol.p_and_T(y2)
        pc, z = stouffer(p1, T1, p2, T2)
        fires = pc < ALPHA
        fp.append(fires & (z > 0))
        wd.append(fires & (z <= 0))
        eq.append((~fires) & tost_ok(y1, eq_bound) & tost_ok(y2, eq_bound))
        done += C
    return (np.concatenate(fp), np.concatenate(wd), np.concatenate(eq))


def se_of(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def build(regimes, key, rng):
    masks = regimes["masks"][key]
    U = regimes["regimes"][key]["pythia_union"]
    return (Family(U, [masks[t] for t in PYTHIA], rng),
            Family(masks[OLMO], [masks[OLMO]], rng))


def main():
    reg = json.load(open(HERE / "theory_v3_regimes.json", encoding="utf-8"))
    rng = np.random.default_rng(20260822)
    N_SIM = 40000
    EQ = 0.5
    out = {"B": B, "alpha": ALPHA, "n_sim": N_SIM, "eq_bound": EQ,
           "se_at_p_half": se_of(0.5, N_SIM), "se_at_p_05": se_of(0.05, N_SIM)}
    print("B=%d  n_sim=%d  SE<=%.5f at p=.5, %.5f at p=.05" % (
        B, N_SIM, se_of(0.5, N_SIM), se_of(0.05, N_SIM)), flush=True)

    deltas = [0.0, 0.15, 0.30, 0.50, 0.80]
    rhos = [0.0, 0.5, 0.8]     # the true rho_fam is a GPU-A anchor; bracketed here
    cells = {}                 # (key, rho, delta) -> (fp, wd, eq) indicators

    for key, label in (("A_gold30", "REGIME A (PRIMARY) gold caliper .30"),
                       ("B_dual", "REGIME B (SENSITIVITY) gold .20 + entropy .30")):
        fpy, fol = build(reg, key, rng)
        print("", flush=True)
        print("=== %s ===  Pythia n=%s (union %d), OLMo n=%d" % (
            label, list(fpy.n.astype(int)), fpy.m, int(fol.n[0])), flush=True)
        print("rho_fam  delta  R_specific  R_entangled  wrong_dir   no_rule", flush=True)
        rows = []
        for rf in rhos:
            for d in deltas:
                fp, wd, eq = track(fpy, fol, d, rf, EQ, N_SIM, rng)
                cells[(key, rf, d)] = (fp, wd, eq)
                nr = 1.0 - fp.mean() - wd.mean() - eq.mean()
                print("%7.2f %6.2f %11.4f %12.4f %10.4f %9.4f" % (
                    rf, d, fp.mean(), eq.mean(), wd.mean(), nr), flush=True)
                rows.append({"rho_fam": rf, "delta": d,
                             "R_specific": float(fp.mean()),
                             "R_entangled": float(eq.mean()),
                             "wrong_direction": float(wd.mean()),
                             "no_rule": float(nr),
                             "se_max": se_of(0.5, N_SIM)})
        out[key] = rows

    print("", flush=True)
    print("=== N-a: provenance track as a NECESSARY condition for R-specific ===",
          flush=True)
    l0n_u = ["u%d" % i for i in range(8)]
    ppy = Family(l0n_u, [l0n_u[:7], l0n_u[:6], l0n_u[:5], l0n_u[:2]], rng)
    pol = Family(["o%d" % i for i in range(4)], [["o%d" % i for i in range(4)]], rng)
    print("provenance track n: Pythia %s (union %d), OLMo %d" % (
        list(ppy.n.astype(int)), ppy.m, int(pol.n[0])), flush=True)
    dgs = [0.0, 0.30, 0.50, 0.80]
    prov = {}
    for dp in dgs:
        prov[dp] = track(ppy, pol, dp, 0.8, EQ, N_SIM, rng)
        print("  prov track delta=%.2f fires=%.4f" % (dp, prov[dp][0].mean()), flush=True)
    print("d_gold d_prov  R_specific  R_entangled  wrong_dir  no_rule   "
          "(gold alone) (prov alone)", flush=True)
    joint = []
    for dg in dgs:
        gfp, gwd, geq = cells[("A_gold30", 0.8, dg)]
        for dp in dgs:
            pfp = prov[dp][0]
            spec = float(np.mean(gfp & pfp))        # provenance is NECESSARY
            wrong = float(gwd.mean())
            ent = float(np.mean(geq & ~pfp))
            nr = 1.0 - spec - wrong - ent
            print("%6.2f %6.2f %11.4f %12.4f %10.4f %8.4f %12.4f %12.4f" % (
                dg, dp, spec, ent, wrong, nr, gfp.mean(), pfp.mean()), flush=True)
            joint.append({"delta_gold": dg, "delta_prov": dp,
                          "R_specific": spec, "R_entangled": ent,
                          "wrong_direction": wrong, "no_rule": nr,
                          "gold_track_alone": float(gfp.mean()),
                          "prov_track_alone": float(pfp.mean())})
    out["N_a_joint"] = joint

    o = Path(__file__).with_suffix(".json")
    json.dump(out, open(o, "w", encoding="utf-8"), indent=1)
    print("", flush=True)
    print("-> " + str(o), flush=True)


if __name__ == "__main__":
    main()
