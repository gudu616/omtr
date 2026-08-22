"""R1 adversarial verification, part 2: the collinearity and matched-predictability tests."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata, mannwhitneyu, norm, wilcoxon

PROJ = Path(__file__).resolve().parents[2]
TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b", "EleutherAI_pythia-1.4b",
        "EleutherAI_pythia-2.8b", "allenai_OLMo-2-0425-1B"]
SHORT = {t: t.split("_")[-1] for t in TAGS}
G, D, E, C = ("gold_logprob_per_token", "tf_depth_tau_0.1_mean",
              "tf_final_entropy_mean", "tf_frac_at_cap")

OUT = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.append(s)

tf = {t: [r for r in json.load(open(PROJ / "results" / "night" / f"tf_depth_{t}.json",
                                    encoding="utf-8"))["records"] if "error" not in r]
      for t in TAGS}
greedy = {t: {r["id"]: r for r in json.load(
    open(PROJ / "results" / "raw" / f"pilot_{t}.json", encoding="utf-8"))["records"]
    if "error" not in r} for t in TAGS}
old = {t: json.load(open(PROJ / "results" / f"analysis_{t}.json", encoding="utf-8")) for t in TAGS}

def sub(t, lv): return [r for r in tf[t] if r["level"] == lv]
def col(rs, k): return np.array([r[k] for r in rs], float)

def fisher_z(r):
    r = min(max(r, -0.999999), 0.999999)
    return 0.5 * math.log((1 + r) / (1 - r))

def rank_biserial(a, b):
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    r, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)

# ---------------------------------------------------------------- A
P("## A. Is 'memory strength' distinguishable from 'final-layer confidence'?")
P("")
P("Teacher-forced mean gold logprob is the negative cross-entropy of the model on the")
P("gold span; tf_final_entropy_mean is the mean entropy of the same predictive")
P("distributions. For a model that is accurate on those tokens the two are the same")
P("number up to the calibration gap, so they cannot be treated as independent variables.")
P("")
P("| model | cond | rho(gold, -H) | mean(gold + H) | sd(gold + H) | OLS slope gold on -H | R2 |")
P("|---|---|---|---|---|---|---|")
for t in TAGS:
    for lv in ("L0", "L0N"):
        rs = sub(t, lv); g, e = col(rs, G), col(rs, E)
        r, _ = spearmanr(g, -e)
        X = np.column_stack([np.ones(len(e)), -e])
        coef, *_ = np.linalg.lstsq(X, g, rcond=None)
        pred = X @ coef
        r2 = 1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum()
        P(f"| {SHORT[t]} | {lv} | {r:+.3f} | {np.mean(g + e):+.3f} | {np.std(g + e, ddof=1):.3f} "
          f"| {coef[1]:+.3f} | {r2:.3f} |")
P("")

P("### A1. Swap the predictor: does entropy predict depth as well as gold logprob does?")
P("")
P("| model | cond | rho(gold, depth) | rho(H, depth) | rho(gold,depth \\| H) | rho(H,depth \\| gold) |")
P("|---|---|---|---|---|---|")
for t in TAGS:
    for lv in ("L0", "L0N"):
        rs = sub(t, lv); g, d, e = col(rs, G), col(rs, D), col(rs, E)
        a, pa = spearmanr(g, d); b, pb = spearmanr(e, d)
        c, pc = partial_spearman(g, d, e); f_, pf = partial_spearman(e, d, g)
        P(f"| {SHORT[t]} | {lv} | {a:+.3f} ({pa:.3f}) | {b:+.3f} ({pb:.3f}) "
          f"| {c:+.3f} ({pc:.3f}) | {f_:+.3f} ({pf:.3f}) |")
P("")

# ---------------------------------------------------------------- B
P("## B. Matched-predictability test: at equal gold logprob, does memorized text")
P("converge earlier?")
P("")
P("Each L0N item is matched to its nearest L0 item in gold logprob (with replacement);")
P("pairs with |dgold| > 0.5 nats are dropped as unmatchable. Positive dDepth means the")
P("memorized item converged LATER, i.e. against the hypothesis.")
P("")
P("| model | n pairs | mean \\|dgold\\| | median dDepth (L0 - L0N) | Wilcoxon p | median dCap |")
P("|---|---|---|---|---|---|")
for t in TAGS:
    l0, l0n = sub(t, "L0"), sub(t, "L0N")
    dd, dc, dg = [], [], []
    for r in l0n:
        j = int(np.argmin([abs(x[G] - r[G]) for x in l0]))
        gap = abs(l0[j][G] - r[G])
        if gap > 0.5:
            continue
        dd.append(l0[j][D] - r[D]); dc.append(l0[j][C] - r[C]); dg.append(gap)
    if len(dd) >= 5:
        try:
            _, p = wilcoxon(dd)
        except Exception:
            p = float("nan")
        P(f"| {SHORT[t]} | {len(dd)} | {np.mean(dg):.3f} | {np.median(dd):+.4f} | {p:.3f} "
          f"| {np.median(dc):+.3f} |")
    else:
        P(f"| {SHORT[t]} | {len(dd)} | - | too few matchable pairs | - | - |")
P("")

# ---------------------------------------------------------------- C
P("## C. Range restriction, matched on gold SPREAD (nats) rather than item count")
P("")
P("For every contiguous-in-gold window of L0 items whose gold spread is within 25% of")
P("the L0N spread for that model, report the within-window rho.")
P("")
P("| model | L0N spread | L0N rho | n matched windows | median window rho | window rho range |")
P("|---|---|---|---|---|---|")
for t in TAGS:
    l0, l0n = sub(t, "L0"), sub(t, "L0N")
    gl = col(l0n, G); spread = gl.max() - gl.min()
    srt = sorted(l0, key=lambda r: r[G])
    rhos = []
    for i in range(len(srt)):
        for j in range(i + 5, len(srt) + 1):
            w = srt[i:j]
            s = w[-1][G] - w[0][G]
            if abs(s - spread) / spread <= 0.25:
                r, _ = spearmanr(col(w, G), col(w, D))
                if not np.isnan(r):
                    rhos.append(r)
    rl0n, _ = spearmanr(gl, col(l0n, D))
    if rhos:
        P(f"| {SHORT[t]} | {spread:.2f} | {rl0n:+.3f} | {len(rhos)} | {np.median(rhos):+.3f} "
          f"| [{min(rhos):+.3f}, {max(rhos):+.3f}] |")
    else:
        P(f"| {SHORT[t]} | {spread:.2f} | {rl0n:+.3f} | 0 | - | - |")
P("")

# ---------------------------------------------------------------- D
P("## D. Bootstrap CI on the Fisher-z difference (asymptotic SE is optimistic at n=11-17)")
P("")
rng = np.random.default_rng(20260820)
P("| model | dz observed | bootstrap 95% CI | frac bootstrap dz < 0 |")
P("|---|---|---|---|")
for t in TAGS:
    l0, l0n = sub(t, "L0"), sub(t, "L0N")
    r0 = spearmanr(col(l0, G), col(l0, D))[0]
    r1 = spearmanr(col(l0n, G), col(l0n, D))[0]
    obs = fisher_z(r0) - fisher_z(r1)
    bs = []
    for _ in range(20000):
        a = [l0[i] for i in rng.integers(0, len(l0), len(l0))]
        b = [l0n[i] for i in rng.integers(0, len(l0n), len(l0n))]
        ra = spearmanr(col(a, G), col(a, D))[0]
        rb = spearmanr(col(b, G), col(b, D))[0]
        if np.isnan(ra) or np.isnan(rb):
            continue
        bs.append(fisher_z(ra) - fisher_z(rb))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    P(f"| {SHORT[t]} | {obs:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {np.mean(bs < 0):.3f} |")
P("")

# ---------------------------------------------------------------- E
P("## E. Convention check: does my rank-biserial reproduce the frozen greedy contrasts?")
P("")
P("| model | my greedy depth rb | frozen analysis value | my greedy entropy rb | frozen |")
P("|---|---|---|---|---|")
for t in TAGS:
    a = [greedy[t][r["id"]] for r in sub(t, "L0") if r["id"] in greedy[t]]
    b = [greedy[t][r["id"]] for r in sub(t, "L0N") if r["id"] in greedy[t]]
    da = np.array([x["depth"]["depth_tau_0.1"] for x in a])
    db = np.array([x["depth"]["depth_tau_0.1"] for x in b])
    ea = np.array([x["final_entropy_mean"] for x in a])
    eb = np.array([x["final_entropy_mean"] for x in b])
    fr = old[t]["contrasts_rank_biserial"]["L0_vs_L0N"]
    P(f"| {SHORT[t]} | {rank_biserial(da, db):+.3f} | {fr['depth.depth_tau_0.1']:+.3f} "
      f"| {rank_biserial(ea, eb):+.3f} | {fr['final_entropy']:+.3f} |")
P("")

# ---------------------------------------------------------------- F
P("## F. Effect size of the memory-specific residual (what is left after predictability)")
P("")
P("Difference of Fisher-z (L0 minus L0N) is the memory-specific increment in the")
P("dose-response slope; dR2 of condition in the pooled rank model is the memory-specific")
P("increment in explained variance. Both, side by side, tf vs greedy.")
P("")
P("| model | tf dz (L0-L0N) | greedy dz | tf dR2 cond | tf partial rho cond\\|gold |")
P("|---|---|---|---|---|")
fup = json.load(open(PROJ / "archive" / "followup_report.json", encoding="utf-8"))
for t in TAGS:
    l0, l0n = sub(t, "L0"), sub(t, "L0N")
    r0 = spearmanr(col(l0, G), col(l0, D))[0]; r1 = spearmanr(col(l0n, G), col(l0n, D))[0]
    dz = fisher_z(r0) - fisher_z(r1)
    t1 = fup[t]["T1_within"]
    dzg = fisher_z(t1["L0"]["rho"]) - fisher_z(t1["L0N"]["rho"])
    rs = l0 + l0n
    y = rankdata(col(rs, D)); xg = rankdata(col(rs, G))
    xc = np.array([1.0 if r["level"] == "L0" else 0.0 for r in rs])
    def r2(X):
        X1 = np.column_stack([np.ones(len(y))] + X)
        coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return 1 - (y - X1 @ coef).var() / y.var()
    dr2 = r2([xg, xc]) - r2([xg])
    pc, ppc = partial_spearman(xc, col(rs, D), col(rs, G))
    P(f"| {SHORT[t]} | {dz:+.3f} | {dzg:+.3f} | {dr2:+.4f} | {pc:+.3f} (p={ppc:.3f}) |")
P("")

(Path(__file__).parent / "r1_out2.md").write_text("\n".join(OUT), encoding="utf-8")
print("\nWROTE r1_out2.md")
