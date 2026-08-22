"""R5: range matching on the reviewed (greedy) measure.

Scoped to the frozen claim: within-condition dose-response on depth_tau_0.1 from
results/raw/pilot_*.json, the values shipped in results/analysis_*.json and
archive/followup_report.json (relocated since this pass was written). Asks whether
"the matched control stayed flat" is a
property of the control condition or of its restricted gold range / tied depth values.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent.parent
TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b", "EleutherAI_pythia-1.4b",
        "EleutherAI_pythia-2.8b", "allenai_OLMo-2-0425-1B"]
SHORT = {t: t.split("_")[-1] for t in TAGS}

OUT = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.append(s)

pilot = {}
for t in TAGS:
    recs = json.load(open(PROJ / "results" / "raw" / f"pilot_{t}.json",
                          encoding="utf-8"))["records"]
    pilot[t] = [r for r in recs if "error" not in r
                and r.get("gold_logprob_per_token") is not None]

def sub(t, lv):
    return [r for r in pilot[t] if r["level"] == lv]

def gold(rs):
    return np.array([r["gold_logprob_per_token"] for r in rs], float)

def depth(rs):
    return np.array([r["depth"]["depth_tau_0.1"] for r in rs], float)

rng = np.random.default_rng(20260820)

P("## R5 — range matching on the reviewed (greedy) measure")
P("")
P("Scope: the frozen pilot values only. y = `depth.depth_tau_0.1` from")
P("`results/raw/pilot_*.json` (the per-position mean as shipped), x =")
P("`gold_logprob_per_token`. These are the numbers behind `followup_report.json` T1 and")
P("the draft's \"matched control stayed flat\" sentence. Item sets are identical to the")
P("R1 teacher-forced run (verified id-by-id), so nothing below is a sampling difference.")
P("")

# ---------------------------------------------------------------- baseline
P("### R5.0 Baseline (reproduces followup_report.json T1)")
P("")
P("| model | n L0 | rho L0 (p) | n L0N | rho L0N (p) |")
P("|---|---|---|---|---|")
base = {}
for t in TAGS:
    a, b = sub(t, "L0"), sub(t, "L0N")
    r0, p0 = spearmanr(gold(a), depth(a))
    r1, p1 = spearmanr(gold(b), depth(b))
    base[t] = (float(r0), float(r1))
    P(f"| {SHORT[t]} | {len(a)} | {r0:+.3f} ({p0:.4f}) | {len(b)} | {r1:+.3f} ({p1:.4f}) |")
P("")

# ---------------------------------------------------------------- (a) spread
P("### R5.a Gold-logprob spread and overlap, L0 vs L0N")
P("")
P("| model | L0 range | L0 spread | L0 SD | L0N range | L0N spread | L0N SD | overlap | L0 items in L0N range |")
P("|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    a, b = gold(sub(t, "L0")), gold(sub(t, "L0N"))
    lo = max(a.min(), b.min()); hi = min(a.max(), b.max())
    ov = max(0.0, hi - lo)
    union = max(a.max(), b.max()) - min(a.min(), b.min())
    n_in = int(((a >= b.min()) & (a <= b.max())).sum())
    P(f"| {SHORT[t]} | [{a.min():.2f}, {a.max():.2f}] | {a.max()-a.min():.2f} | {a.std(ddof=1):.2f} "
      f"| [{b.min():.2f}, {b.max():.2f}] | {b.max()-b.min():.2f} | {b.std(ddof=1):.2f} "
      f"| {ov:.2f} nats ({ov/union*100:.0f}% of union) | {n_in}/{len(a)} |")
P("")
P("The gold logprob values are byte-identical between the greedy pilot and R1 (the")
P("teacher-forced run recomputed the same quantity and reproduced it to 0.00e+00), so")
P("the non-overlap is a property of the battery, not of either measurement. L0N occupies")
P("the low-predictability tail; only a handful of L0 items ever reach into it.")
P("")

# ---------------------------------------------------------------- (b) windows
P("### R5.b Range-matched L0 windows vs the L0N correlation")
P("")
P("Every contiguous-in-gold window of L0 items (min 5 items) whose gold spread falls")
P("within 25% of that model's L0N spread. `pct <= L0N` is the share of matched windows")
P("whose rho is at least as close to zero as the observed L0N rho, i.e. an empirical")
P("(descriptive, overlapping-window) p-value for \"the control is flatter than a")
P("range-matched slice of the memorized condition\".")
P("")
P("| model | L0N spread | L0N rho | n windows | median window rho | window rho range | pct <= L0N |")
P("|---|---|---|---|---|---|---|")
win = {}
for t in TAGS:
    a, b = sub(t, "L0"), sub(t, "L0N")
    gb = gold(b); spread = gb.max() - gb.min()
    srt = sorted(a, key=lambda r: r["gold_logprob_per_token"])
    rhos = []
    for i in range(len(srt)):
        for j in range(i + 5, len(srt) + 1):
            w = srt[i:j]
            s = w[-1]["gold_logprob_per_token"] - w[0]["gold_logprob_per_token"]
            if abs(s - spread) / spread <= 0.25:
                r, _ = spearmanr(gold(w), depth(w))
                if not np.isnan(r):
                    rhos.append(float(r))
    r1 = base[t][1]
    win[t] = rhos
    if rhos:
        frac = float(np.mean([r >= r1 for r in rhos]))
        P(f"| {SHORT[t]} | {spread:.2f} | {r1:+.3f} | {len(rhos)} | {np.median(rhos):+.3f} "
          f"| [{min(rhos):+.3f}, {max(rhos):+.3f}] | {frac:.2f} |")
    else:
        P(f"| {SHORT[t]} | {spread:.2f} | {r1:+.3f} | 0 | - | - | - |")
P("")

P("Random (non-contiguous) spread-matched subsets of L0, n = n_L0N, 20000 draws kept if")
P("their gold spread is within 25% of the L0N spread:")
P("")
P("| model | n kept | median rho | 5th-95th pct | pct >= L0N rho |")
P("|---|---|---|---|---|")
for t in TAGS:
    a, b = sub(t, "L0"), sub(t, "L0N")
    gb = gold(b); spread = gb.max() - gb.min(); k = len(b)
    ga, da = gold(a), depth(a)
    keep = []
    for _ in range(20000):
        idx = rng.choice(len(a), size=k, replace=False)
        g_, d_ = ga[idx], da[idx]
        s = g_.max() - g_.min()
        if abs(s - spread) / spread <= 0.25:
            r, _ = spearmanr(g_, d_)
            if not np.isnan(r):
                keep.append(float(r))
    r1 = base[t][1]
    if len(keep) >= 30:
        lo, hi = np.percentile(keep, [5, 95])
        P(f"| {SHORT[t]} | {len(keep)} | {np.median(keep):+.3f} | [{lo:+.3f}, {hi:+.3f}] "
          f"| {np.mean([r >= r1 for r in keep]):.3f} |")
    else:
        P(f"| {SHORT[t]} | {len(keep)} | too few matched draws | - | - |")
P("")

# ---------------------------------------------------------------- (c) ties
P("### R5.c The other reason a correlation goes flat: tied depth values")
P("")
P("`depth_tau_0.1` is a mean over positions of a quantity taking n_layers discrete")
P("values, and in the control condition nearly every position sits at the cap, so the")
P("per-item means collapse onto a handful of values. Max attainable |rho| is the")
P("Spearman you would get if the observed depth values were arranged in perfect")
P("monotone order against gold — the ceiling the tie structure imposes.")
P("")
P("| model | cond | n | distinct depth values | largest tie group | depth SD | max attainable \\|rho\\| | observed rho |")
P("|---|---|---|---|---|---|---|---|")
for t in TAGS:
    for lv in ("L0", "L0N"):
        rs = sub(t, lv)
        d = depth(rs); g = gold(rs)
        vals, counts = np.unique(np.round(d, 9), return_counts=True)
        best = spearmanr(np.sort(g), np.sort(np.round(d, 9))[::-1])[0]
        obs = base[t][0 if lv == "L0" else 1]
        P(f"| {SHORT[t]} | {lv} | {len(rs)} | {len(vals)} | {counts.max()}/{len(d)} "
          f"| {d.std(ddof=1):.4f} | {abs(best):.3f} | {obs:+.3f} |")
P("")

P("### R5.d Discretisation test: put L0 on the control's depth grid")
P("")
P("Take the L0 items — where the effect is claimed — and round each item's depth to the")
P("nearest value actually observed in that model's L0N depth support, leaving gold")
P("untouched. If the L0 correlation dies, the control's flatness is explained by the")
P("granularity of the measure in that range rather than by the absence of a relation.")
P("Second column additionally restricts L0 to the L0N gold range (both handicaps at once).")
P("")
P("| model | L0 rho | L0 rho on L0N depth grid | L0 rho, grid + L0N gold range (n) | L0N rho |")
P("|---|---|---|---|---|")
for t in TAGS:
    a, b = sub(t, "L0"), sub(t, "L0N")
    ga, da = gold(a), depth(a)
    grid = np.unique(np.round(depth(b), 9))
    snapped = np.array([grid[np.argmin(np.abs(grid - v))] for v in da])
    r_snap, p_snap = spearmanr(ga, snapped)
    gb = gold(b); lo, hi = gb.min(), gb.max()
    m = (ga >= lo) & (ga <= hi)
    if m.sum() >= 5 and len(set(snapped[m])) > 1:
        r_both, p_both = spearmanr(ga[m], snapped[m])
        s_both = f"{r_both:+.3f} (p={p_both:.3f}, n={int(m.sum())})"
    else:
        s_both = f"undefined (n={int(m.sum())}, {len(set(snapped[m]))} distinct depths)"
    P(f"| {SHORT[t]} | {base[t][0]:+.3f} | {r_snap:+.3f} (p={p_snap:.3f}) | {s_both} "
      f"| {base[t][1]:+.3f} |")
P("")

(Path(__file__).parent / "r5_out.md").write_text("\n".join(OUT), encoding="utf-8")
print("\nWROTE r5_out.md")
