"""OMTR pilot 分析器 v1（純指標，不出圖）。

三個核心問題：
1. 記憶劑量反應：gold_logprob（記憶強度連續尺）是否預測收斂深度／熵？
   L0（語料驗證已背）vs L0N（同文體未背對照）是分布上的兩端。
2. 條件簽名：四條件（L0/L0N/L1/L4）的逐層 KL 曲線形狀是否可分？
3. 表徵幾何：哪一層開始，條件之間在殘差流上分得開？（分離度 = 質心距 / 平均組內散佈）

用法：
    .venv/Scripts/python.exe harness/analyze_pilot.py --model EleutherAI/pythia-1.4b
輸出：
    results/analysis_<model>.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent


def rank_biserial(a, b):
    """Mann-Whitney 的效應量（-1..1）；小樣本 pilot 用效應量不追 p 值。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None
    greater = sum((x > y) for x in a for y in b)
    less = sum((x < y) for x in a for y in b)
    return float((greater - less) / (len(a) * len(b)))


def summarize(values):
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def separation_curve(resids_by_cond):
    """每層的條件分離度：質心距 / pooled 組內平均距（>1 = 分得開）。"""
    conds = sorted(resids_by_cond)
    n_layers = next(iter(resids_by_cond.values()))[0].shape[0]
    out = {}
    for c1, c2 in combinations(conds, 2):
        curve = []
        for layer in range(n_layers):
            x1 = np.stack([r[layer] for r in resids_by_cond[c1]])
            x2 = np.stack([r[layer] for r in resids_by_cond[c2]])
            # 每層先做 z-score（跨兩組），避免尺度差異主導
            allx = np.concatenate([x1, x2])
            mu, sd = allx.mean(0), allx.std(0) + 1e-8
            z1, z2 = (x1 - mu) / sd, (x2 - mu) / sd
            centroid_dist = float(np.linalg.norm(z1.mean(0) - z2.mean(0)))
            within = []
            for z in (z1, z2):
                if len(z) > 1:
                    c = z.mean(0)
                    within.extend(np.linalg.norm(z - c, axis=1))
            within_mean = float(np.mean(within)) if within else 1.0
            curve.append(centroid_dist / (within_mean + 1e-8))
        out[f"{c1}_vs_{c2}"] = [round(x, 4) for x in curve]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--raw", default=str(PROJ / "results" / "raw"))
    ap.add_argument("--out", default=None,
                    help="輸出路徑（預設 results/analysis_<model>.json）。"
                         "測試用別的檔名，才不會蓋掉正式結果")
    args = ap.parse_args()
    tag = args.model.replace("/", "_")
    raw = Path(args.raw)

    data = json.load(open(raw / f"pilot_{tag}.json", encoding="utf-8"))
    records = [r for r in data["records"] if "error" not in r]
    errors = [r for r in data["records"] if "error" in r]
    if not records:
        # 0 筆可用紀錄還照寫，等於用一份空分析覆蓋掉正式結果——寧可什麼都不寫。
        print(f"0 ok records in {raw / f'pilot_{tag}.json'} "
              f"({len(errors)} error records); refusing to write an analysis file.\n"
              f"  Fix the run first: writing here would overwrite live results "
              f"with an empty analysis.", file=sys.stderr)
        raise SystemExit(2)
    resid_npz = np.load(raw / f"resid_mean_{tag}.npz")

    by_level: dict = {}
    for r in records:
        by_level.setdefault(r["level"], []).append(r)

    metrics = ("final_entropy_mean", "final_margin_mean")
    depth_keys = None
    for r in records:
        if r.get("depth"):
            depth_keys = list(r["depth"].keys())
            break

    analysis = {"model": args.model, "n_ok": len(records), "n_error": len(errors),
                "levels": {}}
    for lv, rs in sorted(by_level.items()):
        entry = {"n": len(rs)}
        for m in metrics:
            entry[m] = summarize([r.get(m) for r in rs])
        for dk in depth_keys or []:
            entry[f"depth.{dk}"] = summarize([r["depth"].get(dk) for r in rs if r.get("depth")])
        golds = [r.get("gold_logprob_per_token") for r in rs]
        if any(g is not None for g in golds):
            entry["gold_logprob"] = summarize(golds)
        manip = [r.get("manipulation", {}) for r in rs]
        l1p = [m.get("l1_expected_in_output") for m in manip if "l1_expected_in_output" in m]
        if l1p:
            entry["l1_pass_rate"] = float(np.mean(l1p))
        l4p = [m.get("l4_both_domains") for m in manip if "l4_both_domains" in m]
        if l4p:
            entry["l4_both_domains_rate"] = float(np.mean(l4p))
        analysis["levels"][lv] = entry

    # 條件對比效應量（深度與熵）
    contrasts = {}
    pairs = [("L0", "L0N"), ("L0", "L1"), ("L1", "L4"), ("L0N", "L4")]
    for a, b in pairs:
        if a in by_level and b in by_level:
            row = {}
            for dk in depth_keys or []:
                row[f"depth.{dk}"] = rank_biserial(
                    [r["depth"][dk] for r in by_level[a] if r.get("depth")],
                    [r["depth"][dk] for r in by_level[b] if r.get("depth")])
            row["final_entropy"] = rank_biserial(
                [r["final_entropy_mean"] for r in by_level[a]],
                [r["final_entropy_mean"] for r in by_level[b]])
            contrasts[f"{a}_vs_{b}"] = row
    analysis["contrasts_rank_biserial"] = contrasts

    # 記憶劑量反應：gold_logprob × 深度（L0+L0N 合併，Spearman）
    dose = [(r["gold_logprob_per_token"], r["depth"]["depth_tau_0.1"], r["level"])
            for r in records
            if r.get("gold_logprob_per_token") is not None and r.get("depth")]
    if len(dose) >= 5:
        from scipy.stats import spearmanr
        g = [d[0] for d in dose]
        rho, p = spearmanr(g, [d[1] for d in dose])
        # 組內相關（事實查核要求：排除「組間位移」解釋。實測 L0 組內反而更強）
        within = {}
        for lv in sorted({d[2] for d in dose}):
            sel = [d for d in dose if d[2] == lv]
            if len(sel) >= 6:
                w_rho, w_p = spearmanr([d[0] for d in sel], [d[1] for d in sel])
                within[lv] = {"n": len(sel), "rho": float(w_rho), "p": float(w_p)}
        analysis["dose_response"] = {
            "x": "gold_logprob_per_token", "y": "depth_tau_0.1",
            "n": len(dose), "spearman_rho": float(rho), "p": float(p),
            "within_condition": within,
            "points": [{"gold": round(a, 3), "depth": round(b, 3), "level": c}
                       for a, b, c in dose],
        }

    # 逐層條件平均 KL 曲線
    layer_curves = {}
    for lv, rs in by_level.items():
        profs = [r["layer_profile"] for r in rs if r.get("layer_profile")]
        if profs:
            n_layers = len(profs[0])
            layer_curves[lv] = {
                "kl_to_final": [round(float(np.mean([p[l]["kl_to_final"] for p in profs])), 4)
                                for l in range(n_layers)],
                "entropy": [round(float(np.mean([p[l]["entropy"] for p in profs])), 4)
                            for l in range(n_layers)],
            }
    analysis["layer_curves"] = layer_curves

    # 表徵分離度
    resids_by_cond = {}
    for r in records:
        if r["id"] in resid_npz:
            resids_by_cond.setdefault(r["level"], []).append(resid_npz[r["id"]])
    resids_by_cond = {k: v for k, v in resids_by_cond.items() if len(v) >= 4}
    if len(resids_by_cond) >= 2:
        analysis["separation_curves"] = separation_curve(resids_by_cond)

    out_path = Path(args.out) if args.out else PROJ / "results" / f"analysis_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"analysis -> {out_path}")
    for lv, e in analysis["levels"].items():
        gold = e.get("gold_logprob")
        print(f"  {lv}: n={e['n']}"
              + (f" gold_logprob={gold['mean']:.2f}" if gold else "")
              + (f" depth_tau0.1={e['depth.depth_tau_0.1']['mean']:.3f}"
                 if e.get("depth.depth_tau_0.1") else ""))
    if "dose_response" in analysis:
        d = analysis["dose_response"]
        print(f"  dose-response: rho={d['spearman_rho']:.3f} (n={d['n']}, p={d['p']:.4f})")


if __name__ == "__main__":
    main()
