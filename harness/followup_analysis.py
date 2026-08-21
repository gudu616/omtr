"""三顆實彈的鑑別分析（attack harvest 對草稿的三發射擊）。

彈藥 1（L0N 也有劑量反應）→ 三個鑑別測試：
  T1 完整組內表（含 L0N，5 模型）——先把被略過的那欄攤開
  T2 記憶特異變量：L0 組內「語料重複次數」（L0N 依建構無此變量）vs depth/gold
  T3 偏相關：L0 組內 depth~gold 控制 final_entropy（排除「終點銳度」共同因）
彈藥 2/M1（循環量測疑慮）→
  T4 depth 只用位置 1..7（排除與 gold 首 token 共享分布的位置 0）重算劑量反應
輸出 results/followup_report.json ＋ 印表。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

PROJ = Path(__file__).resolve().parent.parent
TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b", "EleutherAI_pythia-1.4b",
        "EleutherAI_pythia-2.8b", "allenai_OLMo-2-0425-1B"]
CORPUS_OF = {"pythia": "pile", "OLMo": "olmo_mix"}


def partial_spearman(x, y, z):
    """rank 之後對 z 殘差化再相關（Spearman 偏相關的標準近似）。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    r, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)


def main():
    battery = {it["id"]: it for it in
               json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))["items"]}
    ver = {r["title"]: r for r in
           json.load(open(PROJ / "battery" / "l0_verification.json", encoding="utf-8"))}
    report = {}
    for tag in TAGS:
        corpus = "pile" if "pythia" in tag else "olmo_mix"
        pilot = json.load(open(PROJ / "results" / "raw" / f"pilot_{tag}.json", encoding="utf-8"))
        recs = {r["id"]: r for r in pilot["records"] if "error" not in r}
        entry = {}

        def rows(level):
            out = []
            for r in recs.values():
                if r["level"] != level or r.get("gold_logprob_per_token") is None:
                    continue
                out.append(r)
            return out

        # T1 組內表（L0 / L0N / L1）
        t1 = {}
        for lv in ("L0", "L0N", "L1"):
            rs = rows(lv)
            if len(rs) >= 6:
                rho, p = spearmanr([r["gold_logprob_per_token"] for r in rs],
                                   [r["depth"]["depth_tau_0.1"] for r in rs])
                t1[lv] = {"n": len(rs), "rho": round(float(rho), 3), "p": round(float(p), 4)}
        entry["T1_within"] = t1

        # T2 語料重複次數（記憶特異 dose）
        l0 = rows("L0")
        counts, depths, golds = [], [], []
        for r in l0:
            title = battery.get(r["id"], {}).get("title")
            v = ver.get(title)
            if not v:
                continue
            ok = [p_["count"] for p_ in v["probes"].get(corpus, []) if p_["status"] == "ok"]
            if not ok:
                continue
            counts.append(float(np.median(ok)))
            depths.append(r["depth"]["depth_tau_0.1"])
            golds.append(r["gold_logprob_per_token"])
        if len(counts) >= 6:
            rho_cd, p_cd = spearmanr(counts, depths)
            rho_cg, p_cg = spearmanr(counts, golds)
            entry["T2_dupcount"] = {
                "n": len(counts),
                "count_vs_depth": {"rho": round(float(rho_cd), 3), "p": round(float(p_cd), 4)},
                "count_vs_gold": {"rho": round(float(rho_cg), 3), "p": round(float(p_cg), 4)},
            }

        # T3 偏相關：depth~gold | final_entropy（L0 組內）
        if len(l0) >= 8:
            g = [r["gold_logprob_per_token"] for r in l0]
            d = [r["depth"]["depth_tau_0.1"] for r in l0]
            e = [r["final_entropy_mean"] for r in l0]
            r_, p_ = partial_spearman(np.array(g), np.array(d), np.array(e))
            entry["T3_partial_gold_depth_ctrl_entropy"] = {
                "n": len(l0), "rho": round(r_, 3), "p": round(p_, 4)}

        # T4 排除位置 0 的 depth 重算劑量反應（L0+L0N+L1 與 L0 組內）
        def depth_pos1plus(r):
            ppd = r.get("per_position_depths") or []
            vals = [d["depth_tau_0.1"] for d in ppd[1:]]
            return float(np.mean(vals)) if vals else None
        for scope, levels in (("all", ("L0", "L0N", "L1")), ("L0", ("L0",))):
            xs, ys = [], []
            for lv in levels:
                for r in rows(lv):
                    dp = depth_pos1plus(r)
                    if dp is not None:
                        xs.append(r["gold_logprob_per_token"])
                        ys.append(dp)
            if len(xs) >= 8:
                rho, p = spearmanr(xs, ys)
                entry[f"T4_pos1plus_{scope}"] = {
                    "n": len(xs), "rho": round(float(rho), 3), "p": round(float(p), 4)}
        report[tag] = entry

    out = PROJ / "results" / "followup_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for tag, e in report.items():
        print(f"=== {tag} ===")
        for k, v in e.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
