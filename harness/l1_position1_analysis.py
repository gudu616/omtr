"""R3（審查後探索）：L1 修正版分析——r-stats 裁定的「可辯護主量測」。

裁定：L1 的可辯護量測 = 位置 1（答案 token 本身）× 只取雙 tokenizer 皆單 token 的題目。
純重分析（用既有 raw），輸出 results/night/l1_position1.json。
"""
import json
from pathlib import Path

from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent
TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b", "EleutherAI_pythia-1.4b",
        "EleutherAI_pythia-2.8b", "allenai_OLMo-2-0425-1B"]

battery = {it["id"]: it for it in
           json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))["items"]}
single_both = {iid for iid, it in battery.items()
               if it["level"] == "L1" and it.get("gold_tokens")
               and all(v == 1 for v in it["gold_tokens"].values())}
print(f"L1 items single-token on BOTH tokenizers: {len(single_both)}/24")

report = {"single_token_both_ids": sorted(single_both), "models": {}}
for tag in TAGS:
    data = json.load(open(PROJ / "results" / "raw" / f"pilot_{tag}.json", encoding="utf-8"))
    rows = []
    for r in data["records"]:
        if (r.get("level") == "L1" and "error" not in r and r["id"] in single_both
                and r.get("per_position_depths") and r.get("gold_logprob_per_token") is not None):
            rows.append((r["gold_logprob_per_token"],
                         r["per_position_depths"][0]["depth_tau_0.1"]))
    depths = [d for _, d in rows]
    distinct = len(set(round(d, 6) for d in depths))
    entry = {"n": len(rows), "distinct_depth_values": distinct}
    if distinct <= 1:
        entry["verdict"] = "instrument has no resolution (constant depth at answer position)"
    elif len(rows) >= 6:
        rho, p = spearmanr([g for g, _ in rows], depths)
        entry.update({"rho": round(float(rho), 3), "p": round(float(p), 4),
                      "verdict": "measurable but see distinct-value count"})
    report["models"][tag] = entry
    print(f"  {tag}: n={entry['n']} distinct={distinct} "
          + (f"rho={entry.get('rho')} p={entry.get('p')}" if "rho" in entry else entry["verdict"]))

out = PROJ / "results" / "night" / "l1_position1.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"-> {out}")
