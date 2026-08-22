"""verbcue 2×2 聚合結果（工項④，theory_verbcue_prereg_v1.md §2/§5，
poolgen §0-0-b）。讀四個模型分片，出四格聚合結果——只備料不判定：
四格映射（§5 格①②③④）由主線按凍結判準對，本檔不寫「支持/反對」。

輸出兩軸 Δ（主線 2026-08-22 指示）：
  Δ_verb / Δ_proper（逐格，L0 vs L0N 缺口，§2 原定義的推廣——原定義只寫給
  verb 列，這裡對 verb/proper 兩列各自的通用/修正兩欄都算，共 4 個數）；
  Δ_pool（逐類，通用 vs 修正 缺口，§7 H2 的軸，verb/proper 各一個數，
  L0+L0N 題集池化後比較兩欄）。

三分母（§5 強制回報）、drop 分布與適足下限連動（共同主要條件 A）、
v2.8 停機規則旗標比例——全部逐格輸出。

用法：
  .venv/Scripts/python.exe harness/verbcue_aggregate.py
輸出：
  results/causal/verbcue_aggregate.json
  planning/office_reports/verbcue_aggregate_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verbcue_main import aggregate_arm, cluster_bootstrap_delta, ADEQUACY_FLOOR_NATS  # noqa: E402
import causal_analysis as ca  # noqa: E402  單一出處：SEED/B_PERM

PROJ = Path(__file__).resolve().parent.parent
MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
         "EleutherAI/pythia-1.4b", "allenai/OLMo-2-0425-1B"]


def load_shards():
    shards = {}
    for m in MODELS:
        path = PROJ / "results" / "causal" / f"verbcue_patch_{m.replace('/', '_')}.json"
        shards[m] = json.load(open(path, encoding="utf-8"))
    return shards


def aggregate_cell_across_models(shards, cell_name, item_ids=None):
    """攤平成 {"item::model": record}，用既有 aggregate_arm 分類（三分母/p/
    drop_median 邏輯與單模型時完全相同，只是 key 帶了 model，純字典鍵不影響
    判斷邏輯）。另外逐 item 池化（跨模型加總 n_survivors/n_candidates，只計
    entered 的 cell）供 cluster bootstrap 用（§2：以題為叢集單位）。"""
    flat = {}
    for model, shard in shards.items():
        cell = shard.get("cells", {}).get(cell_name, {})
        for iid, rec in cell.items():
            if item_ids is not None and iid not in item_ids:
                continue
            flat[f"{iid}::{model}"] = rec
    agg = aggregate_arm(flat)
    per_item = {}
    for key, info in agg["per_item"].items():
        if info["status"] != "entered":
            continue
        iid = key.split("::")[0]
        d = per_item.setdefault(iid, {"n_survivors": 0, "n_candidates": 0})
        d["n_survivors"] += info["n_survivors"]
        d["n_candidates"] += info["n_candidates"]
    return agg, per_item


def main():
    shards = load_shards()
    items = json.load(open(PROJ / "battery" / "verbcue_items.json", encoding="utf-8"))

    l0_by_class = {"verb": set(items["verb"]["L0"]), "proper": set(items["proper"]["L0_including_L0-14"])}
    l0n_by_class = {"verb": set(items["verb"]["L0N"]), "proper": set(items["proper"]["L0N"])}

    cell_class = {"verb_universal": "verb", "verb_corrected": "verb",
                 "proper_universal": "proper", "proper_corrected": "proper"}
    cell_column = {"verb_universal": "universal", "verb_corrected": "corrected",
                  "proper_universal": "proper", "proper_corrected": "corrected"}
    # 修正筆誤：proper_universal 的欄應為 universal
    cell_column["proper_universal"] = "universal"

    result = {"models": MODELS, "cells": {}, "delta_verb_proper_per_cell": {},
             "delta_pool_per_class": {}}

    ss = np.random.SeedSequence(ca.SEED)
    seeds = ss.spawn(8)
    seed_i = 0

    per_item_by_cell = {}   # (cell_name) -> {item_id: {n_survivors,n_candidates}}（entered 的池化）
    for cell_name in ("verb_universal", "verb_corrected", "proper_universal", "proper_corrected"):
        cls = cell_class[cell_name]
        agg_all, per_item_all = aggregate_cell_across_models(shards, cell_name)
        per_item_by_cell[cell_name] = per_item_all
        l0_ids = l0_by_class[cls]
        l0n_ids = l0n_by_class[cls]
        agg_l0, per_item_l0 = aggregate_cell_across_models(shards, cell_name, l0_ids)
        agg_l0n, per_item_l0n = aggregate_cell_across_models(shards, cell_name, l0n_ids)

        result["cells"][cell_name] = {
            "class": cls, "column": cell_column[cell_name],
            "overall": agg_all, "L0": agg_l0, "L0N": agg_l0n,
        }

        boot = cluster_bootstrap_delta(per_item_l0, per_item_l0n, ca.B_PERM,
                                       np.random.default_rng(seeds[seed_i]))
        seed_i += 1
        label = f"delta_{cls}_{cell_column[cell_name]}"
        result["delta_verb_proper_per_cell"][label] = boot

    for cls in ("verb", "proper"):
        univ_cell = f"{cls}_universal"
        corr_cell = f"{cls}_corrected"
        per_item_univ = per_item_by_cell[univ_cell]
        per_item_corr = per_item_by_cell[corr_cell]
        boot = cluster_bootstrap_delta(per_item_univ, per_item_corr, ca.B_PERM,
                                       np.random.default_rng(seeds[seed_i]))
        seed_i += 1
        boot["note"] = ("Δ_pool = p_通用 - p_修正（同類，L0+L0N 題集池化）。"
                        + boot.get("note", ""))
        result["delta_pool_per_class"][f"delta_pool_{cls}"] = boot

    out_path = PROJ / "results" / "causal" / "verbcue_aggregate.json"

    def _json_safe(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, (np.floating, np.integer)):
            return _json_safe(o.item())
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json_safe(v) for v in o]
        return o

    json.dump(_json_safe(result), open(out_path, "w", encoding="utf-8"),
             ensure_ascii=False, indent=1)

    # ---- 人讀報告 ----
    lines = ["# verbcue 2×2 聚合結果（只備料，不判定）\n",
            "四格映射（§5 格①②③④）由主線按凍結判準對，本檔不寫「支持/反對」。\n"]
    for cell_name, c in result["cells"].items():
        o, l0, l0n = c["overall"], c["L0"], c["L0N"]
        lines.append(f"## {cell_name}（{c['class']} × {c['column']}）\n")
        lines.append(f"- 整格：嘗試總格數(item×model)={o['n_total_attempted']}、"
                     f"**三分母強制回報**——進場={o['n_entered']}、"
                     f"適足出局={o['n_adequacy_excluded']}、"
                     f"pool_too_small旗標={o['n_pool_too_small_flagged']}、"
                     f"其他錯誤(no_candidate等)={o['n_total_attempted']-o['n_entered']-o['n_adequacy_excluded']}、"
                     f"p={o['p_candidate_level']}、drop中位數={o['drop_median_nats']}、"
                     f"條件A(腐蝕適足)={o['condition_A_adequate']}（下限={ADEQUACY_FLOOR_NATS}）、"
                     f"停機旗標比例={o['stop_rule_flagged_fraction']}、"
                     f"停機觸發={o['stop_rule_triggered']}")
        lines.append(f"- L0 子集：進場={l0['n_entered']}/{l0['n_total_attempted']}、"
                     f"p={l0['p_candidate_level']}、"
                     f"drop中位數={l0['drop_median_nats']}、條件A={l0['condition_A_adequate']}")
        lines.append(f"- L0N 子集：進場={l0n['n_entered']}/{l0n['n_total_attempted']}、"
                     f"p={l0n['p_candidate_level']}、"
                     f"drop中位數={l0n['drop_median_nats']}、條件A={l0n['condition_A_adequate']}")
        lines.append("")

    lines.append("## Δ（L0 vs L0N，逐格，cluster bootstrap 90% CI）\n")
    for label, b in result["delta_verb_proper_per_cell"].items():
        if b.get("status") != "OK":
            lines.append(f"- {label}: {b}")
            continue
        lines.append(f"- {label}: p_L0={b['p_L0']:.4f} p_L0N={b['p_L0N']:.4f} "
                     f"Δ={b['delta_verb_pp']:.2f}pp CI={[round(x,2) for x in b['ci_90_pp']]} "
                     f"(n_L0={b['n_items_L0']}, n_L0N={b['n_items_L0N']})")
    lines.append("")

    lines.append("## Δ_pool（通用 vs 修正，逐類，cluster bootstrap 90% CI）\n")
    for label, b in result["delta_pool_per_class"].items():
        if b.get("status") != "OK":
            lines.append(f"- {label}: {b}")
            continue
        lines.append(f"- {label}: p_通用={b['p_L0']:.4f} p_修正={b['p_L0N']:.4f} "
                     f"Δ={b['delta_verb_pp']:.2f}pp CI={[round(x,2) for x in b['ci_90_pp']]} "
                     f"(n={b['n_items_L0']})")
    lines.append("")

    report_path = PROJ / "planning" / "office_reports" / "verbcue_aggregate_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n->", out_path)
    print("->", report_path)


if __name__ == "__main__":
    main()
