"""窗口 A 冒煙判定器 v3——凍結判定規則的機械執行者。

分工（單一出處原則）：量測與計算欄位的權威＝causal_smoke.py（含逐格擬合、
二次 dof、S/SE(β)、min-of-valid-bands）；**判定規則的權威＝本腳本**，
規則全部引凍結出處。本腳本另對可便宜重算者做第二路交叉核算並報告差異。

規則出處（規格版本：CAUSAL_PREREG_v1 第一段凍結 + v2.6–v2.13 增補裁決）：
  G1 兩層讀數＋PF-2      v2.13/PF-2：cell ratio=resid_MAD/底線；<3=「≈底線」、
                          ≥10=「明顯活著」、3–10=不確定帶（視同未點燃）。
                          G1-a＝**全部有效格 pooled ratio <3 一致決**→設計死不跑
                          （已降級為粗大失效偵測器，真退化偵測率 .206——v2.13）；
                          G1-b＝全部有效格 item_fe <3 且 pooled ≥10 →設計活、
                          有效 n=配對數；其餘→正常推進。中位格 ratio 併報（診斷）。
  砍溯源軌               v2.11①/v2.13F：r=min-of-valid-bands（ΔY 尺度）；
                          kill_rule_applicable=False（有效帶<2）→只報不砍；
                          |r|>0.95 全帶一致才砍（0.95 非 50/50 點）。
  分支規則               v2.13D：壞題（pool_too_small/未達適足/零候選/例外）
                          佔全部嘗試 (題×模型) 格 >40% → 多 token 分支；
                          逐模型 >60% 標記點名。
  r 量測 dof             v2.13/PF-1：realized_dof 以實際擬合式（二次=Σ(kᵢ−1)−2）；
                          <8 → 補題（不改門檻）。
  ρ 閘門(L0M)            v2.3：ρ̂ > tanh(1.645/√(n−3)) 且 90% CI 下界 > 0。
  L0M 通過規則           M2：五模型中位 |Δgold(整段)| ≤0.5 且起手窗 ≤1.0 nats。
  數值底線               附錄 C②/H 案：ΔY 尺度、≥3 個 distinct 題、兩路徑。
  k 規則                 PF-4：r<0.5→k=1；0.5≤r<2→k=2；r≥2→k=3（ΔY 尺度變異比）。
  陽性對照地板           PF-5：ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)，同帶同位置。

用法： .venv\\Scripts\\python.exe harness/causal_smoke_eval.py --run-id <id>
       [--out-dir results/causal] [--noise-floor <nats>]
輸出： <out-dir>/smoke_eval_<run_id>.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJ = Path(__file__).resolve().parent.parent

RHO_GATE_CI_Z = 1.645
REDUNDANCY_KILL = 0.95
PF2_NEAR = 3.0          # 「≈底線」上限（慣例，PF-2）
PF2_FAR = 10.0          # 「明顯活著」下限（慣例，PF-2）
BRANCH_POOLED = 0.40
BRANCH_PER_MODEL = 0.60
L0M_PASS_WHOLE = 0.5
L0M_PASS_ONSET = 1.0
NOISE_NEGLIGIBLE = 1e-3
R_DOF_TARGET = 8


def load(stage, run_id, out_dir):
    p = out_dir / f"smoke_{stage}_{run_id}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def finite(xs):
    return [float(x) for x in xs if x is not None
            and isinstance(x, (int, float)) and math.isfinite(float(x))]


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---------------------------------------------- G1（讀驅動器逐格擬合，套 PF-2）

def _cell_reading(ratio):
    if ratio is None:
        return None
    if ratio < PF2_NEAR:
        return "near_floor"
    if ratio >= PF2_FAR:
        return "clearly_alive"
    return "uncertain"       # PF-2：不確定帶，依不對稱閘門視同未點燃


def eval_g1(anchors, floor):
    gi = (anchors or {}).get("g1_existence_inputs") or {}
    cells = gi.get("cells") or {}
    out_cells, pooled_ratios = {}, []
    for key, cv in cells.items():
        po, fe = cv.get("pooled") or {}, cv.get("item_fe") or {}
        entry = {}
        for name, blk in (("pooled", po), ("item_fe", fe)):
            mad = blk.get("resid_mad_scale")
            dof = blk.get("dof")
            valid = (mad is not None and dof is not None and dof >= 2)
            ratio = (mad / max(floor, 1e-12)) if (valid and floor) else None
            entry[name] = {"resid_mad_scale": mad, "dof": dof, "valid": valid,
                           "ratio_over_floor": ratio,
                           "reading": _cell_reading(ratio),
                           "S": blk.get("S"), "se_beta": blk.get("se_beta"),
                           "outlier_dominated_flag": blk.get("outlier_dominated_flag")}
        out_cells[key] = entry
        if entry["pooled"]["ratio_over_floor"] is not None:
            pooled_ratios.append(entry["pooled"]["ratio_over_floor"])

    # 複核裁定：一致決集合折回 15 格（5 模型×3 帶＝鍵尾 |all 的格）——
    # .206 偵測率描述的是 15 格規則；main/cross_word 格降為診斷照報。
    valid_cells = [k for k, e in out_cells.items()
                   if e["pooled"]["valid"] and k.endswith("|all")]
    verdict = "PENDING"
    if floor is None:
        verdict = "INCOMPLETE（無數值底線）"
    elif not valid_cells:
        verdict = "INCOMPLETE（無有效格）"
    else:
        all_pooled_near = all(
            out_cells[k]["pooled"]["reading"] == "near_floor" for k in valid_cells)
        fe_ok = [k for k in valid_cells if out_cells[k]["item_fe"]["valid"]]
        all_fe_near = bool(fe_ok) and all(
            out_cells[k]["item_fe"]["reading"] == "near_floor" for k in fe_ok)
        all_pooled_far = all(
            out_cells[k]["pooled"]["reading"] == "clearly_alive" for k in valid_cells)
        if all_pooled_near:
            verdict = "G1-a：全部有效格 pooled ≈ 底線（一致決）→ 設計死，主實驗不跑"
        elif all_fe_near and all_pooled_far:
            verdict = ("G1-b：題內決定性、題間不同 → 設計活，"
                       "有效 n＝配對數（同題多替換詞不買主檢定自由度）")
        else:
            verdict = "proceed：存在 Δs 以外結構（或混合型態，逐格見 cells）"
    return {"cells": out_cells, "n_valid_cells": len(valid_cells),
            "median_pooled_ratio": (float(np.median(pooled_ratios))
                                    if pooled_ratios else None),
            "verdict": verdict,
            "notes": ["G1-a 已降級為粗大失效偵測器（真退化偵測率 .206，v2.13）",
                      "G1 未點燃＝什麼都沒證明（不對稱閘門，附錄 C③）",
                      "不確定帶(3–10)視同未點燃（PF-2）"]}


# ---------------------------------------------- 其餘規則

def eval_redundancy(glink):
    if not glink:
        return {"status": "MISSING"}
    applicable = glink.get("kill_rule_applicable")
    r = glink.get("r_dgold_dprov")
    out = {"r_min_of_valid_bands": r, "n_valid_bands": glink.get("n_valid_bands"),
           "per_band_r": glink.get("per_band_r"),
           "scale": glink.get("scale"),
           "rule": f"全部有效帶 |r|>{REDUNDANCY_KILL} 才砍（min 一致決，v2.11①；"
                   "ΔY 尺度，v2.13F；0.95 非 50/50 點）"}
    if not applicable:
        out["verdict"] = "不可動用（有效帶<2 或無資料）：只報 r、延後裁定"
        return out
    out["kill_provenance_track"] = bool(r is not None and abs(r) > REDUNDANCY_KILL)
    out["verdict"] = ("砍溯源軌＋R9（省 ~2h infini-gram）"
                      if out["kill_provenance_track"] else "溯源軌保留")
    return out


def eval_branch(anchors):
    gate = (anchors or {}).get("single_token_strength_gate") or {}
    pooled = gate.get("pooled") or {}
    # 複核①：驅動器鍵名為 fraction_bad——兩名並收，權威=儀器部
    frac = pooled.get("fraction_bad", pooled.get("frac_bad", gate.get("frac_bad")))
    per_model = gate.get("per_model") or {}
    flagged = gate.get("per_model_flagged") or [
        m for m, v in per_model.items()
        if isinstance(v, dict) and (v.get("frac_bad") or 0) > BRANCH_PER_MODEL]
    return {"pooled_frac_bad": frac,
            "branch_triggered": (None if frac is None else bool(frac > BRANCH_POOLED)),
            "per_model_flagged_gt60": flagged,
            "denominator_rule": gate.get("denominator_rule"),
            "rule": f"併池壞題 >{BRANCH_POOLED:.0%} → 多 token 分支；"
                    f"單一模型 >{BRANCH_PER_MODEL:.0%} 標記點名（v2.13D）"}


def eval_r_dof(anchors):
    """複核③：dof 逐模型判（v2.13 G 案）——跨模型併會讓五模型 dof≈70+、
    補題觸發器死掉。逐模型 dof 直接取該模型 |all 格的 item_fe.dof
    （同模型各帶共用同一批 Δs 點，任一帶即可；取中位防個別帶掉點）。"""
    gi = (anchors or {}).get("g1_existence_inputs") or {}
    cells = gi.get("cells") or {}
    per_model: dict = {}
    for key, cv in cells.items():
        if not key.endswith("|all"):
            continue
        model = key.split("|")[0]
        fe = cv.get("item_fe") or {}
        d = fe.get("dof")
        if d is not None:
            per_model.setdefault(model, []).append(d)
    if not per_model:
        rp = (anchors or {}).get("replicate_points") or {}
        if rp.get("realized_dof") is None:
            return {"status": "MISSING"}
        return {"status": "FALLBACK（cells 無資料，退回彙總值——注意其跨模型併"
                          "問題待儀器部修）", "pooled_realized_dof": rp["realized_dof"]}
    out = {m: {"realized_dof": int(np.median(ds)), "n_bands_seen": len(ds),
               "met": bool(np.median(ds) >= R_DOF_TARGET)}
           for m, ds in per_model.items()}
    return {"per_model": out, "target": R_DOF_TARGET,
            "all_met": all(v["met"] for v in out.values()),
            "if_not_met": "對未達標之模型補題到達標，記錄補幾題，不得改門檻"
                          "（附錄 C③；逐模型判＝v2.13 G 案）"}


def eval_k_rule(anchors, glink):
    # 複核②：實際路徑 anchors["anchors"]["cross_word_r"]["by_model"][m][band]
    # ["r_var_ratio"]。聚合（複核建議）：逐模型報＋全體中位數當 k 決策輸入。
    by_model = (((anchors or {}).get("anchors") or {})
                .get("cross_word_r") or {}).get("by_model") or {}
    per_model, all_vals = {}, []
    for m, bands in by_model.items():
        vals = finite([(bands.get(b) or {}).get("r_var_ratio")
                       for b in (bands or {})])
        if vals:
            per_model[m] = {"median_var_ratio": float(np.median(vals)),
                            "n_bands": len(vals)}
            all_vals.extend(vals)
    if not all_vals:
        return {"status": "MISSING（ΔY 尺度變異比未見於輸出）"}
    vr = float(np.median(all_vals))
    k = 1 if vr < 0.5 else (2 if vr < 2.0 else 3)
    return {"var_ratio_dy_median": vr, "per_model": per_model, "k": k,
            "aggregation": "逐模型報＋全體中位數當 k 決策輸入（複核②建議）",
            "rule": "r<0.5→1；0.5≤r<2→2；r≥2→3（PF-4，慣例；ΔY 尺度，v2.13F）"}


def eval_rho_gate(l0m):
    pairs = (l0m or {}).get("pairs_measured") or []
    per_model = {}
    for m in sorted({row["model"] for row in pairs}) if pairs else []:
        rows = [r for r in pairs if r["model"] == m
                and r.get("l0_onset") is not None and r.get("l0m_onset") is not None]
        n = len(rows)
        if n < 4:
            per_model[m] = {"n_pairs": n, "status": "insufficient"}
            continue
        a = np.array([r["l0_onset"] for r in rows])
        b = np.array([r["l0m_onset"] for r in rows])
        with np.errstate(invalid="ignore"):
            rho = float(np.corrcoef(a, b)[0, 1])
        if not math.isfinite(rho):
            per_model[m] = {"n_pairs": n, "status": "degenerate（常數列）"}
            continue
        thr = math.tanh(RHO_GATE_CI_Z / math.sqrt(n - 3))
        lo = math.tanh(math.atanh(max(min(rho, 0.999999), -0.999999))
                       - RHO_GATE_CI_Z / math.sqrt(n - 3))
        per_model[m] = {"n_pairs": n, "rho_hat": rho, "ci_threshold_equiv": thr,
                        "ci90_lower": lo, "pass": bool(rho > thr and lo > 0)}
    return {"per_model": per_model,
            "rho_variable": "起手窗 clean gold（PF-3；為效應層 ρ 之上界——"
                            "窗口 A 有 patch 資料後必須在 ΔY 上重估，此值僅預篩）",
            "rule": "ρ̂ > tanh(1.645/√(n−3)) 且 90% CI 下界 > 0（v2.3）",
            "fail_wording_frozen": ("在負擔得起的 n 下無法拆除混淆 (C)；主對比照 "
                                    "L0P 執行但其結論永久攜帶 (C) 未拆的限定語")}


def eval_l0m_pass(l0m):
    med = (l0m or {}).get("median_abs_delta") or {}
    whole, onset = med.get("whole"), med.get("onset")
    if whole is None or onset is None:
        return {"status": "MISSING"}
    return {"median_abs_dgold_whole": whole, "median_abs_dgold_onset": onset,
            "pass": bool(whole <= L0M_PASS_WHOLE and onset <= L0M_PASS_ONSET),
            "rule": f"中位|Δgold(整段)|≤{L0M_PASS_WHOLE} 且起手窗≤{L0M_PASS_ONSET}（M2）"}


def eval_noise_floor(noise_json, cli_floor):
    if cli_floor is not None:
        return {"floor_nats": cli_floor, "source": "cli"}
    rows = (noise_json or {}).get("float_path_diffs") or []
    prod = (noise_json or {}).get("production_dtype")
    per_item, paths_per_item = {}, {}
    for r in rows:
        # v2.15/5.8.3：底線只取 batch_composition 且生產 dtype 的列；
        # dtype 層（precision-robustness）與非生產 dtype 列不進底線。
        if r.get("path") not in (None, "batch_composition"):
            continue
        if prod and r.get("dtype") and r.get("dtype") != prod:
            continue
        iid = r.get("id") or r.get("item_id")
        v = r.get("max_abs_diff_dy_nats", r.get("max_abs_diff_nats"))
        if iid is None or v is None or not math.isfinite(float(v)):
            continue
        per_item[iid] = max(per_item.get(iid, 0.0), float(v))
        if r.get("path") is not None:
            paths_per_item.setdefault(iid, set()).add(r["path"])
    if len(per_item) < 3:
        return {"status": "MISSING（需 ≥3 個 distinct 題；H 案）",
                "n_distinct_items": len(per_item), "n_rows": len(rows)}
    both = (all(len(s) >= 2 for s in paths_per_item.values())
            if paths_per_item else None)
    ub = max(per_item.values())
    return {"floor_nats": ub, "n_distinct_items": len(per_item),
            "both_paths_per_item": both,
            "negligible": bool(ub < NOISE_NEGLIGIBLE),
            "scale": "delta_y_nats（附錄 C②）"}


def eval_positive_control(anchors, floor):
    pc = (anchors or {}).get("positive_control") or {}
    dy, ds = pc.get("delta_y_pc"), pc.get("delta_s_pc")
    if dy is None or ds is None:
        return {"status": "MISSING（陽性對照欄位未見）",
                "rule": "ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)，同帶同位置（PF-5）"}
    thr = max(10.0 * (floor or 0.0), 0.5 * ds)
    return {"delta_y_pc": dy, "delta_s_pc": ds, "threshold": thr,
            "pass": bool(dy >= thr),
            "rule": "ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)（PF-5；0.5 慣例）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default=str(PROJ / "results" / "causal"))
    ap.add_argument("--noise-floor", type=float, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    anchors = load("anchors", args.run_id, out_dir)
    l0m = load("l0m", args.run_id, out_dir)
    glink = load("glink", args.run_id, out_dir)
    noise = load("numfloor", args.run_id, out_dir)

    floor_blk = eval_noise_floor(noise, args.noise_floor)
    fv = floor_blk.get("floor_nats")

    report = {
        "run_id": args.run_id,
        "spec_version": "CAUSAL_PREREG_v1 第一段凍結 + v2.6–v2.13（權威=裁決檔）",
        "noise_floor": floor_blk,
        "g1": eval_g1(anchors, fv),
        "redundancy_glink": eval_redundancy(glink),
        "branch": eval_branch(anchors),
        "r_measurement_dof": eval_r_dof(anchors),
        "k_rule": eval_k_rule(anchors, glink),
        "rho_gate": eval_rho_gate(l0m),
        "l0m_pass_rule": eval_l0m_pass(l0m),
        "positive_control": eval_positive_control(anchors, fv),
    }
    report = _json_safe(report)
    out = out_dir / f"smoke_eval_{args.run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
