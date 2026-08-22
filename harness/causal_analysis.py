"""因果階段主分析——把凍結的統計機器接到真資料上。

【單一出處原則】統計機器不在這裡重寫：本腳本 import
planning/office_reports/theory_v3_sim.py 的 t_from_A / Family.p_and_T /
stouffer，讓「模擬時驗證過的那段程式」與「跑真資料的那段程式」是同一段。
凍結參數一律引用出處，不在此處發明。

凍結出處
  配對內條件標籤翻轉置換、聯集讀法、家族統計量=各 run studentized 均值
                                    附錄 D3/E2（v2.13 G 案）＋theory_v3_sim.py
  B=20000、seed=20260822            v2.13 N-g
  出口 E2＝Stouffer 合併雙尾 p<.05  v2.13 B2（不加各家族 p≤.166 連言）
  端點＝起手窗 ΔY（nats）；R 僅描述  v2.8②／附錄 C2（R 無有限變異數）
  TOST 邊界＝附錄 C 表 × σ_DiD      σ_DiD=0.343（附錄 J3），逐家族按實際 n
  溯源軌＝R-specific 的必要條件      v2.13 N-a；含「矛盾格」
  Regime A 主／B 敏感度並列          v2.13 N1
  2.8b 描述性（不入主裁決）          v2.19（承 J7）
  410m L0-01／L0P-09 預見旗標        v2.20
  G1／分支／PC／k                    只引用 smoke_eval，不重算（v2.13）
  MDE=0.302 nats（純 fp32 臂）       附錄 J7(b)

【解盲紀律】本檔撰寫期間未讀取任何實際數值（僅讀 schema：鍵名與型別）。
自測用合成資料（--self-test）。凍結後才對真資料執行。

用法
  .venv/Scripts/python.exe harness/causal_analysis.py --self-test
  .venv/Scripts/python.exe harness/causal_analysis.py --run-id main_winB1
輸出
  results/causal/analysis_<run_id>.json  ＋ stdout 摘要
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJ = Path(__file__).resolve().parent.parent
FROZEN_SIM = PROJ / "planning" / "office_reports" / "theory_v3_sim.py"

# ---- 凍結常數（全部引用出處，不在此發明）-------------------------------
B_PERM = 20000                 # v2.13 N-g
SEED = 20260822                # v2.13 N-g
ALPHA = 0.05                   # v2.13 B2
SIGMA_DID = 0.343              # 附錄 J3
MDE_NATS = 0.302               # 附錄 J7(b)：純 fp32 臂
DESCRIPTIVE_MODELS = ("EleutherAI/pythia-2.8b",)          # v2.19
FORESEEN_FLAGS = {("EleutherAI/pythia-410m", "L0-01"),    # v2.20
                  ("EleutherAI/pythia-410m", "L0P-09")}
PYTHIA_PREFIX = "EleutherAI/pythia"
# 附錄 C 表：等價檢定要 80% 檢定力所需的邊界（σ 為單位），按每家族實際 n
TOST_BOUND_SIGMA = {8: 1.5, 16: 0.9, 24: 0.7}


def load_frozen_machinery():
    spec = importlib.util.spec_from_file_location("theory_v3_sim", FROZEN_SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.B = B_PERM
    return mod


# ------------------------------------------------------------------ 載入 --

def load_patch_files(run_id, out_dir):
    """schema（僅依鍵名，未讀值）：
    <out_dir>/<run_id>_patch_<tag>.json
      .model .dtype .records{item_id: rec}
      rec.level, rec.twin_of, rec.pairs[{regime, role, partner}]
      rec.patches[{band, launch{numer,denom,R,denom_unstable}, g1_point{...}}]
      rec.calibration{disqualify_item, pool_too_small_final, ...}
    """
    files = sorted(Path(out_dir).glob(f"{run_id}_patch_*.json"))
    if not files:
        raise SystemExit(f"no patch files matching {run_id}_patch_*.json in {out_dir}")
    runs, skipped_files = {}, []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        # 同一個 glob 也會撈到 *_patch_summary.json（無 model/records）——
        # 依鍵名判斷，不依檔名猜；撈錯會在載入時就炸而不是在統計裡。
        if "model" not in d or "records" not in d:
            skipped_files.append(f.name)
            continue
        runs[d["model"]] = {"dtype": d.get("dtype"), "records": d["records"],
                            "file": f.name, "config": d.get("config", {})}
    if not runs:
        raise SystemExit(f"no usable patch files ({len(skipped_files)} skipped: "
                         f"{skipped_files})")
    if skipped_files:
        print("skipped non-record files:", skipped_files)
    return runs


def _endpoint(rec, band):
    """主要終點＝起手窗 ΔY（nats）＝patches[band].launch.numer。
    R 一律不進門檻（v2.8②）；delta_s 只作為預先宣告的敏感度共變量。"""
    for p in rec.get("patches", []):
        if p.get("band") == band:
            lam = p.get("launch") or {}
            g1 = p.get("g1_point") or {}
            return {"dy": lam.get("numer"), "ds": lam.get("denom"),
                    "R": lam.get("R"), "denom_unstable": lam.get("denom_unstable"),
                    "g1_dy": g1.get("delta_y"), "g1_ds": g1.get("delta_s")}
    return None


def _usable(rec):
    cal = rec.get("calibration") or {}
    if cal.get("disqualify_item"):
        return False, "disqualified"
    if cal.get("pool_too_small_final"):
        return False, "pool_too_small_final"
    if (cal.get("fallback") or "") == "no_candidate_within_rank_guard":
        return False, "rank_guard_zero_survivor"
    return True, None


def build_pairs(runs, regime, band):
    """回傳 {model: {l0_item_id: paired_difference}}。
    配對方向凍結為 ΔY(L0) − ΔY(對照)；L0 是 role=='l0' 的那一側。
    regime ∈ {'A_gold30','B_dual','L0M','L0N_prov'}（rec.pairs[].regime）。"""
    out, skipped = {}, []
    for model, blob in runs.items():
        recs = blob["records"]
        diffs = {}
        for iid, rec in recs.items():
            if rec.get("level") != "L0":
                continue
            for pr in rec.get("pairs", []):
                if pr.get("regime") != regime or pr.get("role") != "l0":
                    continue
                partner = pr.get("partner")
                other = recs.get(partner)
                if other is None:
                    skipped.append((model, iid, partner, "partner_missing"))
                    continue
                for who, r in ((iid, rec), (partner, other)):
                    ok, why = _usable(r)
                    if not ok:
                        skipped.append((model, who, partner, why))
                        break
                else:
                    a, b = _endpoint(rec, band), _endpoint(other, band)
                    if a is None or b is None or a["dy"] is None or b["dy"] is None:
                        skipped.append((model, iid, partner, "no_endpoint"))
                        continue
                    diffs[iid] = {"diff": float(a["dy"]) - float(b["dy"]),
                                  "ds_diff": (None if a["ds"] is None or b["ds"] is None
                                              else float(a["ds"]) - float(b["ds"])),
                                  "partner": partner}
        if diffs:
            out[model] = diffs
    return out, skipped


# ---------------------------------------------------------------- 統計 --

def family_test(sim, diffs_by_model, rng):
    """一個證據家族的配對翻轉置換檢定，聯集讀法。
    直接複用凍結的 Family.p_and_T——只是把模擬產生的 y 換成真實配對差。"""
    models = sorted(diffs_by_model)
    union = sorted(set().union(*[set(diffs_by_model[m]) for m in models]))
    m_idx = {k: i for i, k in enumerate(union)}
    R, mm = len(models), len(union)
    y = np.zeros((R, mm), np.float32)
    M = np.zeros((R, mm), np.float32)
    for r, model in enumerate(models):
        for iid, rec in diffs_by_model[model].items():
            y[r, m_idx[iid]] = rec["diff"]
            M[r, m_idx[iid]] = 1.0
    fam = sim.Family.__new__(sim.Family)     # 不呼叫 __init__（那是模擬用的抽樣）
    fam.m = mm
    fam.M = M
    fam.n = M.sum(1)
    fam.S = rng.choice([-1.0, 1.0], size=(B_PERM, mm)).astype(np.float32)
    if (fam.n < 2).any():
        return {"error": "a run has fewer than 2 pairs; studentized statistic undefined",
                "per_run_n": fam.n.tolist(), "models": models}
    p, T = fam.p_and_T(y[None, :, :] if y.ndim == 2 else y)
    p = float(np.atleast_1d(p)[0]); T = float(np.atleast_1d(T)[0])
    pooled = np.concatenate([[v["diff"] for v in diffs_by_model[m].values()]
                             for m in models])
    return {"models": models, "union_size": mm, "per_run_n": fam.n.astype(int).tolist(),
            "permutation_floor_p": float(2.0 ** (1 - mm)),
            "p_two_sided": p, "T_studentized_mean": T,
            "mean_diff_nats": float(pooled.mean()),
            "sd_diff_nats": float(pooled.std(ddof=1)) if len(pooled) > 1 else None,
            "n_pairs_pooled": int(len(pooled))}


def tost_family(diffs_by_model, bound):
    """TOST（等價）於一個家族；跨家族用交集聯集檢定（水準恰好 α，附錄 E1/D）。"""
    v = np.concatenate([[x["diff"] for x in diffs_by_model[m].values()]
                        for m in sorted(diffs_by_model)])
    n = len(v)
    if n < 2:
        return {"pass": False, "reason": "n<2", "n": n}
    mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return {"pass": False, "reason": "zero se", "n": n}
    p_lo = float(tdist.sf((mean + bound) / se, n - 1))
    p_hi = float(tdist.cdf((mean - bound) / se, n - 1))
    return {"pass": bool(p_lo < ALPHA and p_hi < ALPHA), "n": n, "bound_nats": bound,
            "mean_nats": mean, "se_nats": se, "p_lower": p_lo, "p_upper": p_hi}


def tost_bound_for(n):
    """附錄 C 表 × sigma_DiD，【逐家族】用該家族自己的 n（凍結④）。

    內插規則（F4 重裁）：取【>= n 的最小表格鍵】對應的邊界。
    方向說明——邊界越【寬】，等價越容易通過，就越容易宣告 R-entangled，
    那是【反】保守；所以保守＝取較【窄】的邊界＝較大 n 那一檔。
    （舊版註解寫「取較大邊界=保守」，方向寫反了，已更正。）
    n 小於表格最小鍵時仍用 1.5σ（附錄 C3.4：S3 L0M n=8 -> 1.5σ），
    此時邊界比該 n 真正需要的還窄 -> 等價檢定檢定力不足 -> 仍是保守方向，
    但必須標記 below_table。"""
    keys = sorted(TOST_BOUND_SIGMA)
    ge = [k for k in keys if k >= n]
    k = ge[0] if ge else keys[-1]
    return TOST_BOUND_SIGMA[k] * SIGMA_DID, k, bool(n < keys[0])


def calibration_ci(mean, sd, n, conf=0.90):
    if n < 2 or sd is None:
        return None
    h = tdist.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)
    return [float(mean - h), float(mean + h)]


def split_families(diffs_by_model, drop_descriptive=True):
    py, ol, dropped = {}, {}, []
    for model, d in diffs_by_model.items():
        if drop_descriptive and model in DESCRIPTIVE_MODELS:
            dropped.append(model); continue
        (py if model.startswith(PYTHIA_PREFIX) else ol)[model] = d
    return py, ol, dropped


def run_track(sim, runs, regime, band, rng, label):
    diffs, skipped = build_pairs(runs, regime, band)
    py, ol, dropped = split_families(diffs)
    out = {"label": label, "regime": regime, "band": band,
           "descriptive_models_excluded": dropped,
           "n_skipped_pairs": len(skipped), "skipped": skipped[:40],
           # n1：預見旗標兩側都要掃（L0 側與 partner 側）
           "foreseen_flags_present": sorted(set(
               f"{m}|{x}" for m in diffs for i, rec in diffs[m].items()
               for x in (i, rec["partner"]) if (m, x) in FORESEEN_FLAGS))}
    if not py or not ol:
        out["status"] = "INCOMPLETE：缺少一個家族的資料，出口不可裁定"
        out["families_present"] = {"pythia": bool(py), "olmo": bool(ol)}
        return out
    fp = family_test(sim, py, rng)
    fo = family_test(sim, ol, rng)
    out["pythia"], out["olmo"] = fp, fo
    if "error" in fp or "error" in fo:
        out["status"] = "INCOMPLETE：家族統計量無法計算"
        return out
    # n4：合併用凍結模組的 stouffer()，不在此重抄公式
    p_comb, z = sim.stouffer(np.array([fp["p_two_sided"]]),
                             np.array([fp["T_studentized_mean"]]),
                             np.array([fo["p_two_sided"]]),
                             np.array([fo["T_studentized_mean"]]))
    p_comb, z = float(np.atleast_1d(p_comb)[0]), float(np.atleast_1d(z)[0])
    out["stouffer_z"] = z
    out["p_combined_two_sided"] = p_comb
    out["fires"] = bool(p_comb < ALPHA)
    out["direction_positive"] = bool(z > 0)
    n_tot = fp["n_pairs_pooled"] + fo["n_pairs_pooled"]
    b_py, k_py, below_py = tost_bound_for(fp["n_pairs_pooled"])   # 逐家族（凍結④）
    b_ol, k_ol, below_ol = tost_bound_for(fo["n_pairs_pooled"])
    t_py, t_ol = tost_family(py, b_py), tost_family(ol, b_ol)
    out["tost"] = {"sigma_did": SIGMA_DID,
                   "pythia": dict(t_py, bound_nats=b_py, table_n=k_py, below_table=below_py),
                   "olmo": dict(t_ol, bound_nats=b_ol, table_n=k_ol, below_table=below_ol),
                   "interpolation_rule": "取 >= n 的最小表格鍵（保守＝較窄邊界）",
                   "equivalent_IU": bool(t_py.get("pass") and t_ol.get("pass"))}
    mean_all = float(np.mean([fp["mean_diff_nats"], fo["mean_diff_nats"]]))
    sd_all = np.nanmean([x for x in (fp["sd_diff_nats"], fo["sd_diff_nats"]) if x])
    out["effect_size"] = {"mean_paired_diff_nats": mean_all,
                          "in_sigma_units": mean_all / SIGMA_DID,
                          "calibration_ci_90": calibration_ci(mean_all, sd_all, n_tot)}
    out["status"] = "OK"
    return out


JOINT_TABLE_SOURCE = "v2.13 N-a（附錄 E2.1，含矛盾格）"


def joint_verdict(gold, prov, pc_pass=None):
    """gold 軌 × 溯源軌，凍結聯合表【七格】（v2.13 N-a／附錄 E2.1）。
    第 7 格＝R-noeffect：同帶同位置的陽性對照未達地板（PF-5）——
    它【優先於】其他格，因為陽性對照沒過就代表儀器沒證明自己動得了東西。
    pc_pass 由 smoke_eval.positive_control.pass 引用而來（不重算）。"""
    if pc_pass is False:
        return {"verdict": "R-noeffect",
                "why": "同層帶同位置帶的陽性對照未達預凍結地板（PF-5）——"
                       "儀器未證明自己動得了東西，其餘各格不予解讀",
                "source": JOINT_TABLE_SOURCE, "positive_control_pass": False}
    if gold.get("status") != "OK":
        return {"verdict": "INCOMPLETE", "why": "gold 軌不可裁定", "source": JOINT_TABLE_SOURCE}
    pv_ok = prov.get("status") == "OK"
    pv_fire = bool(pv_ok and prov.get("fires") and prov.get("direction_positive"))
    if gold["fires"] and gold["direction_positive"]:
        if not pv_ok:
            return {"verdict": "無規則點燃", "why": "溯源軌無資料——必要條件無法滿足；"
                    "gold 軌方向與大小僅列探索性", "source": JOINT_TABLE_SOURCE}
        return ({"verdict": "R-specific", "why": "gold 軌點燃且方向為正、溯源軌一致點燃"}
                if pv_fire else
                {"verdict": "無規則點燃", "why": "gold 軌點燃但溯源軌未點燃（必要條件未滿足）；"
                 "gold 軌僅列探索性"}) | {"source": JOINT_TABLE_SOURCE}
    if gold["fires"] and not gold["direction_positive"]:
        return {"verdict": "反方向點燃", "why": "獨立一格，不併入 none（v0.7 教訓）",
                "source": JOINT_TABLE_SOURCE}
    if gold["tost"]["equivalent_IU"]:
        if pv_fire:
            return {"verdict": "矛盾格 → 無規則點燃",
                    "why": "gold 軌判等價但溯源軌點燃——兩軌互相打架，是設計訊號不是裁決，必須報告",
                    "source": JOINT_TABLE_SOURCE}
        # 凍結規則：溯源軌只約束 R-specific 的宣告，不擋 R-entangled（v2.13 N-a）。
        # 但「溯源軌沒跑」與「跑了沒點燃」對讀者意義不同，必須在輸出裡分開。
        return {"verdict": "R-entangled",
                "why": ("gold 軌等價且溯源軌未點燃" if pv_ok else
                        "gold 軌等價；溯源軌【無資料】——依凍結規則溯源只約束 specific 的宣告，"
                        "故不擋此裁決，但本判定是在沒有溯源證據的情況下作出的，須原樣披露"),
                "provenance_had_data": bool(pv_ok), "source": JOINT_TABLE_SOURCE}
    return {"verdict": "無規則點燃", "why": "差異未點燃且等價未通過（第四格）",
            "source": JOINT_TABLE_SOURCE}


# ------------------------------------------------------- 溯源軌（分開載） --

def load_provenance(run_id, out_dir):
    """schema（依 smoke_provenance / provenance_baseline 的鍵名）：
      runs{model:{rows:[{id, level, frac_present{clean, corrupted,
                          'patched::early', 'patched::mid', 'patched::late'}}]}}
    找不到檔案時回 None——溯源軌記為 INCOMPLETE，不猜。"""
    cands = sorted(Path(out_dir).glob(f"*provenance*{run_id}*.json")) \
        or sorted(Path(out_dir).glob(f"{run_id}*provenance*.json"))
    if not cands:
        return None
    d = json.load(open(cands[0], encoding="utf-8"))
    if "runs" not in d:
        return None
    return {"file": cands[0].name, "runs": d["runs"]}


def build_provenance_pairs(prov, runs, band):
    """溯源端點【凍結定義】＝ f_patched − f_corrupted（與 gold 軌同為【分子】）。

    出處：v2.13 F 案；`smoke_glink_*.json` 的 `scale` 欄實測為
    「delta_y_nats（兩軌都取分子；v2.13 F 案）」。
    ⚠ 本腳本第一版寫成 clean − patched（殘存損害），**與凍結定義不同量、
    方向語意可能相反**——那會讓聯合表的「方向一致」在解盲後才需要解釋。
    已改正；此註記保留為可見更正。

    真檔 schema（F1）：frac_present 是【巢狀物件】
      frac_present[arm] = {frac_present, n_probes, n_words, n_unknown,
                           max_count, resolution, too_few_probes}
    取值＝ [arm]["frac_present"]；`too_few_probes` 為真的列一律排除
    （儀器效度旗標，且在任何 patch 結果之前就決定——與 n_survivors 同類，
    符合附錄 G3 的「觸發量不涉出口變數」四要件）。排除數逐項記錄。"""
    if prov is None:
        return {}, [("all", None, None, "provenance_file_missing")], {}
    key = f"patched::{band}"
    val, skipped = {}, []
    excl = {"too_few_probes": 0, "missing_arm": 0, "not_a_dict": 0}

    def _fp(entry):
        """巢狀取值；平面 float 也接受（自測用），型別不符回 None。"""
        if isinstance(entry, dict):
            if entry.get("too_few_probes"):
                excl["too_few_probes"] += 1
                return None
            v = entry.get("frac_present")
            return None if v is None else float(v)
        if isinstance(entry, (int, float)):
            return float(entry)
        excl["not_a_dict"] += 1
        return None

    for model, blob in prov["runs"].items():
        by_id = {}
        for row in blob.get("rows", []):
            fp = row.get("frac_present") or {}
            if key not in fp or "corrupted" not in fp:
                excl["missing_arm"] += 1
                skipped.append((model, row.get("id"), None, "missing_arm"))
                continue
            a, c = _fp(fp[key]), _fp(fp["corrupted"])
            if a is None or c is None:
                skipped.append((model, row.get("id"), None, "unusable_frac_present"))
                continue
            by_id[row["id"]] = a - c          # 分子：patched - corrupted
        recs = (runs.get(model) or {}).get("records", {})
        diffs = {}
        for iid, rec in recs.items():
            if rec.get("level") != "L0":
                continue
            for pr in rec.get("pairs", []):
                if pr.get("regime") != "L0N_prov" or pr.get("role") != "l0":
                    continue
                partner = pr.get("partner")
                other = recs.get(partner)
                # n6：溯源軌也套 gold 軌同一套 _usable（失格/池太小/rank 護欄零存活）
                bad = None
                for who, r in ((iid, rec), (partner, other)):
                    if r is None:
                        bad = "partner_missing"; break
                    ok, why = _usable(r)
                    if not ok:
                        bad = why; break
                if bad:
                    skipped.append((model, iid, partner, bad)); continue
                if iid in by_id and partner in by_id:
                    diffs[iid] = {"diff": by_id[iid] - by_id[partner],
                                  "ds_diff": None, "partner": partner}
                else:
                    skipped.append((model, iid, partner, "no_provenance_row"))
        if diffs:
            val[model] = diffs
    return val, skipped, excl


def run_provenance_track(sim, runs, prov, band, rng):
    diffs, skipped, excl = build_provenance_pairs(prov, runs, band)
    py, ol, dropped = split_families(diffs)
    out = {"label": "provenance", "band": band,
           "endpoint": "frac_present patched-minus-corrupted (numerator; v2.13 F)",
           "exclusions": excl,
           "source_file": (prov or {}).get("file"),
           "descriptive_models_excluded": dropped,
           "n_skipped_pairs": len(skipped), "skipped": skipped[:40]}
    if not py or not ol:
        out["status"] = ("INCOMPLETE：溯源軌缺少一個家族的資料"
                         "（R-specific 的必要條件無法滿足）")
        return out
    fp, fo = family_test(sim, py, rng), family_test(sim, ol, rng)
    out["pythia"], out["olmo"] = fp, fo
    if "error" in fp or "error" in fo:
        out["status"] = "INCOMPLETE：家族統計量無法計算"
        return out
    p_comb, z = sim.stouffer(np.array([fp["p_two_sided"]]),
                             np.array([fp["T_studentized_mean"]]),
                             np.array([fo["p_two_sided"]]),
                             np.array([fo["T_studentized_mean"]]))
    out["stouffer_z"] = float(np.atleast_1d(z)[0])
    out["p_combined_two_sided"] = float(np.atleast_1d(p_comb)[0])
    out["fires"] = bool(out["p_combined_two_sided"] < ALPHA)
    out["direction_positive"] = bool(z > 0)
    out["status"] = "OK"
    return out


def quote_smoke(run_id, out_dir):
    """G1／分支／PC／k／redundancy 一律【引用】不重算（v2.13）。
    鍵名依 smoke_eval 真檔（F3 更正：redundancy 的真鍵是 `redundancy_glink`，
    舊版用 `redundancy` 恆取到 None；PC 與 k 先前根本沒引）。"""
    for cand in (Path(out_dir) / f"smoke_eval_{run_id}.json",
                 Path(out_dir) / "smoke_eval_winA2-20260822.json",
                 Path(out_dir) / "smoke_eval_winA20260822.json"):
        if not cand.exists():
            continue
        d = json.load(open(cand, encoding="utf-8"))
        pc = d.get("positive_control") or {}
        return {"quoted_from": cand.name,
                "g1_verdict": (d.get("g1") or {}).get("verdict"),
                "g1_median_pooled_ratio": (d.get("g1") or {}).get("median_pooled_ratio"),
                "branch_triggered": (d.get("branch") or {}).get("branch_triggered"),
                "branch_per_model_flagged": (d.get("branch") or {}).get("per_model_flagged_gt60"),
                "redundancy_glink": d.get("redundancy_glink"),
                "positive_control": pc,
                "positive_control_pass": pc.get("pass"),
                "k_rule": d.get("k_rule"),
                "noise_floor": d.get("noise_floor"),
                "rho_gate_rule": (d.get("rho_gate") or {}).get("rule"),
                "l0m_pass_rule": d.get("l0m_pass_rule"),
                "r_measurement_dof": d.get("r_measurement_dof"),
                "note": "引用，未重算（v2.13）"}
    return {"quoted_from": None, "positive_control_pass": None,
            "note": "smoke_eval 未找到——G1／分支／PC／k 欄留白，不得由本腳本補算"}


def sensitivity_sentence(gold):
    s = (f"本設計在最終題庫與量到的 σ_DiD={SIGMA_DID} nats 下，"
         f"每家族實際 n 對應的 80% 檢定力最小可偵測效應為 {MDE_NATS} nats（純 fp32 臂，附錄 J7b）。"
         f"低於 {MDE_NATS} nats 的「無效應」不具資訊量。")
    if gold.get("status") == "OK":
        e = gold["effect_size"]["mean_paired_diff_nats"]
        if abs(e) < MDE_NATS and not gold.get("fires"):
            s += "（本次觀察到的效應量小於該界線，因此本次的未點燃不得讀成「沒有效應」。）"
    return s


NARRATIVE = {
    "R-specific": "§11-A：記憶專屬成分在行為上被驗證（gold 軌點燃且方向為正，"
                  "溯源軌一致點燃）。仍受本設計已披露的限定：溯源軌 n 小、"
                  "兩軌連言的實際虛無率遠低於 .05（.0008–.0254）。",
    "R-entangled": "§11-B：在本設計的等價邊界內分不開記憶專屬與一般可預測性。"
                   "等價邊界由 σ_DiD 與各家族實際 n 決定，非「沒有效應」。",
    "反方向點燃": "§11-C：差異出口點燃但方向與記憶假設相反。此格獨立報告，"
                  "不併入「無規則點燃」（v0.7 教訓：正號一致曾同時擋掉兩個出口）。",
    "矛盾格 → 無規則點燃": "§11-D：兩軌互相打架（gold 判等價、溯源點燃）。"
                           "這是設計出問題的訊號，不是一個裁決，必須原樣報告。",
    "無規則點燃": "§11-E：第四格。預先模擬即顯示這是最可能的結果"
                  "（糾纏為真時 .877／.946），所以它不是意外，也不是證據。",
    "INCOMPLETE": "§11-F：資料不足以套用凍結規則，照實報「不可裁定」，不以任何方式代打。",
}


def descriptive_2p8b(sim, runs, regime, band, rng):
    """n2：2.8b 不入主裁決（v2.19），但【必須輸出】——解盲後才補＝違反凍結。
    單一模型無法做兩家族 Stouffer，所以只報配對差的描述統計，不報出口。"""
    diffs, _ = build_pairs(runs, regime, band)
    d = {m: v for m, v in diffs.items() if m in DESCRIPTIVE_MODELS}
    if not d:
        return {"status": "no_pairs"}
    out = {}
    for m, v in d.items():
        arr = np.array([x["diff"] for x in v.values()])
        out[m] = {"n_pairs": int(len(arr)), "mean_nats": float(arr.mean()),
                  "sd_nats": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                  "items": sorted(v)}
    return {"status": "descriptive_only", "reason": "v2.19：2.8b fp16 偏差不可量，"
            "退出主裁決但照實輸出", "per_model": out}


def analyse(run_id, out_dir):
    sim = load_frozen_machinery()
    runs = load_patch_files(run_id, out_dir)
    prov = load_provenance(run_id, out_dir)
    smoke = quote_smoke(run_id, out_dir)
    pc_pass = smoke.get("positive_control_pass")
    ss = np.random.SeedSequence(SEED)
    res = {"run_id": run_id, "spec": {
        "B": B_PERM, "seed": SEED, "alpha": ALPHA, "sigma_did": SIGMA_DID,
        "mde_nats": MDE_NATS, "endpoint_gold": "launch-window ΔY (nats); R descriptive only",
        "endpoint_provenance": "frac_present patched-minus-corrupted (numerator; v2.13 F)",
        "machinery_source": str(FROZEN_SIM.relative_to(PROJ)),
        "primary_arm": "L0M（M3 凍結原文：配平對內比較 L0 vs L0M 為主）",
        "primary_arm_opchar": "planning/office_reports/theory_v3_l0m_opchar.json",
        "secondary_arms": ["A_gold30 (L0P, Regime A)", "B_dual (L0P, Regime B)"],
        "descriptive_models": list(DESCRIPTIVE_MODELS),
        "foreseen_flags": sorted(f"{m}|{i}" for m, i in FORESEEN_FLAGS)},
        "models_loaded": {m: v["dtype"] for m, v in runs.items()},
        "smoke_quoted": smoke, "bands": {}}
    for band in ("early", "mid", "late"):
        c0, c1, c2, c3, c4 = ss.spawn(5)
        gold_m = run_track(sim, runs, "L0M", band,
                           np.random.default_rng(c0), "gold|L0M(REGISTERED PRIMARY, M3)")
        gold_a = run_track(sim, runs, "A_gold30", band,
                           np.random.default_rng(c1), "gold|L0P RegimeA(secondary)")
        gold_b = run_track(sim, runs, "B_dual", band,
                           np.random.default_rng(c2), "gold|L0P RegimeB(sensitivity)")
        pv = run_provenance_track(sim, runs, prov, band, np.random.default_rng(c3))
        res["bands"][band] = {
            "L0M_primary": gold_m,
            "regime_A_secondary": gold_a, "regime_B_sensitivity": gold_b,
            "provenance": pv,
            "descriptive_2p8b": {
                "L0M": descriptive_2p8b(sim, runs, "L0M", band, np.random.default_rng(c4)),
                "A_gold30": descriptive_2p8b(sim, runs, "A_gold30", band,
                                             np.random.default_rng(c4))},
            "joint_verdict_primary": joint_verdict(gold_m, pv, pc_pass),
            "joint_verdict_secondary_A": joint_verdict(gold_a, pv, pc_pass),
            "joint_verdict_sensitivity_B": joint_verdict(gold_b, pv, pc_pass),
            "sensitivity_bound_sentence": sensitivity_sentence(gold_m)}
        res["bands"][band]["frozen_narrative"] = NARRATIVE.get(
            res["bands"][band]["joint_verdict_primary"]["verdict"], NARRATIVE["INCOMPLETE"])
    return res


def report(res):
    print("run_id:", res["run_id"], "| machinery:", res["spec"]["machinery_source"])
    print("models:", {k.split('/')[-1]: v for k, v in res["models_loaded"].items()})
    print("smoke (quoted, not recomputed):", res["smoke_quoted"].get("quoted_from"))
    for band, b in res["bands"].items():
        print("")
        print("=== band:", band, "===")
        for key in ("L0M_primary", "regime_A_secondary", "regime_B_sensitivity"):
            g = b[key]
            if g.get("status") != "OK":
                print("  %-26s %s" % (key, g.get("status"))); continue
            print("  %-22s p=%.5f  z=%+.3f  mean=%+.4f nats (%.2f sigma)  TOST=%s" % (
                key, g["p_combined_two_sided"], g["stouffer_z"],
                g["effect_size"]["mean_paired_diff_nats"],
                g["effect_size"]["in_sigma_units"], g["tost"]["equivalent_IU"]))
        pv = b["provenance"]
        print("  %-26s %s" % ("provenance", pv.get("status") if pv.get("status") != "OK"
                              else "p=%.5f fires=%s" % (pv["p_combined_two_sided"], pv["fires"])))
        print("  VERDICT L0M (PRIMARY):", b["joint_verdict_primary"]["verdict"],
              "—", b["joint_verdict_primary"]["why"])
        print("  VERDICT A (secondary) :", b["joint_verdict_secondary_A"]["verdict"])
        print("  VERDICT B (sensitivity):", b["joint_verdict_sensitivity_B"]["verdict"])
        print("  2.8b descriptive      :", b["descriptive_2p8b"]["L0M"].get("status"))
        print("  ", b["sensitivity_bound_sentence"])


# ------------------------------------------------------------- 自測 --

def _synth(tmp, run_id, delta_gold, delta_prov, seed=0, n_l0=10, nested=True):
    """造出與真檔【鍵名相同】的合成資料——自測只用這個，不碰真資料。"""
    rng = np.random.default_rng(seed)
    models = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
              "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
              "allenai/OLMo-2-0425-1B"]
    bands = ("early", "mid", "late")
    prov_runs = {}
    for mi, model in enumerate(models):
        recs = {}
        for i in range(n_l0):
            l0, ctl = f"L0-{i:02d}", f"CTL-{i:02d}"
            base = rng.standard_normal()
            for iid, role, lvl in ((l0, "l0", "L0"), (ctl, "l0m", "L0M")):
                shift = delta_gold if role == "l0" else 0.0
                recs[iid] = {
                    "id": iid, "level": lvl, "model": model, "twin_of": None,
                    "n_prompt_tokens": 50, "n_gold_tokens": 54, "n_layers": 16,
                    "hook_kind": "resid_pre", "radius": 2, "launch_k": 2,
                    "backend_dtype": "torch.float32", "n_candidates": 24,
                    "n_forwards": 10, "seconds": 1.0,
                    "bands": {b: [1, 2] for b in bands},
                    "calibration": {"disqualify_item": False,
                                    "pool_too_small_final": False, "fallback": None},
                    "corruption": {"key": "k"},
                    "pairs": ([{"regime": r, "role": "l0", "partner": ctl}
                               for r in ("L0M", "A_gold30", "B_dual", "L0N_prov")]
                              if role == "l0" else
                              [{"regime": r, "role": "l0m", "partner": l0}
                               for r in ("L0M", "A_gold30", "B_dual", "L0N_prov")]),
                    "clean": {}, "corrupted": {},
                    "patches": [{"band": b, "band_layers": [1, 2], "positions": [3],
                                 "launch": {"numer": float(base + shift
                                                           + 0.3 * rng.standard_normal()),
                                            "denom": float(1.0 + 0.2 * rng.random()),
                                            "R": 0.8, "denom_unstable": False},
                                 "g1_point": {"R": 0.8, "delta_s": 1.0, "delta_y": 0.8},
                                 "segment": {}} for b in bands]}
        json.dump({"meta": {}, "model": model, "dtype": "torch.float32",
                   "config": {}, "records": recs},
                  open(tmp / f"{run_id}_patch_{model.replace('/', '_')}.json",
                       "w", encoding="utf-8"))
        rows = []
        for i in range(n_l0):
            b0 = rng.random()
            for iid, sh in ((f"L0-{i:02d}", delta_prov), (f"CTL-{i:02d}", 0.0)):
                # 端點是 patched - corrupted，所以 sh 要加在 patched 上
                vals = {"clean": 1.0, "corrupted": 0.3}
                for b in bands:
                    vals[f"patched::{b}"] = float(np.clip(
                        0.3 + b0 * 0.2 + sh + 0.05 * rng.standard_normal(), 0, 1))
                def wrap(v):
                    if not nested:
                        return v                       # 平面 float（舊 schema）
                    return {"frac_present": v, "n_probes": 13, "n_words": 45,
                            "n_unknown": 0, "max_count": 5, "resolution": 1 / 13,
                            "too_few_probes": False}
                rows.append({"id": iid, "level": "L0", "index": "x",
                             "frac_present": {k: wrap(v) for k, v in vals.items()}})
        prov_runs[model] = {"index": "x", "rows": rows, "min_probes": 6}
    json.dump({"stage": "provenance", "run_id": run_id, "runs": prov_runs},
              open(tmp / f"{run_id}_provenance_all.json", "w", encoding="utf-8"))


def self_test():
    """自測（全合成資料）。終審要求補的分支：巢狀 frac_present、平面→巢狀、
    「gold 點燃＋無溯源→無規則點燃」、「entangled＋無資料披露」、R-noeffect。"""
    import tempfile
    global B_PERM
    ok = True
    print("(1) 校準：虛無世界 40 次（自測用 B=2000 加速）")
    B_keep, B_PERM = B_PERM, 2000
    fired = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k in range(40):
            _synth(tmp, "T", 0.0, 0.0, seed=1000 + k)
            g = analyse("T", tmp)["bands"]["mid"]["L0M_primary"]
            fired += bool(g.get("fires"))
            for f in tmp.glob("T_*"):
                f.unlink()
    rate = fired / 40
    calib_ok = 0.0 <= rate <= 0.20
    ok &= calib_ok
    print("    L0M 主臂差異出口虛無點燃率 = %.3f (%d/40)  %s"
          % (rate, fired, "OK" if calib_ok else "OUT OF RANGE"))
    B_PERM = B_keep

    print("(2) 通路：B=%d" % B_PERM)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, dg, dp, expect, sd, nested in (
                ("strong both", 1.2, 0.5, ("R-specific",), 11, True),
                ("gold only(no prov effect)", 1.2, 0.0, ("無規則點燃",), 12, True),
                ("gold negative", -1.2, 0.5, ("反方向點燃",), 13, True),
                ("contradiction", 0.0, 0.8, ("矛盾格 → 無規則點燃", "R-entangled"), 14, True),
                ("FLAT frac_present", 1.2, 0.5, ("R-specific",), 11, False)):
            _synth(tmp, "T", dg, dp, seed=sd, nested=nested)
            r = analyse("T", tmp)
            b = r["bands"]["mid"]
            v = b["joint_verdict_primary"]["verdict"]
            good = v in expect
            ok &= good
            print("    %-26s -> %-22s %s" % (
                name, v, "OK" if good else "UNEXPECTED (want %s)" % (expect,)))
            assert b["L0M_primary"]["descriptive_models_excluded"] == list(DESCRIPTIVE_MODELS)
            assert b["descriptive_2p8b"]["L0M"]["status"] == "descriptive_only",                 b["descriptive_2p8b"]["L0M"]
            for f in tmp.glob("T_*"):
                f.unlink()

        print("(3) 溯源軌缺席的兩個分支")
        _synth(tmp, "T", 1.2, 0.5, seed=21)
        for f in tmp.glob("T_provenance*"):
            f.unlink()
        b = analyse("T", tmp)["bands"]["mid"]
        v1 = b["joint_verdict_primary"]
        c1 = v1["verdict"] == "無規則點燃" and "溯源軌無資料" in v1["why"]
        ok &= c1
        print("    gold 點燃＋溯源無資料 -> %-14s %s" % (v1["verdict"], "OK" if c1 else "FAIL"))
        for f in tmp.glob("T_*"):
            f.unlink()
        _synth(tmp, "T", 0.0, 0.0, seed=22)
        for f in tmp.glob("T_provenance*"):
            f.unlink()
        b2 = analyse("T", tmp)["bands"]["mid"]["joint_verdict_primary"]
        c2 = (b2["verdict"] != "R-entangled") or (
            b2.get("provenance_had_data") is False and "無資料" in b2["why"])
        ok &= c2
        print("    entangled＋無資料披露      -> %-14s %s"
              % (b2["verdict"], "OK" if c2 else "FAIL"))
        for f in tmp.glob("T_*"):
            f.unlink()

        print("(4) R-noeffect（陽性對照未達地板，優先於其他格）")
        _synth(tmp, "T", 1.2, 0.5, seed=23)
        json.dump({"positive_control": {"pass": False}, "g1": {}, "branch": {}},
                  open(tmp / "smoke_eval_T.json", "w", encoding="utf-8"))
        v3 = analyse("T", tmp)["bands"]["mid"]["joint_verdict_primary"]
        c3 = v3["verdict"] == "R-noeffect"
        ok &= c3
        print("    PC fail -> %-14s %s" % (v3["verdict"], "OK" if c3 else "FAIL"))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")   # n3：不給預設；非自測時強制要求（見下）
    ap.add_argument("--out-dir", default=str(PROJ / "results" / "causal"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.run_id:                      # n3：真資料一定要明寫 run-id，不給預設
        raise SystemExit("--run-id is required (no default; refuse to guess the run)")
    res = analyse(a.run_id, Path(a.out_dir))
    out = Path(a.out_dir) / f"analysis_{a.run_id}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    report(res)
    print("\n->", out)


if __name__ == "__main__":
    main()
