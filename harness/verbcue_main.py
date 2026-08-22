"""動詞偏置 2×2 runner（工項④，theory_verbcue_prereg_v1.md §2-§7）。

規格權威：
  §2（第109-143行）：Δ_verb 定義、候選層級存活率、以題為叢集單位。
  §3（第146-193行）：題集裁定（v2.21 §2/§3-0 語料閘沿用，不另立）。
  §5（第226-260行）：共同主要條件 A（腐蝕適足）＋三分母強制回報。
  §6（第267-295行）：n_repl_per_cue=整池、逐題候選數/存活數/pool_too_small 實測。
  §6.6（第392-429行）：材料可換、判準不可換——本檔常數全部 import 自
  causal_patch，不重抄任何門檻。

【現況（2026-08-22）】②「verb×修正池」與 proper 修正池的精確構造仍待統籌代理/
理論部裁定（已發訊詢問），本檔的四格定義因此設計成**外部傳入**（見
`CellDef`），不寫死在邏輯裡——裁定一出爐，換設定檔即可，runner 骨架不用重寫。

【單一出處】PatchConfig 的凍結欄位（quantile/min_survivors/rank_guard/
adequacy_multiple/n_cue）全部用 causal_patch.PatchConfig 的預設值，不重抄。
數值底線 1.434e-4 nats、適足下限=10×底線——shared.md 凍結值速查，唯一出處。

【只備料，不判定】本檔輸出 p、三分母、drop 中位數、條件A、停機規則旗標、
Δ_verb 的 cluster bootstrap CI——不寫「支持/反對詞池歸因」，§5 的格①②③④
映射由統籌代理按凍結文字執行。

【全程 CPU，不碰 GPU】本檔的 `main()`／`run_cell()` 需要載入真實模型才能真的
跑；本次交付只到「自測全過、骨架就緒」，不執行任何模型載入或前向——
點火令另外等作者時段。

用法：
  自測：.venv/Scripts/python.exe harness/verbcue_main.py --self-test
  （真跑需要 GPU 與作者點火令，本檔尚未在本次交付中執行）
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_patch as cp  # noqa: E402  單一出處：PatchConfig 凍結欄位、run_item

PROJ = Path(__file__).resolve().parent.parent

# 數值底線與適足下限——shared.md「凍結值速查」唯一出處，本檔不重新推導。
NUMERICAL_FLOOR_NATS = 1.434e-4        # fp32 batch 路徑
ADEQUACY_FLOOR_NATS = cp.ADEQUACY_MULTIPLE * NUMERICAL_FLOOR_NATS   # 10 × 底線
STOP_RULE_THRESHOLD = 0.40              # v2.8 停機規則：>40%
DEFAULT_BANDS = {"early": (0.25, 0.45), "mid": (0.45, 0.65), "late": (0.70, 0.90)}
SEED = 20260822

NO_CANDIDATE_ERRORS = {"no_valid_corrupt_candidate", "no_positive_drop",
                       "no_candidate_within_rank_guard"}


def _abort(msg: str):
    raise SystemExit(f"[verbcue_main ABORT] {msg}")


def make_patch_config(pool_size: int) -> cp.PatchConfig:
    """n_repl_per_cue = 整池（工項④要求）；其餘用 PatchConfig 凍結預設。"""
    return cp.PatchConfig(bands=dict(DEFAULT_BANDS), n_cue=6,
                          n_repl_per_cue=pool_size,
                          numerical_floor_nats=NUMERICAL_FLOOR_NATS,
                          seed=SEED, gen_tokens=0)   # §3-b：不做生成/溯源


def prepare_item(base_item: dict, cue_word: str, pool_class: str, pool_words) -> dict:
    """淺拷貝＋覆寫，不動 battery.json 原件（causal_patch.py:292-303 支援的
    item['cue_word']/['cue_pool_class']/['corrupt_pool'] 逐題覆寫）。"""
    it = dict(base_item)
    it["cue_word"] = cue_word
    it["cue_pool_class"] = pool_class
    it["corrupt_pool"] = list(pool_words)
    return it


# ============================================================ 逐格執行 ==

def run_cell(model_obj, cell_items: dict, pool_class: str, pool_words=None,
            pool_words_by_item: dict | None = None) -> dict:
    """cell_items: {item_id: {"item": battery_dict, "cue_word": str}}。
    通用欄傳 `pool_words`（單一 list，整格共用）；修正欄傳 `pool_words_by_item`
    （{item_id: list}，逐題各自池、尺寸可不同——例如 L0N-06/verb 因通用欄撞名
    截短為 23，見 build_verbcue_genre_pools.py 的 §0-b 類推截短）。
    n_repl_per_cue（＝整池，工項④要求）因此逐題各自設定，不是整格一個常數。
    回傳 {item_id: record}（record 結構＝ causal_patch.run_item 的輸出）。"""
    if (pool_words is None) == (pool_words_by_item is None):
        _abort("run_cell 需要恰好一種池來源：pool_words（通用欄）或 pool_words_by_item（修正欄）")
    records = {}
    for iid, spec in cell_items.items():
        this_pool = pool_words if pool_words is not None else pool_words_by_item.get(iid)
        if this_pool is None:
            _abort(f"{iid}: pool_words_by_item 缺這一題的池，無法跑")
        cfg = make_patch_config(len(this_pool))
        item = prepare_item(spec["item"], spec["cue_word"], pool_class, this_pool)
        rec = cp.run_item(model_obj, item, cfg)
        rec["id"] = iid
        records[iid] = rec
    return records


# ============================================================ 聚合／三分母 ==

def aggregate_arm(records: dict) -> dict:
    """§5 強制回報：三分母（進場/適足出局/pool_too_small）＋條件A＋停機旗標。
    輸入 records: {item_id: run_item 輸出的 record dict}（或自測用合成 record，
    schema 相同）。"""
    entered, adequacy_out, pool_too_small = [], [], []
    other_error = []
    n_surv_total = n_cand_total = 0
    selected_drops = []
    per_item = {}

    for iid, rec in records.items():
        if not isinstance(rec, dict):
            _abort(f"{iid}: record 不是 dict（型別={type(rec)}）——上游壞了，不猜")
        err = rec.get("error")
        if err in NO_CANDIDATE_ERRORS:
            per_item[iid] = {"status": "no_candidate", "error": err}
            continue
        if err == "below_adequacy_floor":
            adequacy_out.append(iid)
            per_item[iid] = {"status": "adequacy_excluded", "error": err}
            continue
        if err:
            other_error.append(iid)
            per_item[iid] = {"status": "other_error", "error": err}
            continue
        cal = rec.get("calibration")
        if cal is None:
            _abort(f"{iid}: 無 error 但也無 calibration——不符合 run_item 的輸出契約，"
                   "停止（這是防禦性斷言，不是規格條文，但寧可停也不猜）")
        n_surv = cal.get("n_survivors")
        n_cand = cal.get("n_candidates")
        if n_surv is None or n_cand is None:
            _abort(f"{iid}: calibration 缺 n_survivors/n_candidates，schema 不符，停止")
        entered.append(iid)
        n_surv_total += n_surv
        n_cand_total += n_cand
        flagged_small = bool(cal.get("pool_too_small"))
        if flagged_small:
            pool_too_small.append(iid)
        drop = cal.get("selected_drop_launch")
        if drop is not None:
            selected_drops.append(drop)
        per_item[iid] = {"status": "entered", "n_survivors": n_surv, "n_candidates": n_cand,
                         "pool_too_small": flagged_small, "selected_drop_launch": drop,
                         "meets_adequacy": cal.get("meets_adequacy")}

    n_total = len(records)
    n_entered = len(entered)
    p = (n_surv_total / n_cand_total) if n_cand_total else None
    drop_median = float(np.median(selected_drops)) if selected_drops else None
    condition_A_adequate = (drop_median is not None and drop_median >= ADEQUACY_FLOOR_NATS)
    n_flagged_for_stop = len(adequacy_out) + len(pool_too_small)
    stop_frac = (n_flagged_for_stop / n_total) if n_total else None
    stop_rule_triggered = bool(stop_frac is not None and stop_frac > STOP_RULE_THRESHOLD)

    return {
        "n_total_attempted": n_total,
        "n_entered": n_entered,                          # 分母①
        "n_adequacy_excluded": len(adequacy_out),         # 分母②
        "n_pool_too_small_flagged": len(pool_too_small),  # 分母③
        "n_other_error": len(other_error),
        "adequacy_excluded_items": sorted(adequacy_out),
        "pool_too_small_items": sorted(pool_too_small),
        "other_error_items": sorted(other_error),
        "n_survivors_total": n_surv_total, "n_candidates_total": n_cand_total,
        "p_candidate_level": p,
        "drop_median_nats": drop_median, "adequacy_floor_nats": ADEQUACY_FLOOR_NATS,
        "condition_A_adequate": condition_A_adequate,
        "stop_rule_flagged_fraction": stop_frac,
        "stop_rule_threshold": STOP_RULE_THRESHOLD,
        "stop_rule_triggered": stop_rule_triggered,
        "per_item": per_item,
    }


# =========================================== Δ_verb 的 cluster bootstrap CI ==

def cluster_bootstrap_delta(l0_per_item: dict, l0n_per_item: dict, B: int, rng,
                            conf: float = 0.90):
    """§2：Δ_verb = p_L0 − p_L0N，pp 為單位，以題為叢集單位。
    l0_per_item/l0n_per_item：{item_id: {"n_survivors":int,"n_candidates":int}}
    （只放「entered」的題；適足出局/pool_too_small 不影響這裡的資格判斷——
    pool_too_small 仍算入 p，適足出局的題本來就已經被 aggregate_arm 排除掉了）。
    """
    def arm_arrays(per_item):
        ids = sorted(per_item)
        surv = np.array([per_item[i]["n_survivors"] for i in ids], dtype=float)
        cand = np.array([per_item[i]["n_candidates"] for i in ids], dtype=float)
        return ids, surv, cand

    l0_ids, l0_s, l0_c = arm_arrays(l0_per_item)
    l0n_ids, l0n_s, l0n_c = arm_arrays(l0n_per_item)
    if len(l0_ids) == 0 or len(l0n_ids) == 0:
        return {"status": "INCOMPLETE", "reason": "一臂沒有可用題目"}

    p_l0 = float(l0_s.sum() / l0_c.sum()) if l0_c.sum() else None
    p_l0n = float(l0n_s.sum() / l0n_c.sum()) if l0n_c.sum() else None
    if p_l0 is None or p_l0n is None:
        return {"status": "INCOMPLETE", "reason": "一臂候選總數為 0"}
    delta_obs_pp = (p_l0 - p_l0n) * 100.0

    m0, m0n = len(l0_ids), len(l0n_ids)
    idx0 = rng.integers(0, m0, size=(B, m0))
    idx0n = rng.integers(0, m0n, size=(B, m0n))
    boot_s0 = l0_s[idx0].sum(1); boot_c0 = l0_c[idx0].sum(1)
    boot_s0n = l0n_s[idx0n].sum(1); boot_c0n = l0n_c[idx0n].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_p0 = np.where(boot_c0 > 0, boot_s0 / np.maximum(boot_c0, 1), np.nan)
        boot_p0n = np.where(boot_c0n > 0, boot_s0n / np.maximum(boot_c0n, 1), np.nan)
    boot_delta_pp = (boot_p0 - boot_p0n) * 100.0
    boot_delta_pp = boot_delta_pp[np.isfinite(boot_delta_pp)]
    lo_pct, hi_pct = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    ci = [float(np.percentile(boot_delta_pp, lo_pct)),
         float(np.percentile(boot_delta_pp, hi_pct))] if len(boot_delta_pp) else [None, None]

    return {"status": "OK", "p_L0": p_l0, "p_L0N": p_l0n,
            "delta_verb_pp": delta_obs_pp, "ci_90_pp": ci, "B": int(B),
            "n_items_L0": m0, "n_items_L0N": m0n,
            "midpoint_pp": 15.7, "anchor_verb_subset_pp": 0.8,
            "anchor_full_pool_pp": 30.6,
            "note": "只備料：CI 端點與兩個既有錨並列，格①②③④映射由統籌代理按 §2/§5 執行"}


# ================================================================ 點火 ==
# 四模型逐一載入（8GB 紀律：一次只載一個），每模型跑完四格裡它語料閘合格
# 的題目。落盤沿用 causal_main.py 的既有慣例（暫存檔+原子改名，被殺不留半個
# JSON）；建議透過 PowerShell Start-Process 獨立行程啟動＋落日誌（rig.md
# 已知坑：工作管理系統會殺重載入）。

MODELS_TO_RUN = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
                 "EleutherAI/pythia-1.4b", "allenai/OLMo-2-0425-1B"]


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


def _save_shard(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def shard_path(out_dir, model: str) -> Path:
    return Path(out_dir) / f"verbcue_patch_{model.replace('/', '_')}.json"


def run_ignition(out_dir, config_path, models=None, device="cuda"):
    """真跑主迴圈：逐模型載入、逐格跑該模型語料閘合格的題目，逐格落盤（可斷點續跑）。
    只執行 causal_patch.run_item，不在這裡做任何判定——輸出留給
    aggregate_arm()/cluster_bootstrap_delta() 事後聚合（只備料不判定）。"""
    import time
    import torch

    models = models or MODELS_TO_RUN
    config = json.load(open(config_path, encoding="utf-8"))
    cells = config["cells"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ignite] 模型清單：{models}", flush=True)
    print(f"[ignite] 設定檔：{config_path}（{len(cells)} 格：{list(cells)}）", flush=True)

    for model_name in models:
        path = shard_path(out_dir, model_name)
        shard = {"model": model_name, "dtype": "fp32", "cells": {}}
        if path.exists():
            shard = json.load(open(path, encoding="utf-8"))
            print(f"[ignite] {model_name}: 續跑既有分片（已有格：{list(shard.get('cells', {}))}）",
                 flush=True)

        t0 = time.time()
        print(f"[ignite] 載入 {model_name}...", flush=True)
        model_obj = cp.load_model(model_name, device, torch.float32)
        load_s = time.time() - t0
        shard["load_seconds"] = load_s
        print(f"[ignite] {model_name} 載入完成，{load_s:.2f}s", flush=True)

        try:
            n_fwd_total = 0
            t1 = time.time()
            for cell_name, cell_def in cells.items():
                if cell_name in shard["cells"] and shard["cells"][cell_name]:
                    print(f"[ignite]   {cell_name}: 已完成（續跑略過）", flush=True)
                    continue
                items_for_model = {iid: spec for iid, spec in cell_def["items"].items()
                                   if spec["corpus_mask"].get(model_name)}
                if not items_for_model:
                    shard["cells"][cell_name] = {}
                    _save_shard(path, shard)
                    print(f"[ignite]   {cell_name}: 該模型語料閘 0 題合格，跳過", flush=True)
                    continue
                if cell_def["column"] == "universal":
                    records = run_cell(model_obj, items_for_model, cell_def["pool_class"],
                                       pool_words=cell_def["pool_words"])
                else:
                    records = run_cell(model_obj, items_for_model, cell_def["pool_class"],
                                       pool_words_by_item=cell_def["pool_words_by_item"])
                for r in records.values():
                    n_fwd_total += r.get("n_forwards", 0)
                shard["cells"][cell_name] = records
                _save_shard(path, shard)
                print(f"[ignite]   {cell_name}: {len(records)} 題完成，累計前向 {n_fwd_total}",
                     flush=True)
            secs = time.time() - t1
            shard.setdefault("progress", {})
            shard["progress"].update({"n_forwards_total": n_fwd_total, "seconds_total": secs})
            _save_shard(path, shard)
            print(f"[ignite] {model_name} 全部格完成：{n_fwd_total} 次前向、{secs:.1f}s -> {path}",
                 flush=True)
        finally:
            del model_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("ALL IGNITE RUNS DONE", flush=True)


# ======================================================= 自測（合成資料） ==
# 照位置檢定那套標準：測行為，不是測「跑完了」。全部用合成 record（schema
# 與 causal_patch.run_item 相同），不載入任何模型、不跑 GPU。

def _synth_record(iid, n_cand, n_surv, drop, pool_too_small=None, meets_adequacy=True,
                  error=None):
    """造一筆與 run_item 輸出 schema 相同的合成 record。"""
    if error is not None:
        return {"id": iid, "error": error}
    if pool_too_small is None:
        pool_too_small = bool(n_surv < cp.MIN_SURVIVORS)
    cal = {"n_candidates": n_cand, "n_survivors": n_surv, "pool_too_small": pool_too_small,
          "selected": f"{iid}-key", "selected_drop_launch": drop,
          "meets_adequacy": meets_adequacy, "adequacy_checked": True,
          "adequacy_floor_nats": ADEQUACY_FLOOR_NATS}
    if not meets_adequacy:
        return {"id": iid, "error": "below_adequacy_floor", "calibration": cal}
    return {"id": iid, "calibration": cal, "n_candidates": n_cand}


def self_test():
    ok = True
    summary = {}

    print("=== 自測 (1)：正常路徑——三分母與 p 算對 ===")
    recs = {
        "A": _synth_record("A", n_cand=24, n_surv=23, drop=0.05),   # 進場，非 pool_too_small
        "B": _synth_record("B", n_cand=24, n_surv=10, drop=0.04),   # 進場，pool_too_small（10<12）
        "C": _synth_record("C", n_cand=24, n_surv=0, drop=0.0, meets_adequacy=False),  # 適足出局
        "D": _synth_record("D", n_cand=0, n_surv=0, drop=None, error="no_positive_drop"),  # 無候選
    }
    agg = aggregate_arm(recs)
    c1 = (agg["n_total_attempted"] == 4 and agg["n_entered"] == 2
         and agg["n_adequacy_excluded"] == 1 and agg["n_pool_too_small_flagged"] == 1
         and agg["n_survivors_total"] == 33 and agg["n_candidates_total"] == 48
         and abs(agg["p_candidate_level"] - 33 / 48) < 1e-9)
    ok &= c1
    summary["basic_denominators_ok"] = c1
    print(f"    n_entered={agg['n_entered']} adequacy_excl={agg['n_adequacy_excluded']} "
         f"pool_too_small={agg['n_pool_too_small_flagged']} p={agg['p_candidate_level']:.4f}"
         f"  {'OK' if c1 else 'FAIL'}")

    print("=== 自測 (2)：pool_too_small 仍計入 p（只標旗標不排除）===")
    c2 = ("B" not in [] and agg["p_candidate_level"] is not None
         and 10 <= agg["n_survivors_total"])  # B 的 10 個存活確實被算進總數
    # 更直接的檢查：把 B 拿掉重算，p 應該不同
    recs_no_b = {k: v for k, v in recs.items() if k != "B"}
    agg_no_b = aggregate_arm(recs_no_b)
    c2 = agg["p_candidate_level"] != agg_no_b["p_candidate_level"]
    ok &= c2
    summary["pool_too_small_counted_in_p"] = c2
    print(f"    含B的p={agg['p_candidate_level']:.4f}  去B的p={agg_no_b['p_candidate_level']:.4f}"
         f"  {'OK' if c2 else 'FAIL'}")

    print("=== 自測 (3)：條件A（drop 中位數 vs 適足下限）===")
    high_drop = {"X": _synth_record("X", 24, 20, drop=ADEQUACY_FLOOR_NATS * 5),
                "Y": _synth_record("Y", 24, 20, drop=ADEQUACY_FLOOR_NATS * 3)}
    low_drop = {"X": _synth_record("X", 24, 20, drop=ADEQUACY_FLOOR_NATS * 0.1),
               "Y": _synth_record("Y", 24, 20, drop=ADEQUACY_FLOOR_NATS * 0.2)}
    a_high = aggregate_arm(high_drop)["condition_A_adequate"]
    a_low = aggregate_arm(low_drop)["condition_A_adequate"]
    c3 = (a_high is True) and (a_low is False)
    ok &= c3
    summary["condition_A_ok"] = c3
    print(f"    高drop條件A={a_high}  低drop條件A={a_low}  {'OK' if c3 else 'FAIL'}")

    print("=== 自測 (4)：停機規則（>40% 旗標觸發，界線兩側各測一次）===")
    # 5 題：2 題旗標(adequacy或pool_too_small) = 40%，不觸發（規則是嚴格>40%）
    five_40 = {
        "1": _synth_record("1", 24, 20, 0.05), "2": _synth_record("2", 24, 20, 0.05),
        "3": _synth_record("3", 24, 20, 0.05),
        "4": _synth_record("4", 24, 5, 0.05),                              # pool_too_small
        "5": _synth_record("5", 24, 0, 0.0, meets_adequacy=False),         # 適足出局
    }
    # 5 題：3 題旗標 = 60%，觸發
    five_60 = dict(five_40)
    five_60["3"] = _synth_record("3", 24, 5, 0.05)   # 也變成 pool_too_small
    agg40 = aggregate_arm(five_40); agg60 = aggregate_arm(five_60)
    c4 = (agg40["stop_rule_flagged_fraction"] == 0.40 and agg40["stop_rule_triggered"] is False
         and agg60["stop_rule_flagged_fraction"] == 0.60 and agg60["stop_rule_triggered"] is True)
    ok &= c4
    summary["stop_rule_ok"] = c4
    print(f"    40%旗標->triggered={agg40['stop_rule_triggered']}（應 False，規則是嚴格>40%）  "
         f"60%旗標->triggered={agg60['stop_rule_triggered']}（應 True）  {'OK' if c4 else 'FAIL'}")

    print("=== 自測 (5)：abort 行為——record 缺 calibration 且無 error ===")
    bad = {"Z": {"id": "Z"}}   # 既無 error 也無 calibration，違反 run_item 輸出契約
    try:
        aggregate_arm(bad)
        a5 = False
    except SystemExit:
        a5 = True
    ok &= a5
    summary["abort_on_malformed_record_ok"] = a5
    print(f"    缺 calibration 且無 error -> abort: {'OK' if a5 else 'FAIL'}")

    print("=== 自測 (6)：abort 行為——record 不是 dict ===")
    bad2 = {"Z": "not-a-dict"}
    try:
        aggregate_arm(bad2)
        a6 = False
    except SystemExit:
        a6 = True
    ok &= a6
    summary["abort_on_non_dict_record_ok"] = a6
    print(f"    record 非 dict -> abort: {'OK' if a6 else 'FAIL'}")

    print("=== 自測 (7)：Δ_verb cluster bootstrap——注入已知方向應在 CI 反映出來 ===")
    rng = np.random.default_rng(1)
    # L0：20/24 存活（高存活）；L0N：5/24 存活（低存活）——注入明顯的正 Δ_verb
    l0_items = {f"L0-{i}": {"n_survivors": 20, "n_candidates": 24} for i in range(3)}
    l0n_items = {f"L0N-{i}": {"n_survivors": 5, "n_candidates": 24} for i in range(3)}
    res = cluster_bootstrap_delta(l0_items, l0n_items, B=5000, rng=rng)
    c7 = (res["status"] == "OK" and res["delta_verb_pp"] > 30
         and res["ci_90_pp"][0] > 0)   # 明顯正差，CI 下界應該也是正的
    ok &= c7
    summary["cluster_bootstrap_direction_ok"] = c7
    print(f"    Δ_verb={res.get('delta_verb_pp')}  CI={res.get('ci_90_pp')}  "
         f"{'OK' if c7 else 'FAIL'}")

    print("=== 自測 (8)：Δ_verb cluster bootstrap——零差異時 CI 應涵蓋 0 附近 ===")
    # 兩臂逐題人為造出一點題間變異（否則 bootstrap 重抽全部同值、CI 退化成一個點，
    # 那是測資設計的問題不是程式的問題——見自測(8)第一版失敗記錄），
    # 但兩臂的「整體」候選層級 p 刻意設成相等，驗證 CI 會涵蓋 0。
    rng8 = np.random.default_rng(3)
    l0_eq = {f"L0-{i}": {"n_survivors": int(v), "n_candidates": 24}
            for i, v in enumerate([16, 18, 20, 22, 24])}     # 均值 20/24
    l0n_eq = {f"L0N-{i}": {"n_survivors": int(v), "n_candidates": 24}
             for i, v in enumerate([24, 22, 20, 18, 16])}    # 均值也是 20/24，逐題不同分佈
    res_eq = cluster_bootstrap_delta(l0_eq, l0n_eq, B=5000, rng=rng8)
    c8 = (res_eq["status"] == "OK" and abs(res_eq["delta_verb_pp"]) < 1e-9
         and res_eq["ci_90_pp"][0] < 0 < res_eq["ci_90_pp"][1])
    ok &= c8
    summary["cluster_bootstrap_null_ok"] = c8
    print(f"    Δ_verb={res_eq.get('delta_verb_pp')}  CI={res_eq.get('ci_90_pp')}  "
         f"{'OK' if c8 else 'FAIL'}")

    print("=== 自測 (9)：run_cell 池來源選擇（通用欄 vs 修正欄逐題各異池），"
         "monkeypatch cp.run_item 避免載入真模型 ===")
    calls = []

    def fake_run_item(model_obj, item, cfg):
        calls.append({"id": item["id"], "corrupt_pool": list(item["corrupt_pool"]),
                     "n_repl_per_cue": cfg.n_repl_per_cue})
        return {"id": item["id"], "calibration": {"n_candidates": len(item["corrupt_pool"]),
                                                   "n_survivors": 1, "pool_too_small": True}}

    real_run_item = cp.run_item
    cp.run_item = fake_run_item
    try:
        cell_items = {"L0N-06": {"item": {"id": "L0N-06", "prompt": "x"}, "cue_word": "baked"},
                      "L0N-09": {"item": {"id": "L0N-09", "prompt": "y"}, "cue_word": "z"}}
        # 通用欄：整格共用一份 24 詞池
        calls.clear()
        run_cell(None, cell_items, "verb", pool_words=list(range(24)))
        c9a = all(c["n_repl_per_cue"] == 24 for c in calls)
        # 修正欄：逐題各自池，L0N-06 刻意給 23（模擬 §0-b 類推截短後的情形）
        calls.clear()
        by_item = {"L0N-06": list(range(23)), "L0N-09": list(range(24))}
        run_cell(None, cell_items, "verb", pool_words_by_item=by_item)
        c9b = (next(c["n_repl_per_cue"] for c in calls if c["id"] == "L0N-06") == 23
              and next(c["n_repl_per_cue"] for c in calls if c["id"] == "L0N-09") == 24)
        # 兩種來源都不給/都給 -> abort
        try:
            run_cell(None, cell_items, "verb")
            c9c = False
        except SystemExit:
            c9c = True
        try:
            run_cell(None, cell_items, "verb", pool_words=[1], pool_words_by_item={"x": [1]})
            c9d = False
        except SystemExit:
            c9d = True
        # 修正欄缺某一題的池 -> abort
        try:
            run_cell(None, cell_items, "verb", pool_words_by_item={"L0N-06": [1] * 23})
            c9e = False
        except SystemExit:
            c9e = True
    finally:
        cp.run_item = real_run_item
    c9 = c9a and c9b and c9c and c9d and c9e
    ok &= c9
    summary["run_cell_pool_source_ok"] = c9
    print(f"    通用欄整格同池={c9a}  修正欄逐題各異池(23 vs 24)={c9b}  "
         f"兩者皆無->abort={c9c}  兩者皆有->abort={c9d}  缺題->abort={c9e}  "
         f"{'OK' if c9 else 'FAIL'}")

    print("=== 自測 (10)：真實設定檔端到端乾跑（monkeypatch cp.run_item，不載模型）===")
    print("    背景：自測(9)用手造合成資料，其 item spec 是我自己組的，不是從")
    print("    真實 verbcue_items.json 流出來的——第一次真點火時就撞到這個缺口")
    print("    （run_cell 要 spec['item']，題集 JSON 當時沒這個欄位，KeyError）。")
    print("    這裡用真實 battery/verbcue_cell_config.json 逐格跑一次，堵住同類缺口再發生。")
    cfg_path = PROJ / "battery" / "verbcue_cell_config.json"
    if not cfg_path.exists():
        c10 = None
        print(f"    {cfg_path} 不存在，跳過（不算失敗，只是這個環境還沒 build 過設定檔）")
    else:
        config = json.load(open(cfg_path, encoding="utf-8"))
        e2e_calls = []

        def fake_run_item_e2e(model_obj, item, cfg):
            assert "prompt" in item, f"item 缺 prompt: {sorted(item.keys())}"
            assert item.get("gold_continuation") or item.get("expected_continuation"), (
                f"item 缺 gold_continuation/expected_continuation: {sorted(item.keys())}")
            assert item.get("cue_word") and item.get("cue_pool_class") and item.get("corrupt_pool")
            e2e_calls.append({"id": item["id"], "n_pool": len(item["corrupt_pool"]),
                             "n_repl_per_cue": cfg.n_repl_per_cue})
            return {"id": item["id"], "calibration": {"n_candidates": len(item["corrupt_pool"]),
                                                       "n_survivors": 1, "pool_too_small": True,
                                                       "selected_drop_launch": 0.0}}

        real_run_item2 = cp.run_item
        cp.run_item = fake_run_item_e2e
        try:
            n_total_records = 0
            for model_name in MODELS_TO_RUN:
                for cell_name, cell_def in config["cells"].items():
                    items_for_model = {iid: spec for iid, spec in cell_def["items"].items()
                                       if spec["corpus_mask"].get(model_name)}
                    if not items_for_model:
                        continue
                    if cell_def["column"] == "universal":
                        recs = run_cell(None, items_for_model, cell_def["pool_class"],
                                        pool_words=cell_def["pool_words"])
                    else:
                        recs = run_cell(None, items_for_model, cell_def["pool_class"],
                                        pool_words_by_item=cell_def["pool_words_by_item"])
                    n_total_records += len(recs)
            c10 = n_total_records > 0 and all(c["n_repl_per_cue"] == c["n_pool"] for c in e2e_calls)
        finally:
            cp.run_item = real_run_item2
        print(f"    {len(e2e_calls)} 次 run_item 呼叫全部通過 schema 斷言，"
             f"n_repl_per_cue 全部等於該次呼叫的池尺寸  {'OK' if c10 else 'FAIL'}")
    if c10 is not None:
        ok &= c10
    summary["e2e_real_config_dry_run_ok"] = c10

    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    summary["overall"] = "PASS" if ok else "FAIL"
    return (0 if ok else 1), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ignite", action="store_true",
                    help="真跑：載入四個模型、實際執行 causal_patch.run_item（需要 GPU）")
    ap.add_argument("--config", default=str(PROJ / "battery" / "verbcue_cell_config.json"))
    ap.add_argument("--out-dir", default=str(PROJ / "results" / "causal"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.self_test:
        code, summary = self_test()
        print("\nself-test summary:", json.dumps(summary, ensure_ascii=False))
        raise SystemExit(code)
    if a.ignite:
        run_ignition(a.out_dir, a.config, device=a.device)
        raise SystemExit(0)
    raise SystemExit("指定 --self-test 或 --ignite。")


if __name__ == "__main__":
    main()
