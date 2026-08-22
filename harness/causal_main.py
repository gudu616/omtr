"""窗口 B 主實驗驅動器（器材部）。

依據規格版本：**CAUSAL_PREREG_v1 第一段凍結 ＋ v2.6–v2.16 增補裁決**
（權威＝`planning/DESIGN_PROPOSAL_v2.2_rulings.md` 對應塊）。

**這支是薄迴圈，不是新方法。** 所有量測邏輯都在**已過審的 `causal_patch`**
核心裡（q=0.75 分位選詞、top-50 rank 護欄、per_tok 全存、單 donor cue 鄰域
單層帶貼回、R 與 ΔY）。本檔只負責：讀配對表 → 逐 (模型, 題) 呼叫核心 →
**增量落盤** → 記前向帳。**不做任何分析判定**（verdict 一律 null）。

## 為什麼迴圈是逐「題」不是逐「配對」

配對是 (L0 receiver, L0P 對照) 的**分析結構**，不是量測單位——同一個 L0 題
可能同時出現在 Regime A 與 B 的配對裡。逐配對跑會把同一題量兩次，
既浪費窗口又讓兩次結果可能不一致。所以：**逐題量一次，配對只當索引**，
配對關係原樣寫進輸出讓分析部組對比。

## 斷點續跑（今晚學到的）

每完成一個 (模型, 題) 就寫一次分片檔 `main_<run_id>_<model>.json`；
重跑時已完成的題直接跳過。窗口被中斷、工作管理系統殺掉、OOM——
都不會讓已經花掉的 GPU 時間白費。`--fresh` 才會重來。

## 段

  budget   只算前向帳與時間投影，不碰 GPU（先報預算再開跑）
  patch    主量測：逐 (模型, 題) 的腐蝕-恢復（fp16、batch=1）
  gen      溯源軌素材：L0 receiver 的 clean/corrupted/patched 自由續寫
           （60 token、**關 KV cache**——patch 按絕對位置貼，開 cache 會對不上）
  control  H2/J4 fp32 對照：保留題上 fp16 與 fp32 各一遍，量 dtype 對 ΔY/DiD 的偏差
  all      ＝ patch + gen（control 另跑，它用保留題不是裁決池）

用法（獨立行程，Start-Process）：
    .venv/Scripts/python.exe harness/causal_main.py --stage budget --run-id winB1
    .venv/Scripts/python.exe harness/causal_main.py --stage patch  --run-id winB1
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import causal_patch as cp  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT_DIR = PROJ / "results" / "causal"
PAIRS_JSON = PROJ / "planning" / "battery_expansion" / "adjudication_pairs.json"
HOLDOUT = PROJ / "planning" / "battery_expansion" / "holdout_12.json"
DEFAULT_MODELS = ("EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
                  "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
                  "allenai/OLMo-2-0425-1B")
# 註：tag（底線形）→ 模型名的對照曾經有一份 TAG2MODEL 常數，但驅動器裡
# 每個用到的地方都是反向（model → tag，直接 replace），沒有一處需要它，
# 留著只會讓人以為某處靠它做對照。已移除（審查指出的死碼之一）。
# 溯源軌的宿主層級（§9 硬約束②：不掛 L0P，其溯源出口是構造性的零）。
# **一個出處**：budget 段與 gen 段都用這個常數——原本各寫各的，budget 只數 L0，
# v2.18 把 L0N 接進來之後就把窗口低估了。同一條規則寫兩次＝遲早會漂。
GEN_HOST_LEVELS = ("L0", "L0N")
# F1：續跑時「哪些 error 算做完了」。
# **規則性結果**＝凍結規則對這一題的判定（選不出腐蝕、護欄擋掉、適足不過…）：
# 決定性的，重跑一次還是同一個答案，所以算做完，不重跑。
# **基礎設施例外**＝OOM／CUDA 掉線／讀檔壞掉那類（存的是 repr(e)）：
# 那是機器的問題不是題目的問題，**不重跑就等於讓一次瞬時 OOM 永久踢掉一題**。
# 新紀錄一律帶 error_kind 明確區分；舊分片沒有這個欄位，就退回比對碼表。
RULE_ERROR_CODES = frozenset({
    "no_valid_corrupt_candidate", "no_candidate_within_rank_guard",
    "no_positive_drop", "pool_too_small", "pool_too_small_final",
    "below_adequacy_floor",
})


def is_retryable_error(rec: dict) -> bool:
    """這筆紀錄的 error 是「機器出包」（該重跑）還是「規則判定」（不該重跑）？"""
    if "error" not in rec:
        return False
    kind = rec.get("error_kind")
    if kind:
        return kind == "exception"
    return rec["error"] not in RULE_ERROR_CODES


# ---------------------------------------------------------------- 共用

def env_meta() -> dict:
    import importlib.metadata as md
    import transformers
    return {"spec_version": cp.SPEC_VERSION,
            "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__,
            "transformer_lens": md.version("transformer-lens"),
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


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


def shard_path(run_id: str, model: str, stage: str, out_dir: Path) -> Path:
    return out_dir / f"main_{run_id}_{stage}_{model.replace('/', '_')}.json"


def load_shard(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 分片檔讀不動（{e!r}），視同從頭跑：{path}")
    return {}


def save_shard(path: Path, payload: dict) -> None:
    """增量落盤。先寫暫存再原子改名——中途被殺不會留下半個 JSON。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- 題目與配對

def load_pairs(path: Path | None = None) -> dict:
    p = Path(path) if path else PAIRS_JSON
    if not p.exists():
        raise SystemExit(f"配對表不存在：{p}\n"
                         "  先跑 harness/build_pairs.py（它會與凍結值對帳）")
    return json.load(open(p, encoding="utf-8"))


def battery_index() -> dict:
    """把所有題目來源併成 {id: item}。裁決池題目散在三個檔裡。"""
    idx = {}
    for p in [PROJ / "battery" / "battery.json",
              PROJ / "battery" / "battery_l0p.json"]:
        if p.exists():
            for it in json.load(open(p, encoding="utf-8")).get("items", []):
                idx[it["id"]] = it
    # L0M 裁決池題目在 l0m_candidates.json 的巢狀物件裡
    lp = PROJ / "planning" / "battery_expansion" / "l0m_candidates.json"
    if lp.exists():
        for pr in json.load(open(lp, encoding="utf-8")).get("pairs", []):
            for side in ("l0", "l0m"):
                o = pr.get(side)
                if isinstance(o, dict) and o.get("id"):
                    idx.setdefault(o["id"], o)
    return idx


def items_for_model(pairs: dict, tag: str) -> tuple[list[str], dict]:
    """該模型要量的題目（去重）＋配對索引。

    **去重是刻意的**：同一題可能同時在 Regime A 與 B 的配對裡出現，
    逐配對跑會量兩次。逐題量一次、配對當索引，兩個 regime 共用同一批量測。
    """
    ids, membership = [], {}
    for rk, rv in pairs["regimes"].items():
        for row in rv["per_run"].get(tag, []):
            for side in ("l0", "l0p"):
                iid = row[side]
                if iid not in membership:
                    membership[iid] = []
                    ids.append(iid)
                membership[iid].append({"regime": rk, "role": side,
                                        "partner": row["l0p" if side == "l0" else "l0"]})
    # v2.18／R1：溯源軌的 L0↔L0N。沒有這一段，gen 段那個 level in ("L0","L0N")
    # 的濾器就是死碼，R-specific／R-entangled 構造上點不出來。
    for row in pairs.get("l0n_provenance", {}).get("per_model", {}).get(tag, []):
        for side in ("l0", "l0n"):
            iid = row[side]
            if iid not in membership:
                membership[iid] = []
                ids.append(iid)
            membership[iid].append({"regime": "L0N_prov", "role": side,
                                    "partner": row["l0n" if side == "l0" else "l0"]})
    for row in pairs.get("l0m_adjudication", {}).get("pairs", []):
        for side in ("l0", "l0m"):
            iid = row[side]
            if iid not in membership:
                membership[iid] = []
                ids.append(iid)
            membership[iid].append({"regime": "L0M", "role": side,
                                    "partner": row["l0m" if side == "l0" else "l0"],
                                    "pair_id": row.get("pair_id")})
    return ids, membership


def make_cfg(args, gen_tokens: int = 0) -> cp.PatchConfig:
    bands = {}
    for spec in args.bands.split(","):
        name, lo, hi = spec.split(":")
        bands[name] = (float(lo), float(hi))
    return cp.PatchConfig(
        bands=bands, hook_kind=args.hook, radius=args.radius,
        quantile=args.quantile, min_survivors=args.min_survivors,
        rank_guard=(None if args.rank_guard <= 0 else args.rank_guard),
        adequacy_multiple=args.adequacy_multiple,
        numerical_floor_nats=args.numerical_floor_nats,
        expand_pool=(not args.no_expand_pool),
        expand_max_candidates=args.expand_max_candidates,
        n_cue=args.n_cue, n_repl_per_cue=args.n_repl, seed=args.seed,
        gen_tokens=gen_tokens, gen_seed=args.seed, gen_sample=False)


def with_model(name: str, dtype_str: str, fn, device: str = "cuda"):
    """一次只載一個模型（8GB 紀律）。載入前重設峰值統計，否則報的是別人的峰值。"""
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda 但看不到 CUDA")
        torch.cuda.reset_peak_memory_stats()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    t0 = time.time()
    model = cp.load_model(name, device, dtype)
    load_s = time.time() - t0
    try:
        return fn(model, str(dtype), load_s)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage: budget

def _measured_spf(args) -> dict:
    """從窗口 A 的 anchors 取逐模型實測秒/前向。

    **同 dtype 才算數**：fp32 與 fp16 的速度不同，拿 fp16 的秒數去投影 fp32
    的跑法會低估。所以只在 anchors 的 dtype 與該模型**實際要跑的** dtype
    相符時採用——比的是逐模型的有效 dtype，不是全域 --dtype：混合配置
    （fp32×4＋2.8b fp16）下拿全域值去比，會把 2.8b 那筆唯一相符的實測擋掉，
    最大的模型反而退回用猜的。
    """
    out = {}
    src = Path(args.spf_source)
    if not src.exists():
        return out
    try:
        d = json.load(open(src, encoding="utf-8"))
    except Exception:
        return out
    for m, fb in (d.get("forward_budget") or {}).items():
        eff = "fp16" if m in getattr(args, "fp16_models_list", set()) else args.dtype
        want = "float32" if eff == "fp32" else "float16"
        run_dtype = str((d.get("runs") or {}).get(m, {}).get("dtype", ""))
        if want not in run_dtype:
            continue
        spf = fb.get("seconds_per_forward")
        if spf:
            out[m] = float(spf)
    return out


def _measured_gen_spf(args) -> dict:
    """從既有的 main gen 分片summary 取逐模型**實測秒/生成步**。

    **為什麼 gen 不能沿用 patch 的秒/前向**（窗口 B 打臉出來的）：gen 關 KV
    cache，每一步重跑整段，而段長隨生成變長——平均一步比一個 patch 前向貴。
    winB1 實測比值 1.21–1.77×（410m 1.56／1b 1.35／1.4b 1.38／2.8b 1.77／
    OLMo 1.21）。用 patch 的秒數投影 gen，會系統性低估。
    """
    out = {}
    src = Path(args.gen_spf_source) if args.gen_spf_source else None
    if not src or not src.exists():
        return out
    try:
        d = json.load(open(src, encoding="utf-8"))
    except Exception:
        return out
    n_bands = len(args.bands.split(","))
    for m, v in (d.get("per_model") or {}).items():
        nd, secs = v.get("n_done"), v.get("seconds")
        if not nd or not secs:
            continue
        steps = nd * args.gen_tokens * (2 + n_bands)
        if steps:
            out[m] = float(secs) / steps
    return out


def stage_budget(args, pairs, bat) -> dict:
    """先報預算再開跑（rig_feasibility §3 的硬前置延伸）。

    **前向數給區間不給單值**（winB1 打臉的結果）：
      下界＝校準掃描只掃 `n_cue×n_repl`；
      上界＝每一題都走 v2.14/v2.15 擴池輪，掃到 `expand_max_candidates`。
    舊版只報下界卻標成「設定上界」——winB1 實際 6954 前向 > 舊版所謂上界 5510
    （+26%），因為擴池輪本來就會超過 n_cue×n_repl。**那個標籤是錯的，已改。**
    實跑後一律以分片檔的 n_forwards 為準。
    """
    n_bands = len(args.bands.split(","))
    n_cal_lo = args.n_cue * args.n_repl
    n_cal_hi = max(n_cal_lo,
                   0 if args.no_expand_pool else args.expand_max_candidates)
    per_item_patch = 1 + n_cal_lo + 1 + n_bands           # 下界
    per_item_patch_hi = 1 + n_cal_hi + 1 + n_bands        # 上界（全部走擴池）
    rows, tot_patch, tot_gen = {}, 0, 0
    for m in args.models:
        tag = m.replace("/", "_")
        ids, _ = items_for_model(pairs, tag)
        have = [i for i in ids if i in bat]
        # gen 段的宿主濾器是 level in ("L0","L0N")（stage_gen），投影就必須數
        # 同一組——只數 L0 會把 v2.18 接進來的 L0N 宿主漏掉，把窗口低估。
        n_gen_hosts = sum(1 for i in have
                          if bat[i].get("level") in GEN_HOST_LEVELS)
        f_patch = len(have) * per_item_patch
        f_patch_hi = len(have) * per_item_patch_hi
        f_gen = n_gen_hosts * args.gen_tokens * (2 + n_bands)
        rows[m] = {"n_items": len(have), "n_missing_from_battery": len(ids) - len(have),
                   "forwards_patch_hi": f_patch_hi,
                   "n_gen_hosts": n_gen_hosts,
                   "n_gen_hosts_by_level": {
                       lv: sum(1 for i in have if bat[i].get("level") == lv)
                       for lv in GEN_HOST_LEVELS},
                   "forwards_patch": f_patch, "forwards_gen": f_gen}
        tot_patch += f_patch
        tot_gen += f_gen
    # 逐模型秒/前向：**優先用實測**（窗口 A anchors 的 forward_budget），
    # 沒有實測才退回 --seconds-per-forward 的假設值。兩者哪一種都標明，
    # 不讓假設的數字冒充量到的。
    measured = _measured_spf(args)
    gen_measured = _measured_gen_spf(args)
    est_patch = est_gen = 0.0
    est_patch_hi = 0.0
    for m, r in rows.items():
        spf = measured.get(m, args.seconds_per_forward)
        # gen 的每步成本另外算（見 _measured_gen_spf）；沒有實測才退回 patch 的
        gspf = gen_measured.get(m, spf)
        r["seconds_per_forward"] = spf
        r["spf_source"] = "measured" if m in measured else "assumed"
        r["seconds_per_gen_step"] = gspf
        r["gen_spf_source"] = ("measured" if m in gen_measured
                               else "fallback=patch_spf（會低估 1.2–1.8 倍）")
        r["minutes_patch"] = r["forwards_patch"] * spf / 60
        r["minutes_patch_hi"] = r["forwards_patch_hi"] * spf / 60
        r["minutes_gen"] = r["forwards_gen"] * gspf / 60
        est_patch += r["forwards_patch"] * spf
        est_patch_hi += r["forwards_patch_hi"] * spf
        est_gen += r["forwards_gen"] * gspf
    return {"per_model": rows,
            "totals": {"forwards_patch": tot_patch,
                       "forwards_patch_hi": sum(r["forwards_patch_hi"]
                                                for r in rows.values()),
                       "forwards_gen": tot_gen},
            "seconds_per_forward_assumed": args.seconds_per_forward,
            "seconds_per_forward_measured": measured,
            "seconds_per_gen_step_measured": gen_measured,
            "dtype": args.dtype,
            "estimate_minutes": {"patch": est_patch / 60,
                                 "patch_hi": est_patch_hi / 60,
                                 "gen": est_gen / 60,
                                 "total": (est_patch + est_gen) / 60,
                                 "total_hi": (est_patch_hi + est_gen) / 60},
            "basis": ("**區間非單值**：patch 下界＝校準只掃 n_cue×n_repl，"
                      "上界＝每題都走擴池輪掃到 expand_max_candidates。"
                      "winB1 實測 6954 前向落在 5510–10070 之間（舊版把下界"
                      "標成『上界』，低估 26%——已更正）。"
                      "gen 用逐模型實測秒/步，沒有實測才退回 patch 秒/前向"
                      "（那會低估 1.2–1.8 倍）。實跑後一律以 n_forwards 為準"),
            "note": ("gen 是大頭（關 KV cache，每步重跑整段）。若窗口不夠，"
                     "**先跑 patch**——G1/校準/配對對比都不吃 gen，只有溯源軌吃。"
                     "／gen 宿主數是上界：patch 段出錯（如 rank 護欄擋掉）的題"
                     "不會進 gen。winB1 投影 95 宿主、實跑 74，所以 gen 投影"
                     "會比實跑高——**這個方向是刻意的**，排窗口寧可估多。")}


# ---------------------------------------------------------------- stage: patch

def stage_patch(args, pairs, bat) -> dict:
    """主量測。逐 (模型, 題) 落分片，可斷點續跑。"""
    summary = {}
    for m in args.models:
        tag = m.replace("/", "_")
        ids, membership = items_for_model(pairs, tag)
        todo_all = [i for i in ids if i in bat]
        missing = [i for i in ids if i not in bat]
        path = shard_path(args.run_id, m, "patch", Path(args.out_dir))
        shard = {} if args.fresh else load_shard(path)
        recs0 = shard.get("records") or {}
        # F1：**基礎設施例外不算做完**，否則一次瞬時 OOM 就讓那題永久出局。
        retry = [i for i, r in recs0.items()
                 if (args.retry_errors and "error" in r) or is_retryable_error(r)]
        done = set(recs0) - set(retry)
        todo = [i for i in todo_all if i not in done]
        if args.limit:
            todo = todo[: args.limit]
        print(f"[patch] {m}: 共 {len(todo_all)} 題，已完成 {len(done)}，本輪跑 {len(todo)}"
              + (f"（其中 {len(retry)} 題是重試：{retry[:4]}）" if retry else "")
              + (f"，battery 缺 {len(missing)} 題 {missing[:4]}" if missing else ""),
              flush=True)
        if not todo:
            summary[m] = {"n_done": len(done), "skipped_all": True}
            continue

        cfg = make_cfg(args, gen_tokens=0)
        shard.setdefault("meta", env_meta())
        shard["model"] = m
        shard["dtype"] = args.dtype
        shard["config"] = {"bands": args.bands, "radius": args.radius,
                           "quantile": args.quantile, "rank_guard": args.rank_guard,
                           "min_survivors": args.min_survivors,
                           "expand_max_candidates": args.expand_max_candidates,
                           "numerical_floor_nats": args.numerical_floor_nats,
                           "seed": args.seed, "dtype": args.dtype}
        shard.setdefault("records", {})
        shard["pair_membership"] = membership
        shard["battery_missing"] = missing

        def _go(model, dtype, load_s):
            n_fwd, t0 = 0, time.time()
            for k, iid in enumerate(todo, 1):
                try:
                    rec = cp.run_item(model, bat[iid], cfg)
                except Exception as e:
                    # error_kind＝exception：機器出包，續跑時要重試（F1）
                    rec = {"id": iid, "error": repr(e), "error_kind": "exception"}
                rec["model"] = m
                rec["level"] = bat[iid].get("level")
                rec["twin_of"] = bat[iid].get("twin_of")
                rec["pairs"] = membership.get(iid)
                rec["backend_dtype"] = args.dtype
                shard["records"][iid] = rec
                n_fwd += rec.get("n_forwards", 0)
                shard["progress"] = {"n_done": len(shard["records"]),
                                     "n_total": len(todo_all),
                                     "n_forwards_this_session": n_fwd,
                                     "seconds_this_session": time.time() - t0,
                                     "load_seconds": load_s, "dtype": dtype}
                save_shard(path, shard)          # ← 每題落盤，斷了就從這裡續
                print(f"  [{k}/{len(todo)}] {iid} ({rec.get('level')})"
                      f"{' ERR ' + rec['error'][:40] if 'error' in rec else ''}", flush=True)
            return n_fwd, time.time() - t0

        dt = "fp16" if m in args.fp16_models_list else args.dtype
        n_fwd, secs = with_model(m, dt, _go, args.device)
        summary[m] = {"n_done": len(shard["records"]), "n_forwards": n_fwd,
                      "seconds": secs, "shard": str(path), "dtype": dt}
        print(f"[patch] {m}: {n_fwd} 次前向 / {secs:.1f}s -> {path}")
    return {"stage": "patch", "per_model": summary, "verdict": None}


# ---------------------------------------------------------------- stage: gen

def stage_gen(args, pairs, bat) -> dict:
    """溯源軌素材：**只對 L0 receiver** 做 clean/corrupted/patched 自由續寫。

    §9 硬約束②：溯源軌不掛 L0P（其溯源出口是構造性的零）。這裡直接濾，
    不靠下游記得濾。介入從 patch 段的分片**重建**（不重跑校準）——
    重跑校準會換介入，續寫就不是同一個介入的續寫了。
    """
    summary = {}
    for m in args.models:
        ppath = shard_path(args.run_id, m, "patch", Path(args.out_dir))
        pshard = load_shard(ppath)
        if not pshard.get("records"):
            print(f"[gen] {m}: 找不到 patch 分片，跳過（先跑 --stage patch）")
            continue
        hosts = [iid for iid, r in pshard["records"].items()
                 if "error" not in r and r.get("level") in GEN_HOST_LEVELS]
        path = shard_path(args.run_id, m, "gen", Path(args.out_dir))
        shard = {} if args.fresh else load_shard(path)
        recs0 = shard.get("records") or {}
        # F1 同理：gen 段的例外（OOM／續寫中斷）也不算做完
        retry_g = [i for i, r in recs0.items()
                   if (args.retry_errors and "error" in r) or is_retryable_error(r)]
        done = set(recs0) - set(retry_g)
        todo = [i for i in hosts if i not in done]
        if args.limit:
            todo = todo[: args.limit]
        print(f"[gen] {m}: 宿主 {len(hosts)} 題"
              f"（已濾非 {'/'.join(GEN_HOST_LEVELS)}），本輪跑 {len(todo)}",
              flush=True)
        if not todo:
            summary[m] = {"n_done": len(done), "skipped_all": True}
            continue
        cfg = make_cfg(args, gen_tokens=args.gen_tokens)
        shard.setdefault("meta", env_meta())
        shard["model"] = m
        shard.setdefault("records", {})
        shard["glink_host_levels"] = list(GEN_HOST_LEVELS)

        def _go(model, dtype, load_s):
            t0 = time.time()
            for k, iid in enumerate(todo, 1):
                try:
                    rec = pshard["records"][iid]
                    tok, clean, cache, ct, corrupted, bands, pos = cp.rebuild_selected(
                        model, bat[iid], cfg, rec)
                    prompt_only = tok.tokens[:, :tok.n_prompt]
                    corr_only = ct[:, :tok.n_prompt]
                    gens = {
                        "clean": cp.free_continuation(model, prompt_only, args.gen_tokens,
                                                      args.seed, False),
                        "corrupted": cp.free_continuation(model, corr_only, args.gen_tokens,
                                                          args.seed, False)}
                    for bname, layers in bands.items():
                        hooks = cp.make_patch_hooks(cache, layers, pos, cfg.hook_kind)
                        gens[f"patched::{bname}"] = cp.free_continuation(
                            model, corr_only, args.gen_tokens, args.seed, False,
                            fwd_hooks=hooks)
                    out = {"id": iid, "level": rec.get("level"),
                           "corruption_key": rec["corruption"]["key"],
                           "generations": gens,
                           "n_words": {k2: len(v.split()) for k2, v in gens.items()}}
                except Exception as e:
                    out = {"id": iid, "error": repr(e), "error_kind": "exception"}
                shard["records"][iid] = out
                shard["progress"] = {"n_done": len(shard["records"]),
                                     "seconds_this_session": time.time() - t0}
                save_shard(path, shard)
                print(f"  [{k}/{len(todo)}] {iid}"
                      f"{' ERR' if 'error' in out else ''}", flush=True)
            return time.time() - t0

        dt = "fp16" if m in args.fp16_models_list else args.dtype
        secs = with_model(m, dt, _go, args.device)
        summary[m] = {"n_done": len(shard["records"]), "seconds": secs, "shard": str(path)}
    return {"stage": "gen", "per_model": summary, "verdict": None,
            "next": "infini-gram 查詢留窗口後 CPU 跑（causal_smoke.py --stage provenance）"}


# ---------------------------------------------------------------- stage: genmerge

def stage_genmerge(args, pairs, bat) -> dict:
    """把逐模型 gen 分片併成 `causal_smoke.stage_provenance` 吃的 **runs 形**。

    **介面契約破裂的實體**（窗口 B 當晚炸出來的）：
      冒煙 gen 產出 = `{"runs": {model: {"records": [rec, ...]}}}`（records 是
      **list**）；causal_main 的 gen 產出 = 逐模型分片
      `{"model": ..., "records": {id: rec}}`（records 是 **dict**）＋一個沒有
      records 的 summary。stage_provenance 直接 `data["runs"]` → KeyError。

    **為什麼修在這邊而不是修 stage_provenance**：`causal_smoke.py` 是產出窗口 A
    凍結數字的那支，解盲前不動它。這裡只做**形狀轉換**——不碰任何數值、
    不重算、不篩選（有 error 的紀錄照樣帶過去，讓下游自己照既有規則濾）。
    """
    out_dir = Path(args.out_dir)

    def _merge(stage: str, extra_keys: tuple[str, ...] = ()) -> tuple[dict, list]:
        runs, missing = {}, []
        for m in args.models:
            path = shard_path(args.run_id, m, stage, out_dir)
            shard = load_shard(path)
            recs = shard.get("records") or {}
            if not recs:
                missing.append({"model": m, "stage": stage, "shard": str(path),
                                "reason": "找不到分片或分片沒有 records"})
                continue
            # dict → list：**保持分片裡的插入順序**，不重排（重排會讓兩次合併的
            # 輸出不一樣，之後對帳會多一個假差異）
            rows = [recs[k] for k in recs]
            block = {"records": rows, "n_records": len(rows),
                     "n_error": sum(1 for r in rows if "error" in r),
                     "shard": str(path)}
            for k in extra_keys:
                block[k] = shard.get(k)
            runs[m] = block
        return runs, missing

    written = {}
    # gen → stage_provenance 的 runs 形；patch → stage_glink 的 anchors 形。
    # **兩個都要**：glink 讀的是 anchors["runs"][model]["records"]（list，
    # 每筆帶 patches[].band 與 patches[].launch.numer），跟 provenance 同一個
    # 形狀契約——溯源段炸過一次之後，下一段會用同一種方式再炸一次，所以
    # 一起補掉，不等它在 06:00 炸給我看。
    for stage, extra in (("gen", ("glink_host_levels",)),
                         ("patch", ("dtype", "config"))):
        runs, missing = _merge(stage, extra)
        if not runs:
            print(f"[genmerge] {stage}: 沒有可併的分片，跳過")
            continue
        merged = {"run_id": args.run_id, "meta": env_meta(),
                  "built_by": f"harness/causal_main.py --stage genmerge（{stage}）",
                  "purpose": ("轉成 causal_smoke 吃的 runs 形（gen→provenance、"
                              "patch→glink 的 --anchors-json）；只轉形狀不動數值"),
                  "runs": runs, "missing": missing}
        mp = out_dir / f"main_{args.run_id}_{stage}_runs.json"
        mp.write_text(json.dumps(_json_safe(merged), ensure_ascii=False, indent=1),
                      encoding="utf-8")
        written[stage] = {"path": str(mp),
                          "per_model": {m: {"n_records": r["n_records"],
                                            "n_error": r["n_error"]}
                                        for m, r in runs.items()},
                          "n_records_total": sum(r["n_records"] for r in runs.values()),
                          "missing": missing}
        print(f"[genmerge] {stage} -> {mp}")
        for m, r in runs.items():
            print(f"    {m}: {r['n_records']} 筆（error {r['n_error']}）")
    gp = written.get("gen", {}).get("path")
    pp = written.get("patch", {}).get("path")
    return {"stage": "genmerge", "written": written,
            "merged_json": gp,          # 舊欄位保留（先前交件引用過）
            "verdict": None,
            "next": {
                "provenance": (f'.venv/Scripts/python.exe harness/causal_smoke.py '
                               f'--stage provenance --run-id {args.run_id} '
                               f'--gen-json {gp}'),
                "glink": (f'.venv/Scripts/python.exe harness/causal_smoke.py '
                          f'--stage glink --run-id {args.run_id} '
                          f'--anchors-json {pp} '
                          f'--provenance-json '
                          f'{out_dir / f"smoke_provenance_{args.run_id}.json"}')}}


# ---------------------------------------------------------------- stage: control

def stage_control(args, pairs, bat) -> dict:
    """H2／J4 的 fp32 對照：**保留題**上 fp16 與 fp32 各一遍。

    量的是「換 dtype 讓 ΔY 與 DiD 動多少」——主實驗跑 fp16，這一段是它的
    防護對照（v2.15：由建議升為必要）。介入從保留題的既有紀錄重建，
    **不重跑校準**（重跑會換介入，量到的就變成效應不是 dtype 偏差）。
    """
    src = Path(args.control_anchors)
    if not src.exists():
        raise SystemExit(f"--control-anchors 不存在：{src}（需要保留題的既有介入紀錄）")
    anch = json.load(open(src, encoding="utf-8"))
    hold = json.load(open(HOLDOUT, encoding="utf-8"))
    hitems = {it["id"]: it for it in (hold["items"] if isinstance(hold, dict) else hold)}
    twins = {it["id"]: it.get("twin_of") for it in hitems.values() if it.get("twin_of")}
    model = args.control_model
    excl = {s.strip() for s in (args.control_exclude_items or "").split(",") if s.strip()}
    recs = {r["id"]: r for r in anch["runs"][model]["records"]
            if "error" not in r and r["id"] not in excl}
    # twin 對只要有一邊被排除就整對不成立（DiD 是對內差）
    twins = {a: b for a, b in twins.items() if a not in excl and b not in excl}
    cfg = cp.PatchConfig(bands={k: tuple(v) for k, v in anch["config"]["bands"].items()},
                         hook_kind=anch["config"]["hook_kind"],
                         radius=anch["config"]["radius"])
    # v2.17：2.8b 的 fp32 只裝得進系統記憶體（11.08GB > 8GB VRAM），fp32 臂只能
    # 在 CPU 重算。fp16 臂不必再算一次——**anchors 裡那筆就是主實驗要用的那條鏈**
    # （同一次介入、同一個 patched_readout 單前向路徑、GPU fp16），直接取用比在
    # CPU 上另跑一條 fp16 kernel 更貼近「生產讀數 vs 真值」這個問題。
    arm_src = {"fp32": "recompute", "fp16": args.control_fp16_source}
    anch_dtype = str(anch["runs"][model].get("dtype", ""))
    if arm_src["fp16"] == "anchors" and "float16" not in anch_dtype:
        raise SystemExit(f"--control-fp16-source anchors 但 {src.name} 的 {model} "
                         f"記的是 {anch_dtype}——不可把非 fp16 的臂標成 fp16")
    dy = {"fp32": {}, "fp16": {}}
    if arm_src["fp16"] == "anchors":
        for iid, rec in recs.items():
            if iid not in hitems:
                continue
            yc = rec["corrupted"]["launch_mean"]
            for p in rec["patches"]:
                dy["fp16"][f'{iid}|{p["band"]}'] = p["patched"]["launch_mean"] - yc
    for dt in [d for d in ("fp32", "fp16") if arm_src[d] == "recompute"]:
        def _go(mo, dtype, load_s):
            for iid, rec in recs.items():
                if iid not in hitems:
                    continue
                try:
                    tok, clean, cache, ct, corrupted, bands, pos = cp.rebuild_selected(
                        mo, hitems[iid], cfg, rec)
                except Exception as e:
                    print(f"  {dt} {iid}: rebuild 失敗 {e!r}"); continue
                for bname, layers in bands.items():
                    pr = cp.patched_readout(mo, ct, tok.n_prompt, cache, layers, pos,
                                            cfg.hook_kind, cfg.launch_k)
                    dy[dt][f"{iid}|{bname}"] = pr.launch_mean - corrupted.launch_mean
            return None
        with_model(model, dt, _go, args.device)

    keys = sorted(set(dy["fp32"]) & set(dy["fp16"]))
    cells = [{"key": k, "dy_fp32": dy["fp32"][k], "dy_fp16": dy["fp16"][k],
              "abs_diff": abs(dy["fp32"][k] - dy["fp16"][k])} for k in keys]
    did = []
    bands_seen = sorted({k.split("|")[1] for k in keys})
    for l0m, l0 in twins.items():
        for b in bands_seen:
            a32, b32 = dy["fp32"].get(f"{l0m}|{b}"), dy["fp32"].get(f"{l0}|{b}")
            a16, b16 = dy["fp16"].get(f"{l0m}|{b}"), dy["fp16"].get(f"{l0}|{b}")
            if None in (a32, b32, a16, b16):
                continue
            did.append({"l0m": l0m, "l0": l0, "band": b,
                        "did_fp32": a32 - b32, "did_fp16": a16 - b16,
                        "abs_diff": abs((a32 - b32) - (a16 - b16))})
    med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    return {"stage": "control", "model": model, "anchors": str(src),
            "arms": {
                "fp32": {"source": arm_src["fp32"], "device": args.device,
                         "dtype": "fp32"},
                "fp16": ({"source": "anchors", "file": str(src),
                          "device": anch["runs"][model].get("device"),
                          "dtype": anch_dtype,
                          "note": "＝主實驗要用的那條鏈（GPU fp16），非 CPU 代打"}
                         if arm_src["fp16"] == "anchors"
                         else {"source": "recompute", "device": args.device,
                               "dtype": "fp16"})},
            "n_cells": len(cells), "n_did_pairs": len(did),
            "cells": cells, "did": did,
            "cell_abs_diff": {"max": max((c["abs_diff"] for c in cells), default=None),
                              "median": med([c["abs_diff"] for c in cells])},
            "did_abs_diff": {"max": max((d["abs_diff"] for d in did), default=None),
                             "median": med([d["abs_diff"] for d in did])},
            "verdict": None,
            "note": ("門檻要不要從推的 0.236 改成量的，是理論部/董事會的裁決；"
                     "這裡只給實測分布。")}


# ---------------------------------------------------------------- main

def build_argparser():
    ap = argparse.ArgumentParser(description="OMTR 因果階段主實驗驅動器（窗口 B）")
    ap.add_argument("--stage", required=True,
                    choices=["budget", "patch", "gen", "genmerge", "control", "all"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--pairs", default=None,
                    help="配對表路徑（預設 adjudication_pairs.json）。v2.18 的"
                         "含 L0N 版另存一檔，點火令下來再決定用哪一份")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                    help="主實驗＝fp16（H2；保留題 fp32 對照另由 --stage control 做）")
    ap.add_argument("--fp16-models", default="",
                    help="強制 fp16 的模型（逗號分隔）")
    ap.add_argument("--fresh", action="store_true", help="忽略既有分片，從頭跑")
    ap.add_argument("--retry-errors", action="store_true",
                    help="續跑時**所有** error 紀錄都重跑（含規則判定的）。"
                         "預設只重跑基礎設施例外（OOM 那類）——規則判定是"
                         "決定性的，重跑一次還是同一個答案，白花前向")
    ap.add_argument("--limit", type=int, default=0,
                    help="每個模型最多跑幾題（0=全部）。**冒煙測試用**。"
                         "⚠ 不要拿它切窗口：它按配對表順序截斷，而表尾正好是"
                         "L0N-only 與 L0M-only 的題，截了會留下「L0 側量了、"
                         "對側沒量」的半套配對——續跑補得回來，但當晚那批資料"
                         "不能分析。要切窗口請切 --models（逐模型跑完整的配對）")
    ap.add_argument("--bands", default="early:0.25:0.45,mid:0.45:0.65,late:0.70:0.90")
    ap.add_argument("--hook", default="resid_pre", choices=["resid_pre", "resid_post"])
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--quantile", type=float, default=cp.CORRUPT_QUANTILE)
    ap.add_argument("--min-survivors", type=int, default=cp.MIN_SURVIVORS)
    ap.add_argument("--rank-guard", type=int, default=cp.RANK_GUARD)
    ap.add_argument("--adequacy-multiple", type=float, default=cp.ADEQUACY_MULTIPLE)
    ap.add_argument("--numerical-floor-nats", type=float, default=None)
    ap.add_argument("--no-expand-pool", action="store_true")
    ap.add_argument("--expand-max-candidates", type=int, default=48)
    ap.add_argument("--n-cue", type=int, default=6)
    ap.add_argument("--n-repl", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--spf-source",
                    default=str(OUT_DIR / "smoke_anchors_winA2.json"),
                    help="逐模型實測秒/前向的來源（只採 dtype 相符的）")
    ap.add_argument("--seconds-per-forward", type=float, default=0.075,
                    help="budget 段用的假設值（窗口 A 實測 0.053–0.121）")
    ap.add_argument("--gen-spf-source",
                    default=str(OUT_DIR / "main_winB1_gen_summary.json"),
                    help="逐模型實測**秒/生成步**的來源（gen 關 KV cache、每步"
                         "重跑整段，比 patch 前向貴 1.2–1.8 倍，不能沿用）")
    ap.add_argument("--control-model", default="EleutherAI/pythia-1b")
    ap.add_argument("--control-anchors",
                    default=str(OUT_DIR / "smoke_anchors_winA2.json"))
    ap.add_argument("--control-exclude-items", default="",
                    help="閘量測要排除的題（逗號分隔）。2.8b 的 CPU fp32 閘協定"
                         "明文排除 **L0M-11**（慢性三題之一，凍結保留集已排除）"
                         "——寫成旗標而不是事後在分析裡濾，協定才可重跑")
    ap.add_argument("--control-fp16-source", default="recompute",
                    choices=["recompute", "anchors"],
                    help="fp16 臂從哪來。recompute＝兩臂都在 --device 上重算"
                         "（1b/1.4b 的 J4 就是這樣跑的，預設不動）；anchors＝直接"
                         "取 --control-anchors 裡既有的 fp16 讀數（2.8b 用：fp32 臂"
                         "只裝得進 CPU，而 anchors 那筆正是主實驗的 GPU fp16 鏈）")
    return ap


def main():
    args = build_argparser().parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.fp16_models_list = {m.strip() for m in args.fp16_models.split(",") if m.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.radius > cp.CUE_RADIUS_MAX:
        raise SystemExit(f"--radius {args.radius} 超過 R4 上限 {cp.CUE_RADIUS_MAX}")

    print(f"=== OMTR 主實驗 | stage={args.stage} run_id={args.run_id}")
    print(f"    spec: {cp.SPEC_VERSION}")
    print(f"    device={args.device} dtype={args.dtype} models={len(args.models)}")
    pairs = load_pairs(args.pairs)
    bat = battery_index()
    print(f"    配對表: {Path(args.pairs or PAIRS_JSON).name}"
          f"（對帳 {pairs.get('crosscheck')}）")
    ln = pairs.get("l0n_provenance")
    print(f"    溯源軌 L0N：{ln['status'] if ln else '本表沒有這一段（gen 的 L0N 濾器會是死碼）'}")
    print(f"    battery 索引 {len(bat)} 題")

    t0 = time.time()
    stages = ["patch", "gen"] if args.stage == "all" else [args.stage]
    written = {}
    for st in stages:
        fn = {"budget": stage_budget, "patch": stage_patch,
              "gen": stage_gen, "genmerge": stage_genmerge,
              "control": stage_control}[st]
        payload = fn(args, pairs, bat)
        payload = {"run_id": args.run_id, "meta": env_meta(),
                   "rerun": " ".join([".venv/Scripts/python.exe",
                                      "harness/causal_main.py"] + sys.argv[1:]),
                   **payload}
        p = out_dir / f"main_{args.run_id}_{st}_summary.json"
        p.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        written[st] = str(p)
        print(f"[{st}] -> {p}")
        if st == "budget":
            e = payload["estimate_minutes"]
            n_meas = len(payload["seconds_per_forward_measured"])
            n_gmeas = len(payload["seconds_per_gen_step_measured"])
            print(f"    投影：patch {e['patch']:.1f}–{e['patch_hi']:.1f} 分"
                  f"（下界＝不擴池／上界＝全擴池）、gen {e['gen']:.1f} 分、"
                  f"合計 {e['total']:.1f}–{e['total_hi']:.1f} 分")
            print(f"    秒/前向：{n_meas}/{len(payload['per_model'])} 個模型用實測；"
                  f"秒/生成步：{n_gmeas}/{len(payload['per_model'])} 個用實測"
                  f"（其餘退回假設 {args.seconds_per_forward}s）")
    print(f"=== done in {time.time()-t0:.1f}s")
    for st, p in written.items():
        print(f"    {st}: {p}")


if __name__ == "__main__":
    main()
