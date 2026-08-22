"""GPU 窗口 A 冒煙驅動器（器材部）。

依據規格版本：**CAUSAL_PREREG_v1 第一段凍結（2026-08-21）**，§8 冒煙協定。

只碰**保留題**（§9），只量、不裁決。要量的東西：

  五錨   ① σ_DiD（起手窗尺度）  ② rho_fam  ③ 腐蝕強度校準曲線
         ④ 起手窗效應錨          ⑤ G-link 兩軌相關 r
  兩檢查 G1-gold 散點（§7.1 不可識別區）、L0M 12 對試做（16 前向/對預算）

分段（每段自己落一份 JSON，可以分開跑、分開重跑）：

  timing      單次前向真實秒數（rig_feasibility §3 列為硬前置）      〔GPU〕
  anchors     逐保留題 clean→校準→腐蝕→patch→R；①②③④＋G1 散點      〔GPU〕
  l0m         L0M 雙胞胎試做（只量 clean 的整段/起手窗 gold）         〔GPU〕
  gen         溯源軌素材：clean/corrupted/patched 的自由續寫          〔GPU〕
  provenance  把 gen 的續寫丟 infini-gram 算 frac_present            〔CPU＋網路〕
  glink       anchors × provenance → ⑤ r                            〔CPU〕
  all         ＝ timing + anchors + l0m + gen（GPU 那四段）

用法（GPU 窗口）：
    .venv/Scripts/python.exe harness/causal_smoke.py --stage all \
        --holdout planning/battery_expansion/holdout_12.json \
        --models EleutherAI/pythia-410m,EleutherAI/pythia-1b,EleutherAI/pythia-1.4b,\
EleutherAI/pythia-2.8b,allenai/OLMo-2-0425-1B

保留題檔還沒齊備時（現況）：可直接注入 id 清單，從既有 battery 撈——
    .venv/Scripts/python.exe harness/causal_smoke.py --stage anchors \
        --holdout-ids L0-01,L0-02,L0P-01 --allow-nonconforming-holdout

⛔ 這支不做裁決：不跑置換、不跑 TOST、不判 L0M 過不過閘、不判 R-specific。
   L0M 那一段只吐「規則要的兩個數字 ＋ 凍結門檻」並排，過不過由分析部判。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import causal_patch as cp  # noqa: E402

for _s in (sys.stdout, sys.stderr):      # Windows cp950：別讓 print 殺掉量測
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT_DIR = PROJ / "results" / "causal"
PYTHIA_FAMILY = ("EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
                 "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b")
DEFAULT_MODELS = PYTHIA_FAMILY + ("allenai/OLMo-2-0425-1B",)

# §9 保留集構成（凍結）：新 L0 5 ＋ L0M 4（必為 L0 保留題的雙胞胎）＋ L0P 3
HOLDOUT_SPEC = {"L0": 5, "L0M": 4, "L0P": 3}
# §9 硬約束②＋N4.4：G-link 宿主只能是 L0 或 L0N，**不得掛 L0P**（其溯源出口
# 是構造性的零：pythia-1.4b 16/18、OLMo 18/18）
GLINK_HOST_LEVELS = ("L0", "L0N")
# §3 L0M 通過規則的兩個凍結門檻（只並排列出，不在這裡判）
L0M_THRESHOLDS = {"median_abs_delta_gold_segment": 0.5, "median_abs_delta_gold_launch": 1.0}
L0M_FORWARD_BUDGET_PER_PAIR = 16     # §8「16 前向/對」


# ---------------------------------------------------------------- 共用

def rerun_cmd(extra: str = "") -> str:
    exe = ".venv/Scripts/python.exe"
    args = " ".join(a if " " not in a else f'"{a}"' for a in sys.argv[1:])
    return f"{exe} harness/causal_smoke.py {args} {extra}".strip()


def env_meta() -> dict:
    import importlib.metadata as md
    import transformers
    return {
        "spec_version": cp.SPEC_VERSION,
        "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformer_lens": md.version("transformer-lens"),
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _json_safe(o):
    """NaN/Inf → null。

    Python 的 json.dump 預設會寫出裸的 `NaN`，那**不是合法 JSON**：分析部
    （還有任何非 Python 的讀者）拿去 parse 會直接炸。R 在分母≈0 時本來就是
    NaN，所以這條路是常態不是例外。numpy 純量也一併轉成 Python 原生型別。
    """
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.floating, np.integer)):
        return _json_safe(o.item())
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def dump(stage: str, run_id: str, payload: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"smoke_{stage}_{run_id}.json"
    payload = {"stage": stage, "run_id": run_id, "meta": env_meta(),
               "rerun": rerun_cmd(), **payload}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[{stage}] -> {path}")
    return path


def finite(xs) -> list[float]:
    """濾掉 None / NaN / Inf。中位數之類的聚合一律先過這一關——
    np.median 碰到一個 NaN 會讓整個中位數變 NaN，靜默地毀掉一個錨。"""
    return [float(x) for x in xs
            if x is not None and isinstance(x, (int, float, np.floating, np.integer))
            and np.isfinite(x)]


def _median(xs):
    v = finite(xs)
    return float(np.median(v)) if v else None


def sd(xs) -> float | None:
    """樣本標準差（ddof=1）。n<2 回 None——不用 0 冒充「沒有變異」。"""
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else None


def pearson(a, b) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- 保留題

def load_holdout(args) -> tuple[list[dict], dict]:
    """讀保留題。兩條路：檔案（正規）或注入 id 清單（保留集尚未齊備時的暫用路）。

    §9 的構成與硬約束在這裡驗；不合就擋下，除非顯式帶
    --allow-nonconforming-holdout（該旗標會寫進輸出 JSON，跑過就賴不掉）。
    """
    prov: dict = {"source": None, "allow_nonconforming": args.allow_nonconforming_holdout}
    if args.holdout:
        path = Path(args.holdout)
        data = json.load(open(path, encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        prov.update({"source": str(path), "holdout_version": (
            data.get("version") if isinstance(data, dict) else None)})
    elif args.holdout_ids:
        want = [s.strip() for s in args.holdout_ids.split(",") if s.strip()]
        pool: dict[str, dict] = {}
        for p in [Path(args.battery), Path(args.battery_l0p)]:
            if p.exists():
                for it in json.load(open(p, encoding="utf-8")).get("items", []):
                    pool[it["id"]] = it
        missing = [w for w in want if w not in pool]
        if missing:
            raise SystemExit(f"這些 id 在 battery 裡找不到：{missing}")
        items = [pool[w] for w in want]
        prov.update({"source": "injected-ids", "ids": want,
                     "batteries": [str(args.battery), str(args.battery_l0p)]})
    else:
        raise SystemExit(
            "要嘛給 --holdout <保留題 json>，要嘛給 --holdout-ids <逗號分隔 id>。\n"
            "  保留集檔案還沒造好時走後者；正式冒煙前必須換成 --holdout（§9 保留集凍結）。")

    counts: dict[str, int] = {}
    for it in items:
        counts[it.get("level", "?")] = counts.get(it.get("level", "?"), 0) + 1
    prov["level_counts"] = counts

    problems = []
    for lv, n in HOLDOUT_SPEC.items():
        if counts.get(lv, 0) != n:
            problems.append(f"{lv} 應為 {n} 題，實際 {counts.get(lv, 0)} 題")
    l0_ids = {it["id"] for it in items if it.get("level") == "L0"}
    for it in items:
        if it.get("level") == "L0M":
            twin = it.get("twin_of")
            if not twin:
                problems.append(f"{it['id']} 沒有 twin_of（§9 硬約束①：L0M 保留題必須是 "
                                "L0 保留題的雙胞胎，ρ 的唯一來源）")
            elif twin not in l0_ids:
                problems.append(f"{it['id']} 的 twin_of={twin} 不在保留集的 L0 題裡")
    # 小項（總審查）：保留題 id 不得與**裁決池** battery id 重疊。
    # §9「保留題造好即凍結、永不進主裁決」——重疊代表同一題兩邊都在，
    # 那會讓冒煙看到的東西進主裁決，是偷看。
    pool_ids = set()
    for bp in (Path(args.battery), Path(args.battery_l0p)):
        if bp.exists():
            try:
                for it in json.load(open(bp, encoding="utf-8")).get("items", []):
                    pool_ids.add(it.get("id"))
            except Exception as e:
                problems.append(f"讀不到裁決池 {bp.name} 做互斥檢查：{e!r}")
    overlap = sorted({it["id"] for it in items} & pool_ids)
    prov["adjudication_pool_overlap"] = overlap
    if overlap and prov.get("source") != "injected-ids":
        problems.append(
            f"保留題與裁決池 battery id 重疊 {overlap}——§9 要求保留題永不進主裁決，"
            "重疊等於偷看")
    prov["composition_problems"] = problems
    if problems and not args.allow_nonconforming_holdout:
        raise SystemExit("保留集不符 §9：\n  " + "\n  ".join(problems) +
                         "\n（真的要用不合規的清單試跑，加 --allow-nonconforming-holdout；"
                         "該旗標會寫進輸出 JSON）")
    if problems:
        print("[warn] 保留集不符 §9，但已顯式放行：\n  " + "\n  ".join(problems))
    return items, prov


def make_cfg(args, gen_tokens: int = 0) -> cp.PatchConfig:
    bands = {}
    for spec in args.bands.split(","):
        name, lo, hi = spec.split(":")
        bands[name] = (float(lo), float(hi))
    return cp.PatchConfig(
        bands=bands, hook_kind=args.hook, radius=args.radius,
        expand_pool=(not args.no_expand_pool),
        expand_max_candidates=args.expand_max_candidates,
        quantile=args.quantile, min_survivors=args.min_survivors,
        rank_guard=(None if args.rank_guard <= 0 else args.rank_guard),
        adequacy_multiple=args.adequacy_multiple,
        numerical_floor_nats=args.numerical_floor_nats,
        denom_eps=args.denom_eps, seed=args.seed,
        n_cue=args.n_cue, n_repl_per_cue=args.n_repl,
        gen_tokens=gen_tokens, gen_seed=args.seed, gen_sample=args.gen_sample)


def _dtype_for(name: str, args) -> str:
    """該模型用哪個後端跑（附錄 H①）。

    G1 鏈跑 fp32；**2.8b 從 G1/底線排除**（fp32 需 11.08 GB，8 GB 裝不下），
    其餘量照 fp16 跑並逐格記 backend。硬條件：**該模型的 G1 資料與底線
    必須同後端**——跨後端的 G1 讀數會讓一致決混入異質格。
    """
    forced = {m.strip() for m in (args.fp16_models or "").split(",") if m.strip()}
    if name in forced:
        return "fp16"
    return args.dtype if args.dtype != "auto" else "fp16"


def with_model(name: str, dtype_str: str, fn, device_arg: str = "auto"):
    """一次只載一個模型（8GB 紀律），用完就放掉。

    device 預設 auto（有卡就用卡）。**CPU 流程測試一定要顯式 --device cpu**：
    `auto` 在這台機器上會直接吃 GPU，「不開 GPU」不能靠預設值來保證。
    """
    device = device_arg if device_arg != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda 但這台機器看不到 CUDA")
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[
        dtype_str if dtype_str != "auto" else ("fp16" if device == "cuda" else "fp32")]
    if device == "cuda":
        # 不重設的話 max_memory_allocated 會是**跨模型累積**的峰值，
        # 於是第二個模型報出來的是第一個模型的數字（總審查小項）。
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = cp.load_model(name, device, dtype)
    load_s = time.time() - t0
    try:
        return fn(model, device, str(dtype), load_s)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage: timing

def stage_timing(args, items, prov) -> dict:
    """單次前向真實秒數。rig_feasibility §3：樂觀 0.15s 與保守 1.0s 差 7 倍，
    這個差距要在花使用者顯卡時間之前先收窄，收窄的辦法就是這一段。"""
    probe = items[0]
    out = {"holdout": prov, "probe_item": probe["id"], "models": {}}
    for name in args.models:
        def _go(model, device, dtype, load_s):
            prompt, gold = cp.item_prompt_gold(probe)
            tok = cp.tokenize_pair(model, prompt, gold)
            t = cp.time_forward(model, tok, n=args.timing_n, warmup=args.timing_warmup)
            t.update({"load_seconds": load_s, "n_layers": int(model.cfg.n_layers)})
            if torch.cuda.is_available():
                t["cuda_mem_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
            return t
        print(f"[timing] {name} ...", flush=True)
        out["models"][name] = with_model(name, _dtype_for(name, args), _go, args.device)
        print(f"[timing] {name}: {out['models'][name]['seconds_per_forward']*1000:.1f} ms/forward")
    # ── I 案：投影 ＝ Σ(各段前向數 × 實測 s/f) ＋ 載入時間 ──
    # 前向數用**每題實測的候選數**估不了（還沒跑），所以用設定上界估，
    # 並在 anchors 跑完後由 forward_budget 的實際值取代。標明是哪一種。
    n_items = len(items)
    n_bands = len(args.bands.split(","))
    per_item_cal = args.n_cue * args.n_repl              # 校準掃描上界
    per_item = 1 + per_item_cal + 1 + n_bands            # clean+校準+腐蝕+逐帶
    cw_extra = (args.cross_word_items * (1 + (args.cross_word_k - 1) * (1 + n_bands))
                if args.cross_word_k > 1 else 0)
    est = {}
    for name, t in out["models"].items():
        spf, load_s = t["seconds_per_forward"], t["load_seconds"]
        anchors_f = n_items * per_item + cw_extra
        l0m_f = 2 * n_items                              # 每對 2 題，上界
        gen_steps = args.gen_tokens * (2 + n_bands) * n_items
        est[name] = {
            "anchors": anchors_f * spf + load_s,
            "l0m": l0m_f * spf + load_s,
            "gen": gen_steps * spf + load_s,
            "numfloor": (2 * (1 + 1 + 2 * n_bands) * args.numfloor_items) * spf + 2 * load_s,
        }
    total_wo_gen = sum(v["anchors"] + v["l0m"] + v["numfloor"] for v in est.values())
    total_gen = sum(v["gen"] for v in est.values())
    budget_s = args.window_minutes * 60
    defer = (total_wo_gen + total_gen) > budget_s
    out["projection"] = {
        "per_model_seconds": est,
        "total_seconds_without_gen": total_wo_gen,
        "total_seconds_gen": total_gen,
        "total_seconds": total_wo_gen + total_gen,
        "window_budget_seconds": budget_s,
        "forward_counts_basis": "設定上界（n_cue×n_repl 等），anchors 跑完後以實測取代",
        "defer_gen_to_next_window": bool(defer),
        "defer_reason": ("投影超出窗口預算 → **gen 段自動移到下一窗**"
                         "（gen 不是判定必要：G1/校準/r/L0M 都不吃它，"
                         "它只餵溯源軌 G-link）" if defer else None),
    }
    print(f"[timing] 投影：不含 gen {total_wo_gen/60:.1f} 分、gen {total_gen/60:.1f} 分、"
          f"合計 {(total_wo_gen+total_gen)/60:.1f} 分 / 預算 {args.window_minutes} 分"
          + ("　→ **gen 自動延後**" if defer else ""))
    out["budget_note"] = ("seconds_per_forward × 該段的 n_forwards ＝ 預估秒數；"
                          "anchors/l0m/gen 跑完後 forward_budget 有實測值，"
                          "屆時以實測為準")
    return out


# ---------------------------------------------------------------- stage: anchors

def stage_anchors(args, items, prov) -> dict:
    """逐保留題跑完整 patch 流程，落下 ①②③④ 與 G1 散點的原始資料。"""
    runs: dict[str, list[dict]] = {}
    cfg0 = make_cfg(args, gen_tokens=0)
    # v2.6 §3：跨替換詞量測改當「選 k」用（4 題 × 3 個進帶替換詞），
    # 不再冒充噪音——換詞＝換介入，那是效應不是噪音。
    kset = {it["id"] for it in items[: args.cross_word_items]} if args.cross_word_k > 1 else set()
    for name in args.models:
        def _go(model, device, dtype, load_s):
            recs = []
            for i, it in enumerate(items):
                print(f"  [{i+1}/{len(items)}] {it['id']} ({it.get('level')})", flush=True)
                cfg = make_cfg(args, gen_tokens=0)
                try:
                    rec = cp.run_item(model, it, cfg)
                except Exception as e:
                    rec = {"id": it["id"], "level": it.get("level"), "error": repr(e)}
                if it["id"] in kset and "error" not in rec:
                    rec["cross_word"] = _cross_word(model, it, cfg, rec, args.cross_word_k)
                    rec["n_forwards"] = rec.get("n_forwards", 0) + 1 + sum(
                        1 + len(r_["patches"]) for r_ in rec["cross_word"])
                rec["model"] = name
                # ⑤ 配對傳播（附錄 H）：holdout 檔本來就有 twin_of，
                # 只是量測輸出沒帶出去，害下游要用「120 種指派」去猜配對。
                rec["twin_of"] = it.get("twin_of")
                rec["backend_dtype"] = _dtype_for(name, args)
                recs.append(rec)
            return {"device": device, "dtype": dtype, "load_seconds": load_s,
                    "n_layers": int(model.cfg.n_layers), "records": recs}
        print(f"[anchors] {name} ...", flush=True)
        runs[name] = with_model(name, _dtype_for(name, args), _go, args.device)
    budget = {}
    for name, run in runs.items():
        f = sum(r.get("n_forwards", 0) for r in run["records"])
        s = sum(r.get("seconds", 0.0) for r in run["records"])
        budget[name] = {"n_forwards": f, "seconds": s,
                        "seconds_per_forward": (s / f) if f else None,
                        "n_items_ok": sum(1 for r in run["records"] if "error" not in r),
                        "n_items_error": sum(1 for r in run["records"] if "error" in r)}
        print(f"[anchors] {name}: {f} 次前向 / {s:.1f}s "
              f"({(s/f*1000 if f else 0):.0f} ms per forward)")
    anchors = summarize_anchors(runs, items)
    return {"holdout": prov, "config": _cfg_json(cfg0, args), "runs": runs,
            "forward_budget": budget,
            "anchors": anchors,
            "single_token_strength_gate": single_token_gate(runs, items, args),
            "g1_scatter": g1_scatter(runs),
            "g1_existence_inputs": g1_existence_inputs(
                runs, g1_backend=("fp32" if args.dtype == "fp32" else args.dtype),
                excluded_models=[m.strip() for m in (args.fp16_models or "").split(",")
                                 if m.strip() and m in runs]),
            # ── 下游契約（harness/causal_smoke_eval.py 直接讀這兩個 top-level 鍵）──
            # 那支是凍結分析腳本，寫在任何 GPU 資料之前；它讀不到就會靜默拿 None，
            # 所以鍵名與欄位名以它為準，不由我這邊定。
            "calibration": calibration_contract(runs),
            "replicate_points": replicate_points_contract(runs),
            "positive_control": positive_control_contract(runs)}


def positive_control_contract(runs) -> dict:
    """陽性對照（M6 ＋ PF-5）：**同層帶、同位置帶**的 patch 在保留題上要達地板。

    M6 的陽性對照不是另一種介入——它**就是主 patch 本身**（同層帶同位置帶），
    只是拿它問一個不同的問題：「這台儀器動得了東西嗎？」動不了就是 R-noeffect，
    量測無效、不對世界下任何判斷。**第 0 層還原式必過的版本明文禁用**，
    而本管線在 `layer_band`／`make_patch_hooks` 兩處都擋掉了它，所以這裡的
    陽性對照是**會失敗的**版本——這正是它有資格當對照的原因。

    判定器（PF-5）讀 `delta_y_pc` 與 `delta_s_pc`，門檻
    `ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)`。**判定不在這裡**，我只給兩個數字：
    取各格 ΔY 的**中位數**與對應 Δs 的中位數（中位而非最大，避免用最好的一格
    充當儀器整體能力）。逐格明細一併附上。
    """
    cells = []
    for model, run in runs.items():
        for r in run["records"]:
            if "error" in r:
                continue
            ds = (r.get("calibration") or {}).get("selected_drop_launch")
            for p_ in r.get("patches", []):
                dy = p_["g1_point"].get("delta_y")
                if dy is None or ds is None:
                    continue
                cells.append({"model": model, "id": r["id"], "level": r.get("level"),
                              "band": p_["band"], "delta_y": dy, "delta_s": ds,
                              "band_layers": p_["band_layers"],
                              "positions": p_["positions"]})
    dy_med = _median([c["delta_y"] for c in cells])
    ds_med = _median([c["delta_s"] for c in cells])
    per_band = {}
    for b in sorted({c["band"] for c in cells}):
        sel = [c for c in cells if c["band"] == b]
        per_band[b] = {"n": len(sel),
                       "delta_y_median": _median([c["delta_y"] for c in sel]),
                       "delta_s_median": _median([c["delta_s"] for c in sel])}
    return {"delta_y_pc": dy_med, "delta_s_pc": ds_med,
            "n_cells": len(cells), "per_band": per_band, "cells": cells,
            "definition": ("＝主 patch 本身（同層帶同位置帶，M6）；"
                           "第 0 層還原式必過版本在 layer_band/make_patch_hooks "
                           "兩處硬擋，所以這個對照是**會失敗的**版本"),
            "aggregation": "各格 ΔY／Δs 的中位數（不用最大值，避免最好的一格充當整體）",
            "verdict": None,
            "note": "門檻 ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)（PF-5）由判定器套用"}


def calibration_contract(runs) -> dict:
    """causal_smoke_eval.eval_calibration 讀的格式：
    {model: [{id, n_survivors, selected_drop, selected_quantile}, ...]}"""
    out = {}
    for model, run in runs.items():
        rows = []
        for r in run["records"]:
            cal = r.get("calibration") or {}
            rows.append({"id": r["id"],
                         "n_survivors": cal.get("n_survivors"),
                         "selected_drop": cal.get("selected_drop_launch"),
                         "selected_quantile": cal.get("selected_quantile"),
                         "pool_too_small": cal.get("pool_too_small"),
                         "pool_too_small_final": cal.get("pool_too_small_final"),
                         "expansion": cal.get("expansion"),
                         "meets_adequacy": cal.get("meets_adequacy"),
                         "adequacy_checked": cal.get("adequacy_checked"),
                         "fallback": cal.get("fallback"),
                         "error": r.get("error")})
        out[model] = rows
    return out


def replicate_points_contract(runs) -> dict:
    """r 量測的實現 dof。**逐模型算，不併池**（v2.13 G 案：dof 逐回歸）。

    ⚠ **修正（本輪自抓）**：先前版本把五個模型的 kᵢ 全丟進一個 Σ(kᵢ−1)，
    報出 dof=51——那正是 N4.2 警告的「五模型 × 12 題 = 60 個觀測」錯誤。
    **跨 run 併池不為 σ 加自由度**（同一批 b[i]，五個模型看到的是同一批題）。
    逐模型的真實 dof 是 6–12，差了一個量級。判定器一直是逐模型算的，是我這邊
    的併池數字錯；現在改成**逐模型為準**，併池值保留但明標不是決策量。

    dof 公式依實際擬合式（v2.13 PF-1）：二次式＋逐題截距 → **Σ(kᵢ−1) − 2**。
    """
    per_model = {}
    for model, run in runs.items():
        ks = []
        for r in run["records"]:
            if "error" in r:
                continue
            k = 1 + len(r.get("cross_word") or [])
            if k >= 2:
                ks.append(k)
        m = len(ks)
        sum_km1 = sum(k - 1 for k in ks)
        dof = (sum_km1 - 2) if m else None
        per_model[model] = {
            "n_items_with_k_ge_2": m,
            "mean_k": (float(np.mean(ks)) if ks else None),
            "sum_k_minus_1": sum_km1, "k_per_item": ks,
            "realized_dof": dof,
            "meets_target": (None if dof is None else bool(dof >= 8)),
        }
    short = sorted(mm for mm, v in per_model.items()
                   if v["meets_target"] is False)
    all_ks = [k for v in per_model.values() for k in v["k_per_item"]]
    return {
        "per_model": per_model,
        "decision_unit": "**逐模型**（v2.13 G 案：dof 逐回歸；跨 run 併池不加自由度，N4.2）",
        "target_dof": 8,
        "all_met": (not short),
        "models_below_target": short,
        "if_not_met": ("對未達標之模型補題到達標、記錄補幾題、**不得改門檻**"
                       "（附錄 C③）"),
        "realized_dof_formula": "Σ(kᵢ−1)−2（二次式＋逐題截距，v2.13 PF-1）",
        # 下游相容欄位：判定器目前讀這兩個做逐模型反推，保留但**不是決策量**
        "n_items_with_k_ge_2": len(all_ks),
        "mean_k": (float(np.mean(all_ks)) if all_ks else None),
        "pooled_note": ("上面兩個併池欄位**不是決策量**——併池會把 dof 從 6–12 "
                        "膨脹到 51。決策一律看 per_model。"),
    }


def positive_control_contract(runs) -> dict:
    """陽性對照（M6 ＋ PF-5）：**同層帶、同位置帶**的 patch 在保留題上要達地板。

    M6 的陽性對照不是另一種介入——它**就是主 patch 本身**（同層帶同位置帶），
    只是拿它問一個不同的問題：「這台儀器動得了東西嗎？」動不了就是 R-noeffect，
    量測無效、不對世界下任何判斷。**第 0 層還原式必過的版本明文禁用**，
    而本管線在 `layer_band`／`make_patch_hooks` 兩處都擋掉了它，所以這裡的
    陽性對照是**會失敗的**版本——這正是它有資格當對照的原因。

    判定器（PF-5）讀 `delta_y_pc` 與 `delta_s_pc`，門檻
    `ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)`。**判定不在這裡**，我只給兩個數字：
    取各格 ΔY 的**中位數**與對應 Δs 的中位數（中位而非最大，避免用最好的一格
    充當儀器整體能力）。逐格明細一併附上。
    """
    cells = []
    for model, run in runs.items():
        for r in run["records"]:
            if "error" in r:
                continue
            ds = (r.get("calibration") or {}).get("selected_drop_launch")
            for p_ in r.get("patches", []):
                dy = p_["g1_point"].get("delta_y")
                if dy is None or ds is None:
                    continue
                cells.append({"model": model, "id": r["id"], "level": r.get("level"),
                              "band": p_["band"], "delta_y": dy, "delta_s": ds,
                              "band_layers": p_["band_layers"],
                              "positions": p_["positions"]})
    dy_med = _median([c["delta_y"] for c in cells])
    ds_med = _median([c["delta_s"] for c in cells])
    per_band = {}
    for b in sorted({c["band"] for c in cells}):
        sel = [c for c in cells if c["band"] == b]
        per_band[b] = {"n": len(sel),
                       "delta_y_median": _median([c["delta_y"] for c in sel]),
                       "delta_s_median": _median([c["delta_s"] for c in sel])}
    return {"delta_y_pc": dy_med, "delta_s_pc": ds_med,
            "n_cells": len(cells), "per_band": per_band, "cells": cells,
            "definition": ("＝主 patch 本身（同層帶同位置帶，M6）；"
                           "第 0 層還原式必過版本在 layer_band/make_patch_hooks "
                           "兩處硬擋，所以這個對照是**會失敗的**版本"),
            "aggregation": "各格 ΔY／Δs 的中位數（不用最大值，避免最好的一格充當整體）",
            "verdict": None,
            "note": "門檻 ΔY_pc ≥ max(10×底線, 0.5×Δs_pc)（PF-5）由判定器套用"}


def calibration_contract(runs) -> dict:
    """causal_smoke_eval.eval_calibration 讀的格式：
    {model: [{id, n_survivors, selected_drop, selected_quantile}, ...]}"""
    out = {}
    for model, run in runs.items():
        rows = []
        for r in run["records"]:
            cal = r.get("calibration") or {}
            rows.append({"id": r["id"],
                         "n_survivors": cal.get("n_survivors"),
                         "selected_drop": cal.get("selected_drop_launch"),
                         "selected_quantile": cal.get("selected_quantile"),
                         "pool_too_small": cal.get("pool_too_small"),
                         "pool_too_small_final": cal.get("pool_too_small_final"),
                         "expansion": cal.get("expansion"),
                         "meets_adequacy": cal.get("meets_adequacy"),
                         "adequacy_checked": cal.get("adequacy_checked"),
                         "fallback": cal.get("fallback"),
                         "error": r.get("error")})
        out[model] = rows
    return out


def replicate_points_contract(runs) -> dict:
    """causal_smoke_eval.eval_r_dof 讀的格式，＋v2.11③/v2.13 PF-1 的正確 dof。

    **dof 依實際使用的擬合式算**（v2.13 PF-1「連帶必改」）：
      二次式＋逐題截距 → **dof = Σ(kᵢ−1) − 2**（一次式才是 −1）
      8 題×3 詞 → Σ(kᵢ−1)=16 → dof=14 過門檻；
      但淘汰到 k=2 時 Σ(kᵢ−1)=8 → dof=6 < 8，**會觸發補題**。

    v2.11③：**輸出 Σ(kᵢ−1) 取代 mean_k**（代數同值，純可讀性）。
    `mean_k` 仍保留——判定器目前用它反推，拿掉會讓它靜默拿 None。
    """
    ks = []
    for run in runs.values():
        for r in run["records"]:
            if "error" in r:
                continue
            k = 1 + len(r.get("cross_word") or [])
            if k >= 2:
                ks.append(k)
    m = len(ks)
    sum_km1 = sum(k - 1 for k in ks)
    mean_k = float(np.mean(ks)) if ks else None
    dof_quad = (sum_km1 - 2) if m else None
    return {"n_items_with_k_ge_2": m, "mean_k": mean_k,
            "sum_k_minus_1": sum_km1, "k_per_item": ks,
            "realized_dof": dof_quad,
            "realized_dof_formula": "Σ(kᵢ−1)−2（二次式＋逐題截距，v2.13 PF-1）",
            "realized_dof_if_linear": ((sum_km1 - 1) if m else None),
            "target_dof": 8,
            "meets_target": (None if dof_quad is None else bool(dof_quad >= 8)),
            "spec": ("附錄 C③＋v2.13 PF-1：實現 dof ≥ 8；起始 8 題×3 詞；"
                     "不足補題、記錄補幾題、**不得改門檻**"),
            "note": ("⚠ 判定器 eval_r_dof 目前用 m(mean_k−1)−1（**一次式**）反推，"
                     "與 PF-1 凍結的二次式差 1。以本欄 realized_dof 為準。")}


def _cross_word(model, item, cfg, rec, n_words) -> list[dict]:
    """跨替換詞量測（v2.6 §3）：同一個 cue、換另外幾個**進帶**的替換詞。

    ⚠ **這個量已經改用途了。** 我原本把它當 §7.1 的噪音尺，理論部否決：
    換替換詞＝換介入，量到的是效應不是噪音。v2.6 把它改成「選 k」的依據
    ——跨替換詞的相關高，代表一題用一個詞就夠；相關低，代表要多取幾個平均。
    噪音底線改由 `numfloor` 段量（同介入同輸入，只換 dtype／batch）。
    """
    cue = rec["corruption"]["key"].split(":")[0]
    cal = rec["calibration"]
    target = cal.get("quantile_target_drop")
    guard = cal.get("rank_guard")
    # v2.8 口徑：同一個 cue、其餘的**存活**候選（drop>0 且過 rank 護欄），
    # 依「離分位目標多近」排序取前幾個。沒有帶可言了（分母全廢），
    # 所以用「最接近選中者的強度」當「等價實現」的操作化。
    pool = [c for c in cal["curve"]
            if c["key"].split(":")[0] == cue
            and c["key"] != rec["corruption"]["key"]
            and c.get("drop_launch", 0) > 0
            and (not guard or (c.get("launch_rank_max") is not None
                               and c["launch_rank_max"] <= guard))]
    if target is not None:
        pool.sort(key=lambda c: (abs(c["drop_launch"] - target), c["key"]))
    else:
        pool.sort(key=lambda c: c["key"])
    out = []
    prompt, gold = cp.item_prompt_gold(item)
    tok = cp.tokenize_pair(model, prompt, gold)
    bands = {k: cp.layer_band(int(model.cfg.n_layers), lo_, hi_, cfg.hook_kind)
             for k, (lo_, hi_) in cfg.bands.items()}
    all_layers = sorted({l for ls in bands.values() for l in ls})
    clean, cache = cp.gold_readout_with_cache(
        model, tok, cp.hook_names_for(all_layers, cfg.hook_kind), cfg.launch_k)
    for cand in pool[: n_words - 1]:
        span = [s for s in cp._word_spans(tok.prompt) if s[2] == cand["cue_word"]]
        span = [s for s in span if str(s[0]) == cue]
        if not span:
            continue
        cc = cp.build_corrupt_candidate(model, tok, span[0], cand["replacement"],
                                        cand["pool_class"])
        if cc is None:
            continue
        ct = model.to_tokens(cc.corrupted_prompt + tok.gold)
        with torch.no_grad():
            corr = cp.gold_readout_from_logits(model(ct), ct, tok.n_prompt, cfg.launch_k)
        positions = cp.cue_neighborhood(cc.diff_token_idx, cfg.radius, tok.n_prompt)
        row = {"key": cc.key, "delta_launch": corr.launch_mean - clean.launch_mean,
               "patches": {}}
        for bname, layers in bands.items():
            pr = cp.patched_readout(model, ct, tok.n_prompt, cache, layers, positions,
                                    cfg.hook_kind, cfg.launch_k)
            row["patches"][bname] = cp.recovery_ratio(
                clean.launch_mean, corr.launch_mean, pr.launch_mean)
        out.append(row)
    return out


def _cfg_json(cfg: cp.PatchConfig, args) -> dict:
    return {"bands": {k: list(v) for k, v in cfg.bands.items()},
            "hook_kind": cfg.hook_kind, "radius": cfg.radius, "launch_k": cfg.launch_k,
            "denom_eps": cfg.denom_eps,
            "denom_eps_status": "器材部佔位值，非凍結值：|Y_clean−Y_corr| 低於它就標 denom_unstable",
            "n_cue": cfg.n_cue, "n_repl_per_cue": cfg.n_repl_per_cue,
            "seed": args.seed, "dtype": args.dtype,
            # v2.8（依 DESIGN_PROPOSAL_v2.2_rulings v2.8／theory_v3_final 附錄 C）
            "spec_amendment": "v2.8",
            "quantile": cfg.quantile,
            "quantile_status": "v2.8 凍結值 0.75（規格明標「慣例」，非導出）",
            "min_survivors": cfg.min_survivors,
            "adequacy_multiple": cfg.adequacy_multiple,
            "numerical_floor_nats": cfg.numerical_floor_nats,
            "adequacy_checked": cfg.numerical_floor_nats is not None,
            "rank_guard": cfg.rank_guard,
            "rank_guard_status": "v2.6 凍結、v2.8 維持（top-50），絕對護欄只留上限",
            "abolished_in_v2.8": "Δs_max 相對帶 [0.30,0.70]（分母全廢）",
            "cross_word_k": args.cross_word_k, "cross_word_items": args.cross_word_items,
            "gate_frac": args.gate_frac, "gate_model_frac": args.gate_model_frac,
            "probe_words": args.probe_words, "probe_stride": args.probe_stride}


def _per_item(runs, band, field) -> dict[str, dict[str, float]]:
    """{model: {item_id: value}}；field 走 launch 的 R / denom。"""
    out: dict[str, dict[str, float]] = {}
    for model, run in runs.items():
        d = {}
        for r in run["records"]:
            if "error" in r:
                continue
            p = next((p for p in r.get("patches", []) if p["band"] == band), None)
            if p:
                d[r["id"]] = p["launch"][field]
        out[model] = d
    return out


def summarize_anchors(runs, items) -> dict:
    """① σ_DiD ② rho_fam ③ 校準曲線摘要 ④ 起手窗效應錨。

    全部是**描述統計**（SD、相關、中位數）。不含檢定、不含 CI、不含裁決。
    每個錨都附 n，n 小的時候 n 本身就是那個數字最重要的限定語。
    """
    levels = {it["id"]: it.get("level") for it in items}
    twins = {it["id"]: it.get("twin_of") for it in items if it.get("level") == "L0M"}
    bands = sorted({p["band"] for run in runs.values() for r in run["records"]
                    for p in r.get("patches", [])})
    out: dict = {"bands": bands, "sigma": {}, "rho_fam": {}, "launch_effect": {},
                 "calibration": {}}

    for band in bands:
        R = _per_item(runs, band, "R")
        D = _per_item(runs, band, "denom")
        # ① σ：條件 × run 格內的題間 SD（N4.1 的 sigma^2 = tau^2 + sigma_e^2）
        cell = {}
        for model, d in R.items():
            for lv in sorted(set(levels.values())):
                ids = [i for i in d if levels.get(i) == lv]
                if len(ids) >= 2:
                    cell[f"{model}|{lv}"] = {
                        "n": len(ids), "sd_R": sd([d[i] for i in ids]),
                        "sd_drop": sd([D[model][i] for i in ids if i in D[model]]),
                        "median_R": _median([d[i] for i in ids])}
        # ① σ_DiD：雙胞胎對內差（L0M − L0）的 SD。
        # **v2 修復（附錄 H⑤）：用 holdout 檔裡的真 `twin_of` 算點估計**，
        # 不再需要「120 種指派」的區間——配對本來就記著，只是先前沒被傳播出來。
        # **尺度＝ΔY（nats）**（v2.13 F 案：比值尺度有假相關，門檻量一律走分子）。
        DY = _per_item(runs, band, "numer")
        did = {}
        for model in R:
            d = DY.get(model, {})
            pairs, diffs = [], []
            for l0m_id, l0_id in twins.items():
                if l0_id and l0m_id in d and l0_id in d:
                    diffs.append(d[l0m_id] - d[l0_id])
                    pairs.append({"l0m": l0m_id, "l0": l0_id,
                                  "dy_l0m": d[l0m_id], "dy_l0": d[l0_id],
                                  "did": d[l0m_id] - d[l0_id]})
            did[model] = {"n_pairs": len(diffs),
                          "sigma_did_dy_nats": sd(diffs),
                          "mean_did_dy_nats": (float(np.mean(finite(diffs)))
                                               if finite(diffs) else None),
                          "pairs": pairs,
                          "scale": "delta_y_nats（v2.13 F 案）",
                          "pairing_source": "holdout 檔的 twin_of（真配對，非指派枚舉）",
                          # 舊鍵保留相容，但標明它是比值尺度、不得進門檻
                          "sd_did_R_descriptive_only": sd(
                              [R[model][a] - R[model][b] for a, b in twins.items()
                               if b and a in R[model] and b in R[model]])}
        # ② rho_fam：Pythia run 之間、逐題統計量的平均兩兩相關。
        # 用名稱前綴認家族（不寫死那四個），模型清單換了照樣算得出來；
        # 實際併了哪幾個 run 一律寫進 runs_used，不讓讀的人用猜的。
        py = sorted(m for m in R if m.lower().startswith("eleutherai/pythia"))
        common = set.intersection(*[set(R[m]) for m in py]) if len(py) >= 2 else set()
        pairs = []
        for i in range(len(py)):
            for j in range(i + 1, len(py)):
                ids = sorted(common)
                r = pearson([R[py[i]][k] for k in ids], [R[py[j]][k] for k in ids])
                if r is not None:
                    pairs.append({"runs": [py[i], py[j]], "r": r})
        out["sigma"][band] = {"per_cell": cell, "did_twin_pairs": did}
        out["rho_fam"][band] = {
            "n_runs": len(py), "runs_used": py, "n_common_items": len(common),
            "pairwise": pairs,
            "mean_r": float(np.mean([p["r"] for p in pairs])) if pairs else None,
            "note": "四個 Pythia run 之間逐題 R 的平均兩兩相關；OLMo 不併（跨家族尺度不可比）",
        }
        # ④ 起手窗效應錨：腐蝕落差與恢復量，nats
        eff = {}
        for model, d in R.items():
            drops = [v for v in D[model].values()]
            eff[model] = {
                "n": len(drops),
                "median_corruption_drop_nats": _median(drops),
                "iqr_corruption_drop_nats": (
                    float(np.percentile(finite(drops), 75) - np.percentile(finite(drops), 25))
                    if len(finite(drops)) >= 4 else None),
                "median_R": _median(list(d.values())),
                "frac_R_above_0.9": (float(np.mean([v >= 0.9 for v in finite(d.values())]))
                                     if finite(d.values()) else None),
                "n_denom_unstable": sum(
                    1 for r in runs[model]["records"] if "error" not in r
                    for p in r.get("patches", []) if p["band"] == band
                    and p["launch"]["denom_unstable"]),
            }
        out["launch_effect"][band] = eff

    # ③ 校準曲線摘要（整條原始曲線在 runs[*]['records'][*]['calibration']['curve']）
    for model, run in runs.items():
        rows = []
        for r in run["records"]:
            c = r.get("calibration", {}).get("curve") or []
            if not c:
                continue
            drops = [x.get("drop_launch", -x["delta_launch"]) for x in c]
            pos = [x for x in drops if x > 0]
            rows.append({"id": r["id"], "n_candidates": len(c),
                         # drop = Y_clean − Y_corrupted，正值＝腐蝕真的弄壞了 gold
                         "min_drop": min(drops), "max_drop": max(drops),
                         "median_drop": _median(drops),
                         "n_positive_drop": len(pos),
                         "n_nonpositive_drop": len(drops) - len(pos),
                         "selected": r.get("corruption", {}).get("key"),
                         "selected_drop": (
                             r["clean"]["launch_mean"] - r["corrupted"]["launch_mean"]
                             if "corrupted" in r else None),
                         "fallback": r.get("calibration", {}).get("fallback")})
        out["calibration"][model] = rows

    # v2.6 §3 / 附錄 B2.4：跨替換詞 **變異比 r = σ²_word / σ²_item**（用途＝選 k）
    # **v2.13 F 案：改在分子 ΔY（nats）上算**——共用隨機分母＋非零均值分子
    # ⇒ 比值假相關；R 尺度會系統性汙染這個量，與 G-link 同病同修。
    cw: dict = {}
    for model, run in runs.items():
        per_band: dict[str, dict] = {}
        for band in bands:
            cols: list[list[float]] = []
            for r in run["records"]:
                words = r.get("cross_word") or []
                if not words:
                    continue
                main = next((p["launch"]["numer"] for p in r.get("patches", [])
                             if p["band"] == band), None)
                vals = [main] + [w["patches"][band].get("numer") for w in words
                                 if band in w["patches"]]
                vals = finite(vals)
                if len(vals) >= 2:
                    cols.append(vals)
            if len(cols) < 2:
                continue
            per_band[band] = variance_components(cols)
        cw[model] = per_band
    out["cross_word_r"] = {
        "by_model": cw,
        "purpose": "選 k（一題要取幾個替換詞平均）——v2.6 §3／附錄 B2.4",
        "estimand": "r = σ²_word / σ²_item（單向隨機效果分解），**不是相關係數**",
        "scale": "**ΔY（nats）**——v2.13 F 案：比值尺度有假相關，改在分子上算",
        "spec_n": "附錄 C③／v2.13：實現 dof ≥ 8；起始 8 題×3 詞（二次式 dof=Σ(kᵢ−1)−2）",
        "verdict": None,
        "note": ("⛔ **這不是噪音尺。** 換替換詞＝換介入，差異是效應不是噪音；"
                 "器材部原先把它當 §7.1 噪音門檻的做法已被理論部否決"
                 "（會系統性高估噪音→閘門永不點燃→死閘）。"
                 "噪音底線改由 numfloor 段量。**k 值由分析部依 σ_eff 倍率表選"
                 "（§12 白名單⑤，k∈{1,2,3}），器材部只給 r 與倍率。**"),
    }
    return out


def variance_components(groups: Sequence[Sequence[float]]) -> dict:
    """單向隨機效果分解：組＝題、組內＝替換詞。回傳 σ²_word、σ²_item、r 與 σ_eff 倍率。

    theory_v3_final 附錄 B2.4 要的 r 是**變異比** σ²_word/σ²_item，不是相關係數。
    σ²_eff = σ²_item + σ²_word / k，倍率＝sqrt(σ²_eff(k) / σ²_eff(1))。
    （拿 B2.4 的表驗過：r=1 時 k=3 → 0.816；r=2 時 k=4 → 0.707。）

    不平衡時取各組共同的最小 k 截斷（截掉的數量一併回報），因為平衡設計的
    MSB/MSW 分解才有這個乾淨的閉式。
    """
    k = min(len(g) for g in groups)
    n_trunc = sum(len(g) - k for g in groups)
    if k < 2 or len(groups) < 2:
        return {"n_items": len(groups), "k": k, "reason": "insufficient_groups_or_words"}
    G = np.array([g[:k] for g in groups], float)
    N = G.shape[0]
    means = G.mean(1)
    msw = float(((G - means[:, None]) ** 2).sum() / (N * (k - 1)))
    msb = float(k * np.var(means, ddof=1))
    var_word = msw
    var_item_raw = (msb - msw) / k
    var_item = max(0.0, var_item_raw)
    r = (var_word / var_item) if var_item > 0 else None
    ratios = {}
    for kk in (1, 2, 3, 4):
        eff = var_item + var_word / kk
        base = var_item + var_word
        ratios[f"k={kk}"] = float(np.sqrt(eff / base)) if base > 0 else None
    return {"n_items": N, "k": k, "n_truncated_values": n_trunc,
            "df_within": N * (k - 1),
            "MSW": msw, "MSB": msb,
            "var_word": var_word, "var_item": var_item,
            "var_item_raw": float(var_item_raw),
            "var_item_clamped_at_zero": bool(var_item_raw < 0),
            "r_var_ratio": r,
            "sigma_eff_ratio_vs_k1": ratios}


def single_token_gate(runs, items, args) -> dict:
    """v2.8 §1 分支規則素材（v2.13 D 案修正分母與逐模型旗標）。

    規則：保留題中 **pool_too_small 或未達適足下限**者 > 40% → 判單 token
    腐蝕強度不足，走多 token 分支。

    **v2.13 D 案兩處修正：**
      ①**分母＝全部嘗試過的 (題 × 模型) 格，含例外題**。零候選、rank 護欄
        全擋、適足不過——這些都是 **bad 且入分母**，不能因為「那題出局了」
        就從分母消失（那會讓壞題越多、分母越小、比例越好看）。
      ②**併池 >40% 是主判定，另外逐模型 >60% 必須標記點名**——防「兩個模型
        全死被三個模型平均掉」。

    底線未注入時 `adequacy_checked=false`，只算 pool_too_small 那一半並標
    `incomplete`。**verdict 一律 null。**
    """
    out = {"threshold_frac": args.gate_frac,
           "per_model_flag_frac": args.gate_model_frac,
           "rule": (f"**只有 adequacy_fail 計入分支**；> {args.gate_frac:.0%} → 單 token "
                    f"腐蝕強度不足，走多 token 分支；另逐模型 > {args.gate_model_frac:.0%} "
                    "必須標記點名（v2.13 D②）"),
           "rule_status": "v2.14（附錄 G）壞格三分法；v2.8 合併定義作廢",
           "cell_classes": {
               "adequacy_fail": "計入分支（真的強度不足）",
               "rank_guard_zero_survivor": "**排除不計入**（腐蝕過強／clean 排名低；多 token 是反向補救）",
               "pool_too_small": "走擴池，不計入分支",
               "ok": "正常"},
           "denominator_rule": ("全部嘗試過的 (題 × 模型) 格，**扣除 "
                                "rank_guard_zero_survivor 那些格**（v2.14）"),
           "numerical_floor_nats": args.numerical_floor_nats,
           "verdict": None, "by_model": {}}
    tot_bad = tot_n = 0
    flagged = []
    for model, run in runs.items():
        rows, n_bad, n_cells, any_unchecked = [], 0, 0, False
        for r in run["records"]:
            cal = r.get("calibration") or {}
            err = r.get("error")
            pool_small = cal.get("pool_too_small")
            adequacy_ok = cal.get("meets_adequacy")
            checked = bool(cal.get("adequacy_checked"))
            if not checked:
                any_unchecked = True
            # ── v2.14（附錄 G）壞格三分法，v2.8 的合併定義作廢 ──
            # 先前裁定曾將「強度不足」與「選詞精度不足」誤併一條，
            # 於是**量測預算不足會被誤診成設計失敗**。三類的補救方向完全不同：
            #   adequacy_fail                  → 計入多 token 分支（真的強度不足）
            #   no_candidate_within_rank_guard → **排除不計入**（腐蝕過強／clean
            #                                     排名低，多 token 是反向補救）
            #   pool_too_small(1≤存活<12)      → 走**擴池**，不計入分支
            adequacy_fail = bool(checked and adequacy_ok is False)
            rank_guard_zero = (err == "no_candidate_within_rank_guard")
            cls = ("adequacy_fail" if adequacy_fail else
                   "rank_guard_zero_survivor" if rank_guard_zero else
                   "pool_too_small" if pool_small else
                   ("other_error:" + err) if err else "ok")
            counts_toward_branch = adequacy_fail
            excluded_from_denominator = rank_guard_zero
            rows.append({"id": r["id"], "n_survivors": cal.get("n_survivors"),
                         "pool_too_small": pool_small,
                         "adequacy_checked": checked, "meets_adequacy": adequacy_ok,
                         "selected_drop": cal.get("selected_drop_launch"),
                         "exhaustion_type": cal.get("exhaustion_type"),
                         "zero_survivor_rank_diagnostic": cal.get(
                             "zero_survivor_rank_diagnostic"),
                         "error": err, "cell_class": cls,
                         "counts_toward_branch": counts_toward_branch,
                         "excluded_from_denominator": excluded_from_denominator,
                         "bad": counts_toward_branch})
            if excluded_from_denominator:
                continue                      # 該格整個排除，不進分子也不進分母
            n_bad += int(counts_toward_branch); n_cells += 1
        frac = (n_bad / n_cells) if n_cells else None
        if frac is not None and frac > args.gate_model_frac:
            flagged.append({"model": model, "fraction_bad": frac,
                            "n_bad": n_bad, "n_cells": n_cells})
        out["by_model"][model] = {
            "n_cells": n_cells, "n_bad": n_bad, "fraction_bad": frac,
            "exceeds_per_model_flag": (None if frac is None
                                       else bool(frac > args.gate_model_frac)),
            "incomplete": any_unchecked,
            "incomplete_reason": ("數值底線未注入，適足下限那一半沒判；"
                                  "現值只反映 pool_too_small 與例外題"
                                  if any_unchecked else None),
            "rows": rows}
        tot_bad += n_bad; tot_n += n_cells
    out["pooled"] = {"n_cells": tot_n, "n_bad": tot_bad,
                     "fraction_bad": (tot_bad / tot_n) if tot_n else None}
    out["per_model_flagged"] = flagged
    out["per_model_flag_note"] = (
        f"以下模型壞題比例 > {args.gate_model_frac:.0%}，**必須點名不得被併池平均掉**："
        + ", ".join(f"{x['model']}({x['fraction_bad']:.0%})" for x in flagged)
        if flagged else None)
    return out


def _fit_block(pts, mad_k=1.4826, quadratic=True, with_item_fe=False) -> dict:
    """一個回歸格：ΔY ~ f(Δs)（＋逐題截距）。回傳 dof / S / SE(β) / 兩個 σ̂。

    v2.13 PF-1 補簽：**f 凍結為二次式**（理由換掉——瓶頸假設涵蓋「Δs 的任意
    函數」，在 Δs 上給彈性是正確建模；剛性的真風險是誤放：真退化＋h 有彎時
    一次式讀 17.7×底線＝誤判活著＝燒掉註定沒結果的 GPU 窗）。
    **dof 依實際使用的擬合式算**（二次＋逐題截距＝Σ(kᵢ−1)−2）。

    v2.12：
      ①**S 只數該回歸實際保留的 distinct (item, replacement)**——Δs 與層帶
        無關，照 scatter 列算會三倍重複計數。三帶 S 通常相同，**不同即某帶
        掉點的診斷訊號**，所以逐格各算各的、不共用。
      ②**SE(β) = sqrt(RSS/dof)/sqrt(S)，配 OLS β̂**；G1 殘差尺度維持 MAD。
        兩用途兩尺度不是不一致：G1 問「殘差主體 vs 底線」（離群不算結構→MAD），
        SE 問「β̂ 的不確定度」（OLS 估計量→OLS 尺度）。
        **兩個 σ̂ 都輸出；RSS/MAD > 1.5 → 標記「精度陳述由離群點主導」，
        兩種讀法並列，不排除任何點。**
      ③kᵢ≤1 的題貢獻恰 0 dof 且吃一截距 → **唯一排除**（v2.11-a②），分開報。
    """
    ids = sorted({q["id"] for q in pts})
    x = np.array([q["delta_s"] for q in pts], float)
    y = np.array([q["delta_y"] for q in pts], float)
    keyed: dict[str, set] = {}
    for q in pts:
        keyed.setdefault(q["id"], set()).add(q.get("replacement_key") or "__main__")
    k = {i: len(v) for i, v in keyed.items()}
    contributing = sorted(i for i in ids if k[i] >= 2)
    dropped_k1 = sorted(i for i in ids if k[i] < 2)
    sum_km1 = sum(k[i] - 1 for i in contributing)
    n_par_x = 3 if quadratic else 2
    dof = (sum_km1 - (n_par_x - 1)) if with_item_fe else (len(pts) - n_par_x)

    per_item_S, S = {}, 0.0
    for i in contributing:
        seen, vals = set(), []
        for q in pts:
            if q["id"] != i:
                continue
            kk = q.get("replacement_key") or "__main__"
            if kk in seen:
                continue
            seen.add(kk); vals.append(q["delta_s"])
        v = np.array(vals, float)
        si = float(((v - v.mean()) ** 2).sum())
        per_item_S[i] = si
        S += si

    out = {"n_points": len(pts), "n_items": len(ids),
           "n_items_contributing": len(contributing),
           "n_items_k_le_1": len(dropped_k1), "items_k_le_1": dropped_k1,
           "sum_k_minus_1": sum_km1, "k_per_item": k,
           "fit_form": "quadratic" if quadratic else "linear",
           "dof_formula": ("Σ(kᵢ−1)−2（逐題截距＋二次）" if with_item_fe and quadratic
                           else ("Σ(kᵢ−1)−1（逐題截距＋一次）" if with_item_fe
                                 else f"n−{n_par_x}（併池）")),
           "dof": int(dof), "S": S,
           "S_unit": "distinct(item, replacement)，逐回歸各算（v2.12①）"}
    if per_item_S and S > 0:
        top = max(per_item_S.items(), key=lambda t: t[1])
        out["S_share_per_item"] = {i: v / S for i, v in per_item_S.items()}
        out["max_item_S_share"] = top[1] / S
        out["concentration_flag"] = bool(top[1] / S > 0.50)
        out["concentration_item"] = top[0]
        out["concentration_note"] = ("最大單題佔 S >50%——照實標記，**不排除任何點**"
                                     "（v2.11-a④）" if top[1] / S > 0.50 else None)
    # v2.14 / 附錄 F②：**逐題截距的讀數只能吃 kᵢ≥2 的題，而且要在擬合之前就排除**。
    # 每題只有 1 點時，該題的截距會把那一點完全吃掉 → 殘差恆 0；把這種題留在
    # 擬合裡會做出「殘差 0、dof 為負」的**保證誤殺格**（理論部 F0 在 winA 資料上
    # 重算證實）。main 的點每題只有 1 個，所以 FE 讀數實務上只吃 cross_word。
    fit_pts = pts
    if with_item_fe:
        keep = set(contributing)
        fit_pts = [q for q in pts if q["id"] in keep]
        out["fe_excluded_points"] = len(pts) - len(fit_pts)
        out["fe_fit_note"] = ("逐題截距只吃 kᵢ≥2 的題（擬合前排除）；"
                              "kᵢ=1 會讓殘差恆 0＝保證誤殺（附錄 F②）")
        ids = contributing
        x = np.array([q["delta_s"] for q in fit_pts], float)
        y = np.array([q["delta_y"] for q in fit_pts], float)
    if dof < 2 or len(fit_pts) < n_par_x + 2:
        out["status"] = "insufficient_dof"
        return out

    cols = [np.ones_like(x), x] + ([x ** 2] if quadratic else [])
    if with_item_fe:
        idx = {i: j for j, i in enumerate(ids)}
        ii = np.array([idx[q["id"]] for q in fit_pts])
        for j in range(1, len(ids)):
            cols.append((ii == j).astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    rss = float((res ** 2).sum())
    sigma_rss = float(np.sqrt(rss / dof)) if dof > 0 else None
    sigma_mad = float(mad_k * np.median(np.abs(res - np.median(res)))) if len(res) >= 4 else None
    out.update({
        "n_points_fitted": len(fit_pts),
        "beta_slope": float(beta[1]),
        "beta_quad": (float(beta[2]) if quadratic else None),
        "RSS": rss, "sigma_hat_rss": sigma_rss, "resid_mad_scale": sigma_mad,
        "se_beta": (sigma_rss / float(np.sqrt(S)))
                   if (sigma_rss is not None and S > 0) else None,
        "se_beta_formula": "sqrt(RSS/dof)/sqrt(S)，配 OLS β̂（v2.12②）",
    })
    if sigma_rss is not None and sigma_mad and sigma_mad > 0:
        ratio = sigma_rss / sigma_mad
        out["sigma_rss_over_mad"] = ratio
        out["outlier_dominated_flag"] = bool(ratio > 1.5)
        if ratio > 1.5:
            out["outlier_note"] = ("RSS/MAD > 1.5：**精度陳述由離群點主導**，"
                                   "兩種讀法並列，不排除任何點（v2.12②）")
    return out


def g1_existence_inputs(runs, g1_backend: str = "fp32",
                       excluded_models: Sequence[str] = ()) -> dict:
    """G1 存在性檢定素材（器材部側對帳；正式判定在 causal_smoke_eval.py）。

    **v2.13 G 案：擬合單位＝逐 (模型, 帶, source)。** 五模型共用截距會讓
    模型間整 nat 量級的差灌進殘差，把 G1-a 殺閘關掉——與「換詞當噪音」同病。
    合成＝15 格全一致決（沿 B2.5/D1），**合成與判定都在分析部**，這裡只出格。

    尺度：**ΔY（nats）**（v2.8 §2：R 不得進入任何門檻）。
    """
    excl = set(excluded_models)
    pts = []
    for model, run in runs.items():
        # 附錄 H①：**跨後端的 G1 讀數不得混進一致決集合**。
        # 該模型的 G1 資料與底線必須同後端；2.8b 因 fp32 裝不下 8GB 而排除，
        # 它其餘的量照跑照報，只是不進 G1。
        if model in excl:
            continue
        for r in run["records"]:
            if "error" in r:
                continue
            for p_ in r.get("patches", []):
                ds, dy = p_["g1_point"]["delta_s"], p_["g1_point"].get("delta_y")
                if dy is not None and np.isfinite(dy) and np.isfinite(ds):
                    pts.append({"model": model, "id": r["id"], "band": p_["band"],
                                "delta_s": ds, "delta_y": dy, "source": "main",
                                "replacement_key": "__main__"})
            for w in (r.get("cross_word") or []):
                ds_w = -w["delta_launch"]
                for band, rr in (w.get("patches") or {}).items():
                    dy_w = rr.get("numer")
                    if dy_w is not None and np.isfinite(dy_w) and np.isfinite(ds_w):
                        pts.append({"model": model, "id": r["id"], "band": band,
                                    "delta_s": ds_w, "delta_y": dy_w,
                                    "source": "cross_word",
                                    "replacement_key": w.get("key")})

    out = {"scale": "delta_y_nats（v2.8 §2：R 不得進入任何門檻）",
           "fit_unit": "逐 (model, band, source)（v2.13 G 案）",
           "fit_form": "quadratic（v2.13 PF-1 補簽凍結；「二次」仍標慣例）",
           "resid_scale": "G1 用 MAD；SE(β) 用 OLS（v2.12②，兩用途兩尺度）",
           "composition": ("一致決集合＝**同後端**的 (模型 × 帶) 格"
                           "（v2.13 G 案；後端異質格不得混入，附錄 H①）"),
           "g1_backend": g1_backend,
           "excluded_models": sorted(excl),
           "excluded_reason": ("fp32 裝不下 8GB → 該模型 G1 資料與底線無法同後端，"
                               "依附錄 H① 預設排除；其餘量照跑照報"),
           "cells": {}, "verdict": None,
           "authority": "正式判定在 harness/causal_smoke_eval.py；本節僅供對帳"}
    cells = {}
    for model in sorted({q["model"] for q in pts}):
        for band in sorted({q["band"] for q in pts if q["model"] == model}):
            for src in ("main", "cross_word", "all"):
                sel = [q for q in pts if q["model"] == model and q["band"] == band
                       and (src == "all" or q["source"] == src)]
                if not sel:
                    continue
                key = f"{model}|{band}|{src}"
                cells[key] = {
                    "pooled": _fit_block(sel, with_item_fe=False),
                    "item_fe": _fit_block(sel, with_item_fe=True),
                }
    out["cells"] = cells
    return out


def g1_scatter(runs) -> list[dict]:
    """§7.1 G1-gold 散點：每個 patch 一個點（腐蝕落差 Δs, 恢復比例 R），帶條件標籤。"""
    pts = []
    for model, run in runs.items():
        for r in run["records"]:
            if "error" in r:
                continue
            for p in r.get("patches", []):
                pts.append({"model": model, "id": r["id"], "level": r.get("level"),
                            "band": p["band"], "delta_s": p["g1_point"]["delta_s"],
                            # v2.8 §2：門檻在分子 ΔY（nats）上；R 只留作描述
                            "delta_y": p["g1_point"].get("delta_y"),
                            "R": p["g1_point"]["R"],
                            "source": "main",
                            "denom_unstable": p["launch"]["denom_unstable"]})
            # v2.8 §4：「G1-b 讀數與 r 量測共用同批點」——跨替換詞的點**必須**
            # 進散點，否則逐題截距那一支恆為不可辨識（每題只有一個 Δs），
            # G1 的兩層讀數就塌回一層。這些點與主選點是同型的 (Δs, ΔY)。
            for w in (r.get("cross_word") or []):
                ds_w = -w["delta_launch"]
                for band, rr in (w.get("patches") or {}).items():
                    pts.append({"model": model, "id": r["id"], "level": r.get("level"),
                                "band": band, "delta_s": ds_w,
                                "delta_y": rr.get("numer"), "R": rr.get("R"),
                                "source": "cross_word", "replacement_key": w.get("key"),
                                "denom_unstable": rr.get("denom_unstable")})
    return pts


# ---------------------------------------------------------------- stage: l0m

def stage_l0m(args, items, prov) -> dict:
    """L0M 12 對試做（§3／M2）。只量 clean 的整段與起手窗 gold——通過規則要的
    就是這兩個量在五個模型上的中位數。**這裡不判過不過閘**（§任務邊界）。"""
    pairs_src = Path(args.l0m_pairs) if args.l0m_pairs else None
    if pairs_src:
        data = json.load(open(pairs_src, encoding="utf-8"))
        pairs = data["pairs"] if isinstance(data, dict) else data
    else:   # 從保留題裡撈雙胞胎（保留集的 4 對；12 對候選檔另給）
        by_id = {it["id"]: it for it in items}
        pairs = [{"l0": by_id[it["twin_of"]], "l0m": it} for it in items
                 if it.get("level") == "L0M" and it.get("twin_of") in by_id]
    if not pairs:
        raise SystemExit("找不到 L0M 雙胞胎對；用 --l0m-pairs 指定 12 對候選檔"
                         '（格式：{"pairs":[{"l0":{...},"l0m":{...}}, ...]}）')
    print(f"[l0m] {len(pairs)} 對 × {len(args.models)} 模型")

    runs: dict[str, list[dict]] = {}
    for name in args.models:
        def _go(model, device, dtype, load_s):
            rows, n_fwd = [], 0
            for pr in pairs:
                row = {"l0_id": pr["l0"]["id"], "l0m_id": pr["l0m"]["id"]}
                try:
                    for side in ("l0", "l0m"):
                        p, g = cp.item_prompt_gold(pr[side])
                        tok = cp.tokenize_pair(model, p, g)
                        rd = cp.gold_readout(model, tok)
                        n_fwd += 1
                        row[side] = {"launch_mean": rd.launch_mean,
                                     "segment_mean": rd.segment_mean,
                                     "per_tok": rd.per_tok,
                                     "n_gold_tokens": rd.n_gold_tokens}
                    row["delta_segment"] = row["l0m"]["segment_mean"] - row["l0"]["segment_mean"]
                    row["delta_launch"] = row["l0m"]["launch_mean"] - row["l0"]["launch_mean"]
                except Exception as e:
                    row["error"] = repr(e)
                rows.append(row)
            return {"device": device, "dtype": dtype, "n_forwards": n_fwd, "rows": rows}
        print(f"[l0m] {name} ...", flush=True)
        runs[name] = with_model(name, _dtype_for(name, args), _go, args.device)

    # 通過規則要的兩個數字：五模型「中位」的 |Δ|。並排列出門檻，**不判定**。
    def med_abs(field):
        per_model = []
        for name, run in runs.items():
            vals = [abs(r[field]) for r in run["rows"] if field in r]
            if vals:
                per_model.append({"model": name, "median_abs": _median(vals),
                                  "n_pairs": len(vals)})
        overall = _median([m["median_abs"] for m in per_model])
        return {"per_model": per_model, "median_across_models": overall}

    n_fwd_total = sum(r["n_forwards"] for r in runs.values())
    # ── 下游契約（causal_smoke_eval.eval_rho_gate / eval_l0m_pass）──
    pairs_measured = []
    for name, run in runs.items():
        for row in run["rows"]:
            if "error" in row:
                continue
            pairs_measured.append({
                "model": name, "l0_id": row["l0_id"], "l0m_id": row["l0m_id"],
                "l0_onset": row["l0"]["launch_mean"],
                "l0m_onset": row["l0m"]["launch_mean"],
                "l0_whole": row["l0"]["segment_mean"],
                "l0m_whole": row["l0m"]["segment_mean"],
                "delta_onset": row["delta_launch"],
                "delta_whole": row["delta_segment"]})

    def _med_across_models(field):
        per = []
        for name, run in runs.items():
            v = [abs(r[field]) for r in run["rows"] if field in r]
            if v:
                per.append(float(np.median(v)))
        return _median(per)

    return {
        "holdout": prov, "n_pairs": len(pairs),
        "pairs_measured": pairs_measured,
        "median_abs_delta": {"whole": _med_across_models("delta_segment"),
                             "onset": _med_across_models("delta_launch")},
        "pairs": [{"l0": p["l0"]["id"], "l0m": p["l0m"]["id"]} for p in pairs],
        "runs": runs,
        "pass_rule_inputs": {
            "median_abs_delta_gold_segment": med_abs("delta_segment"),
            "median_abs_delta_gold_launch": med_abs("delta_launch"),
            "frozen_thresholds": L0M_THRESHOLDS,
            "verdict": None,
            "note": ("規則（§3 凍結）：五模型中位 |Δgold(整段)| ≤ 0.5 nats **且** "
                     "起手窗 Δ ≤ 1.0 nats，否則降級為未配平描述性對照。"
                     "**過不過由分析部判，器材部只把兩個數字與門檻並排放著。**"),
        },
        "forward_budget": {
            "budget_per_pair": L0M_FORWARD_BUDGET_PER_PAIR,
            "budget_total": L0M_FORWARD_BUDGET_PER_PAIR * len(pairs),
            "actual_total": n_fwd_total,
            "actual_per_pair": n_fwd_total / len(pairs),
            "note": ("§8 寫 16 前向/對（12 對 ≈192）。實際＝2 題 × 模型數，"
                     "一次 forward 同時給整段與起手窗，不必分兩次。"),
        },
    }


# ---------------------------------------------------------------- stage: gen

def stage_gen(args, items, prov) -> dict:
    """溯源軌素材：clean / corrupted / patched 的自由續寫。

    §9 硬約束②：**G-link 不掛 L0P**——這裡直接把 L0P 濾掉，不是靠下游記得濾。
    """
    hosts = [it for it in items if it.get("level") in GLINK_HOST_LEVELS]
    dropped = [it["id"] for it in items if it.get("level") not in GLINK_HOST_LEVELS]
    if not hosts:
        raise SystemExit(f"保留題裡沒有 {GLINK_HOST_LEVELS} 的 G-link 宿主")
    print(f"[gen] 宿主 {len(hosts)} 題；濾掉（非 G-link 宿主）{dropped}")
    cfg = make_cfg(args, gen_tokens=args.gen_tokens)
    runs = {}
    for name in args.models[: args.gen_models]:
        def _go(model, device, dtype, load_s):
            recs = []
            for i, it in enumerate(hosts):
                print(f"  [{i+1}/{len(hosts)}] {it['id']}", flush=True)
                try:
                    rec = cp.run_item(model, it, cfg)
                except Exception as e:
                    rec = {"id": it["id"], "level": it.get("level"), "error": repr(e)}
                recs.append(rec)
            return {"device": device, "dtype": dtype, "records": recs}
        print(f"[gen] {name} ...", flush=True)
        runs[name] = with_model(name, _dtype_for(name, args), _go, args.device)
    return {"holdout": prov, "config": _cfg_json(cfg, args),
            "gen_tokens": args.gen_tokens, "gen_sample": args.gen_sample,
            "seed": args.seed, "excluded_ids": dropped,
            "glink_host_levels": list(GLINK_HOST_LEVELS), "runs": runs}


# ---------------------------------------------------------------- stage: numfloor

def _sel_ds(rec) -> float | None:
    """該題選定腐蝕的 Δs（nats）。底線 v2 要按 Δs 分箱，這是分箱鍵。"""
    return (rec.get("calibration") or {}).get("selected_drop_launch")


def _numfloor_select(pool, by_id, target: int) -> list:
    """底線 v2 的取樣協定（附錄 F③）。

    舊版取「前 N 題」＝**全是 LOWGOLD、而且 Δs 只落在某一段**，量出來的底線
    不能代表整個 Δs 範圍。v2 要求：
      ①每個條件（L0 / L0M / L0P …）**至少 1 題**——不能全是 LOWGOLD
      ②**涵蓋 Δs 高低兩端**——底線隨 Δs 單調成長（理論部量到 0.06→0.028、
        1.24→0.112、2.50→0.229），只取中段會低估高 Δs 格的底線
    作法：先每個條件各取一題（取該條件 Δs 最大者，確保上端有代表），
    再用剩餘名額補「全體 Δs 最小」與「全體 Δs 最大」，最後按 Δs 均勻補齊。
    順序完全決定於 (Δs, id)，無亂數。
    """
    have = [r for r in pool if _sel_ds(r) is not None]
    if not have:
        return pool[:target]
    by_level: dict = {}
    for r in have:
        by_level.setdefault(by_id[r["id"]].get("level", "?"), []).append(r)
    picked, seen = [], set()

    def take(r):
        if r["id"] not in seen:
            seen.add(r["id"]); picked.append(r)

    for lv in sorted(by_level):                      # ① 每條件至少一題
        take(max(by_level[lv], key=lambda r: (_sel_ds(r), r["id"])))
    order = sorted(have, key=lambda r: (_sel_ds(r), r["id"]))
    take(order[0]); take(order[-1])                  # ② 兩端
    for r in order:                                  # 均勻補齊
        if len(picked) >= max(target, len(by_level)):
            break
        take(r)
    return picked


def _floor_bins(rows, recs) -> dict:
    """底線按 Δs 分箱（附錄 F③）：底線隨 Δs 單調成長，G1 要用該格 Δs 中位數
    對應的箱，而不是全體一個數字。箱界固定，不隨資料動（看資料前就定）。"""
    ds = {r["id"]: _sel_ds(r) for r in recs}
    edges = [0.0, 0.5, 1.5, 3.0, float("inf")]
    names = ["ds<0.5", "0.5-1.5", "1.5-3.0", "ds>=3.0"]
    out = {n: {"n": 0, "max_abs_diff_dy_nats": None, "items": []} for n in names}
    for r in rows:
        v = ds.get(r["id"])
        if v is None:
            continue
        for i in range(len(names)):
            if edges[i] <= v < edges[i + 1]:
                b = out[names[i]]
                b["n"] += 1
                b["max_abs_diff_dy_nats"] = max(b["max_abs_diff_dy_nats"] or 0.0,
                                                r["abs_diff"])
                if r["id"] not in b["items"]:
                    b["items"].append(r["id"])
                break
    return {"bin_edges_nats": edges[:-1] + ["inf"], "bins": out,
            "usage": "G1 用該格 Δs 中位數落在哪一箱，就取那一箱的底線（附錄 F③）"}


def stage_numfloor(args, items, prov) -> dict:
    """v2.6 §2 數值底線：**同介入、同輸入**，只換 dtype 或 batch，看 R 差多少。

    這是 G1 存在性檢定要比對的那條線。兩條路徑：
      路徑 A（dtype）：fp32 跑一次、fp16 跑一次，相減。
      路徑 B（batch）：同 dtype，改 batch 組成/size 再跑一次，相減。
    ≥3 題 × 2 路徑，取**差異上界**當底線。

    關鍵是「同介入」：腐蝕與 patch 位置都從既有的 anchors 紀錄原封不動重建
    （`cp.rebuild_selected`），**不重跑校準**——重跑校準會換掉介入，那量到的
    就變成效應而不是數值噪音，正是被否決的那個錯。

    所以本段需要 `--anchors-json`（同一批題、同一組設定跑出來的）。
    """
    if not args.anchors_json:
        raise SystemExit("numfloor 段要 --anchors-json（介入必須從既有紀錄重建，不可重跑校準）")
    anchors = json.load(open(args.anchors_json, encoding="utf-8"))
    by_id = {it["id"]: it for it in items}
    cfg = make_cfg(args, gen_tokens=0)
    prod_dt = args.dtype if args.dtype != "auto" else (
        "fp16" if (args.device == "cuda" or
                   (args.device == "auto" and torch.cuda.is_available())) else "fp32")
    out: dict = {"holdout": prov, "anchors_json": args.anchors_json,
                 "production_dtype": prod_dt,
                 "config": _cfg_json(cfg, args), "paths": {}, "n_items_target": args.numfloor_items}

    for model, run in anchors["runs"].items():
        if model not in args.models:
            continue
        pool = [r for r in run["records"] if "error" not in r and r["id"] in by_id]
        recs = _numfloor_select(pool, by_id, args.numfloor_items)
        if not recs:
            continue
        print(f"[numfloor] {model}: {len(recs)} 題 "
              f"{[(r['id'], r.get('level'), round(_sel_ds(r) or 0, 3)) for r in recs]}",
              flush=True)
        per_dtype: dict[str, dict] = {}
        for dt in ("fp32", "fp16"):
            def _go(model_obj, device, dtype, load_s):
                rows = []
                for r in recs:
                    row = {"id": r["id"], "R": {}, "R_batch": {},
                           "y_patched": {}, "y_patched_batch": {}, "denom": {}}
                    try:
                        (tok, clean, cache, ct, corrupted, bands,
                         positions) = cp.rebuild_selected(model_obj, by_id[r["id"]], cfg, r)
                        for bname, layers in bands.items():
                            p1 = cp.patched_readout(model_obj, ct, tok.n_prompt, cache,
                                                    layers, positions, cfg.hook_kind,
                                                    cfg.launch_k, batch=1)
                            rr = cp.recovery_ratio(clean.launch_mean, corrupted.launch_mean,
                                                   p1.launch_mean, cfg.denom_eps)
                            row["R"][bname] = rr["R"]
                            row["denom"][bname] = rr["denom"]
                            # v2.8 §2 的量是**分子 ΔY = Y_patched − Y_corrupted**，
                            # 不是 Y_patched。跨 dtype 時 Y_corrupted 也會動，
                            # 只比 Y_patched 會把腐蝕端的浮點差漏掉。
                            row["y_patched"][bname] = p1.launch_mean - corrupted.launch_mean
                            pb = cp.patched_readout(model_obj, ct, tok.n_prompt, cache,
                                                    layers, positions, cfg.hook_kind,
                                                    cfg.launch_k, batch=args.numfloor_batch)
                            row["R_batch"][bname] = cp.recovery_ratio(
                                clean.launch_mean, corrupted.launch_mean,
                                pb.launch_mean, cfg.denom_eps)["R"]
                            row["y_patched_batch"][bname] = pb.launch_mean - corrupted.launch_mean
                    except Exception as e:
                        row["error"] = repr(e)
                    rows.append(row)
                return {"device": device, "dtype": dtype, "rows": rows}
            try:
                per_dtype[dt] = with_model(model, dt, _go, args.device)
            except Exception as e:
                per_dtype[dt] = {"error": repr(e)}
                print(f"[numfloor] {model} {dt} 失敗：{e!r}")

        def _rows(getter_a, getter_b, tag):
            res = []
            for dt, blk in per_dtype.items():
                for row in blk.get("rows", []):
                    for b in (row.get("R") or {}):
                        v, v2 = getter_a(row, b), getter_b(row, b)
                        dn = (row.get("denom") or {}).get(b)
                        if (v is None or v2 is None or not np.isfinite(v)
                                or not np.isfinite(v2)):
                            continue
                        res.append({"dtype": dt, "id": row["id"], "band": b,
                                    "abs_diff": abs(v - v2), "denom": dn,
                                    "denom_unstable": bool(dn is None
                                                           or abs(dn) < args.denom_eps)})
            return res

        # 路徑 B：同 dtype，batch 1 vs N
        pb_R = _rows(lambda r, b: r["R"].get(b), lambda r, b: r["R_batch"].get(b), "R")
        pb_Y = _rows(lambda r, b: r["y_patched"].get(b),
                     lambda r, b: r["y_patched_batch"].get(b), "Y")
        # 路徑 A：同 batch=1，fp32 vs fp16
        def _pair(field):
            a = {(r["id"], b): v for r in per_dtype.get("fp32", {}).get("rows", [])
                 for b, v in (r.get(field) or {}).items()}
            c = {(r["id"], b): v for r in per_dtype.get("fp16", {}).get("rows", [])
                 for b, v in (r.get(field) or {}).items()}
            dn = {(r["id"], b): v for r in per_dtype.get("fp32", {}).get("rows", [])
                  for b, v in (r.get("denom") or {}).items()}
            res = []
            for k in sorted(set(a) & set(c)):
                x, y = a[k], c[k]
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                d = dn.get(k)
                res.append({"id": k[0], "band": k[1], "abs_diff": abs(x - y),
                            "denom": d,
                            "denom_unstable": bool(d is None or abs(d) < args.denom_eps)})
            return res
        pa_R, pa_Y = _pair("R"), _pair("y_patched")

        def _ub(rows, stable_only=False):
            v = [r["abs_diff"] for r in rows if not (stable_only and r["denom_unstable"])]
            return max(v) if v else None

        allR, allY = pa_R + pb_R, pa_Y + pb_Y
        out["paths"][model] = {
            "per_dtype": per_dtype,
            "path_A_dtype": {"n_R": len(pa_R), "rows_R": pa_R, "rows_Y": pa_Y,
                             "max_abs_diff_R": _ub(pa_R), "max_abs_diff_Y_nats": _ub(pa_Y)},
            "path_B_batch": {"batch": args.numfloor_batch, "n_R": len(pb_R),
                             "rows_R": pb_R, "rows_Y": pb_Y,
                             "max_abs_diff_R": _ub(pb_R), "max_abs_diff_Y_nats": _ub(pb_Y)},
            # **三個底線都給，不挑。** 理由見 note。
            "numerical_floor_R": _ub(allR),
            "numerical_floor_R_denom_stable_only": _ub(allR, stable_only=True),
            # ── 附錄 F③：底線＝**只算 batch 路徑**，dtype 層剔除 ──
            # dtype（fp32 vs fp16）量的是「換一種權重處理精度」，那不是同一條
            # 管線的執行噪音，是**兩條不同精度的管線**；把它算進底線會讓底線
            # 吃到一個生產上不會發生的差異。改標 precision_robustness 照報。
            # 底線必須量在**生產實際用的 dtype** 上。numfloor 兩個 dtype 都跑，
            # 但若把 fp16 的 batch 列也算進來，底線會被 fp16 拉高約 1000 倍，
            # 「搬 fp32 把底線縮回 1e-4」這件事就被自己抵銷掉了。
            "numerical_floor_dy_nats": _ub([r for r in pb_Y
                                            if r["dtype"] == prod_dt]),
            "production_dtype": prod_dt,
            "numerical_floor_basis": (
                f"batch 路徑、**生產 dtype={prod_dt}**（同 dtype、同介入、同輸入）"),
            "floor_by_dtype": {dt: _ub([r for r in pb_Y if r["dtype"] == dt])
                               for dt in sorted({r["dtype"] for r in pb_Y})},
            "precision_robustness_dy_nats": _ub(pa_Y),
            "precision_robustness_note": (
                "fp32 vs fp16 的差＝兩條不同精度管線的落差，**不計入底線**，"
                "另行照報（附錄 F③）"),
            "floor_by_delta_s_bin": _floor_bins([r for r in pb_Y if r["dtype"] == prod_dt], recs),
            "numerical_floor_Y_nats": _ub(allY),   # 舊鍵名（含 dtype 層），保留相容
            "denom_eps": args.denom_eps,
            # theory 附錄 B2.2 的唯一數值門檻，**單位是 nats**：
            # 「若兩條路徑差異 < 1e-3 nats，即可宣告數值噪音相對於任何實質
            # 效應可忽略，並把這件事寫進報告」。這是規格給的宣告條件，
            # 所以在 nats 尺度上判；判定本身仍留給分析部（verdict=null）。
            "negligible_threshold_nats": args.negligible_nats,
            "below_negligible_threshold": (
                None if _ub(allY) is None else bool(_ub(allY) < args.negligible_nats)),
            "verdict": None,
            "note": (
                "三個底線都給、器材部不挑，因為它們量的不是同一件事：\n"
                "① numerical_floor_R：規格字面（R 尺度）。但 **R 是比值，分母 Δs 小的時候"
                "分子上一點點數值差會被放大成很大的 ΔR**——本段實測就被單一小分母的題"
                "拉到 24.2，那不是雜訊變大，是除以小數。\n"
                "② ..._denom_stable_only：同樣是 R 尺度，但只收 |Δs| ≥ denom_eps 的列。\n"
                "③ numerical_floor_Y_nats：改量分子本身（patched 的 gold log-prob，nats），"
                "不受分母放大影響，但**與 R 尺度的殘差 SD 不能直接比**。\n"
                "要拿哪一個去跟 G1 的殘差 SD 比，是分析部的裁決。\n"
                "另：**CPU 上的路徑 B 可能恆為 0**（batch 維不改變規約順序），"
                "而 CPU 的 fp16 是模擬出來的、精度與 GPU 原生 fp16 不同——"
                "所以 CPU 這一輪只證明程式跑得動，**底線的數值必須在 GPU 上重量**。"),
        }
        # ── 下游契約（causal_smoke_eval.eval_noise_floor 讀 top-level
        #    float_path_diffs[].max_abs_diff_dy_nats，需 ≥3 筆）──
        # v2.8 §2：底線在 ΔY（nats）上，R 尺度僅描述。
        for row in pa_Y:
            out.setdefault("float_path_diffs", []).append(
                {"path": "dtype_fp32_vs_fp16", "model": model, "id": row["id"],
                 "band": row["band"], "max_abs_diff_dy_nats": row["abs_diff"]})
        for row in pb_Y:
            out.setdefault("float_path_diffs", []).append(
                {"path": "batch_composition", "model": model, "id": row["id"],
                 "band": row["band"], "dtype": row["dtype"],
                 "max_abs_diff_dy_nats": row["abs_diff"]})
        pm = out["paths"][model]
        print(f"[numfloor] {model}: 底線(batch,ΔY)={pm['numerical_floor_dy_nats']} nats"
              f" | precision-robustness(dtype)={pm['precision_robustness_dy_nats']} nats")
        for bn, bv in pm["floor_by_delta_s_bin"]["bins"].items():
            if bv["n"]:
                print(f"    箱 {bn:<9} n={bv['n']:2d} 底線={bv['max_abs_diff_dy_nats']:.6f}")
    return out


# ------------------------------------------------------- stage: provenance（無 GPU）

def stage_provenance(args, *_ignored) -> dict:
    """把 gen 段的續寫丟 infini-gram 算 frac_present。**CPU＋網路，不佔 GPU**。

    沿用 harness/verify_battery.py 的 check_novelty（同一支驗題器，同一個門檻
    語意），index 依模型家族選：Pythia→pile，OLMo→olmo_mix。
    """
    from verify_battery import check_novelty
    src = Path(args.gen_json)
    data = json.load(open(src, encoding="utf-8"))
    out = {"gen_json": str(src), "gen_run_id": data.get("run_id"),
           "probe_params": {"n_words": args.probe_words, "stride": args.probe_stride,
                            "source": "prereg §2 規格補完（6 詞窗、stride 4）；"
                                      "顯式傳參，未改 verify_battery 預設"},
           "runs": {}}
    for model, run in data["runs"].items():
        index_key = "olmo_mix" if "olmo" in model.lower() else "pile"
        rows = []
        for r in run["records"]:
            if "error" in r or "generations" not in r:
                continue
            row = {"id": r["id"], "level": r.get("level"), "index": index_key,
                   "frac_present": {}}
            for cond, text in r["generations"].items():
                nv = check_novelty(text, index_key,
                                   n_words=args.probe_words, stride=args.probe_stride)
                row["frac_present"][cond] = {
                    "frac_present": nv["frac_present"], "n_probes": nv["n_probes"],
                    "n_unknown": nv["n_unknown"], "max_count": nv["max_count"],
                    "n_words": len(text.split()),
                    # frac_present 的解析度＝1/n_probes。n_probes=2 時它只能取
                    # {0, .5, 1} 三個值，拿這種東西去跟連續的 R_gold 算相關，
                    # 算出來的 r 主要反映量化格線，不是兩軌的關係。
                    "too_few_probes": nv["n_probes"] < args.min_probes,
                    "resolution": (1.0 / nv["n_probes"]) if nv["n_probes"] else None}
                print(f"  {r['id']:12s} {cond:16s} frac_present="
                      f"{nv['frac_present']} (n={nv['n_probes']}, unk={nv['n_unknown']})",
                      flush=True)
            rows.append(row)
        thin = [(r["id"], c) for r in rows for c, v in r["frac_present"].items()
                if v["too_few_probes"]]
        out["runs"][model] = {"index": index_key, "rows": rows,
                              "min_probes": args.min_probes, "too_few_probes": thin}
        if thin:
            print(f"[warn] {model}: {len(thin)} 個續寫的探針數 < {args.min_probes}，"
                  f"frac_present 解析度不足，glink 會分開報：{thin[:6]}")
    return out


# ---------------------------------------------------------- stage: glink（無 GPU）

def stage_glink(args, *_ignored) -> dict:
    """⑤ G-link：兩軌相關 r。**v2.13 F 案：算在分子 ΔY（nats）尺度上。**

    尺度（F 案，比 C2 更硬）：共用隨機分母＋非零均值分子 ⇒ 比值假相關——
    實測兩軌完全獨立時 **R 尺度單帶有 25–38% 機率假性衝破 0.95**（誤砍方向），
    且 D1 誤砍率表的 Fisher-z SD 在比值尺度實測是名目的 2.2 倍。
    §7.2 凍的是**門檻 0.95**，不是尺度；尺度從未被凍——補完非改判準。
      gold 軌分子 ΔY_gold = Y_patched − Y_corrupted（nats）
      溯源軌分子 ΔY_prov = f_patched − f_corrupted（frac_present 差）

    砍軌規則（v2.11 ①）：**min-of-valid-bands**——所有有效帶皆 |r| > 0.95
    才砍溯源軌（max 在真 r=0.90 時誤砍 36.5%、min 僅 0.3%；後果不對稱：
    誤砍＝丟掉唯一能識別記憶的出口，漏砍＝多花 ~2h）。**殺人的規則要一致決。**
    兩條伴隨凍結：①**有效帶 <2 不得動用砍軌規則**，只報 r 延後；
    ②0.95 **不是 50/50 點**（真 r=0.95 時 min-of-3 僅點燃 12.5%），
    規則天生保守，不得讀成中位判準。

    **判定仍不在這裡**：這裡給逐帶 r、有效帶數、與 min，`verdict` 為 null。
    """
    anchors = json.load(open(args.anchors_json, encoding="utf-8"))
    provdata = json.load(open(args.provenance_json, encoding="utf-8"))
    # gold 軌取**分子**（v2.13 F 案），不再取 R
    gold: dict[tuple, float] = {}
    for model, run in anchors["runs"].items():
        for r in run["records"]:
            if "error" in r:
                continue
            for p_ in r.get("patches", []):
                gold[(model, r["id"], p_["band"])] = p_["launch"]["numer"]

    rows = []
    for model, run in provdata["runs"].items():
        for row in run["rows"]:
            fp = row["frac_present"]
            if "clean" not in fp or "corrupted" not in fp:
                continue
            fc, fk = fp["clean"]["frac_present"], fp["corrupted"]["frac_present"]
            if fc is None or fk is None:
                continue
            for cond, v in fp.items():
                if not cond.startswith("patched::") or v["frac_present"] is None:
                    continue
                band = cond.split("::", 1)[1]
                n_probes = [fp["clean"]["n_probes"], fp["corrupted"]["n_probes"],
                            v["n_probes"]]
                g = gold.get((model, row["id"], band))
                rows.append({"model": model, "id": row["id"], "level": row["level"],
                             "band": band,
                             # 兩軌都用分子（v2.13 F 案）
                             "dy_gold": g,
                             "dy_prov": v["frac_present"] - fk,
                             "f_clean": fc, "f_corrupted": fk,
                             "f_patched": v["frac_present"],
                             "n_probes_min": min(n_probes),
                             "thin_probes": min(n_probes) < args.min_probes})

    by_band: dict[str, dict] = {}
    for band in sorted({r["band"] for r in rows}):
        sel = [r for r in rows if r["band"] == band and r["dy_gold"] is not None]
        clean_rows = [r for r in sel if not r["thin_probes"]]
        r_all = pearson([r["dy_gold"] for r in sel], [r["dy_prov"] for r in sel])
        r_ok = pearson([r["dy_gold"] for r in clean_rows],
                       [r["dy_prov"] for r in clean_rows])
        # 有效帶＝探針足夠的子集算得出 r 的帶（pearson 內含 n≥3 與零變異守門）
        by_band[band] = {"n_all": len(sel), "n_enough_probes": len(clean_rows),
                         "r_all": r_all, "r_enough_probes": r_ok,
                         "r_used": r_ok, "is_valid_band": r_ok is not None,
                         "scale": "delta_y_nats（v2.13 F 案）"}

    valid = {b: v["r_used"] for b, v in by_band.items() if v["is_valid_band"]}
    n_valid = len(valid)
    min_abs_r = min((abs(x) for x in valid.values()), default=None)
    out = {
        "anchors_json": args.anchors_json, "provenance_json": args.provenance_json,
        "scale": "delta_y_nats（兩軌都取分子；v2.13 F 案）",
        "rows": rows, "by_band": by_band,
        "valid_bands": sorted(valid), "n_valid_bands": n_valid,
        "per_band_r": valid,
        # ── 下游契約：判定器讀 top-level r_dgold_dprov ──
        # v2.11 ①：合成＝min-of-valid-bands（不是 max）
        "r_dgold_dprov": min_abs_r,
        "r_source": (f"min-of-valid-bands（{n_valid} 個有效帶：{sorted(valid)}）"
                     if n_valid else "無有效帶"),
        "redundancy_threshold": 0.95,
        "kill_rule_applicable": bool(n_valid >= 2),
        "kill_rule_note": (
            "有效帶 <2 → **不得動用砍軌規則**，只報 r 延後（v2.11 ①伴隨凍結①）"
            if n_valid < 2 else
            "所有有效帶皆 |r|>0.95 才砍（min-of-valid-bands，一致決）"),
        "threshold_note": ("0.95 **不是 50/50 點**：真 r=0.95 時 min-of-3 只點燃 12.5%，"
                           "規則天生保守，**不得讀成中位判準**（v2.11 ①伴隨凍結②）"),
        "verdict": None,
        "note": "砍不砍是分析部/作者的裁決；這裡只給逐帶 r、有效帶數與 min。",
    }
    return out


# ---------------------------------------------------------------- main

def build_argparser():
    ap = argparse.ArgumentParser(
        description="OMTR 因果階段冒煙驅動器（CAUSAL_PREREG_v1 §8）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["timing", "anchors", "l0m", "gen", "numfloor",
                             "provenance", "glink", "all"])
    ap.add_argument("--run-id", default=None, help="輸出檔名尾碼，預設 UTC 時戳")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="輸出目錄。**CPU 形狀測試請改到別的目錄**，"
                         "別讓玩具模型的數字跟真冒煙結果躺在同一層")
    # 保留題
    ap.add_argument("--holdout", default=None, help="保留題 JSON（§9 正規路徑）")
    ap.add_argument("--holdout-ids", default=None, help="逗號分隔 id，從 battery 撈（暫用路徑）")
    ap.add_argument("--battery", default=str(PROJ / "battery" / "battery.json"))
    ap.add_argument("--battery-l0p", default=str(PROJ / "battery" / "battery_l0p.json"))
    ap.add_argument("--allow-nonconforming-holdout", action="store_true",
                    help="放行不符 §9 構成的清單（旗標會寫進輸出 JSON）")
    ap.add_argument("--l0m-pairs", default=None, help="L0M 12 對候選檔")
    # 模型
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto＝有卡就吃卡。**CPU 流程測試必須顯式 --device cpu**")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"])
    ap.add_argument("--fp16-models", default="EleutherAI/pythia-2.8b",
                    help="強制跑 fp16 的模型（逗號分隔）。這些模型**自動排除於 "
                         "G1／底線**（附錄 H①：2.8b fp32 需 11.08GB 裝不下 8GB）")
    ap.add_argument("--gen-models", type=int, default=1,
                    help="gen 段只跑前幾個模型（自由續寫關 KV cache，很貴）")
    # patch 設定
    ap.add_argument("--bands", default="early:0.25:0.45,mid:0.45:0.65,late:0.70:0.90",
                    help="相對深度層帶，格式 name:lo:hi 逗號分隔")
    ap.add_argument("--hook", default="resid_pre", choices=["resid_pre", "resid_post"])
    ap.add_argument("--radius", type=int, default=2, help="cue 鄰域半徑（R4 上限 3）")
    # ── v2.6：相對帶 + rank 護欄（依 CAUSAL_PREREG_v1 v2.6 更正／theory_v3_final 附錄 B）
    ap.add_argument("--quantile", type=float, default=cp.CORRUPT_QUANTILE,
                    help="v2.8 分位制：選存活候選中 Δs 的 q 分位者（凍結 0.75，慣例）")
    ap.add_argument("--no-expand-pool", action="store_true",
                    help="關掉擴池輪（v2.14 附錄 G3 預設開啟）")
    ap.add_argument("--expand-max-candidates", type=int, default=48,
                    help="擴池每題候選上限（v2.14 凍結 48；實測存活率 0.374 → P(≥12)=.976）")
    ap.add_argument("--min-survivors", type=int, default=cp.MIN_SURVIVORS,
                    help="存活候選 <N → pool_too_small 照實報（v2.8 凍結 12）")
    ap.add_argument("--adequacy-multiple", type=float, default=cp.ADEQUACY_MULTIPLE,
                    help="適足下限＝選中 Δs ≥ N × 數值底線（v2.8 凍結 10）")
    ap.add_argument("--numerical-floor-nats", type=float, default=None,
                    help="從 numfloor 段注入的數值底線（nats）；不給就不判適足性")
    ap.add_argument("--rank-guard", type=int, default=50,
                    help="絕對護欄上限：腐蝕後 gold 起手窗排名須在 top-N（v2.6）；<=0 關閉")
    ap.add_argument("--gate-model-frac", type=float, default=0.60,
                    help="逐模型壞題比例超過此值必須標記點名（v2.13 D②）")
    ap.add_argument("--gate-frac", type=float, default=0.40,
                    help="v2.8 分支規則：壞題（pool_too_small 或未達適足下限）>N → 多 token 分支")
    ap.add_argument("--cross-word-k", type=int, default=3,
                    help="每題取幾個進帶替換詞（v2.6 §3 選 k 用，非噪音）")
    ap.add_argument("--cross-word-items", type=int, default=8,
                    help="跨替換詞量測用前幾題（v2.8 §4 起始配置：8 題 × 3 詞，規劃 dof=15）")
    ap.add_argument("--numfloor-items", type=int, default=3,
                    help="數值底線用幾題（v2.6 §2：≥3 題）")
    ap.add_argument("--numfloor-batch", type=int, default=3,
                    help="數值底線路徑 B 的 batch size")
    ap.add_argument("--negligible-nats", type=float, default=1e-3,
                    help="數值噪音可忽略的宣告門檻（nats）——theory 附錄 B2.2 寫死的 1e-3")
    ap.add_argument("--denom-eps", type=float, default=cp.DENOM_EPS,
                    help="R 的分母低於此值就標 denom_unstable（nats，佔位值）")
    ap.add_argument("--n-cue", type=int, default=6)
    ap.add_argument("--n-repl", type=int, default=4)
    # --repeats 已退場：它量的是「換替換詞」，理論部 v2.6 判定那是效應不是噪音。
    # 換詞的量測改由 --cross-word-k/--cross-word-items 承接（用途＝選 k）；
    # 噪音底線改由 --stage numfloor 量。
    ap.add_argument("--seed", type=int, default=20260822, help="顯式 seed（§5 凍結值）")
    # 生成 / 溯源
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--gen-sample", action="store_true", help="用取樣續寫（預設貪婪，決定性）")
    ap.add_argument("--gen-json", default=None, help="provenance 段的輸入")
    ap.add_argument("--anchors-json", default=None, help="glink 段的輸入")
    ap.add_argument("--provenance-json", default=None, help="glink 段的輸入")
    ap.add_argument("--probe-words", type=int, default=6,
                    help="溯源探針窗長（prereg §2：6 詞）")
    ap.add_argument("--probe-stride", type=int, default=4,
                    help="溯源探針 stride。**顯式 4**（prereg §2 規格補完：6 詞窗、"
                         "stride 4）。verify_battery 的預設是 2 且**不動**——"
                         "那是已發表工具，改呼叫端顯式傳參（v2.13 C 案）")
    ap.add_argument("--prov-eps", type=float, default=0.05)
    ap.add_argument("--min-probes", type=int, default=8,
                    help="frac_present 至少要幾個探針才算有解析度（低於此值另外分開報）")
    # 計時
    ap.add_argument("--window-minutes", type=float, default=30.0,
                    help="窗口 A 預算（分鐘）。投影超出時 gen 段自動延後（v2.13 I 案）")
    ap.add_argument("--force-gen", action="store_true",
                    help="即使投影超窗也照跑 gen（覆寫 I 案的自動延後）")
    ap.add_argument("--timing-n", type=int, default=20)
    ap.add_argument("--timing-warmup", type=int, default=3)
    return ap


def _timing_says_defer(written: dict, args) -> bool:
    """讀同一輪 timing 段的投影，決定 gen 要不要延到下一窗（v2.13 I 案）。

    讀不到投影時**保守照跑**——延後是省時間的最佳化，不是安全機制，
    讀不到就不該自作主張砍掉一整段量測。
    """
    path = written.get("timing")
    if not path:
        return False
    try:
        d = json.load(open(path, encoding="utf-8"))
        return bool((d.get("projection") or {}).get("defer_gen_to_next_window"))
    except Exception:
        return False


def main():
    args = build_argparser().parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if args.radius > cp.CUE_RADIUS_MAX:
        raise SystemExit(f"--radius {args.radius} 超過 R4 上限 {cp.CUE_RADIUS_MAX}")

    print(f"=== OMTR causal smoke | stage={args.stage} run_id={run_id}")
    print(f"    spec: {cp.SPEC_VERSION}")
    print(f"    cuda_available={torch.cuda.is_available()} --device={args.device} "
          f"models={args.models}")
    t0 = time.time()

    if args.stage in ("provenance", "glink"):
        if args.stage == "provenance":
            if not args.gen_json:
                raise SystemExit("provenance 段要 --gen-json")
            dump("provenance", run_id, stage_provenance(args), Path(args.out_dir))
        else:
            if not (args.anchors_json and args.provenance_json):
                raise SystemExit("glink 段要 --anchors-json 與 --provenance-json")
            dump("glink", run_id, stage_glink(args), Path(args.out_dir))
        print(f"=== done in {time.time()-t0:.1f}s")
        return

    items, prov = load_holdout(args)
    print(f"    保留題 {len(items)}：{prov['level_counts']}")
    stages = ["timing", "anchors", "l0m", "gen"] if args.stage == "all" else [args.stage]
    written, deferred = {}, []
    for st in stages:
        # I 案：timing 的投影說超窗就把 gen 移下一窗（gen 非判定必要）
        if st == "gen" and not args.force_gen and _timing_says_defer(written, args):
            print("[gen] 依 timing 投影超出窗口預算，**本窗跳過 gen**"
                  "（--force-gen 可覆寫）。G1／校準／r／L0M 都不吃 gen，"
                  "只有溯源軌 G-link 需要它。")
            deferred.append("gen")
            continue
        fn = {"timing": stage_timing, "anchors": stage_anchors,
              "l0m": stage_l0m, "gen": stage_gen,
              "numfloor": stage_numfloor}[st]
        st0 = time.time()
        try:
            payload = fn(args, items, prov)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{st}] 失敗：{e!r}")
            payload = {"error": repr(e), "holdout": prov}
        payload["stage_seconds"] = time.time() - st0
        written[st] = str(dump(st, run_id, payload, Path(args.out_dir)))
    print(f"=== done in {time.time()-t0:.1f}s")
    for st, p in written.items():
        print(f"    {st}: {p}")
    if deferred:
        print(f"    ⏭ 本窗延後：{deferred}（投影超窗；下一窗補跑 --stage gen）")
    if "gen" in written:
        print("\n下一步（不佔 GPU，可白天跑）：")
        print(f"  .venv/Scripts/python.exe harness/causal_smoke.py --stage provenance "
              f"--run-id {run_id} --gen-json {written['gen']}")
        if "anchors" in written:
            print(f"  .venv/Scripts/python.exe harness/causal_smoke.py --stage glink "
                  f"--run-id {run_id} --anchors-json {written['anchors']} "
                  f"--provenance-json {Path(args.out_dir) / f'smoke_provenance_{run_id}.json'}")


if __name__ == "__main__":
    main()
