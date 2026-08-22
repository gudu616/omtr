"""重建裁決池的配對清單，落成機器可讀檔（窗口 B 驅動器的輸入）。

依據規格版本：CAUSAL_PREREG_v1 第一段凍結 ＋ v2.6–v2.16 增補裁決。

**為什麼要這支**：`theory_v3_regimes.json` 只存了逐 run 的 **L0 側遮罩**
（哪些 L0 題入選），沒有存「哪個 L0 配哪個 L0P」。主實驗要逐配對跑，
所以得把配對關係重建出來。

**單一出處紀律**：配對邏輯**直接 import 理論部的 `theory_v3_regimes`**
（`load` / `compatible` / `max_match`），不自己抄一份——抄一份就會有兩個
版本各自漂移。該模組的 `main()` 有 `__name__` 守衛，import 不會觸發重算。

重建結果會**逐 run 對照 regimes JSON 的 n_pairs 與遮罩**，對不上就報錯不落檔
（配對數是凍結數字，重建對不上代表我這支寫錯了，不是資料變了）。

L0M 裁決池 8 對另從 `l0m_candidates.json` 取（pool=adjudication）。

用法：
    .venv/Scripts/python.exe harness/build_pairs.py
輸出：planning/battery_expansion/adjudication_pairs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "planning" / "office_reports"))

import theory_v3_regimes as TR  # noqa: E402  單一出處：配對邏輯用理論部那份

OUT = PROJ / "planning" / "battery_expansion" / "adjudication_pairs.json"
REGIMES_JSON = PROJ / "planning" / "office_reports" / "theory_v3_regimes.json"
L0M_JSON = PROJ / "planning" / "battery_expansion" / "l0m_candidates.json"
# v2.18／R1：溯源軌的 L0↔L0N 配對。裁決文寫的路徑是 planning/ 底下，
# 理論部實際落在 planning/office_reports/——以實際落點為準（已核對內容）。
L0N_JSON = PROJ / "planning" / "office_reports" / "theory_v3_l0n_pairs.json"
HOLDOUT = PROJ / "planning" / "battery_expansion" / "holdout_12.json"


def holdout_ids() -> set[str]:
    """保留題永不進主裁決（§9）——重建出的配對必須把它們濾掉。"""
    if not HOLDOUT.exists():
        return set()
    d = json.load(open(HOLDOUT, encoding="utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    ids = {it["id"] for it in items}
    ids |= {it["twin_of"] for it in items if it.get("twin_of")}
    return ids


def build_regime(name: str, gold: float, ent: float | None, hold: set[str]) -> dict:
    """逐 run 重建 (L0, L0P) 配對。回傳 {tag: [{l0, l0p, dgold, dent}, ...]}。"""
    per_run, counts = {}, {}
    for tag in TR.TAGS:
        a, b = TR.load(tag, "L0"), TR.load(tag, "L0P")
        ok = TR.compatible(a, b, gold, ent)
        matched = TR.max_match(ok)
        rows = []
        for u, v in matched:
            ra, rb = a[u], b[v]
            if ra["id"] in hold or rb["id"] in hold:
                continue                      # §9：保留題不得進主裁決
            rows.append({
                "l0": ra["id"], "l0p": rb["id"],
                "dgold": float(ra[TR.X] - rb[TR.X]),
                "dent": float(ra[TR.Y] - rb[TR.Y]),
                "l0_gold": float(ra[TR.X]), "l0p_gold": float(rb[TR.X]),
            })
        rows.sort(key=lambda r: (r["l0"], r["l0p"]))
        per_run[tag] = rows
        counts[tag] = len(matched)            # 未濾保留題的原始配對數，用來對帳
    return {"caliper": {"gold": gold, "ent": ent},
            "per_run": per_run, "raw_match_counts": counts}


def crosscheck(built: dict, regime_key: str) -> list[str]:
    """與凍結的 regimes JSON 對帳：配對數與 L0 遮罩都要一致。"""
    problems = []
    if not REGIMES_JSON.exists():
        return ["theory_v3_regimes.json 不存在，無法對帳"]
    ref = json.load(open(REGIMES_JSON, encoding="utf-8"))
    ref_n = {r["tag"]: r["n_pairs"] for r in ref["regimes"][regime_key]["per_model"]}
    ref_masks = ref["masks"][regime_key]
    for tag, n in built["raw_match_counts"].items():
        if tag in ref_n and n != ref_n[tag]:
            problems.append(f"{tag}: 重建配對數 {n} ≠ 凍結值 {ref_n[tag]}")
        # 遮罩比對用「未濾保留題」的 L0 集合才公平
        got = sorted({r["l0"] for r in built["per_run"][tag]})
        ref_mask = sorted(ref_masks.get(tag, []))
        if ref_mask and not set(got) <= set(ref_mask):
            extra = sorted(set(got) - set(ref_mask))
            problems.append(f"{tag}: 重建出凍結遮罩之外的 L0 題 {extra}")
    return problems


def l0m_pairs(hold: set[str]) -> dict:
    """L0M 裁決池配對（pool=adjudication）。保留集那幾對要濾掉。"""
    if not L0M_JSON.exists():
        return {"status": "MISSING", "path": str(L0M_JSON), "pairs": []}
    d = json.load(open(L0M_JSON, encoding="utf-8"))
    raw = d.get("pairs") or []
    out, skipped = [], []
    for p in raw:
        pool = (p.get("pool") or "").lower()
        # l0 / l0m 是**巢狀題目物件**（不是字串 id）——第一版我當成字串，
        # 直接 TypeError。取 .id，取不到就整筆列進 skipped 讓人看見，
        # 不猜、不靜默跳過。
        l0 = (p.get("l0") or {}).get("id") if isinstance(p.get("l0"), dict) else p.get("l0")
        l0m = (p.get("l0m") or {}).get("id") if isinstance(p.get("l0m"), dict) else p.get("l0m")
        if not l0 or not l0m:
            skipped.append({"pair_id": p.get("pair_id"), "reason": "取不到 id"})
            continue
        if "adjud" not in pool:
            skipped.append({"pair_id": p.get("pair_id"), "reason": f"pool={pool!r} 非裁決池"})
            continue
        if l0m in hold or l0 in hold:
            skipped.append({"pair_id": p.get("pair_id"), "reason": "保留題（§9 不得進主裁決）"})
            continue
        out.append({"pair_id": p.get("pair_id"), "l0m": l0m, "l0": l0,
                    "pool": pool, "in_main_12": p.get("in_main_12")})
    return {"status": "ok" if out else "EMPTY_AFTER_FILTER",
            "n": len(out), "pairs": out, "n_skipped": len(skipped), "skipped": skipped,
            "source": str(L0M_JSON.relative_to(PROJ)),
            "declared_n_pairs": d.get("n_pairs"),
            "note": "只收 pool 含 adjud 者；被濾掉的逐筆列在 skipped，不靜默丟"}


def battery_ids() -> dict:
    """題庫 id → item（只為了驗「配對表點到的題真的存在且家族對得上」）。"""
    idx = {}
    for p in [PROJ / "battery" / "battery.json", PROJ / "battery" / "battery_l0p.json"]:
        if p.exists():
            for it in json.load(open(p, encoding="utf-8")).get("items", []):
                idx[it["id"]] = it
    return idx


def family_rep(tag: str) -> str:
    """L0/L0N 的 eligible_models 寫的是**語料家族代表**不是逐 checkpoint
    （run_pilot.py:291 原文）。所以 410m 要查的是 pythia-1.4b 那筆。"""
    return ("allenai/OLMo-2-0425-1B" if "olmo" in tag.lower()
            else "EleutherAI/pythia-1.4b")


def l0n_pairs(hold: set[str]) -> dict:
    """溯源軌的 L0↔L0N 配對（v2.18／R1）。

    清單由理論部定版（最大匹配有平手解——計數唯一但配對不唯一，所以**不能**
    讓這支自己重配，會得到合法但不同的解）。這支只做三件事：讀、對帳、落檔。
    """
    if not L0N_JSON.exists():
        return {"status": "MISSING", "path": str(L0N_JSON), "per_model": {}}
    d = json.load(open(L0N_JSON, encoding="utf-8"))
    bat = battery_ids()
    exp = d.get("frozen_counts_expected", {})
    per_model, problems, skipped, distinct = {}, [], [], set()
    for tag in TR.TAGS:
        blk = d["per_model"].get(tag)
        if blk is None:
            problems.append(f"{tag}: 清單裡沒有這個 run")
            continue
        if blk.get("n_pairs") != len(blk["pairs"]):
            problems.append(f"{tag}: 宣告 n_pairs {blk.get('n_pairs')} "
                            f"≠ 實際 {len(blk['pairs'])} 筆")
        rows = []
        for pr in blk["pairs"]:
            l0, l0n = pr["l0"], pr["l0n"]
            miss = [i for i in (l0, l0n) if i not in bat]
            if miss:
                problems.append(f"{tag}: 題庫查無 {miss}")
                continue
            if l0 in hold or l0n in hold:
                skipped.append({"tag": tag, "l0": l0, "l0n": l0n,
                                "reason": "保留題（§9 不得進主裁決）"})
                continue
            rep = family_rep(tag)
            for i in (l0, l0n):
                el = bat[i].get("eligible_models")
                if el and rep not in el:
                    problems.append(f"{tag}: {i} 的 eligible_models {el} "
                                    f"不含家族代表 {rep}")
            rows.append({"l0": l0, "l0n": l0n, "dgold": pr.get("dgold")})
            distinct.add(l0n)
        per_model[tag] = rows
    # 凍結計數對帳：pythia 四個照 TAGS 序、olmo 一個
    got_py = [len(per_model.get(t, [])) for t in TR.TAGS if "olmo" not in t.lower()]
    got_ol = [len(per_model.get(t, [])) for t in TR.TAGS if "olmo" in t.lower()]
    if exp.get("pythia") and got_py != list(exp["pythia"]):
        problems.append(f"pythia 配對數 {got_py} ≠ 凍結值 {exp['pythia']}")
    if exp.get("olmo") is not None and got_ol and got_ol[0] != exp["olmo"]:
        problems.append(f"olmo 配對數 {got_ol[0]} ≠ 凍結值 {exp['olmo']}")
    return {"status": "ok" if not problems else "CROSSCHECK_FAILED",
            "source": str(L0N_JSON.relative_to(PROJ)),
            "pairing_defined_by": "理論部定版清單（最大匹配平手解不唯一，本支不重配）",
            "n_distinct_l0n": len(distinct),
            "distinct_l0n": sorted(distinct),
            "per_model": per_model, "problems": problems,
            "n_skipped": len(skipped), "skipped": skipped}


def main() -> int:
    out_path = OUT
    argv = sys.argv[1:]
    if "--out" in argv:
        out_path = Path(argv[argv.index("--out") + 1])
    hold = holdout_ids()
    print(f"保留題（不得進主裁決）{len(hold)} 個")
    out = {"spec_version": "CAUSAL_PREREG_v1 第一段凍結＋v2.6–v2.18",
           "built_by": "harness/build_pairs.py",
           "pairing_logic_source": "planning/office_reports/theory_v3_regimes.py（直接 import，不抄）",
           "holdout_excluded": sorted(hold),
           "regimes": {}, "crosscheck": {}}
    for key, (g, e) in {"A_gold30": (0.30, None), "B_dual": (0.20, 0.30)}.items():
        built = build_regime(key, g, e, hold)
        probs = crosscheck(built, key)
        out["regimes"][key] = built
        out["crosscheck"][key] = probs or "OK：逐 run 配對數與遮罩皆與凍結值一致"
        tot = sum(len(v) for v in built["per_run"].values())
        print(f"{key}: 逐 run 配對 "
              f"{ {t: len(v) for t, v in built['per_run'].items()} } 合計 {tot}")
        for p in probs:
            print("  ⚠ " + p)
    out["l0m_adjudication"] = l0m_pairs(hold)
    print(f"L0M 裁決池：{out['l0m_adjudication'].get('n', 0)} 對 "
          f"({out['l0m_adjudication']['status']})")
    out["l0n_provenance"] = l0n_pairs(hold)
    ln = out["l0n_provenance"]
    print(f"L0N 溯源軌：逐 run "
          f"{ {t: len(v) for t, v in ln.get('per_model', {}).items()} }"
          f" distinct L0N {ln.get('n_distinct_l0n')} ({ln['status']})")
    for p in ln.get("problems", []):
        print("  ⚠ " + p)

    hard = [p for k, v in out["crosscheck"].items() if isinstance(v, list) for p in v]
    hard += ln.get("problems", [])
    if hard:
        print("\n⛔ 對帳不通過，**不落檔**——重建對不上凍結值代表這支寫錯了：")
        for p in hard:
            print("   " + p)
        return 2
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
