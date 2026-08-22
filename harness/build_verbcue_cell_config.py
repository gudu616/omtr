"""組出 2×2 四格的設定檔（§7-c 讀法B：verb/proper 兩列的兩欄都是「通用24」vs
「取自文體24」，欄尺寸配平）。統籌代理 2026-08-22 訊息要求：「設定檔結構預留
逐格池定義＋池尺寸欄位，免得配平裁下來又動骨架」——本檔就是那個結構。

【結構說明，2026-08-22 二版修正】「通用」欄的池對整格所有題共用（一份
flat list）；「修正」欄的池是**逐題各自一份**（每題從自己的 Gutenberg 全文
產生，見 theory_verbcue_poolgen_v1.md）。第一版把兩者用同一個 `pool_words`
欄位裝，修正欄裝進去的其實是逐題 dict，`pool_size` 因此被誤算成「題數」
不是「池尺寸」——**自己開檔核對時抓到，二版修正**：`pool_words`（flat list，
通用欄用）與 `pool_words_by_item`（{item_id:[words]}，修正欄用）分開放，
runner 讀取時依 column 欄位判斷用哪一個，不會混淆。

proper 臂 L0 題集用 8 題版（L0-14 union+遮罩，ec35d1e 裁定採用）。

用法：
  .venv/Scripts/python.exe harness/build_verbcue_cell_config.py
輸出：
  battery/verbcue_cell_config.json
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
POOLGEN_OUTPUT = PROJ / "battery" / "verbcue_pools_corrected.json"


def main():
    items = json.load(open(PROJ / "battery" / "verbcue_items.json", encoding="utf-8"))
    pools = json.load(open(PROJ / "battery" / "verbcue_pools.json", encoding="utf-8"))
    verb_pool = pools["verb"]["final_pool"]
    proper_pool = pools["proper"]["final_pool"]

    poolgen_available = POOLGEN_OUTPUT.exists()
    corrected = json.load(open(POOLGEN_OUTPUT, encoding="utf-8")) if poolgen_available else None

    verb_items = dict(items["verb"]["L0"], **items["verb"]["L0N"])
    # ⚠ 修正（2026-08-22，聚合階段抓到）：第一版只塞了 L0（8題版，ec35d1e
    # 裁定採用），漏了 L0N 的 5 題——導致第一次點火的 proper_universal／
    # proper_corrected 兩格完全沒收集 L0N 資料。跟 verbcue_precheck.py 的
    # proper_items_all 是同一種遺漏，但那次只修了診斷腳本，沒回頭查真正決定
    # 真跑內容的這支——本檔才是決定 run_ignition() 實際跑哪些題的來源。
    proper_items = dict(items["proper"]["L0_including_L0-14"], **items["proper"]["L0N"])

    def universal_cell(pool_class, item_dict, pool_words):
        return {"pool_class": pool_class, "column": "universal",
                "pool_size": len(pool_words), "pool_words": pool_words,
                "pool_words_by_item": None, "status": "ready",
                "n_items": len(item_dict), "item_ids": sorted(item_dict), "items": item_dict}

    def corrected_cell(pool_class, item_dict):
        if not poolgen_available:
            return {"pool_class": pool_class, "column": "corrected",
                    "pool_size": None, "pool_words": None, "pool_words_by_item": None,
                    "status": "pending_poolgen",
                    "n_items": len(item_dict), "item_ids": sorted(item_dict), "items": item_dict}
        per_item = corrected.get(pool_class, {})
        by_item = {}
        sizes = set()
        missing = []
        for iid in item_dict:
            entry = per_item.get(iid)
            if entry is None:
                missing.append(iid)
                continue
            by_item[iid] = entry["final_pool"]
            sizes.add(len(entry["final_pool"]))
        return {"pool_class": pool_class, "column": "corrected",
                "pool_size": ("mixed" if len(sizes) > 1 else (sizes.pop() if sizes else None)),
                "pool_size_by_item": {iid: len(w) for iid, w in by_item.items()},
                "pool_words": None, "pool_words_by_item": by_item,
                "status": ("ready" if not missing else "incomplete"),
                "missing_items": missing,
                "n_items": len(item_dict), "item_ids": sorted(item_dict), "items": item_dict}

    config = {
        "reading": "B（§7-c 裁定①，77316c7）：欄因子＝替換池與該題文體合不合；"
                  "verb×通用＝proper×通用＝24 詞、與文體無關；"
                  "verb×修正＝proper×修正＝取自該題文體、逐題各自的尺寸"
                  "（正常24；L0N-06/verb 因通用欄撞名截短為23，見 verbcue_genre_pools.md）",
        "proper_L0_item_set_used": "L0_including_L0-14（8題版，ec35d1e 裁定：union+遮罩、"
                                   "L0-14 只入 OLMo）",
        "cells": {
            "verb_universal": universal_cell("verb", verb_items, verb_pool),
            "proper_universal": universal_cell("proper", proper_items, proper_pool),
            "verb_corrected": corrected_cell("verb", verb_items),
            "proper_corrected": corrected_cell("proper", proper_items),
        },
        "poolgen_output_path": str(POOLGEN_OUTPUT), "poolgen_available": poolgen_available,
    }
    out_path = PROJ / "battery" / "verbcue_cell_config.json"
    json.dump(config, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for name, c in config["cells"].items():
        print(f"{name}: status={c['status']} pool_size={c.get('pool_size')} "
             f"n_items={c['n_items']}")
    print("->", out_path)


if __name__ == "__main__":
    main()
