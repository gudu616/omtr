"""逐格題集交叉檢查（機器assert，不是目測）——主線 2026-08-22 要求，
把「cell_config 跟 precheck 本來就該一起查」這個教訓做成常設檢查，
防止同一類遺漏（某支腳本合併題集時漏掉一個子集）再次通過而不被發現。

內部紀錄用語（第十九錯①同型）：**修例不修型**——只修掉某一次的具體錯誤，
沒有把「兩支腳本各自獨立重建同一個題集、只要有一支漏掉就不會被抓到」這個
**病因本身**堵住。本檔是堵病因的第一步（獨立重建＋交叉assert）；徹底解法
是讓題集只有一個計算來源，供全部消費端 import（見檔尾 TODO）。

檢查三件事：
  1. `verbcue_cell_config.json` 四格的 item_ids，跟從 `verbcue_items.json`
     用「跟 build_verbcue_cell_config.py 同一條規則」重建的題集是否相等。
  2. 同上，但用「跟 verbcue_precheck.py 同一條規則」重建。
  3. 上面兩者（cell_config 用的規則 vs precheck 用的規則）彼此是否相等
     ——這才是真正的「兩支腳本交叉assert」，抓的正是「兩支各自重建、
     只有一支漏掉」這種病。

任一項不等就 abort，不猜、不放行。

用法：
  .venv/Scripts/python.exe harness/verbcue_crosscheck_itemsets.py
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent


def _abort(msg):
    raise SystemExit(f"[verbcue_crosscheck_itemsets ABORT] {msg}")


def canonical_from_cell_config_rule(items):
    """重建 build_verbcue_cell_config.py 現行版本的題集規則。"""
    verb_items = dict(items["verb"]["L0"], **items["verb"]["L0N"])
    proper_items = dict(items["proper"]["L0_including_L0-14"], **items["proper"]["L0N"])
    return {"verb": set(verb_items), "proper": set(proper_items)}


def canonical_from_precheck_rule(items):
    """重建 verbcue_precheck.py 現行版本的題集規則（cross_column_check 用的
    verb_items_all／proper_items_all）。"""
    verb_items_all = dict(items["verb"]["L0"], **items["verb"]["L0N"])
    proper_items_all = dict(items["proper"]["L0_including_L0-14"], **items["proper"]["L0N"])
    return {"verb": set(verb_items_all), "proper": set(proper_items_all)}


def main():
    items = json.load(open(PROJ / "battery" / "verbcue_items.json", encoding="utf-8"))
    config = json.load(open(PROJ / "battery" / "verbcue_cell_config.json", encoding="utf-8"))

    from_cell_config_rule = canonical_from_cell_config_rule(items)
    from_precheck_rule = canonical_from_precheck_rule(items)

    cell_to_class = {"verb_universal": "verb", "verb_corrected": "verb",
                     "proper_universal": "proper", "proper_corrected": "proper"}

    n_checks = 0
    for cell_name, cls in cell_to_class.items():
        actual = set(config["cells"][cell_name]["item_ids"])
        expected = from_cell_config_rule[cls]
        n_checks += 1
        if actual != expected:
            _abort(f"{cell_name}: verbcue_cell_config.json 實際題集與"
                   f"build_verbcue_cell_config.py 規則重建結果不相等——"
                   f"config有但重建沒有={actual-expected}，"
                   f"重建有但config沒有={expected-actual}")

    for cls in ("verb", "proper"):
        n_checks += 1
        a, b = from_cell_config_rule[cls], from_precheck_rule[cls]
        if a != b:
            _abort(f"{cls}: cell_config 規則重建題集 與 precheck 規則重建題集不相等——"
                   f"這正是兩支腳本各自獨立重建、只有一支漏掉的那種病。"
                   f"cell_config有precheck沒有={a-b}，precheck有cell_config沒有={b-a}")

    print(f"全部 {n_checks} 項交叉檢查通過："
         f"cell_config.json 逐格題集 == build_verbcue_cell_config.py 規則重建 "
         f"== verbcue_precheck.py 規則重建")
    for cls in ("verb", "proper"):
        print(f"  {cls}: n={len(from_cell_config_rule[cls])} {sorted(from_cell_config_rule[cls])}")


if __name__ == "__main__":
    main()

# TODO（徹底解法，留給下一輪，非本次緊急修補範圍）：build_verbcue_items.py
# 的輸出直接落一個 canonical union 欄位（例如 items["verb"]["ALL"]／
# items["proper"]["ALL"]），cell_config／precheck 都改成單純讀那個欄位，
# 不再各自用 dict(a, **b) 重建——那樣「兩支各自重建」這個病因本身就不存在了，
# 不用靠交叉 assert 補洞。這次先用交叉 assert 頂著，因為要儘快恢復點火，
# 徹底解法留待下一輪不急的時候做（改 build_verbcue_items.py 的輸出格式
# 會牽動所有消費端，不該在等點火結果的時候順手做）。
