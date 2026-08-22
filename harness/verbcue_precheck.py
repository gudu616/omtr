"""點火前置檢查：逐 (題, cue) 印出可用替換數，不只比池檔名目長度
（§7-c 裁定②，77316c7；理論部抓到的陷阱：`causal_patch.py:268` 的自我替換
防護 `if replacement.lower() == word.lower(): return None` 會讓 cue 詞恰好
落在池裡的題，可用數＝名目數−1，且是靜默發生的）。

本檔純字串比對，不載入模型、不碰 GPU——這條檢查不需要 tokenizer 對齊，
自我替換防護在 `build_corrupt_candidate` 裡排在對齊檢查之前，字串相等就
會被擋掉，與 token 化結果無關（真正的 token 對齊過濾仍需要模型，
不在本檔範圍內，那部分會讓可用數比這裡算的更低，本檔只抓「名目數不等於
可用數」這一種已知、可不靠模型就抓到的陷阱）。

用法：
  .venv/Scripts/python.exe harness/verbcue_precheck.py
輸出：
  planning/office_reports/verbcue_precheck_report.md
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent


def load():
    items = json.load(open(PROJ / "battery" / "verbcue_items.json", encoding="utf-8"))
    pools = json.load(open(PROJ / "battery" / "verbcue_pools.json", encoding="utf-8"))
    return items, pools


def available_count(cue_word: str, pool_words) -> tuple[int, bool]:
    """回傳 (可用數, 是否命中自我替換). 純字串比對，比照
    causal_patch.py:268 `replacement.lower() == word.lower()`。"""
    hit = any(w.lower() == cue_word.lower() for w in pool_words)
    return (len(pool_words) - (1 if hit else 0)), hit


def check_cell(label, item_dict, pool_words):
    pool_size = len(pool_words)
    rows = []
    n_collision = 0
    for iid in sorted(item_dict):
        spec = item_dict[iid]
        cue = spec["cue_word"]
        avail, hit = available_count(cue, pool_words)
        if hit:
            n_collision += 1
        rows.append({"item": iid, "cue_word": cue, "pool_size_nominal": pool_size,
                    "available": avail, "self_replacement_collision": hit})
    return {"label": label, "pool_size_nominal": pool_size, "n_items": len(rows),
            "n_collisions": n_collision, "rows": rows,
            "all_available_equal_nominal": n_collision == 0}


def load_corrected():
    path = PROJ / "battery" / "verbcue_pools_corrected.json"
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def check_cell_percell_pool(label, item_dict, per_item_pools):
    """修正池版：每題自己的池（不是共用一份），逐題各自算可用數。"""
    rows = []
    n_collision = 0
    for iid in sorted(item_dict):
        spec = item_dict[iid]
        cue = spec["cue_word"]
        pool_entry = per_item_pools.get(iid)
        if pool_entry is None:
            rows.append({"item": iid, "cue_word": cue, "pool_size_nominal": None,
                        "available": None, "self_replacement_collision": None,
                        "note": "該題不在修正池產出中"})
            continue
        pool_words = pool_entry["final_pool"]
        avail, hit = available_count(cue, pool_words)
        if hit:
            n_collision += 1
        rows.append({"item": iid, "cue_word": cue, "pool_size_nominal": len(pool_words),
                    "available": avail, "self_replacement_collision": hit})
    return {"label": label, "n_items": len(rows), "n_collisions": n_collision, "rows": rows}


def main():
    items, pools = load()
    verb_pool = pools["verb"]["final_pool"]
    proper_pool = pools["proper"]["final_pool"]
    corrected = load_corrected()

    cells = {
        "verb x 通用池 / L0": (items["verb"]["L0"], verb_pool),
        "verb x 通用池 / L0N": (items["verb"]["L0N"], verb_pool),
        "proper x 通用池 / L0N": (items["proper"]["L0N"], proper_pool),
        "proper x 通用池 / L0（排L0-14）": (items["proper"]["L0_excluding_L0-14"], proper_pool),
        "proper x 通用池 / L0（含L0-14，8題版，已裁定採用）": (items["proper"]["L0_including_L0-14"], proper_pool),
    }
    results = {label: check_cell(label, item_dict, pool) for label, (item_dict, pool) in cells.items()}

    cross_column_check = None
    if corrected:
        verb_items_all = dict(items["verb"]["L0"], **items["verb"]["L0N"])
        # ⚠ 修正：第一版漏了 proper 臂的 L0N 5 題，只查了 L0 的 8 題（8+8=16 對，
        # 少查了本該有的 13 題 proper 全集）。跟統籌代理訊息裡的「10對」（理論部逐
        # cue 廣義查核，含 L0N-06 的 baked+boiled 兩個 verb cue）對不上時發現的。
        proper_items_all = dict(items["proper"]["L0_including_L0-14"],
                                **items["proper"]["L0N"])
        corrected_cells = {
            "verb x 修正池": (verb_items_all, corrected["verb"]),
            "proper x 修正池": (proper_items_all, corrected["proper"]),
        }
        for label, (item_dict, per_item_pools) in corrected_cells.items():
            results[label] = check_cell_percell_pool(label, item_dict, per_item_pools)

        # §4-b 檢查②：同一題同一cue，通用欄與修正欄的可用數必須相等
        cross_column_check = {"mismatches": [], "n_checked": 0}
        pairs = [("verb", verb_items_all, verb_pool, corrected["verb"]),
                ("proper", proper_items_all, proper_pool, corrected["proper"])]
        for cls, item_dict, univ_pool, corr_pools in pairs:
            for iid in sorted(item_dict):
                cue = item_dict[iid]["cue_word"]
                univ_avail, _ = available_count(cue, univ_pool)
                corr_entry = corr_pools.get(iid)
                if corr_entry is None:
                    continue
                corr_avail, _ = available_count(cue, corr_entry["final_pool"])
                cross_column_check["n_checked"] += 1
                if univ_avail != corr_avail:
                    cross_column_check["mismatches"].append(
                        {"item": iid, "cls": cls, "cue": cue,
                         "universal_available": univ_avail, "corrected_available": corr_avail})

    lines = ["# 點火前置檢查：逐 (題, cue) 可用替換數（§7-c 裁定②＋poolgen §2-d/§4-b②）\n"]
    if corrected:
        lines.append("poolgen 已落檔（a9364e7），本次檢查涵蓋通用池與修正池共 6 個格"
                     "（含 proper 的 L0-14 兩案供對照，裁定採 8 題版）。\n")
    else:
        lines.append("只查通用池兩欄——修正池尚未落檔。\n")
    any_collision = False
    for label, res in results.items():
        any_collision = any_collision or res["n_collisions"] > 0
        lines.append(f"## {label}（題數={res['n_items']}，撞名衝突={res['n_collisions']}）\n")
        lines.append("| 題 | cue 詞 | 名目數 | 可用數 | 自我替換衝突 |")
        lines.append("|---|---|---|---|---|")
        for r in res["rows"]:
            flag = "**是**" if r["self_replacement_collision"] else ("否" if r["available"] is not None else "N/A")
            lines.append(f"| {r['item']} | {r['cue_word']} | {r['pool_size_nominal']} | "
                         f"{r['available']} | {flag} |")
        lines.append("")

    lines.append("## §4-b 檢查②：同一題同一 cue，通用欄與修正欄可用數是否相等\n")
    if cross_column_check is not None:
        lines.append(f"逐 (題,類) 核對 {cross_column_check['n_checked']} 對，"
                     f"不相等數 = {len(cross_column_check['mismatches'])}\n")
        if cross_column_check["mismatches"]:
            lines.append("| 題 | 類 | cue | 通用欄可用數 | 修正欄可用數 |")
            lines.append("|---|---|---|---|---|")
            for m in cross_column_check["mismatches"]:
                lines.append(f"| {m['item']} | {m['cls']} | {m['cue']} | "
                             f"{m['universal_available']} | {m['corrected_available']} |")
        lines.append(f"\n**{'⛔ 有不相等，依 poolgen §2-d 規定不得點火' if cross_column_check['mismatches'] else '✅ 全部相等，此項檢查通過'}**\n")
    else:
        lines.append("修正池尚未落檔，此項未執行。\n")

    lines.append("## 總結\n")
    lines.append(f"**通用池兩格（verb/proper）是否可用數＝名目數（24/24）："
                 f"{'否，有衝突，見上表' if any_collision else '是，無衝突'}**"
                 f"（唯一已知衝突：verb×通用池的 L0N-06，cue=baked，23/24）\n")
    lines.append("**未涵蓋範圍**：token 對齊檢查（`build_corrupt_candidate` 的三個對齊條件）"
                 "仍需要真實模型才能查，本檔只抓「名目數≠可用數」這一種不靠模型就能抓到的陷阱"
                 "（自我替換）。真實候選數在點火後才會逐題確定，可能比這裡算的可用數更低"
                 "（不會更高）。\n")

    report_path = PROJ / "planning" / "office_reports" / "verbcue_precheck_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    for label, res in results.items():
        print(f"{label}: n_items={res['n_items']} n_collisions={res['n_collisions']}")
        for r in res["rows"]:
            if r["self_replacement_collision"]:
                print(f"    !! {r['item']}: cue={r['cue_word']!r} 命中池內同名詞，"
                     f"可用數 {r['available']}/{r['pool_size_nominal']}")
    if cross_column_check is not None:
        print(f"\n§4-b②交叉核對：checked={cross_column_check['n_checked']} "
             f"mismatches={len(cross_column_check['mismatches'])}")
        for m in cross_column_check["mismatches"]:
            print(f"    !! {m}")
    print("\n->", report_path)
    return results, cross_column_check


if __name__ == "__main__":
    main()
