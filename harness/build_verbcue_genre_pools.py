"""修正池（文體匹配替換詞池）生成（theory_verbcue_poolgen_v1.md，HEAD a9364e7 凍結）。

規格權威逐條對應：
  §1：來源＝該題 battery.json `source` 欄所指的 Gutenberg 全文。
  §2-a：演算法（斷詞/類別過濾/停用詞/長度/排除集/計次/頻次序破字母平/取前N）。
  §2-c：排除集＝(該題n_cue=6全部cue候選) ∪ (prompt全部詞) ∪ (gold全部詞)。
  §0-b：N>=24 正常；12<=N<24 該題所有格改用N（通用池同步截斷為凍結24詞順序
        前N個）；N<12 該題全格退出。
  §1-d：抓取沿用既有禮儀（build_l0_gutenberg.fetch_book：指數退避+三鏡像），
        落盤存檔（不入git，只雜湊入git）。

【啟動斷言（§2-a 要求，取不到即 abort，不得 fallback 副本）】
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_patch as cp  # noqa: E402
from build_l0_gutenberg import fetch_book  # noqa: E402  單一出處：既有抓取禮儀

assert hasattr(cp, "_WORD_RE") and hasattr(cp, "STOPWORDS"), (
    "causal_patch 缺 _WORD_RE 或 STOPWORDS——§2-a 規定取不到就 abort，不得 fallback 副本")

PROJ = Path(__file__).resolve().parent.parent
RAW_DIR = PROJ / "hf-cache" / "gutenberg_verbcue"     # 落盤但不入 git（.gitignore 已排除 hf-cache/）
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_N = 24

VERB_ITEMS = ["L0-11", "L0-19", "L0-20", "L0N-06", "L0N-09", "L0N-10", "L0N-11", "L0N-12"]
PROPER_ITEMS = ["L0-03", "L0-05", "L0-06", "L0-10", "L0-11", "L0-14", "L0-15", "L0-17",
               "L0N-02", "L0N-03", "L0N-04", "L0N-08", "L0N-11"]


def load_battery():
    d = json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))
    return {it["id"]: it for it in d["items"]}


def parse_gid(source: str) -> int:
    assert source.startswith("gutenberg:"), f"source 格式不是 gutenberg:<id>：{source!r}"
    return int(source.split(":", 1)[1])


_FETCH_CACHE: dict = {}


def get_text(gid: int) -> str:
    """落盤快取優先；沒有才照 §1-d 用既有禮儀（退避+鏡像）向 Gutenberg 抓。"""
    path = RAW_DIR / f"{gid}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = fetch_book(gid, _FETCH_CACHE)
    path.write_text(text, encoding="utf-8")
    return text


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclude_set(item: dict) -> set:
    """§2-c：三者聯集，全部小寫。"""
    prompt = item["prompt"]
    gold = item.get("gold_continuation") or item.get("expected_continuation") or ""
    cue_spans = cp.cue_word_candidates(prompt, 6)
    cue_words = {w.lower() for _, _, w in cue_spans}
    prompt_words = {m.group().lower() for m in cp._WORD_RE.finditer(prompt)}
    gold_words = {m.group().lower() for m in cp._WORD_RE.finditer(gold)}
    return cue_words | prompt_words | gold_words


def candidate_pool(item: dict, cls: str, full_text: str):
    """§2-a 步驟1-7，回傳 (排序後的候選清單[(word,freq),...], 排除集大小)。"""
    all_words = cp._WORD_RE.findall(full_text)
    freq = Counter(w.lower() for w in all_words)

    seen_form = {}
    for w in all_words:
        if cp.guess_pool_class(w) != cls:
            continue
        if w.lower() in cp.STOPWORDS:
            continue
        if len(w) <= 2:
            continue
        key = w.lower()
        if key not in seen_form:
            seen_form[key] = w              # 保留該詞第一次出現時的原大小寫

    excl = exclude_set(item)
    candidates = [k for k in seen_form if k not in excl]
    candidates_sorted = sorted(candidates, key=lambda k: (-freq[k], k))   # 步驟7
    return [(seen_form[k], freq[k]) for k in candidates_sorted], len(excl)


def build_all():
    battery = load_battery()
    results = {}       # (item_id, cls) -> {"N":..., "pool":[(word,freq),...], "excl_size":..., "gid":...}
    hashes = {}

    print("=== 高風險題優先檢查：L0-11（verb + proper 兩列都用得到）===")
    it = battery["L0-11"]
    gid = parse_gid(it["source"])
    text = get_text(gid)
    path = RAW_DIR / f"{gid}.txt"
    hashes[str(gid)] = file_sha256(path)
    n_words_total = len(cp._WORD_RE.findall(text))
    print(f"L0-11 source=gutenberg:{gid}，全文詞數(粗算,_WORD_RE)={n_words_total}")
    for cls in ("verb", "proper"):
        pool, excl_size = candidate_pool(it, cls, text)
        N = len(pool)
        print(f"  cls={cls}: N={N}（排除集={excl_size}）"
             f"{'  !! N<24，觸發 §0-b 情形2/3' if N < TARGET_N else ''}")
        results[("L0-11", cls)] = {"N": N, "pool": pool, "excl_size": excl_size, "gid": gid}

    print("\n=== 其餘題目 ===")
    for iid in sorted(set(VERB_ITEMS + PROPER_ITEMS) - {"L0-11"}):
        it = battery[iid]
        gid = parse_gid(it["source"])
        text = get_text(gid)
        path = RAW_DIR / f"{gid}.txt"
        if str(gid) not in hashes:
            hashes[str(gid)] = file_sha256(path)
        classes = []
        if iid in VERB_ITEMS:
            classes.append("verb")
        if iid in PROPER_ITEMS:
            classes.append("proper")
        for cls in classes:
            pool, excl_size = candidate_pool(it, cls, text)
            N = len(pool)
            print(f"{iid} cls={cls}: N={N}（排除集={excl_size}，gid={gid}）"
                 f"{'  !! N<24' if N < TARGET_N else ''}")
            results[(iid, cls)] = {"N": N, "pool": pool, "excl_size": excl_size, "gid": gid}

    return results, hashes, battery


def apply_size_rule(results: dict):
    """§0-b：N>=24 用24；12<=N<24 該題全格用N；N<12 該題退出。"""
    decisions = {}
    for (iid, cls), r in results.items():
        N = r["N"]
        if N >= TARGET_N:
            decisions[(iid, cls)] = {"final_n": TARGET_N, "case": "normal"}
        elif N >= 12:
            decisions[(iid, cls)] = {"final_n": N, "case": "truncated_2"}
        else:
            decisions[(iid, cls)] = {"final_n": N, "case": "dropped_3"}
    return decisions


def apply_universal_collision_truncation(decisions, results, battery):
    """§0-b 類推（2026-08-22 統籌代理裁定，非理論部逐字條文，但統籌代理已給明確推理，
    見對應工作紀錄）：若某題的 cue 詞恰好落在「通用池」裡（causal_patch.py:268
    自我替換防護），通用欄的可用數會靜默少 1（例：verb×通用池的 L0N-06，
    cue=baked，可用=23/24）。poolgen §4-b 檢查②要求同一題同一cue兩欄可用數
    相等，而修正欄的候選生成已經把cue排除在外（§2-c第1條），天生不會撞——
    兩欄因此不等。**處置：套用§0-b「同一題在它出現的每一格都改用N」的邏輯，
    以兩欄可用數的較小值為準**，修正欄對應截短1個（丟最後一名，即頻次序+
    字母序排最後的那個）。這不是理論部逐字凍結的規則，是統籌代理指示的類推應用，
    範圍僅限於已知的通用池撞名案例（本批資料只有 L0N-06/verb 一例）。"""
    universal = json.load(open(PROJ / "battery" / "verbcue_pools.json", encoding="utf-8"))
    items_data = json.load(open(PROJ / "battery" / "verbcue_items.json", encoding="utf-8"))
    univ_pool = {"verb": universal["verb"]["final_pool"],
                "proper": universal["proper"]["final_pool"]}

    def cue_word_for(iid, cls):
        for arm in (items_data.get(cls, {}).get("L0", {}), items_data.get(cls, {}).get("L0N", {}),
                   items_data.get(cls, {}).get("L0_including_L0-14", {})):
            if iid in arm:
                return arm[iid]["cue_word"]
        return None

    adjustments = []
    for (iid, cls), dec in decisions.items():
        cue = cue_word_for(iid, cls)
        if cue is None or dec["case"] == "dropped_3":
            continue
        hit = any(w.lower() == cue.lower() for w in univ_pool[cls])
        if not hit:
            continue
        univ_avail = len(univ_pool[cls]) - 1
        if dec["final_n"] > univ_avail:
            old_n = dec["final_n"]
            dec["final_n"] = univ_avail
            dec["case"] = dec["case"] + "+universal_collision_truncated"
            adjustments.append({"item": iid, "cls": cls, "cue": cue,
                               "old_final_n": old_n, "new_final_n": univ_avail})
    return adjustments


def main():
    results, hashes, battery = build_all()
    decisions = apply_size_rule(results)
    collision_adjustments = apply_universal_collision_truncation(decisions, results, battery)
    if collision_adjustments:
        print("\n=== §0-b 類推：通用池撞名，修正欄同步截短 ===")
        for a in collision_adjustments:
            print(f"  {a['item']}/{a['cls']} cue={a['cue']!r}: "
                 f"final_n {a['old_final_n']}->{a['new_final_n']}")

    # 落雜湊清單（檔案本身不入 git）
    hash_path = PROJ / "battery" / "verbcue_gutenberg_hashes.json"
    json.dump({"files": hashes, "raw_dir": str(RAW_DIR)},
             open(hash_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 落最終凍結物：逐詞表（§4-a 格式）
    genre_pools = {}
    lines = ["# 修正池（文體匹配）逐題逐類詞表\n",
            "來源：`theory_verbcue_poolgen_v1.md`（HEAD a9364e7）§1/§2 演算法，"
            "逐題 Gutenberg 全文，SHA-256 見 `battery/verbcue_gutenberg_hashes.json`。\n"]
    truncated, dropped = [], []
    for (iid, cls), dec in sorted(decisions.items()):
        r = results[(iid, cls)]
        final_n = dec["final_n"]
        pool_words = r["pool"][:final_n] if dec["case"] != "dropped_3" else []
        genre_pools.setdefault(cls, {})[iid] = {
            "final_pool": [w for w, _ in pool_words], "N": r["N"], "final_n": final_n,
            "case": dec["case"], "gid": r["gid"], "excl_size": r["excl_size"]}
        lines.append(f"## {iid} / {cls}（N={r['N']}，最終尺寸={final_n}，"
                     f"情形={dec['case']}，gutenberg:{r['gid']}，排除集={r['excl_size']}）\n")
        if dec["case"] == "dropped_3":
            lines.append("**該題自全部格次退出（N<12）。**\n")
            dropped.append((iid, cls, r["N"]))
            continue
        if dec["case"] == "truncated_2":
            truncated.append((iid, cls, r["N"]))
        lines.append("| # | 詞 | 出現次數 |")
        lines.append("|---|---|---|")
        for i, (w, f) in enumerate(pool_words, 1):
            lines.append(f"| {i} | {w} | {f} |")
        lines.append("")

    lines.append("## §0-b 類推：通用池撞名，修正欄同步截短（統籌代理 2026-08-22 裁定，非理論部逐字條文）\n")
    if collision_adjustments:
        for a in collision_adjustments:
            lines.append(f"- {a['item']}/{a['cls']}：cue={a['cue']!r} 撞通用池，"
                         f"修正欄 final_n 由 {a['old_final_n']} 截短為 {a['new_final_n']}"
                         f"（與通用欄可用數 {a['new_final_n']} 打平，滿足 poolgen §4-b 檢查②）")
    else:
        lines.append("（本輪無撞名，未觸發）")
    lines.append("")

    lines.append("## 總表：情形2(截斷)與情形3(退題)\n")
    lines.append(f"截斷（12≤N<24）：{truncated}\n")
    lines.append(f"退題（N<12）：{dropped}\n")

    out_json = PROJ / "battery" / "verbcue_pools_corrected.json"
    json.dump(genre_pools, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report_path = PROJ / "planning" / "office_reports" / "verbcue_genre_pools.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== 摘要 ===")
    print("截斷 (12<=N<24):", truncated)
    print("退題 (N<12):", dropped)
    print("->", out_json)
    print("->", hash_path)
    print("->", report_path)


if __name__ == "__main__":
    main()
