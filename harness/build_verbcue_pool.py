"""通用詞池機械擴充：verb（工項①）＋ proper（§7-c 裁定②新增）。

規格權威：`planning/office_reports/theory_verbcue_prereg_v1.md`
§6.5-d（第 374-388 行，「加材料不是降門檻」）＋
§6.6（第 424-429 行，「擴充 verb 池的機械規則（凍結）」，逐字）：

    取現有 verb 12 詞，加上 verb_ing 10 詞的 -ed 形（`carrying→carried` 等
    已重複者去除），不足 24 個時，由單一指定來源（英文最常見規則動詞過去式
    表）按字母序補足到 24。不得由任何人「挑看起來合適的」。
    這條規則本身要在點火前寫成程式並印出最終詞表，附進報告。

【單一出處】`CORRUPT_POOL["verb"]`／`CORPUT_POOL["verb_ing"]` 直接 import
自 `causal_patch.py`，不另抄。

【指定來源】englishstudyhere.com「50 Most Commonly Used Regular Verbs in
Past」（2026-08-22 WebFetch 全文取得，見下方 SOURCE_50 常數逐字抄錄＋URL）。
選它的理由：公開、免費、標題明寫「regular verbs」（已先排除不規則動詞的
篩選工作，不需要我自己做語言學判斷）、給出完整的過去式拼法。

【機械規則執行，逐步可查】：
  1. 現有 12 詞（原樣，不重排）。
  2. verb_ing 的 10 個 -ed 形——逐一與現有 12 詞比對，**10 個全部重複**
     （carrying→carried、opening→opened、walking→walked、turning→turned、
     answering→answered、watching→watched、following→followed、
     entering→entered、offering→offered、listening→listened，
     全部已在現有 12 詞裡），貢獻 0 個新詞。
  3. 對照 SOURCE_50，逐字檢查是否為真正的規則動詞（發現一個來源本身的
     事實錯誤：見 SOURCE_ERRATA），剔除錯誤項與跟現有池重複的項。
  4. 剩餘候選按過去式字母序排序，取前 12 個補進池，湊滿 24。

【proper 池同一規則,§7-c 裁定②(commit 77316c7)】：「通用 proper 池（現 10 詞）
也要照同一機械規則擴到 24」。現有 10 詞是人名（Thomas, Margaret, Henry,
Eleanor, William, Charlotte, Edward, Frances, Arthur, Beatrice），非 -ed
動詞，§6.6 逐字規則（verb_ing 的 -ed 形那步）不適用；套用的是**同一種精神**
（單一指定、公開、可引用的來源；機械處理；不得手挑；逐詞落 provenance）。
指定來源：eslyes.com「50 most Common English Names」（Male First Names 25
＋Female First Names 25，2026-08-22 WebFetch 全文取得，逐字抄錄）。

⚠ **§7-c 裁定②另附「可用數非名目數」的提醒**：`causal_patch.py:268` 的自我
替換防護會讓某些 (題, cue) 組合的可用替換數 < 24（cue 詞本身若剛好落在池裡）。
本檔只管池的內容；可用數檢查在 `harness/verbcue_precheck.py`。

用法：
  .venv/Scripts/python.exe harness/build_verbcue_pool.py
輸出：
  battery/verbcue_pools.json （verb+proper 兩個池，逐詞 provenance，供 runner 讀取）
  planning/office_reports/verbcue_pool_expansion.md （人讀報告，含完整詞表）
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_patch import CORRUPT_POOL  # noqa: E402  單一出處：既有 12/10 詞不重抄

PROJ = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------
# 指定來源（2026-08-22 WebFetch 全文取得，逐字抄錄，不得事後增刪）：
#   https://englishstudyhere.com/verbs/50-most-commonly-used-regular-verbs-in-past/
#   標題：「50 Most Commonly Used Regular Verbs in Past」
# 格式：(base_form, past_tense_form)，順序＝原頁面出現順序。
SOURCE_50 = [
    ("Accept", "Accepted"), ("Act", "Acted"), ("Bake", "Baked"),
    ("Behave", "Behaved"), ("Close", "Closed"), ("Compare", "Compared"),
    ("Compete", "Competed"), ("Die", "Died"), ("Disagree", "Disagreed"),
    ("Disturb", "Disturbed"), ("Dress", "Dressed"), ("Dry", "Dried"),
    ("Eliminate", "Eliminated"), ("End", "Ended"), ("Enjoy", "Enjoyed"),
    ("Fix", "Fixed"), ("Follow", "Followed"), ("Freeze", "Freezed"),
    ("Fry", "Fried"), ("Greet", "Greeted"), ("Guess", "Guessed"),
    ("Hunt", "Hunted"), ("Identify", "Identified"), ("Imagine", "Imagined"),
    ("Invite", "Invited"), ("Join", "Joined"), ("Jump", "Jumped"),
    ("Knock", "Knocked"), ("Love", "Loved"), ("Manage", "Managed"),
    ("Mark", "Marked"), ("Match", "Matched"), ("Name", "Named"),
    ("Need", "Needed"), ("Open", "Opened"), ("Order", "Ordered"),
    ("Organize", "Organized"), ("Pack", "Packed"), ("Paint", "Painted"),
    ("Pass", "Passed"), ("Perform", "Performed"), ("Persuade", "Persuaded"),
    ("Program", "Programmed"), ("Protect", "Protected"), ("Review", "Reviewed"),
    ("Shop", "Shopped"), ("Slow", "Slowed"), ("Turn", "Turned"),
    ("Underline", "Underlined"), ("Want", "Wanted"),
]
assert len(SOURCE_50) == 50, f"抄錄長度不對：{len(SOURCE_50)}（來源頁面標題就叫 50，須逐字核對）"

# 來源本身的事實錯誤（不是喜好篩選，是語法糾正——freeze 是不規則動詞
# froze/frozen，不是 freezed；任何英文文法參考書都可查證，不涉主觀判斷）：
SOURCE_ERRATA = {"Freeze": "irregular verb (froze/frozen), not 'freezed'; excluded as a factual error "
                           "in the source, not a preference choice"}

# ------------------------------------------------------------------------
# proper 池的指定來源（2026-08-22 WebFetch 全文取得，逐字抄錄）：
#   https://eslyes.com/namesdict/popular_names.htm
#   標題：「50 most Common English Names」——Male First Names(25)
#   + Female First Names(25)，順序＝原頁面出現順序。不用該頁另列的
#   「Last Names」50 個姓氏——CORRUPT_POOL["proper"] 現有 10 詞全是名不是姓，
#   維持同一詞性類別，不混用姓氏（這是類別一致性，不是喜好篩選）。
SOURCE_NAMES_50 = (
    ["James", "John", "Robert", "Michael", "William", "David", "Richard",
     "Charles", "Joseph", "Thomas", "Christopher", "Daniel", "Paul", "Mark",
     "Donald", "George", "Kenneth", "Steven", "Edward", "Brian", "Ronald",
     "Anthony", "Kevin", "Jason", "Jeff"]
    + ["Mary", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer",
       "Maria", "Susan", "Margaret", "Dorothy", "Lisa", "Nancy", "Karen",
       "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon",
       "Michelle", "Laura", "Sarah", "Kimberly", "Deborah"]
)
assert len(SOURCE_NAMES_50) == 50, f"抄錄長度不對：{len(SOURCE_NAMES_50)}"

N_TARGET = 24


def build_verb_pool():
    """回傳 (final_words: tuple[str], provenance: list[dict])。"""
    existing = list(CORRUPT_POOL["verb"])          # 12 詞，原樣、不重排
    verb_ing = list(CORRUPT_POOL["verb_ing"])       # 10 詞

    provenance = [{"word": w, "origin": "existing_verb_pool", "step": 1} for w in existing]
    have = {w.lower() for w in existing}

    # 步驟2：verb_ing 的 -ed 形——用最常見的規則變化（去 -ing 加 -ed；
    # 這 10 個詞全部是「去尾 e 再加 -ed」或「單純加 -ed」的標準規則變化，
    # 手動核對，不靠自動詞形還原庫（本機沒裝，§9 已記這是既有政策）：
    ing_to_ed = {
        "carrying": "carried", "opening": "opened", "walking": "walked",
        "turning": "turned", "answering": "answered", "watching": "watched",
        "following": "followed", "entering": "entered", "offering": "offered",
        "listening": "listened",
    }
    assert set(ing_to_ed) == set(verb_ing), "verb_ing 常數與硬編對照表不一致，需要重新核對變化規則"
    ing_dupes = []
    for ing in verb_ing:
        ed = ing_to_ed[ing]
        if ed.lower() in have:
            ing_dupes.append((ing, ed))
        else:
            provenance.append({"word": ed, "origin": f"verb_ing:{ing}", "step": 2})
            have.add(ed.lower())
    # 已量：全部 10 個都重複，貢獻 0 新詞（見模組 docstring）。
    assert len(ing_dupes) == 10, f"verb_ing 重複數預期 10，實得 {len(ing_dupes)}——規則庫或詞池已變，需重查"

    n_have = len(have)
    n_need = N_TARGET - n_have
    if n_need <= 0:
        return (tuple(existing[:N_TARGET]), provenance[:N_TARGET],
               {"candidates_sorted": [], "excluded": [], "ing_dupes": ing_dupes})

    # 步驟3+4：對照指定來源，剔除錯誤與重複，按過去式字母序排序，取前 n_need 個。
    candidates = []
    excluded = []
    for base, past in SOURCE_50:
        if base in SOURCE_ERRATA:
            excluded.append({"word": past, "reason": SOURCE_ERRATA[base]})
            continue
        if past.lower() in have:
            excluded.append({"word": past, "reason": f"與現有池重複（{past.lower()}）"})
            continue
        candidates.append(past.lower())
    candidates_sorted = sorted(candidates)          # 按過去式字母序（機械規則，非手挑）
    picked = candidates_sorted[:n_need]
    for w in picked:
        provenance.append({"word": w, "origin": "designated_source:englishstudyhere.com",
                           "step": 3, "source_rank_in_alpha_order": candidates_sorted.index(w) + 1})
        have.add(w)

    final = tuple(existing) + tuple(picked)
    if len(final) != N_TARGET:
        raise SystemExit(f"擴充後詞數不是 {N_TARGET}，實得 {len(final)}——機械規則跑出非預期結果，停止")
    return final, provenance, {"candidates_sorted": candidates_sorted, "excluded": excluded,
                               "ing_dupes": ing_dupes}


def build_proper_pool():
    """回傳 (final_words, provenance, diag)——proper 池版，§7-c 裁定②新增。
    無 verb_ing 那一步（proper 沒有對應的『動名詞形』可轉換），直接進指定來源。
    """
    existing = list(CORRUPT_POOL["proper"])         # 10 詞，原樣、不重排
    provenance = [{"word": w, "origin": "existing_proper_pool", "step": 1} for w in existing]
    have = {w.lower() for w in existing}

    n_need = N_TARGET - len(have)
    if n_need <= 0:
        return tuple(existing[:N_TARGET]), provenance[:N_TARGET], {}

    # ⚠ 專有名詞維持原大寫（來源本來就是大寫）——不像 verb 池要轉小寫配合現有
    # 池的慣例；這裡若轉小寫，連 guess_pool_class 自己都不會把它認成 proper 類
    # （guess_pool_class: word[:1].isupper() 才判 proper），那就不是同一類詞了。
    candidates, excluded = [], []
    for name in SOURCE_NAMES_50:
        if name.lower() in have:
            excluded.append({"word": name, "reason": f"與現有池重複（{name.lower()}）"})
            continue
        candidates.append(name)                      # 保留原大寫
    candidates_sorted = sorted(candidates, key=str.lower)   # 大小寫不敏感排序，值保留大寫
    picked = candidates_sorted[:n_need]
    for w in picked:
        provenance.append({"word": w, "origin": "designated_source:eslyes.com",
                           "step": 2, "source_rank_in_alpha_order": candidates_sorted.index(w) + 1})
        have.add(w.lower())

    final = tuple(existing) + tuple(picked)
    if len(final) != N_TARGET:
        raise SystemExit(f"proper 池擴充後詞數不是 {N_TARGET}，實得 {len(final)}——停止")
    return final, provenance, {"candidates_sorted": candidates_sorted, "excluded": excluded}


def main():
    v_final, v_prov, v_diag = build_verb_pool()
    assert len(v_final) == len(set(w.lower() for w in v_final)), "verb 池有重複詞，機械規則出錯"
    p_final, p_prov, p_diag = build_proper_pool()
    assert len(p_final) == len(set(w.lower() for w in p_final)), "proper 池有重複詞，機械規則出錯"

    verb_out = {
        "target": N_TARGET, "final_pool": list(v_final),
        "provenance": v_prov,
        "rule_source": ("theory_verbcue_prereg_v1.md §6.6 第424-429行"
                        "（現有12+verb_ing的-ed形去重，不足補指定來源按字母序）"
                        "＋§7-c 裁定③追認"),
        "designated_source": {
            "name": "englishstudyhere.com: 50 Most Commonly Used Regular Verbs in Past",
            "url": "https://englishstudyhere.com/verbs/50-most-commonly-used-regular-verbs-in-past/",
            "fetched": "2026-08-22", "n_entries": len(SOURCE_50),
        },
        "source_errata": SOURCE_ERRATA,
        "excluded_from_source": v_diag["excluded"],
        "verb_ing_all_duplicated": v_diag["ing_dupes"],
    }
    proper_out = {
        "target": N_TARGET, "final_pool": list(p_final),
        "provenance": p_prov,
        "rule_source": ("theory_verbcue_prereg_v1.md §7-c 裁定②"
                        "（77316c7：通用proper池同一機械規則精神擴到24）"),
        "designated_source": {
            "name": "eslyes.com: 50 most Common English Names "
                    "(Male First Names 25 + Female First Names 25)",
            "url": "https://eslyes.com/namesdict/popular_names.htm",
            "fetched": "2026-08-22", "n_entries": len(SOURCE_NAMES_50),
        },
        "source_errata": {},
        "excluded_from_source": p_diag["excluded"],
    }

    out_path = PROJ / "battery" / "verbcue_pools.json"
    existing_data = {}
    if out_path.exists():
        existing_data = json.load(open(out_path, encoding="utf-8"))
    existing_data["verb"] = verb_out
    existing_data["proper"] = proper_out
    json.dump(existing_data, open(out_path, "w", encoding="utf-8"),
             ensure_ascii=False, indent=1)

    def section(title, out_d, prov, diag_d, extra_note=""):
        lines = [f"## {title}\n",
                f"指定來源：[{out_d['designated_source']['name']}]"
                f"({out_d['designated_source']['url']})（2026-08-22 WebFetch 全文取得）。\n"]
        if extra_note:
            lines.append(extra_note + "\n")
        lines += ["### 最終 24 詞\n", "| # | 詞 | 來源 |", "|---|---|---|"]
        for i, p in enumerate(prov, 1):
            lines.append(f"| {i} | {p['word']} | {p['origin']} |")
        lines.append("\n### 來源 50 詞中被排除的項（含理由）\n")
        lines.append("| 詞 | 排除理由 |")
        lines.append("|---|---|")
        for e in diag_d.get("excluded", []):
            lines.append(f"| {e['word']} | {e['reason']} |")
        return lines

    lines = ["# 通用詞池擴充到 24（verb 工項①＋proper §7-c 裁定②，機械規則，可重跑）\n"]
    lines += section("verb 池", verb_out, v_prov, v_diag,
                     f"verb_ing 的 10 個 -ed 形全部與現有池重複（貢獻 0 新詞）：{v_diag['ing_dupes']}")
    lines.append("\n---\n")
    lines += section("proper 池", proper_out, p_prov, p_diag,
                     "proper 沒有 verb_ing 那一步（無對應動名詞形可轉換），"
                     "直接從現有 10 詞進指定來源補 14 個。")
    report_path = PROJ / "planning" / "office_reports" / "verbcue_pool_expansion.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"verb 最終 {len(v_final)} 詞：{v_final}")
    print(f"proper 最終 {len(p_final)} 詞：{p_final}")
    print("->", out_path)
    print("->", report_path)


if __name__ == "__main__":
    main()
