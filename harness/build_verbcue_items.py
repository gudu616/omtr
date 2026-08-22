"""2×2 題集備料（工項③，theory_verbcue_prereg_v1.md §3-a／§6.5-c／§7）。

規格權威：
  §3-a（第 150-175 行）：verb 臂題集裁定與已量表（L0 3 題、L0N 3 題於 winB1 内）。
  §6.5-c（第 353-372 行）：納入 winB1 沒跑過的 L0N-10/12（L0N-10 只進 Pythia，
  union+遮罩讀法，不得假設四個 run 同題）。
  §7（第 433-454 行）：2×2 需要 proper 類同一套題集邏輯（原文未逐題列出，
  本檔獨立重推，見下方「本檔的推論，非逐字抄」段落）。

本檔用 `causal_patch.cue_word_candidates`＋`guess_pool_class` 對 `battery.json`
逐題重跑分類（純文字分析，n_cue=6 凍結值，不碰任何 per_tok/GPU），
產出 verb／proper 兩類 × L0／L0N 兩臂的題集，含每題的目標 cue 詞與逐模型
語料閘遮罩。

【已與理論部已發表數字交叉核對，全部相符（見 verbcue_items_report.md）】：
  verb 臂：L0N 3→5 題（納入 L0N-10/12 後）、L0 3 題——與 §3-a／§6.5-c 逐字相符。
  proper 臂：L0N 5 題——與 §6.5-b「proper×原池 7+5」的「5」相符。

【本檔的推論，非逐字抄——標明供覆核】：
  1. **proper 臂 L0 側題數**：n_cue=6 下 L0 有 8 題含 proper 類 cue
     （L0-03/05/06/10/11/14/15/17），但 L0-14 的 `verified_in=['olmo_mix']`
     （不含 pile），若排除 L0-14，恰為 7 題——與 §6.5-b「proper×原池 7+5」
     的「7」相符。**但排除 L0-14 這個動作本身，本檔目前找不到逐字凍結出處**
     ——§6.5-c 對 L0N-10（同類語料閘缺陷）採的是「不排除，union+遮罩」，
     若對 L0-14 一致適用同一政策，L0 proper 題數應為 8（僅排除 Pythia 三 run，
     保留 OLMo run），不是 7。**本檔兩案都算，見 include_l0_14 開關**，
     何者為準待理論部/主線裁定。
  2. **一題有多個目標類cue 時，選哪一個當 cue_word**：例如 L0N-06 同時有
     baked／boiled 兩個 verb cue、L0N-11 有 bewildered／squealed。
     本檔選**離 gold 邊界最近的那個**（`cue_word_candidates` 本身就是
     「由近而遠」排序，即該類別 cue 在排序後第一個出現的）——這是本檔的
     construction 選擇，不是逐字凍結規則，同樣待確認。

用法：
  .venv/Scripts/python.exe harness/build_verbcue_items.py
輸出：
  battery/verbcue_items.json（供 runner 讀取）
  planning/office_reports/verbcue_items_report.md（人讀摘要，含交叉核對）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_patch import cue_word_candidates, guess_pool_class  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
N_CUE = 6                          # 既凍值，§3-a 裁定不放寬
PYTHIA_PREFIX = "EleutherAI/pythia"
OLMO_MODEL = "allenai/OLMo-2-0425-1B"
ALL_MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
             "EleutherAI/pythia-1.4b", OLMO_MODEL]   # fp32 主臂四模型（v2.19/v2.21 §5）
CORPUS_OF = {m: "pile" for m in ALL_MODELS[:3]}
CORPUS_OF[OLMO_MODEL] = "olmo_mix"

# 已量到、與理論部已發表數字核對用（見模組 docstring）
EXPECTED = {
    "verb": {"L0": {"L0-11", "L0-19", "L0-20"},
             "L0N": {"L0N-06", "L0N-09", "L0N-10", "L0N-11", "L0N-12"}},
    "proper": {"L0N": {"L0N-02", "L0N-03", "L0N-04", "L0N-08", "L0N-11"}},
}


def classify_item(item):
    """回傳該題在 n_cue=6 下每個 pool_class 的 cue 清單（近→遠）。"""
    spans = cue_word_candidates(item["prompt"], N_CUE)
    by_class = {}
    for s, e, w in spans:
        cls = guess_pool_class(w)
        by_class.setdefault(cls, []).append(w)
    return by_class


def corpus_mask(item):
    verified = set(item.get("verified_in") or [])
    return {m: (CORPUS_OF[m] in verified) for m in ALL_MODELS}


def build(include_l0_14: bool):
    battery = json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))
    items = {it["id"]: it for it in battery["items"]}

    result = {"L0": {}, "L0N": {}}
    for iid, it in items.items():
        level = it.get("level")
        if level not in ("L0", "L0N"):
            continue
        by_class = classify_item(it)
        mask = corpus_mask(it)
        result[level][iid] = {"by_class": by_class, "corpus_mask": mask,
                              "verified_in": it.get("verified_in"), "item": it}

    def cell_items(level, cls, exclude=()):
        out = {}
        for iid, d in result[level].items():
            if iid in exclude:
                continue
            cues = d["by_class"].get(cls)
            if not cues:
                continue
            out[iid] = {"cue_word": cues[0],           # 近→遠第一個（construction 選擇②）
                        "n_cues_this_class": len(cues),
                        "all_cues_this_class": cues,
                        "corpus_mask": d["corpus_mask"],
                        "verified_in": d["verified_in"],
                        "item": d["item"]}             # 完整 battery 題目 dict，run_cell() 需要
        return out

    verb_l0 = cell_items("L0", "verb")
    verb_l0n = cell_items("L0N", "verb")
    proper_l0n = cell_items("L0N", "proper")
    proper_l0_incl = cell_items("L0", "proper")                       # 含 L0-14
    proper_l0_excl = cell_items("L0", "proper", exclude={"L0-14"})    # 排 L0-14

    out = {
        "n_cue": N_CUE,
        "verb": {"L0": verb_l0, "L0N": verb_l0n},
        "proper": {
            "L0N": proper_l0n,
            "L0_including_L0-14": proper_l0_incl,
            "L0_excluding_L0-14": proper_l0_excl,
            "note": "L0-14 verified_in=['olmo_mix']（不含 pile）。是否排除待裁，"
                    "兩案都給；預設 runner 讀哪一案由呼叫端決定，見模組 docstring 推論1。",
        },
        "cross_check": {
            "verb_L0_match": set(verb_l0) == EXPECTED["verb"]["L0"],
            "verb_L0N_match": set(verb_l0n) == EXPECTED["verb"]["L0N"],
            "proper_L0N_match": set(proper_l0n) == EXPECTED["proper"]["L0N"],
            "verb_L0_ids": sorted(verb_l0), "verb_L0N_ids": sorted(verb_l0n),
            "proper_L0N_ids": sorted(proper_l0n),
            "proper_L0_incl_ids": sorted(proper_l0_incl),
            "proper_L0_excl_ids": sorted(proper_l0_excl),
        },
    }
    return out


def main():
    out = build(include_l0_14=True)
    out_path = PROJ / "battery" / "verbcue_items.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cc = out["cross_check"]
    lines = ["# 2×2 題集備料（工項③，可重跑）\n",
            "## 與理論部已發表數字交叉核對\n",
            f"- verb 臂 L0：{cc['verb_L0_ids']}（與 §3-a 相符：{cc['verb_L0_match']}）",
            f"- verb 臂 L0N：{cc['verb_L0N_ids']}（與 §6.5-c 相符：{cc['verb_L0N_match']}）",
            f"- proper 臂 L0N：{cc['proper_L0N_ids']}"
            f"（與 §6.5-b「7+5」的 5 相符：{cc['proper_L0N_match']}）",
            f"- proper 臂 L0（排 L0-14）：{cc['proper_L0_excl_ids']}"
            f"（n={len(cc['proper_L0_excl_ids'])}，與 §6.5-b「7」相符："
            f"{len(cc['proper_L0_excl_ids']) == 7}）",
            f"- proper 臂 L0（含 L0-14，union+遮罩）：{cc['proper_L0_incl_ids']}"
            f"（n={len(cc['proper_L0_incl_ids'])}——若採此案，L0-14 只能配 OLMo run，"
            "三個 Pythia run 遮罩掉，比照 L0N-10 的處置）",
            "\n## L0-14 排除與否，本檔未擅自決定（見程式 docstring 推論1）\n",
            "兩案都已落檔於 verbcue_items.json 的 proper.L0_including_L0-14／"
            "proper.L0_excluding_L0-14，待理論部/主線裁定用哪一案。\n"]
    report_path = PROJ / "planning" / "office_reports" / "verbcue_items_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("verb L0:", cc["verb_L0_ids"], "match:", cc["verb_L0_match"])
    print("verb L0N:", cc["verb_L0N_ids"], "match:", cc["verb_L0N_match"])
    print("proper L0N:", cc["proper_L0N_ids"], "match:", cc["proper_L0N_match"])
    print("proper L0 (excl L0-14):", cc["proper_L0_excl_ids"],
         "n=", len(cc["proper_L0_excl_ids"]))
    print("proper L0 (incl L0-14):", cc["proper_L0_incl_ids"],
         "n=", len(cc["proper_L0_incl_ids"]))
    print("->", out_path)
    print("->", report_path)


if __name__ == "__main__":
    main()
