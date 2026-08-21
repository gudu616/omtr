"""battery v2 總組裝器（整合 harness 審查 16 項 + QC 29 項的所有修正）。

輸入：
  battery/l0_candidates.json + battery/l0_verification.json        （L0 v3）
  battery/l0_control_candidates.json + battery/l0n_verification.json（L0N 對照組）
  battery/items_draft.json                                          （3 designer 原稿）
  battery/qc_report.json                                            （24 題 L1 cloze + 鷹架）
  battery/items_v2_patch.json                                       （QC 手術包）
輸出：
  battery/battery.json（version v2，含 audit 與 per-model token 共變量）
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
BAT = PROJ / "battery"

MODELS = {"pile": "EleutherAI/pythia-1.4b", "olmo_mix": "allenai/OLMo-2-0425-1B"}
ALL_MODELS = list(MODELS.values())

# L1 長度墊料：語意惰性前言，把 cloze 提示墊到接近 L0 的 token 預算
L1_PREAMBLE = ("The following short statements were collected for a "
               "general-knowledge reader. They are unrelated to one another and "
               "appear in no particular order. Each was checked for accuracy "
               "before being included in the collection.\n\n")


def load(name):
    with open(BAT / name, encoding="utf-8") as f:
        return json.load(f)


def draft_id(it, counter):
    key = f"{it['topic'][:4]}-{it['level']}"
    counter[key] = counter.get(key, 0) + 1
    return f"{it['level']}-{it['topic'].replace(' ', '_')}-{counter[key]}"


def passage_items(candidates, verification, level, id_prefix):
    ver = {r["title"]: r for r in verification}
    items, dropped = [], []
    n = 0
    for p in candidates:
        v = ver.get(p["title"], {}).get("verified", {})
        eligible = [MODELS[c] for c, ok in v.items() if ok is True]
        unknown = [c for c, ok in v.items() if ok is None]
        if unknown:
            dropped.append({"title": p["title"], "reason": f"verification unknown: {unknown}"})
            continue
        if not eligible:
            dropped.append({"title": p["title"], "reason": "no corpus passed"})
            continue
        n += 1
        items.append({
            "id": f"{id_prefix}-{n:02d}",
            "level": level,
            "topic": "canon" if level == "L0" else "obscure-control",
            "prompt": p["prompt_text"],
            "gold_continuation": p["continuation_text"],
            "expected": "verbatim continuation" if level == "L0"
                        else "any fluent continuation (non-memorized control)",
            "source": p["source_work"],
            "title": p["title"],
            "boundary_word": p.get("boundary_word"),
            "eligible_models": eligible,
            "verified_in": [c for c, ok in v.items() if ok is True],
        })
    return items, dropped


def main():
    l0_items, l0_dropped = passage_items(
        load("l0_candidates.json")["l0_candidates"],
        load("l0_verification.json"), "L0", "L0")
    l0n_items, l0n_dropped = passage_items(
        load("l0_control_candidates.json")["l0_candidates"],
        load("l0n_verification.json"), "L0N", "L0N")

    draft = load("items_draft.json")["items"]
    qc = load("qc_report.json")
    patch = load("items_v2_patch.json")

    reformats = {r["id"]: r for r in qc["l1_reformats"]}
    for r in patch.get("l1_replacements", []):
        reformats[r["id"]] = r
    # 機器可用模板（qc_report 的 level_scaffolds 混有設計註解，不能直接用）
    scaffolds = {k: v.rstrip() for k, v in load("scaffolds_final.json").items()}
    for lv, sc in scaffolds.items():
        assert "{ITEM_PROMPT}" in sc, f"scaffold {lv} missing placeholder"
        assert sc.endswith(":"), f"scaffold {lv} must end with ':' (got ...{sc[-10:]!r})"

    l4_by_topic = {it["topic"]: it for it in patch["l4_items"]}
    l5_by_topic = {it["topic"]: it for it in patch["l5_items"]}
    l2_repl = {it["topic"]: it for it in patch.get("l2_replacements", [])}
    l3_rewd = {it["topic"]: it for it in patch.get("l3_rewordings", [])}

    items = l0_items + l0n_items
    counter: dict = {}
    for it in draft:
        iid = draft_id(it, counter)
        lv, topic = it["level"], it["topic"]
        rec = {"id": iid, "level": lv, "topic": topic,
               "eligible_models": ALL_MODELS}
        if lv == "L1":
            rf = reformats.get(iid)
            if not rf:
                raise SystemExit(f"missing L1 reformat for {iid}")
            answer = rf["answer_token"].strip()
            rec.update({
                "prompt": rf["cloze_prompt"],
                "run_prompt": L1_PREAMBLE + rf["cloze_prompt"],
                "gold_continuation": answer,
                "expected": answer,
                "qc_problem": rf.get("problem"),
            })
        else:
            prompt, expected = it["prompt"], it.get("expected", "")
            if lv == "L2" and topic in l2_repl:
                prompt, expected = l2_repl[topic]["prompt"], l2_repl[topic]["expected"]
            elif lv == "L3" and topic in l3_rewd:
                prompt = l3_rewd[topic]["prompt"]
            elif lv == "L4":
                p4 = l4_by_topic.get(topic)
                if not p4:
                    raise SystemExit(f"patch missing L4 for topic {topic}")
                prompt, expected = p4["prompt"], p4["expected"]
                rec["blend_partner"] = p4.get("blend_partner")
            elif lv == "L5":
                p5 = l5_by_topic.get(topic)
                if not p5:
                    raise SystemExit(f"patch missing L5 for topic {topic}")
                prompt, expected = p5["prompt"], p5["expected"]
            if lv != "L4":
                rec["blend_partner"] = it.get("blend_partner")
            rec.update({
                "prompt": prompt,
                "run_prompt": scaffolds[lv].replace("{ITEM_PROMPT}", prompt),
                "expected": expected,
            })
        items.append(rec)

    # 雙 tokenizer 共變量
    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(m) for m in ALL_MODELS}
    for it in items:
        text = it.get("run_prompt") or it["prompt"]
        it["prompt_tokens"] = {m: len(t.encode(text)) for m, t in toks.items()}
        if it.get("gold_continuation"):
            it["gold_tokens"] = {
                m: len(t.encode(" " + it["gold_continuation"], add_special_tokens=False))
                for m, t in toks.items()}

    out = {
        "version": "v2.1",  # L0N 對照組補上 shift_split 實詞邊界配平
        "built": "2026-08-20",
        "audit": {
            "l0_dropped": l0_dropped,
            "l0n_dropped": l0n_dropped,
            "sources": ["items_draft(3 designers)", "qc_report(Opus QC)",
                        "items_v2_patch(Opus surgery)", "l0 v3 gutenberg",
                        "l0n obscure controls"],
        },
        "items": items,
    }
    with open(BAT / "battery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    by: dict = {}
    for it in items:
        by[it["level"]] = by.get(it["level"], 0) + 1
    print("battery v2:", len(items), "items", dict(sorted(by.items())))
    for corpus, model in MODELS.items():
        for lv in ("L0", "L0N"):
            n = sum(1 for it in items if it["level"] == lv and model in it["eligible_models"])
            print(f"  {lv} eligible[{corpus}]: {n}")
    med = {}
    for lv in sorted(by):
        ts = sorted(it["prompt_tokens"][ALL_MODELS[0]] for it in items if it["level"] == lv)
        med[lv] = ts[len(ts) // 2]
    print("  median pythia prompt tokens:", med)
    if l0_dropped or l0n_dropped:
        print("  dropped:", [d["title"] for d in l0_dropped + l0n_dropped])


if __name__ == "__main__":
    main()
