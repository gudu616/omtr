"""組裝最終 battery.json：L0（含 per-model 語料驗證標記）＋ L1–L5 items。"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
BAT = PROJ / "battery"

with open(BAT / "items_draft.json", encoding="utf-8") as f:
    draft = json.load(f)["items"]
with open(BAT / "l0_candidates.json", encoding="utf-8") as f:
    l0 = json.load(f)["l0_candidates"]
with open(BAT / "l0_verification.json", encoding="utf-8") as f:
    ver = {r["title"]: r for r in json.load(f)}

CORPUS_MODEL = {"pile": "EleutherAI/pythia-1.4b", "olmo_mix": "allenai/OLMo-2-0425-1B"}

items = []
for i, p in enumerate(l0):
    v = ver.get(p["title"], {}).get("verified", {})
    eligible = [CORPUS_MODEL[c] for c, ok in v.items() if ok]
    if not eligible:
        continue  # 兩個語料都沒驗證到 → 不能當 L0
    items.append({
        "id": f"L0-{i+1:02d}",
        "level": "L0",
        "topic": "canon",
        "prompt": p["prompt_text"],
        "gold_continuation": p["continuation_text"],
        "expected": "verbatim continuation of the canonical passage",
        "source": p["source_work"],
        "title": p["title"],
        "eligible_models": eligible,
        "provenance": v,
    })

level_counter: dict = {}
for it in draft:
    lv, topic = it["level"], it["topic"]
    key = f"{topic[:4]}-{lv}"
    level_counter[key] = level_counter.get(key, 0) + 1
    items.append({
        "id": f"{lv}-{topic.replace(' ', '_')}-{level_counter[key]}",
        "level": lv,
        "topic": topic,
        "prompt": it["prompt"],
        "expected": it.get("expected", ""),
        "blend_partner": it.get("blend_partner"),
        "eligible_models": ["EleutherAI/pythia-1.4b", "allenai/OLMo-2-0425-1B"],
    })

out = {"version": "v1", "built": "2026-08-20", "items": items}
with open(BAT / "battery.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

by = {}
for it in items:
    by[it["level"]] = by.get(it["level"], 0) + 1
print("battery.json:", len(items), "items", by)
n_pythia_l0 = sum(1 for it in items if it["level"] == "L0" and "EleutherAI/pythia-1.4b" in it["eligible_models"])
n_olmo_l0 = sum(1 for it in items if it["level"] == "L0" and "allenai/OLMo-2-0425-1B" in it["eligible_models"])
print(f"L0 eligible: pythia={n_pythia_l0}, olmo={n_olmo_l0}")
