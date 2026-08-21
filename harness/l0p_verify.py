"""L0P 新穎性閘門：每題 11 窗（10 個 7 字窗 stride4 掃 gold 段＋1 個跨界窗），
兩份語料各驗一次，門檻照 L0N：**每一個窗 count ≤ 2** 才合格（對照組必須乾淨）。

同時做欄位對齊：battery_l0p.json 的每題補上 run_pilot 需要的 gold_continuation。
輸出：battery/l0p_verification.json；不合格題直接列名（預註冊 R 規則會剔除）。
"""
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "harness"))
from verify_battery import INDEXES, probes_from, query, _edge_trim  # noqa: E402

MAX_COUNT = 2
bat_path = PROJ / "battery" / "battery_l0p.json"
bat = json.load(open(bat_path, encoding="utf-8"))

# 欄位對齊（run_pilot 讀 gold_continuation）
changed = False
for it in bat["items"]:
    if "gold_continuation" not in it:
        it["gold_continuation"] = it["expected_continuation"]
        changed = True
if changed:
    json.dump(bat, open(bat_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[ok] gold_continuation 欄位補齊")

out = []
bad = []
for it in bat["items"]:
    cont_words = it["gold_continuation"].split()
    prompt_words = it["prompt"].split()
    probes = probes_from(it["gold_continuation"], n_words=7, stride=4)[:10]
    boundary = _edge_trim(prompt_words[-3:] + cont_words[:4])
    if boundary:
        probes.append(" ".join(boundary))  # _edge_trim 回傳字串陣列，API 要句子
    rec = {"id": it["id"], "tier": it["tier"], "probes": {}}
    verdicts = {}
    for key, index in INDEXES.items():
        rs = []
        for p in probes:
            status, count = query(index, p)
            rs.append({"probe": p, "status": status, "count": count})
            time.sleep(0.6)
        rec["probes"][key] = rs
        oks = [r for r in rs if r["status"] == "ok"]
        if len(oks) != len(rs):
            verdicts[key] = None
        else:
            verdicts[key] = all(r["count"] <= MAX_COUNT for r in oks)
    rec["verified"] = verdicts
    out.append(rec)
    flag = "OK " if all(v is True for v in verdicts.values()) else "FAIL"
    if flag == "FAIL":
        bad.append(it["id"])
    mx = {k: max((r["count"] for r in rec["probes"][k] if r["status"] == "ok"),
                 default=-1) for k in INDEXES}
    print(f"{flag} {it['id']} ({it['tier']:<6}) max_count={mx}")

json.dump(out, open(PROJ / "battery" / "l0p_verification.json", "w",
                    encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n{len(out) - len(bad)}/{len(out)} 合格；不合格：{bad or '無'}")
sys.exit(1 if bad else 0)
