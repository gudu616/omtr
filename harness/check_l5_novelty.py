"""L5 新穎度溯源：模型「發明」的東西，語料裡真的查無嗎？

對每個 L5 生成（greedy + 1 個取樣）做滑動 n-gram 溯源，
對應各模型自己的訓練語料索引。輸出連續量（frac_present 等），不做二元判定。

用法： .venv/Scripts/python.exe harness/check_l5_novelty.py
輸出： archive/novelty_report.json（historical，GPT 二審後移出 results/；
       這是一次性的 L5 溯源查證，不是現行 pipeline 的一步，故落點跟著移）
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from verify_battery import check_novelty  # noqa: E402

MODELS = {
    "EleutherAI_pythia-1.4b": "pile",
    "allenai_OLMo-2-0425-1B": "olmo_mix",
}


def main():
    report = []
    for tag, corpus in MODELS.items():
        path = PROJ / "results" / "raw" / f"pilot_{tag}.json"
        data = json.load(open(path, encoding="utf-8"))
        for r in data["records"]:
            if r.get("level") != "L5" or "error" in r:
                continue
            texts = {"greedy": r.get("greedy_continuation", "")}
            if r.get("samples"):
                texts["sample0"] = r["samples"][0]
            for kind, text in texts.items():
                if not text or len(text.split()) < 12:
                    report.append({"model": tag, "id": r["id"], "kind": kind,
                                   "status": "too_short"})
                    continue
                print(f"[{tag}] {r['id']} ({kind}) ...", flush=True)
                nv = check_novelty(text, index_key=corpus, n_words=6, stride=4)
                report.append({
                    "model": tag, "id": r["id"], "kind": kind, "corpus": corpus,
                    "n_probes": nv["n_probes"], "n_unknown": nv["n_unknown"],
                    "frac_present": nv["frac_present"],
                    "max_count": nv["max_count"],
                })
                fp = nv["frac_present"]
                print(f"  -> frac_present={fp if fp is None else round(fp, 2)} "
                      f"max_count={nv['max_count']}", flush=True)
    out = PROJ / "archive" / "novelty_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ok = [r for r in report if r.get("frac_present") is not None]
    if ok:
        import statistics
        med = statistics.median(r["frac_present"] for r in ok)
        print(f"done: {len(ok)} checks, median frac_present={med:.2f} -> {out}")


if __name__ == "__main__":
    main()
