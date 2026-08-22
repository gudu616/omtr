"""Provenance baseline for the causal phase, all five runs.

The provenance track's clean-run distribution (frac_present of each model's own
greedy continuation, in that model's own training corpus) existed for only two
runs (planning/office_reports/theory_provenance_floor.py). The causal phase
needs it for every run it will patch. This script produces ONE uniform artifact
covering all five, importing the two already-measured runs' rows verbatim
(origin field records where each row came from; no re-querying).

Protocol (frozen, spec-completion note in CAUSAL_PREREG_v1 §2): frac_present
over 6-word windows at stride 4 of the stored 60-token greedy continuation,
queried against that model's own training-corpus infini-gram index. Identical
to harness/check_l5_novelty.py and the floor script.

Run:  .venv/Scripts/python.exe harness/provenance_baseline.py
Out:  results/causal/provenance_baseline.json   (resumable; finished rows kept)
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from verify_battery import check_novelty  # noqa: E402

MODELS = {
    "EleutherAI_pythia-410m": "pile",
    "EleutherAI_pythia-1b": "pile",
    "EleutherAI_pythia-1.4b": "pile",
    "EleutherAI_pythia-2.8b": "pile",
    "allenai_OLMo-2-0425-1B": "olmo_mix",
}
LEVELS = {"L0": "raw", "L0N": "raw", "L0P": "l0p"}
FLOOR = PROJ / "planning" / "office_reports" / "theory_provenance_floor.json"
OUT = PROJ / "results" / "causal" / "provenance_baseline.json"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else []
    done = set((r["model"], r["id"]) for r in report)
    if FLOOR.exists():
        for r in json.load(open(FLOOR, encoding="utf-8")):
            if (r["model"], r["id"]) not in done:
                r = dict(r)
                r["origin"] = "theory_provenance_floor"
                report.append(r)
                done.add((r["model"], r["id"]))
    for tag, corpus in MODELS.items():
        for level, sub in LEVELS.items():
            path = PROJ / "results" / sub / ("pilot_" + tag + ".json")
            if not path.exists():
                continue  # e.g. no L0P run for the three small Pythias
            data = json.load(open(path, encoding="utf-8"))
            for r in data["records"]:
                if r.get("level") != level or "error" in r:
                    continue
                if (tag, r["id"]) in done:
                    continue
                text = r.get("greedy_continuation", "")
                if len(text.split()) < 12:
                    report.append({"model": tag, "level": level, "id": r["id"],
                                   "status": "too_short", "origin": "baseline"})
                    done.add((tag, r["id"]))
                    continue
                nv = check_novelty(text, index_key=corpus, n_words=6, stride=4)
                row = {"model": tag, "level": level, "id": r["id"],
                       "corpus": corpus, "n_probes": nv["n_probes"],
                       "n_unknown": nv["n_unknown"],
                       "frac_present": nv["frac_present"],
                       "max_count": nv["max_count"], "origin": "baseline"}
                report.append(row)
                done.add((tag, r["id"]))
                print("%s %s %-14s frac_present=%s max=%s" % (
                    tag, level, r["id"], row["frac_present"], row["max_count"]),
                    flush=True)
                json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
    n_live = sum(1 for r in report if r.get("frac_present") is not None)
    print("done: %d rows (%d measured) -> %s" % (len(report), n_live, OUT))


if __name__ == "__main__":
    main()
