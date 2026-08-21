"""Copy vs citation, decided by how the corpus count DECAYS with window length.

If a passage's high 7-gram counts came from N whole-book copies, every window
length would keep hitting those same N documents: the count curve would PLATEAU
at ~N. If they come from independent quotations, longer windows outrun the
quoted fragment and the count falls off a cliff.

For each L0 item we query the same starting offset at increasing lengths and
record the curve, in each model's own training corpus.

CAVEAT that must travel with the numbers: count(long)=0 is ambiguous. It can
mean "no whole copy in the corpus" OR "our source text differs from the corpus
copy by a character somewhere in that span". A single punctuation difference
breaks a 40-token match and leaves a 10-token match intact. So the curve is
read as a SHAPE (plateau vs cliff), and the fraction of items with any positive
long-window count is reported so the reader can judge how much of the zero rate
is edition mismatch.

Run: .venv/Scripts/python.exe harness/gate_decay_probe.py
Resumable: finished items are kept in the json and skipped.
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "harness"))
from verify_battery import query, INDEXES  # noqa: E402

LENGTHS = [7, 12, 20, 30, 44]
CORPORA = ["pile", "olmo_mix"]
OUT = PROJ / "results" / "gate" / "gate_decay.json"


def main():
    bat = {i["id"]: i for i in json.load(
        open(PROJ / "battery" / "battery.json", encoding="utf-8"))["items"]}
    rows = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else []
    done = set((r["id"], r["corpus"]) for r in rows)
    ids = sorted(i for i in bat if i.startswith("L0-"))
    for iid in ids:
        txt = bat[iid].get("gold_continuation") or ""
        w = txt.split()
        if len(w) < 20:
            continue
        for corpus in CORPORA:
            if (iid, corpus) in done:
                continue
            curve = {}
            for n in LENGTHS:
                if n > len(w):
                    continue
                st, c = query(INDEXES[corpus], " ".join(w[:n]))
                curve[str(n)] = c if st == "ok" else None
            rows.append({"id": iid, "corpus": corpus, "n_words": len(w),
                         "curve": curve})
            print("%-10s %-9s %s" % (iid, corpus,
                                     " ".join("%s:%s" % (k, v) for k, v in curve.items())),
                  flush=True)
            json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
    print("done -> " + str(OUT))


if __name__ == "__main__":
    main()
