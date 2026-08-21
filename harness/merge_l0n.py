"""把 v2.1 配平版 L0N 重跑結果併回各模型的主結果檔（舊 L0N 記錄汰換）。"""
import json
import shutil
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "results" / "raw"
NEW = PROJ / "results" / "raw_l0n_v21"

TAGS = ["EleutherAI_pythia-410m", "EleutherAI_pythia-1b", "EleutherAI_pythia-1.4b",
        "EleutherAI_pythia-2.8b", "allenai_OLMo-2-0425-1B"]


def merge_npz(main_path: Path, new_path: Path):
    main = dict(np.load(main_path).items()) if main_path.exists() else {}
    new = dict(np.load(new_path).items())
    main = {k: v for k, v in main.items() if not k.startswith("L0N-")}
    main.update(new)
    np.savez_compressed(main_path, **main)


def main():
    for tag in TAGS:
        main_p = RAW / f"pilot_{tag}.json"
        new_p = NEW / f"pilot_{tag}.json"
        if not (main_p.exists() and new_p.exists()):
            print(f"skip {tag}: missing {'main' if not main_p.exists() else 'new'}")
            continue
        shutil.copy2(main_p, main_p.with_suffix(".json.bak_pre_v21"))
        main = json.load(open(main_p, encoding="utf-8"))
        new = json.load(open(new_p, encoding="utf-8"))
        kept = [r for r in main["records"] if r.get("level") != "L0N"]
        n_old = len(main["records"]) - len(kept)
        main["records"] = kept + new["records"]
        main["meta"]["l0n_version"] = "v2.1-shift-split-matched"
        with open(main_p, "w", encoding="utf-8") as f:
            json.dump(main, f, ensure_ascii=False, indent=2)
        for kind in ("resid_first", "resid_mean"):
            merge_npz(RAW / f"{kind}_{tag}.npz", NEW / f"{kind}_{tag}.npz")
        print(f"{tag}: replaced {n_old} old L0N with {len(new['records'])} v2.1 records")


if __name__ == "__main__":
    main()
