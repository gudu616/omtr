"""R1（審查後探索）：teacher-forced 深度重測——解掉「兩跨度不同源」混淆。

原 pilot 的殘留混淆：記憶強度量在 gold 續文上、收斂深度量在模型自產的 greedy
token 上（大法官 F18 要求揭露、並點名此重測為可選升級）。本腳本把深度改到
**gold 續文本身的位置**上量：一次 forward with cache 蓋 prompt+gold，逐 gold
位置算逐層剖面 → 兩個變量同源，混淆關閉。

僅跑 L0 / L0N（有 gold 的條件）。輸出獨立於審查凍結版：results/night/。
用法： .venv/Scripts/python.exe harness/run_teacher_forced.py --model <name> [--eligible-as ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from run_pilot import Fp32Lens, depth_metrics, load_model, position_profile  # noqa: E402

MAX_GOLD_POSITIONS = 56  # gold 段 ~44 詞≈55 token，設上限防極端值


def run_item_tf(model, lens, item):
    prompt = item["prompt"]
    gold = " " + item["gold_continuation"].strip()
    tokens = model.to_tokens(prompt + gold)
    prompt_tokens = model.to_tokens(prompt)
    n_prompt = prompt_tokens.shape[1]
    if not torch.equal(tokens[0, :n_prompt], prompt_tokens[0]):
        return {"id": item["id"], "level": item["level"],
                "error": "tokenization_boundary_mismatch"}
    n_total = tokens.shape[1]
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("resid_post"))
    # gold 區段的預測位置：n_prompt-1 .. n_total-2（各預測下一個 gold token）
    positions = list(range(n_prompt - 1, min(n_total - 1, n_prompt - 1 + MAX_GOLD_POSITIONS)))
    n_layers = model.cfg.n_layers
    depths, entropies, at_cap = [], [], 0
    cap = (n_layers - 1) / n_layers
    for pos in positions:
        prof = position_profile(lens, cache, logits[0, pos], pos, n_layers)
        d = depth_metrics(prof["layers"], n_layers)
        depths.append(d["depth_tau_0.1"])
        entropies.append(prof["final_entropy"])
        if d["depth_tau_0.1"] >= cap - 1e-9:
            at_cap += 1
    # gold_logprob（同一次 forward）
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    targets = tokens[0, 1:]
    per_tok = logp[torch.arange(targets.shape[0]), targets]
    gold_lp = float(per_tok[n_prompt - 1:].mean())
    return {
        "id": item["id"], "level": item["level"], "n_gold_positions": len(positions),
        "gold_logprob_per_token": gold_lp,
        "tf_depth_tau_0.1_mean": float(np.mean(depths)),
        "tf_final_entropy_mean": float(np.mean(entropies)),
        "tf_frac_at_cap": at_cap / len(positions),
        "per_position_depth": [round(d, 4) for d in depths],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eligible-as", default=None)
    ap.add_argument("--battery", default=str(PROJ / "battery" / "battery.json"))
    args = ap.parse_args()

    eligible_name = args.eligible_as or args.model
    battery = json.load(open(args.battery, encoding="utf-8"))
    items = [it for it in battery["items"]
             if it["level"] in ("L0", "L0N") and it.get("gold_continuation")
             and eligible_name in it.get("eligible_models", [])]
    if not items:
        print(f"0 items selected for {args.model} (eligible-as={eligible_name})",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"model={args.model} items={len(items)} (teacher-forced L0/L0N)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = load_model(args.model, device, dtype)
    lens = Fp32Lens(model)

    records = []
    for i, item in enumerate(items):
        print(f"[{i + 1}/{len(items)}] {item['id']} ({item['level']})", flush=True)
        try:
            records.append(run_item_tf(model, lens, item))
        except Exception as e:
            records.append({"id": item["id"], "level": item["level"], "error": repr(e)})

    out_dir = PROJ / "results" / "night"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_")
    with open(out_dir / f"tf_depth_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "note": "teacher-forced depth (exploratory, post-review)",
                   "records": records}, f, ensure_ascii=False, indent=2)

    # 即時摘要：組內劑量反應（tf 版）
    from scipy.stats import spearmanr
    ok = [r for r in records if "error" not in r]
    for lv in ("L0", "L0N"):
        rs = [r for r in ok if r["level"] == lv]
        if len(rs) >= 6:
            rho, p = spearmanr([r["gold_logprob_per_token"] for r in rs],
                               [r["tf_depth_tau_0.1_mean"] for r in rs])
            rho_c, p_c = spearmanr([r["gold_logprob_per_token"] for r in rs],
                                   [r["tf_frac_at_cap"] for r in rs])
            print(f"  {lv}: n={len(rs)} tf_depth rho={rho:+.3f} (p={p:.4f})"
                  f" | tf_frac_at_cap rho={rho_c:+.3f} (p={p_c:.4f})")
    n_err = len(records) - len(ok)
    print(f"done: {len(ok)}/{len(records)} ok ({n_err} errors) -> {out_dir / f'tf_depth_{tag}.json'}")


if __name__ == "__main__":
    main()
