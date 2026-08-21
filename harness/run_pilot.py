"""OMTR pilot runner v2（依對抗式審查 16 項確認發現重寫）。

主要修正：
- Pythia 載入 shim：TL 3.7.3 的 neox 轉換器仍讀 `embed_out`，transformers 5.x 已改名
  `lm_head` → 自建 HF 模型、加 alias、經 hf_model= 傳入。
- 量測位置：不再只量 prompt 後單一 token，改為「先 greedy 生成 K 個 token、再對
  整段做一次 forward with cache」，在 K 個答案位置逐位置算逐層剖面後聚合——
  各條件的量測 locus 才可比。
- fp32 logit lens：ln_final/unembed 參數複製成 fp32，殘差流轉 fp32 再投影；
  KL/entropy 全走 log_softmax，無 epsilon。
- 收斂深度改為連續量：KL 門檻深度（tau 掃 0.05/0.1/0.5，除以層數正規化）＋
  KL 曲線 AUC ＋ top1-top2 margin 記錄；保留舊 argmax 版本供對照。
- 模型載入用 from_pretrained_no_processing（fp16 下 fold_ln 會累積誤差且 TL 明確警告）。
- battery 先驗證（id 唯一、level 齊全）才碰 GPU；單項錯誤不再可能炸整批。
- 操縱檢查（manipulation check）：L1 記錄 expected 是否出現於生成；L4 記錄雙域
  錨點與句數，分析階段據此分層。
- 輸出記錄完整 metadata（版本、dtype、參數）。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJ / "hf-cache"))

from transformer_lens import HookedTransformer  # noqa: E402

TAUS = (0.05, 0.1, 0.5)
K_ANSWER_TOKENS = 8


# ---------------------------------------------------------------- 模型載入

def load_model(name: str, device: str, dtype):
    """統一走 no_processing；Pythia/NeoX 需要 embed_out alias shim。"""
    hf_model = None
    if "pythia" in name.lower() or "neox" in name.lower():
        from transformers import GPTNeoXForCausalLM
        hf_name = name if "/" in name else f"EleutherAI/{name}"
        hf_model = GPTNeoXForCausalLM.from_pretrained(hf_name, torch_dtype=dtype)
        hf_model.embed_out = hf_model.lm_head  # TL 3.7.3 converter 相容 shim
    model = HookedTransformer.from_pretrained_no_processing(
        name, hf_model=hf_model, device=device, dtype=dtype)
    model.eval()
    return model


class Fp32Lens:
    """fp32 的 ln_final + unembed 複本，避免 fp16 誤差汙染逐層剖面。"""

    def __init__(self, model):
        cfg = model.cfg
        self.norm_type = cfg.normalization_type or "LN"
        self.eps = cfg.eps
        ln = model.ln_final
        self.w = ln.w.detach().float() if hasattr(ln, "w") else None
        self.b = ln.b.detach().float() if hasattr(ln, "b") else None
        self.W_U = model.unembed.W_U.detach().float()
        b_U = getattr(model.unembed, "b_U", None)
        self.b_U = b_U.detach().float() if b_U is not None else None

    def __call__(self, resid: torch.Tensor) -> torch.Tensor:
        x = resid.float()
        if self.norm_type.startswith("RMS"):
            x = x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)
        else:  # LN / LNPre
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(
                x.var(-1, unbiased=False, keepdim=True) + self.eps)
        if self.w is not None:
            x = x * self.w
        if self.b is not None:
            x = x + self.b
        logits = x @ self.W_U
        if self.b_U is not None:
            logits = logits + self.b_U
        return logits


# ---------------------------------------------------------------- 逐層剖面

def position_profile(lens: Fp32Lens, cache, final_logits: torch.Tensor,
                     pos: int, n_layers: int):
    """單一位置的逐層剖面（fp32、log_softmax 為底）。"""
    final_logp = torch.log_softmax(final_logits.float(), -1)
    final_p = final_logp.exp()
    final_top2 = torch.topk(final_logits.float(), 2).values
    final_top1 = int(final_logits.argmax())
    layers = []
    for layer in range(n_layers):
        logits = lens(cache["resid_post", layer][0, pos])
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        top2 = torch.topk(logits, 2).values
        layers.append({
            "entropy": float(-(p * logp).sum()),
            "kl_to_final": float((final_p * (final_logp - logp)).sum()),
            "top1": int(logits.argmax()),
            "top1_prob": float(p.max()),
            "margin": float(top2[0] - top2[1]),
            "matches_final": int(logits.argmax()) == final_top1,
        })
    return {
        "layers": layers,
        "final_entropy": float(-(final_p * final_logp).sum()),
        "final_margin": float(final_top2[0] - final_top2[1]),
    }


def depth_metrics(layers, n_layers: int):
    """連續收斂深度：KL 門檻深度（正規化）+ AUC + 舊版 argmax 深度。"""
    kls = [l["kl_to_final"] for l in layers]
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    out = {"kl_auc_norm": float(trapezoid(kls) / max(1, n_layers - 1))}
    for tau in TAUS:
        depth = 0
        for i in range(n_layers - 1, -1, -1):
            if kls[i] >= tau:
                depth = i + 1
                break
        out[f"depth_tau_{tau}"] = depth / n_layers
    last_false = -1
    for i, l in enumerate(layers):
        if not l["matches_final"]:
            last_false = i
    out["depth_argmax"] = (last_false + 1) / n_layers
    return out


# ---------------------------------------------------------------- 每項執行

def gold_logprob(model, prompt: str, gold: str):
    tokens = model.to_tokens(prompt + gold)
    prompt_tokens = model.to_tokens(prompt)
    n_prompt = prompt_tokens.shape[1]
    if not torch.equal(tokens[0, :n_prompt], prompt_tokens[0]):
        return None, "tokenization_boundary_mismatch"
    with torch.no_grad():
        logits = model(tokens)
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    targets = tokens[0, 1:]
    per_tok = logp[torch.arange(targets.shape[0]), targets]
    return float(per_tok[n_prompt - 1:].mean()), None


def manipulation_check(item, greedy_text: str, samples):
    level = item["level"]
    texts = [greedy_text] + list(samples)
    if level == "L1" and item.get("expected"):
        hit = any(item["expected"].lower() in t.lower() for t in texts)
        return {"l1_expected_in_output": hit}
    if level == "L4":
        anchors = [item.get("topic", ""), item.get("blend_partner", "")]
        both = all(a and any(a.lower().split()[0] in t.lower() for t in texts)
                   for a in anchors)
        n_sent = len(re.findall(r"[.!?]", greedy_text))
        return {"l4_both_domains": both, "l4_n_sentences": n_sent}
    return {}


def run_item(model, lens: Fp32Lens, item, gen_tokens=60, n_samples=3):
    prompt = item.get("run_prompt") or item["prompt"]
    tokens = model.to_tokens(prompt)
    n_prompt = tokens.shape[1]
    with torch.no_grad():
        greedy = model.generate(tokens, max_new_tokens=gen_tokens,
                                do_sample=False, verbose=False)
        full = greedy[:, : n_prompt + K_ANSWER_TOKENS]
        logits, cache = model.run_with_cache(
            full, names_filter=lambda n: n.endswith("resid_post"))
        # 取樣種子由 item id 導出（zlib.crc32，不用 hash()——process-random，
        # 前研究 rollout.py:509 的教訓），讓 samples 可重現
        import zlib
        samples = []
        for k in range(n_samples):
            torch.manual_seed(zlib.crc32(f"{item['id']}#{k}".encode()) % (2 ** 31))
            samples.append(model.to_string(model.generate(
                tokens, max_new_tokens=gen_tokens, do_sample=True,
                temperature=0.8, verbose=False)[0, n_prompt:]))
    greedy_text = model.to_string(greedy[0, n_prompt:])
    n_layers = model.cfg.n_layers
    # 答案位置：預測第 1..K 個答案 token 的位置（n_prompt-1 起）
    n_ans = min(K_ANSWER_TOKENS, full.shape[1] - n_prompt)
    positions = list(range(n_prompt - 1, n_prompt - 1 + n_ans))
    profiles, depths = [], []
    for pos in positions:
        prof = position_profile(lens, cache, logits[0, pos], pos, n_layers)
        profiles.append(prof)
        depths.append(depth_metrics(prof["layers"], n_layers))
    # 逐層聚合（各位置平均）
    agg_layers = [
        {
            "entropy": float(np.mean([p["layers"][l]["entropy"] for p in profiles])),
            "kl_to_final": float(np.mean([p["layers"][l]["kl_to_final"] for p in profiles])),
            "match_rate": float(np.mean([p["layers"][l]["matches_final"] for p in profiles])),
            "margin": float(np.mean([p["layers"][l]["margin"] for p in profiles])),
        }
        for l in range(n_layers)
    ]
    agg_depth = {k: float(np.mean([d[k] for d in depths])) for k in depths[0]} if depths else {}
    rec = {
        "id": item["id"],
        "level": item["level"],
        "topic": item.get("topic"),
        "prompt": prompt,
        "n_prompt_tokens": int(n_prompt),
        "layer_profile": agg_layers,
        "per_position_depths": depths,
        "depth": agg_depth,
        "final_entropy_mean": float(np.mean([p["final_entropy"] for p in profiles])),
        "final_margin_mean": float(np.mean([p["final_margin"] for p in profiles])),
        "first_pos_profile": profiles[0]["layers"] if profiles else None,
        "greedy_continuation": greedy_text,
        "samples": samples,
        "manipulation": manipulation_check(item, greedy_text, samples),
    }
    if item.get("gold_continuation"):
        lp, err = gold_logprob(model, prompt, " " + item["gold_continuation"].strip())
        rec["gold_logprob_per_token"] = lp
        if err:
            rec["gold_logprob_error"] = err
    # RSA 素材：第一個答案位置 + K 位置平均，各層 resid
    first_stack = np.stack([
        cache["resid_post", l][0, positions[0]].float().cpu().numpy()
        for l in range(n_layers)])
    mean_stack = np.stack([
        cache["resid_post", l][0, positions].float().mean(0).cpu().numpy()
        for l in range(n_layers)])
    return rec, first_stack, mean_stack


# ---------------------------------------------------------------- 主流程

def validate_battery(items):
    problems = []
    seen = set()
    for i, it in enumerate(items):
        if not it.get("id"):
            problems.append(f"item[{i}] missing id")
        elif it["id"] in seen:
            problems.append(f"duplicate id: {it['id']}")
        else:
            seen.add(it["id"])
        if not it.get("level"):
            problems.append(f"item[{i}] missing level")
        if not it.get("prompt"):
            problems.append(f"item[{i}] missing prompt")
    if problems:
        raise SystemExit("battery validation failed:\n  " + "\n  ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--battery", default=str(PROJ / "battery" / "battery.json"))
    ap.add_argument("--out", default=str(PROJ / "results" / "raw"))
    ap.add_argument("--levels", default="L0,L0N,L1,L4")
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--eligible-as", default=None,
                    help="以哪個模型名過濾 L0/L0N 資格（同語料家族共用，如 Pythia 系列全部用 EleutherAI/pythia-1.4b）")
    ap.add_argument("--limit", type=int, default=0, help="每個 level 最多跑幾項（0=全部，煙霧測試用）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = set(args.levels.split(","))

    with open(args.battery, encoding="utf-8") as f:
        battery = json.load(f)
    eligible_name = args.eligible_as or args.model
    items = [it for it in battery["items"]
             if it["level"] in levels
             and eligible_name in it.get("eligible_models", [eligible_name])]
    if args.limit:
        by_level: dict = {}
        capped = []
        for it in items:
            by_level.setdefault(it["level"], []).append(it)
        for lv_items in by_level.values():
            capped.extend(lv_items[: args.limit])
        items = capped
    validate_battery(items)
    if not items:
        # 一定要擋在載模型之前：否則跑完會留下 done: 0/0 ok 與三個空產物，
        # 看起來像成功。最常見的原因是 --eligible-as 沒帶（L0/L0N 的
        # eligible_models 寫的是同語料家族的代表模型名，不是每個 checkpoint）。
        hint = ("allenai/OLMo-2-0425-1B" if "olmo" in args.model.lower()
                else "EleutherAI/pythia-1.4b")
        print(f"0 items selected: model={args.model} levels={sorted(levels)} "
              f"eligible_as={eligible_name}\n"
              f"  L0/L0N carry eligible_models for the corpus-family representative only;\n"
              f"  other checkpoints in the family need --eligible-as, e.g.\n"
              f"    --model {args.model} --eligible-as {hint}\n"
              f"  (also check that --levels names levels present in {args.battery})",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"model={args.model} items={len(items)} levels={sorted(levels)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = load_model(args.model, device, dtype)
    lens = Fp32Lens(model)

    records, first_resids, mean_resids = [], {}, {}
    for i, item in enumerate(items):
        item_id = item.get("id") or f"{item.get('level', 'LX')}_{i:03d}"
        print(f"[{i + 1}/{len(items)}] {item_id} ({item.get('level')})", flush=True)
        try:
            rec, first_stack, mean_stack = run_item(
                model, lens, item, gen_tokens=args.gen_tokens)
            records.append(rec)
            first_resids[item_id] = first_stack
            mean_resids[item_id] = mean_stack
        except Exception as e:
            records.append({"id": item_id, "level": item.get("level"),
                            "error": repr(e)})

    import transformer_lens
    import transformers
    import importlib.metadata as md
    meta = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "loader": "from_pretrained_no_processing + neox hf_model shim",
        "transformer_lens": md.version("transformer-lens"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "taus": list(TAUS),
        "k_answer_tokens": K_ANSWER_TOKENS,
        "gen_tokens": args.gen_tokens,
        "battery_version": battery.get("version"),
        "n_items": len(items),
    }
    tag = args.model.replace("/", "_")
    with open(out_dir / f"pilot_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / f"resid_first_{tag}.npz", **first_resids)
    np.savez_compressed(out_dir / f"resid_mean_{tag}.npz", **mean_resids)
    n_ok = sum(1 for r in records if "error" not in r)
    print(f"done: {n_ok}/{len(records)} ok -> {out_dir}")
    _ = transformer_lens


if __name__ == "__main__":
    main()
