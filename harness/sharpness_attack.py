"""攻擊倖存主張：記憶劑量→輸出篤定，是記憶專屬還是一般可預測性？

三個零 GPU 檢驗（全部只用已存檔一手資料）：
  T1  counts→entropy：模型外部的語料次數直接對輸出熵——外部錨點能不能一路
      接到結果變數（目前論文只錨到 gold）。
  T2  偏相關拆解：gold~entropy | counts 與 counts~entropy | gold——
      「模型內部的劑量尺」和「模型外部的劑量尺」誰在扛。
  T3  子詞破碎控制：L0 組內 gold~entropy | (每字平均 token 數)——
      詞彙稀有度混淆吃掉多少。
偏相關 = 先取名次再做部分相關（Spearman 偏相關慣例）。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

PROJ = Path(__file__).resolve().parent.parent
MODELS = [("pythia-410m", "EleutherAI_pythia-410m", "pile"),
          ("pythia-1b", "EleutherAI_pythia-1b", "pile"),
          ("pythia-1.4b", "EleutherAI_pythia-1.4b", "pile"),
          ("pythia-2.8b", "EleutherAI_pythia-2.8b", "pile"),
          ("OLMo-2-1B", "allenai_OLMo-2-0425-1B", "olmo_mix")]


def partial_spear(x, y, z):
    """rho(x, y | z)：名次化後的部分相關。回傳 (r, n)。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    r = float(np.corrcoef(ex, ey)[0, 1])
    return r, len(x)


def pval(r, n, k=1):
    """偏相關的 t 檢定近似（雙尾）。k=控制變數數。"""
    from scipy.stats import t as tdist
    df = n - 2 - k
    if df <= 0 or abs(r) >= 1:
        return float("nan")
    t = r * np.sqrt(df / (1 - r * r))
    return float(2 * tdist.sf(abs(t), df))


battery = {i["id"]: i for i in
           json.load(open(PROJ / "battery" / "battery.json", encoding="utf-8"))["items"]}
ver = {r["title"]: r for r in
       json.load(open(PROJ / "battery" / "l0_verification.json", encoding="utf-8"))}


def counts_for(item_id, corpus):
    title = battery.get(item_id, {}).get("title")
    v = ver.get(title)
    if not v:
        return None
    ok = [p["count"] for p in v["probes"].get(corpus, []) if p.get("status") == "ok"]
    return float(np.median(ok)) if ok else None


print(f"{'model':<12} {'n':>3} | {'T1 cnt~ent':>12} | {'T2a gold~ent|cnt':>17} "
      f"| {'T2b cnt~ent|gold':>17} | {'T3 gold~ent|frag':>17}")
rows = {}
for short, tag, corpus in MODELS:
    recs = [r for r in json.load(open(PROJ / "results" / "raw" / f"pilot_{tag}.json",
                                      encoding="utf-8"))["records"]
            if r.get("level") == "L0" and "error" not in r
            and r.get("gold_logprob_per_token") is not None]
    data = []
    for r in recs:
        c = counts_for(r["id"], corpus)
        if c is None:
            continue
        # 子詞破碎度：gold 段的 token 數 / 英文字數（字數以空白切）
        n_words = len(r["prompt"].split()) or 1
        frag = r["n_prompt_tokens"] / n_words
        data.append((np.log10(c + 1), r["gold_logprob_per_token"],
                     r["final_entropy_mean"], frag))
    cnt, gold, ent, frag = map(np.array, zip(*data))
    t1_r, t1_p = spearmanr(cnt, ent)
    t2a_r, n = partial_spear(gold, ent, cnt); t2a_p = pval(t2a_r, n)
    t2b_r, _ = partial_spear(cnt, ent, gold); t2b_p = pval(t2b_r, n)
    t3_r, _ = partial_spear(gold, ent, frag); t3_p = pval(t3_r, n)
    rows[short] = dict(n=n, t1=(t1_r, t1_p), t2a=(t2a_r, t2a_p),
                       t2b=(t2b_r, t2b_p), t3=(t3_r, t3_p))
    fmt = lambda rp: f"{rp[0]:+.3f} ({rp[1]:.3f})"
    print(f"{short:<12} {n:>3} | {fmt((t1_r, t1_p)):>12} | {fmt((t2a_r, t2a_p)):>17} "
          f"| {fmt((t2b_r, t2b_p)):>17} | {fmt((t3_r, t3_p)):>17}")

out = PROJ / "results" / "relative_depth" / "sharpness_attack.json"
json.dump(rows, open(out, "w", encoding="utf-8"), indent=2, default=float)
print(f"\nwrote {out}")
