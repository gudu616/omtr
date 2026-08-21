"""數字對帳器：把 Markdown 裡引用的每個統計量，回到 raw 重算一次。

存在理由：草稿 `:23` 與 `DISCLOSURE.md` 都宣稱「所有數字都能從 raw JSON 重算」。
這支程式讓那句話變成可執行的斷言，而不是一句自我宣告。

鐵律——只讀原始資料：
  results/raw/pilot_*.json（逐項紀錄）＋ *.bak_pre_v21（v2.1 前的對照組，供歷史值對帳）
  battery/battery.json、battery/l0_verification.json（題庫與語料查證）
  results/novelty_report.json（L5 溯源查詢的原始回傳；frac_present 是量測值不是統計量，
    而且 infini-gram 查詢無法離線重跑，所以它算 primary，不算中間值）
**絕不**讀 results/followup_report.json 或 results/analysis_*.json：那兩份存的是
`round(rho, 3)` 之後的中間值，拿它當真相會二次進位，等於認證掉自己要抓的錯
（審查團就有兩位審查員被 followup_report.json 的 0.645 騙過）。

兩層覆蓋，缺一不可：
  1. anchor 檢查——認得句子，能判斷「這個位置該放哪個統計量」，最精確。
  2. 值域掃描——不看句子，把文件裡每個數字拿去比對整張統計量表的所有合法寫法。
     改寫文件會讓 anchor 失效（v0.3 就是這樣：v0.2 的 anchor 全部落空），
     值域掃描是那時候唯一還在工作的東西，也是「0 命中卻印通過」的解藥。

用法：
    .venv/Scripts/python.exe harness/reconcile.py docs/WRITEUP_v0.5_EN.md
    .venv/Scripts/python.exe harness/reconcile.py docs/WRITEUP_v0.5_EN.md docs/learn/*.md
    .venv/Scripts/python.exe harness/reconcile.py --dump-stats      # 印出全部重算值
    .venv/Scripts/python.exe harness/reconcile.py --list-uncovered docs/learn/08-kl-depth.md

離開碼：0 = 全部對帳通過；1 = 有不符；2 = 用法或資料錯誤。

三種判定：
  PRESENT-AND-CORRECT     文件引用了，而且是從 raw 正確進位而來
  PRESENT-BUT-WRONG       文件引用了，但對不上 raw（附註是否為二次進位所致）
  CLAIMED-BUT-UNVERIFIABLE 文件引用了，但 raw 根本產不出這個量（無法認證）
"""
import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rankdata, spearmanr

PROJ = Path(__file__).resolve().parent.parent

TAGS = {
    "410m": "EleutherAI_pythia-410m",
    "1b": "EleutherAI_pythia-1b",
    "1.4b": "EleutherAI_pythia-1.4b",
    "2.8b": "EleutherAI_pythia-2.8b",
    "olmo": "allenai_OLMo-2-0425-1B",
}
ORDER = ["410m", "1b", "1.4b", "2.8b", "olmo"]        # 論文一律用這個模型順序
CORPUS_OF = {t: ("olmo_mix" if "OLMo" in tag else "pile") for t, tag in TAGS.items()}


# ---------------------------------------------------------------- raw 載入

class Raw:
    """五份 pilot raw + 題庫 + 語料查證。除此之外什麼都不讀。"""

    def __init__(self, proj: Path = PROJ):
        self.proj = proj
        self.pilots, self.n_layers = {}, {}
        for short, tag in TAGS.items():
            path = proj / "results" / "raw" / f"pilot_{tag}.json"
            if not path.exists():
                raise SystemExit(f"[reconcile] 找不到 raw：{path}")
            data = json.load(open(path, encoding="utf-8"))
            recs = [r for r in data["records"] if "error" not in r]
            self.pilots[short] = recs
            self.n_layers[short] = len(recs[0]["layer_profile"])
        self.battery = {it["id"]: it for it in
                        json.load(open(proj / "battery" / "battery.json",
                                       encoding="utf-8"))["items"]}
        ver_path = proj / "battery" / "l0_verification.json"
        self.verification = ({r["title"]: r for r in
                              json.load(open(ver_path, encoding="utf-8"))}
                             if ver_path.exists() else None)
        # v2.1 前的 raw：F17 要公布的舊對照組相關就從這裡重算，不是抄舊報告
        self.pre_v21 = {}
        for short, tag in TAGS.items():
            p = proj / "results" / "raw" / f"pilot_{tag}.json.bak_pre_v21"
            if p.exists():
                self.pre_v21[short] = [r for r in json.load(open(p, encoding="utf-8"))["records"]
                                       if "error" not in r]
        nov = proj / "results" / "novelty_report.json"
        self.novelty = json.load(open(nov, encoding="utf-8")) if nov.exists() else None
        # 盲評分數：逐題 pass/score 的原始評分，不是統計量，所以算 primary
        jp = proj / "results" / "judge_scores.json"
        self.judge = json.load(open(jp, encoding="utf-8"))["scores"] if jp.exists() else None

    def level(self, short: str, lv: str):
        return [r for r in self.pilots[short] if r["level"] == lv]

    def scored(self, short: str, lv: str):
        """有 gold_logprob 且有 depth 的紀錄——劑量反應的分析單位。"""
        return [r for r in self.level(short, lv)
                if r.get("gold_logprob_per_token") is not None and r.get("depth")]

    def cap(self, short: str) -> float:
        """depth_tau_* 的結構天花板 (L-1)/L：最後一層對自己的 KL 恆為 0，永遠跨不過門檻。"""
        n = self.n_layers[short]
        return (n - 1) / n

    def frac_at_cap(self, short: str, rec) -> Optional[float]:
        ppd = rec.get("per_position_depths") or []
        if not ppd:
            return None
        c = self.cap(short)
        return float(np.mean([abs(d["depth_tau_0.1"] - c) < 1e-9 for d in ppd]))

    def anchor_rows(self, short: str):
        """L0 每題的（語料窗口計數中位數, depth, gold）——外部錨點的原料。"""
        if self.verification is None:
            return None
        corpus = CORPUS_OF[short]
        rows = []
        for r in self.scored(short, "L0"):
            title = self.battery.get(r["id"], {}).get("title")
            v = self.verification.get(title)
            if not v:
                continue
            ok = [p["count"] for p in v["probes"].get(corpus, []) if p["status"] == "ok"]
            if not ok:
                continue
            rows.append((float(np.median(ok)), r["depth"]["depth_tau_0.1"],
                         r["gold_logprob_per_token"]))
        return rows or None


def partial_spearman(x, y, z):
    """rank 化後對 z 殘差化再相關（與 followup_analysis.py 同一個估計式）。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    r, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)


def rank_biserial(a, b):
    """與 analyze_pilot.py 的 rank_biserial 同式，避免兩邊算出不同的效應量。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None
    g = sum((x > y) for x in a for y in b)
    l = sum((x < y) for x in a for y in b)
    return float((g - l) / (len(a) * len(b)))


# ---------------------------------------------------------------- 檢查表

DEPTH_VARIANTS = ("kl_auc_norm", "depth_tau_0.05", "depth_tau_0.1",
                  "depth_tau_0.5", "depth_argmax")
LADDER_LEVELS = ("L0", "L2", "L3", "L4", "L5")   # 階梯本身，不含對照 L0N 與填空 L1


def add_v03_stats(raw: Raw, S: dict) -> None:
    """v0.3 新增引用的統計量。與 build_stats 分開只是為了讀得下去。"""
    from scipy.stats import mannwhitneyu, wilcoxon

    shared = {r["id"] for r in raw.level("1.4b", "L0")}     # 四個 Pythia 共用的 17 題

    for t in ORDER:
        # --- A.1 / A.5：三條件 × 五種 depth 變體的完整組內格
        for lv in ("L0", "L0N", "L1"):
            rs = raw.scored(t, lv)
            if len(rs) < 6:
                continue
            g = [r["gold_logprob_per_token"] for r in rs]
            for dv in DEPTH_VARIANTS:
                y = [r["depth"][dv] for r in rs]
                if len(set(y)) < 2:      # 常數欄（例如某些 L1 的 argmax）沒有相關可算
                    continue
                rho, p = spearmanr(g, y)
                S[f"grid.{t}.{lv}.{dv}.rho"] = float(rho)
                S[f"grid.{t}.{lv}.{dv}.p"] = float(p)

        l0 = raw.scored(t, "L0")
        g = [r["gold_logprob_per_token"] for r in l0]
        d = [r["depth"]["depth_tau_0.1"] for r in l0]
        S[f"goldsd.{t}.L0"] = float(np.std(g, ddof=1))   # analyze_pilot.summarize 也用 ddof=1

        # --- A.2 leave-one-out（決定性；bootstrap CI 有 RNG，不進表）
        full = spearmanr(g, d).statistic
        loo = [float(spearmanr(g[:i] + g[i + 1:], d[:i] + d[i + 1:]).statistic)
               for i in range(len(g))]
        S[f"loo.{t}.L0.min"], S[f"loo.{t}.L0.max"] = min(loo), max(loo)
        S[f"loo.{t}.L0.max_shift"] = max(abs(x - full) for x in loo)

        # --- 天花板佔比與有效層級數（§2 / A.6 的「2–5 對 9–13」）
        for lv in ("L0", "L0N", "L1"):
            rs = raw.scored(t, lv)
            if not rs:
                continue
            fc = [raw.frac_at_cap(t, r) for r in rs]
            dd = [r["depth"]["depth_tau_0.1"] for r in rs]
            S[f"fraccapmean.{t}.{lv}"] = float(np.mean(fc))
            S[f"distinct.{t}.{lv}.depth"] = len(set(dd))
            S[f"distinct.{t}.{lv}.fraccap"] = len(set(fc))

        # --- 階梯長度混淆（§1 / TL;DR）
        allr = [r for r in raw.pilots[t] if r.get("layer_profile")]
        rho, p = spearmanr([r["n_prompt_tokens"] for r in allr],
                           [r["final_entropy_mean"] for r in allr])
        S[f"ladder.{t}.len_ent_all.rho"], S[f"ladder.{t}.len_ent_all.p"] = float(rho), float(p)
        sub = [r for r in allr if r["level"] in ("L2", "L3", "L4", "L5")]
        if len(sub) >= 8:
            rho, p = spearmanr([r["n_prompt_tokens"] for r in sub],
                               [r["final_entropy_mean"] for r in sub])
            S[f"ladder.{t}.len_ent_l2l5.rho"] = float(rho)
            S[f"ladder.{t}.len_ent_l2l5.p"] = float(p)
        # 「53–408 tokens」= 階梯各層的 prompt token 中位數範圍（不含 L0N/L1）
        meds = [float(np.median([r["n_prompt_tokens"] for r in raw.level(t, lv)]))
                for lv in LADDER_LEVELS if raw.level(t, lv)]
        if len(meds) >= 4:
            S[f"ladder.{t}.tokmed_min"], S[f"ladder.{t}.tokmed_max"] = min(meds), max(meds)

    # --- OLMo 只用與 Pythia 共用的 17 題（F5 要求先發制人的防禦數字）
    rs = [r for r in raw.scored("olmo", "L0") if r["id"] in shared]
    rho, p = spearmanr([r["gold_logprob_per_token"] for r in rs],
                       [r["depth"]["depth_tau_0.1"] for r in rs])
    S["shared17.olmo.L0.rho"], S["shared17.olmo.L0.p"] = float(rho), float(p)
    S["shared17.olmo.L0.n"] = len(rs)

    # --- frac_at_cap 的 p 上界（§建構效度「p < .007 in 5/5」）
    S["fraccap.p_max"] = max(S[f"fraccap.{t}.gold.p"] for t in ORDER)

    # --- F17：v2.1 前的對照組相關，屬 EXPECTED-HISTORICAL，不是舊值漏改
    if "1.4b" in raw.pre_v21:
        rs = [r for r in raw.pre_v21["1.4b"]
              if r["level"] == "L0N" and r.get("gold_logprob_per_token") is not None]
        rho, p = spearmanr([r["gold_logprob_per_token"] for r in rs],
                           [r["depth"]["depth_tau_0.1"] for r in rs])
        S["precontrol.1.4b.L0N.rho"], S["precontrol.1.4b.L0N.p"] = float(rho), float(p)
        S["precontrol.1.4b.L0N.n"] = len(rs)

    # --- L0/L0N 的詞數配平：以 44 詞的 gold 續文為準（提示區含格式差異，不是配平對象）
    for t in ("1.4b", "olmo"):
        per = {}
        for lv in ("L0", "L0N"):
            per[lv] = [len(raw.battery[r["id"]]["gold_continuation"].split())
                       for r in raw.level(t, lv) if r["id"] in raw.battery]
        S[f"words.{t}.rb"] = rank_biserial(per["L0"], per["L0N"])
        S[f"words.{t}.p"] = float(mannwhitneyu(per["L0"], per["L0N"],
                                               alternative="two-sided").pvalue)

    S["loo.max_shift_all"] = max(S[f"loo.{t}.L0.max_shift"] for t in ORDER)

    # --- Fisher z：L0 的相關是否「不同於」對照與填空（TL;DR 的 .0014 / <.0001）
    from math import atanh, sqrt, erfc
    for t in ORDER:
        n0 = S[f"within.{t}.L0.n"]
        z0 = atanh(max(-0.999999, min(0.999999, S[f"within.{t}.L0.rho"])))
        for lv in ("L0N", "L1"):
            n1 = S[f"within.{t}.{lv}.n"]
            z1 = atanh(max(-0.999999, min(0.999999, S[f"within.{t}.{lv}.rho"])))
            se = sqrt(1 / (n0 - 3) + 1 / (n1 - 3))
            S[f"fisherz.{t}.L0_vs_{lv}.p"] = float(erfc(abs(z0 - z1) / se / sqrt(2)))

    # --- Benjamini–Hochberg over the 15-cell τ=0.1 family（TL;DR 的 q）
    cells = [(f"{t}.{lv}", S[f"within.{t}.{lv}.p"]) for t in ORDER
             for lv in ("L0", "L0N", "L1")]
    cells.sort(key=lambda x: x[1])
    m_ = len(cells)
    prev = 1.0
    for i in range(m_ - 1, -1, -1):
        prev = min(prev, cells[i][1] * m_ / (i + 1))
        S[f"bh_q.{cells[i][0]}"] = float(prev)

    # --- 跨模型的記憶分數一致性（TL;DR「rho 0.76–0.99」）
    common = set.intersection(*[{r["id"] for r in raw.scored(t, "L0")} for t in ORDER])
    ids = sorted(common)
    vecs = {t: [next(r["gold_logprob_per_token"] for r in raw.scored(t, "L0") if r["id"] == i)
                for i in ids] for t in ORDER}
    xs = [float(spearmanr(vecs[a], vecs[b]).statistic)
          for i, a in enumerate(ORDER) for b in ORDER[i + 1:]]
    S["crossmodel.gold.rho_min"], S["crossmodel.gold.rho_max"] = min(xs), max(xs)

    # --- depth 的組內離散度比（§2「sd 比 1.88–7.75×」）
    ratios = []
    for t in ORDER:
        s0 = np.std([r["depth"]["depth_tau_0.1"] for r in raw.scored(t, "L0")], ddof=1)
        for lv in ("L0N", "L1"):
            s1 = np.std([r["depth"]["depth_tau_0.1"] for r in raw.scored(t, lv)], ddof=1)
            if s1 > 0:
                ratios.append(float(s0 / s1))
    S["sdratio.min"], S["sdratio.max"] = min(ratios), max(ratios)

    # --- §5 盲評：L4 通過率（judge_scores.json 逐題 pass）
    if raw.judge:
        for short, tag in (("1.4b", TAGS["1.4b"]), ("olmo", TAGS["olmo"])):
            rows = [x for x in raw.judge
                    if x["model"] == tag and x["id"].startswith("L4")]
            if rows:
                S[f"judge.{short}.l4_pass_rate"] = 100.0 * sum(
                    1 for x in rows if x["pass"]) / len(rows)
                S[f"judge.{short}.l4_n"] = len(rows)
        S["judge.n_scores"] = len(raw.judge)
        S["judge.n_l5"] = sum(1 for x in raw.judge if x["id"].startswith("L5"))

    # --- 語料資格高原：對 l0_verification 的既有窗口計數重跑門檻掃描
    if raw.verification:
        counts = []
        for frac in (0.6, 0.65, 0.7):
            for cnt in (10, 20, 30, 40):
                n = 0
                for rec in raw.verification.values():
                    probes = [p for p in rec["probes"].get("pile", [])
                              if p["status"] == "ok"]
                    if probes and sum(1 for p in probes if p["count"] >= cnt) / len(probes) >= frac:
                        n += 1
                counts.append(n)
        S["plateau.eligible_min"], S["plateau.eligible_max"] = min(counts), max(counts)
        S["plateau.total"] = len(raw.verification)

    # --- 每詞子詞數：prompt token 數 ÷ 固定 44 詞的 gold 續文長度，依語料資格過濾。
    # 這是唯一同時重現 −0.775/−0.818 與 p .0005/.0002 的算法（另外試過 31 種都不對）。
    for short, mk in (("1.4b", "EleutherAI/pythia-1.4b"),
                      ("olmo", "allenai/OLMo-2-0425-1B")):
        per = {}
        for lv in ("L0", "L0N"):
            per[lv] = [it["prompt_tokens"][mk] / len(it["gold_continuation"].split())
                       for it in raw.battery.values()
                       if it["level"] == lv and mk in it.get("prompt_tokens", {})
                       and mk in str(it.get("eligible_models", ""))]
        S[f"tokperword.{short}.rb"] = rank_biserial(per["L0"], per["L0N"])
        S[f"tokperword.{short}.p"] = float(mannwhitneyu(per["L0"], per["L0N"],
                                                        alternative="two-sided").pvalue)

    # --- §5：從 novelty_report.json 的 frac_present 現算 Wilcoxon
    if raw.novelty:
        for t, tag in (("1.4b", TAGS["1.4b"]), ("olmo", TAGS["olmo"])):
            by: dict = {}
            for r in raw.novelty:
                if r["model"] != tag or r.get("frac_present") is None:
                    continue
                by.setdefault(r["id"], {})[r["kind"]] = r["frac_present"]
            pairs = [(v["greedy"], v["sample0"]) for v in by.values()
                     if "greedy" in v and "sample0" in v]
            if len(pairs) >= 6:
                S[f"novelty.{t}.n"] = len(pairs)
                S[f"novelty.{t}.greedy_wins"] = sum(1 for a, b in pairs if a > b)
                S[f"novelty.{t}.wilcoxon_p"] = float(
                    wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue)


def build_stats(raw: Raw) -> dict:
    """宣告式檢查表：名稱 → 從 raw 重算出來的值。None 代表 raw 產不出這個量。"""
    S: dict = {}

    for t in ORDER:
        # --- 組內劑量反應（L0 / L0N / L1，depth_tau_0.1）
        for lv in ("L0", "L0N", "L1"):
            rs = raw.scored(t, lv)
            S[f"within.{t}.{lv}.n"] = len(rs)
            if len(rs) >= 6:
                g = [r["gold_logprob_per_token"] for r in rs]
                d = [r["depth"]["depth_tau_0.1"] for r in rs]
                rho, p = spearmanr(g, d)
                S[f"within.{t}.{lv}.rho"] = float(rho)
                S[f"within.{t}.{lv}.p"] = float(p)

        # --- 同分天花板（R5）：把該條件自己的 depth 值完美單調排列對上 gold，
        #     得到同分結構允許的 |rho| 上限。純粹由 raw 的值多重集決定。
        #     L1（填空）一併算：稿件的天花板但書涵蓋「對照與填空」兩個比較格。
        for lv in ("L0", "L0N", "L1"):
            rs = raw.scored(t, lv)
            if len(rs) >= 6:
                g = sorted(r["gold_logprob_per_token"] for r in rs)
                d = sorted(r["depth"]["depth_tau_0.1"] for r in rs)
                S[f"ceiling.{t}.{lv}"] = float(abs(spearmanr(g, d)[0]))

        # --- 量程重疊率（R5.a 的 ov/union）：L0 與 L0N 的 gold 量程交集佔聯集比例。
        #     注意不是「對照組自身量程/聯集」（那是 45–53%，曾被誤當成同一個量）。
        l0g = [r["gold_logprob_per_token"] for r in raw.scored(t, "L0")]
        l0ng = [r["gold_logprob_per_token"] for r in raw.scored(t, "L0N")]
        if l0g and l0ng:
            ov = max(0.0, min(max(l0g), max(l0ng)) - max(min(l0g), min(l0ng)))
            union = max(max(l0g), max(l0ng)) - min(min(l0g), min(l0ng))
            S[f"rangeoverlap.{t}"] = ov / union

        # --- 混組（L0+L0N+L1）劑量反應
        dose = [(r["gold_logprob_per_token"], r["depth"]["depth_tau_0.1"])
                for r in raw.pilots[t]
                if r.get("gold_logprob_per_token") is not None and r.get("depth")]
        rho, p = spearmanr([x[0] for x in dose], [x[1] for x in dose])
        S[f"pooled.{t}.n"] = len(dose)
        S[f"pooled.{t}.rho"] = float(rho)
        S[f"pooled.{t}.p"] = float(p)

        # --- 各層級的最終熵平均
        for lv in {r["level"] for r in raw.pilots[t]}:
            rs = raw.level(t, lv)
            S[f"entropy.{t}.{lv}.mean"] = float(np.mean([r["final_entropy_mean"] for r in rs]))
            S[f"depthmean.{t}.{lv}.depth_tau_0.1"] = float(
                np.mean([r["depth"]["depth_tau_0.1"] for r in rs if r.get("depth")]))
            golds = [r["gold_logprob_per_token"] for r in rs
                     if r.get("gold_logprob_per_token") is not None]
            if golds:
                S[f"goldmean.{t}.{lv}"] = float(np.mean(golds))

        # --- 條件對比的 rank-biserial（L0 vs L0N、L1 vs L4）
        for a, b in (("L0", "L0N"), ("L1", "L4")):
            ra, rb_ = raw.level(t, a), raw.level(t, b)
            if not ra or not rb_:
                continue
            S[f"rb.{t}.{a}_vs_{b}.final_entropy"] = rank_biserial(
                [r["final_entropy_mean"] for r in ra], [r["final_entropy_mean"] for r in rb_])
            for dk in ("depth_tau_0.05", "depth_tau_0.1", "depth_tau_0.5",
                       "depth_argmax", "kl_auc_norm"):
                S[f"rb.{t}.{a}_vs_{b}.{dk}"] = rank_biserial(
                    [r["depth"][dk] for r in ra if r.get("depth")],
                    [r["depth"][dk] for r in rb_ if r.get("depth")])

        # --- 有 gold 的題數（草稿 `:21` 的分母）
        S[f"golditems.{t}"] = sum(1 for r in raw.pilots[t]
                                  if r.get("gold_logprob_per_token") is not None)

        l0 = raw.scored(t, "L0")
        g = [r["gold_logprob_per_token"] for r in l0]
        d = [r["depth"]["depth_tau_0.1"] for r in l0]
        e = [r["final_entropy_mean"] for r in l0]

        # --- kl_auc_norm 的反向組內結果（F14 要求公布的負面結果）
        rho, p = spearmanr(g, [r["depth"]["kl_auc_norm"] for r in l0])
        S[f"klauc.{t}.L0.rho"], S[f"klauc.{t}.L0.p"] = float(rho), float(p)

        # --- 偏相關：gold~depth | final_entropy
        pr, pp = partial_spearman(np.array(g), np.array(d), np.array(e))
        S[f"partial.{t}.gold_depth.rho"], S[f"partial.{t}.gold_depth.p"] = pr, pp

        # --- frac_at_cap 建構效度（F19；B5 被駁回後改以附錄敏感度分析呈現）
        fc = [raw.frac_at_cap(t, r) for r in l0]
        rho, p = spearmanr(g, fc)
        S[f"fraccap.{t}.gold.rho"], S[f"fraccap.{t}.gold.p"] = float(rho), float(p)
        rho, p = spearmanr(d, fc)
        S[f"fraccap.{t}.depth.rho"], S[f"fraccap.{t}.depth.p"] = float(rho), float(p)
        pr, pp = partial_spearman(np.array(g), np.array(fc), np.array(e))
        S[f"fraccap.{t}.partial.rho"], S[f"fraccap.{t}.partial.p"] = pr, pp
        S[f"fraccap.{t}.cap"] = raw.cap(t)

        # --- 排除位置 0 的 depth 重算（草稿 `:45` 的循環量測對照）
        xs, ys = [], []
        for r in l0:
            ppd = r.get("per_position_depths") or []
            v = [x["depth_tau_0.1"] for x in ppd[1:]]
            if v:
                xs.append(r["gold_logprob_per_token"])
                ys.append(float(np.mean(v)))
        if len(xs) >= 6:
            rho, p = spearmanr(xs, ys)
            S[f"pos1plus.{t}.L0.rho"], S[f"pos1plus.{t}.L0.p"] = float(rho), float(p)

        # --- 外部錨點：語料窗口計數 vs gold / depth
        rows = raw.anchor_rows(t)
        if rows is None:
            S[f"anchor.{t}.count_gold.rho"] = None      # l0_verification.json 不在 → 無法認證
            S[f"anchor.{t}.count_gold.p"] = None
            S[f"anchor.{t}.count_depth.rho"] = None
            S[f"anchor.{t}.count_depth.p"] = None
        else:
            c = [x[0] for x in rows]
            rho, p = spearmanr(c, [x[2] for x in rows])
            S[f"anchor.{t}.count_gold.rho"], S[f"anchor.{t}.count_gold.p"] = float(rho), float(p)
            rho, p = spearmanr(c, [x[1] for x in rows])
            S[f"anchor.{t}.count_depth.rho"], S[f"anchor.{t}.count_depth.p"] = float(rho), float(p)

    # --- 跨模型彙總
    S["golditems.total"] = sum(S[f"golditems.{t}"] for t in ORDER)
    ns = [n for t in ORDER for lv in ("L0", "L0N", "L1", "L4")
          if (n := len(raw.level(t, lv))) > 0]     # 只有 1.4B/OLMo 跑了 L2–L5
    S["conditions.n_min"], S["conditions.n_max"] = min(ns), max(ns)
    ar = [S[f"anchor.{t}.count_gold.rho"] for t in ORDER]
    ap = [S[f"anchor.{t}.count_gold.p"] for t in ORDER]
    if all(v is not None for v in ar):
        S["anchor.count_gold.rho_min"] = min(ar)
        S["anchor.count_gold.rho_max"] = max(ar)
        S["anchor.count_gold.p_max"] = max(ap)
    else:
        S["anchor.count_gold.rho_min"] = S["anchor.count_gold.rho_max"] = None
        S["anchor.count_gold.p_max"] = None
    for lv in ("L0", "L0N", "L5"):
        vals = [S.get(f"entropy.{t}.{lv}.mean") for t in ("1.4b", "olmo")]
        if all(v is not None for v in vals):
            S[f"entropy.flagships.{lv}.min"] = min(vals)
            S[f"entropy.flagships.{lv}.max"] = max(vals)

    # --- 量程重疊率的跨模型極值（稿件寫「13–27%」引用的就是這兩個）
    rc = [S[f"rangeoverlap.{t}"] for t in ORDER if S.get(f"rangeoverlap.{t}") is not None]
    if rc:
        S["rangeoverlap.min"] = min(rc)
        S["rangeoverlap.max"] = max(rc)

    # --- v0.6 自我測試統計量：相對門檻／熵縮放判準，直接由 layer_profile 曲線重算。
    #     定義與 harness/relative_depth_analysis.py 一致：由後往前掃，回傳
    #     kl_to_final > 門檻 的最深層 / 總層數。
    def _lastcross(curve, thr):
        n = len(curve)
        for i in range(n - 1, -1, -1):
            if curve[i]["kl_to_final"] > thr:
                return i / n
        return 0.0

    ge = []
    for t in ORDER:
        rs = [r for r in raw.scored(t, "L0") if r.get("layer_profile")]
        if len(rs) < 6:
            continue
        g = [r["gold_logprob_per_token"] for r in rs]
        S[f"goldent.{t}"] = float(spearmanr(g, [r["final_entropy_mean"] for r in rs])[0])
        ge.append(S[f"goldent.{t}"])
        for a, key in ((0.5, "a50"), (0.25, "a25"), (0.10, "a10"), (0.05, "a05")):
            d = [_lastcross(r["layer_profile"],
                            a * r["layer_profile"][0]["kl_to_final"]) for r in rs]
            if len(set(d)) > 1:
                S[f"relcross.{t}.{key}"] = float(spearmanr(g, d)[0])
        d = [_lastcross(r["layer_profile"], 0.25 * r["final_entropy_mean"]) for r in rs]
        if len(set(d)) > 1:
            S[f"entscaled25.{t}"] = float(spearmanr(g, d)[0])
    if ge:
        S["goldent.min"] = min(ge)   # 最負
        S["goldent.max"] = max(ge)

    # --- 去重語料複製跑（v0.6 引用；results/night 的 deduped pilot 一手輸出）
    for key, fn in (("410m", "pilot_EleutherAI_pythia-410m-deduped.json"),
                    ("1.4b", "pilot_EleutherAI_pythia-1.4b-deduped.json")):
        p = PROJ / "results" / "night" / fn
        if not p.is_file():
            continue
        recs = [r for r in json.load(open(p, encoding="utf-8"))["records"]
                if r.get("level") == "L0" and "error" not in r
                and r.get("gold_logprob_per_token") is not None and r.get("depth")]
        if len(recs) >= 6:
            rho, pv = spearmanr([r["gold_logprob_per_token"] for r in recs],
                                [r["depth"]["depth_tau_0.1"] for r in recs])
            S[f"dedupl0.{key}.rho"] = float(rho)
            S[f"dedupl0.{key}.p"] = float(pv)

    # --- R1 教師強制塌縮（v0.5 但書引用）：tf 下 gold 與最終熵在 L0 內的相關。
    #     來源是 results/night/tf_depth_*.json 的逐項紀錄——一手量測輸出，非中間值。
    tfc = []
    for t in ORDER:
        p = PROJ / "results" / "night" / f"tf_depth_{TAGS[t]}.json"
        if not p.is_file():
            continue
        recs = [r for r in json.load(open(p, encoding="utf-8"))["records"]
                if r.get("level") == "L0" and r.get("gold_logprob_per_token") is not None]
        if len(recs) >= 6:
            rho = spearmanr([r["gold_logprob_per_token"] for r in recs],
                            [r["tf_final_entropy_mean"] for r in recs])[0]
            S[f"tfcollapse.{t}"] = float(rho)
            tfc.append(float(rho))
    if tfc:
        S["tfcollapse.min"] = min(tfc)   # 最負（塌得最死）
        S["tfcollapse.max"] = max(tfc)

    add_v03_stats(raw, S)

    return S


# ---------------------------------------------------------------- 值域索引

def sci_key(v: float, mant_dp: int) -> str:
    """極小值（p 值）的正規寫法，以科學記號為準：4.06e-05 在 1dp 下是 4.1e-5。"""
    d = Decimal(str(abs(v)))
    exp = d.adjusted()
    mant = quant(float(d.scaleb(-exp)), mant_dp, ROUND_HALF_UP)
    return f"{'-' if v < 0 else ''}{canon(mant)}e{exp}"


def _renderings(v: float) -> tuple:
    """回傳 (正確寫法集合, 二次進位寫法集合)。只收 2dp 以上，1dp 太容易撞。"""
    ok, dbl = set(), set()
    if abs(v) < 0.005:                       # p 值：只以科學記號入索引
        return {sci_key(v, d) for d in (1, 2, 3)}, set()
    for d in (2, 3, 4):
        for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
            ok.add(canon(quant(v, d, mode)))
        direct = quant(v, d, ROUND_HALF_UP)
        for k in range(d + 1, d + 4):
            two = quant(quant(v, k, ROUND_HALF_UP), d, ROUND_HALF_UP)
            if abs(two - direct) > 1e-12:
                dbl.add(canon(two))
    return ok, dbl


def indexable(name: str, v) -> bool:
    """值域索引的入場條件。目的是擋掉「跟文件裡每個 1 都撞」的退化統計量。"""
    if v is None or isinstance(v, bool):
        return False
    v = float(v)
    if not np.isfinite(v) or v.is_integer():
        return False
    # 不顯著的 p 值文件都寫 n.s.，不會照抄數字；它們卻會撞上 0.5、0.25、0.2…
    if name.endswith(".p") and v >= 0.2:
        return False
    return abs(v) not in (0.5, 0.25)


def build_value_index(stats: dict) -> tuple:
    """數字寫法 → 可能對應的統計量。文件被改寫、anchor 全部落空時，靠這個維持覆蓋。

    整數量（267、53、17…）不進索引：它們和層數、題號、卡號、行號撞得太厲害，
    交給 anchor 檢查與舊值表處理。
    """
    correct: dict = {}
    double: dict = {}
    for name, v in stats.items():
        if not indexable(name, v):
            continue
        ok, dbl = _renderings(float(v))
        for r in ok - {"0", "-0"}:
            correct.setdefault(r, []).append(name)
        for r in dbl - {"0", "-0"}:
            double.setdefault(r, []).append(name)
    for r in correct:                       # 正確寫法優先，避免互撞誤判成二次進位
        double.pop(r, None)
    return correct, double


# ---------------------------------------------------------------- 引用地點表

@dataclass
class Site:
    """一個引用地點：anchor 之後（同一行）的第 i 個數字字面值，應該等於某個統計量。

    anchor 只負責定位，不參與比對；數字一律從 anchor 之後現場擷取，
    所以文件改寫導致 anchor 失配時會報 ANCHOR-MISS，不會靜靜地放行。
    """
    files: str                      # fnmatch 樣式，比對檔名
    anchor: str                     # 正則；定位用
    slots: list                     # [(index, stat_name, kind)]，kind: value|ub|lb
    label: str = ""
    span_lines: int = 1
    pending: bool = False           # True = 該段文字還沒寫（如 F19），anchor 失配屬預期
    note: str = ""                  # 判讀提醒，會附在這個地點的每一筆結果後面


V, UB = "value", "ub"


def _row(model_re: str, t: str) -> Site:
    """§3 主表的一列：rho, p, L0N rho, L1 rho。

    附錄 A.1／A.2／A.6 的表格也以同樣的模型名開頭，所以必須靠 (n.s.) 這個
    只有 §3 才有的欄位內容把它們區隔開，否則會拿 §3 的欄位定義去讀附錄的表。
    """
    return Site(files="WRITEUP_*.md", label=f"§3 表格列 {t}",
                anchor=rf"^\|\s*{model_re}\s*\|(?=[^\n]*\(n\.s\.\))",
                slots=[(0, f"within.{t}.L0.rho", V),
                       (1, f"within.{t}.L0.p", UB if t == "olmo" else V),
                       (2, f"within.{t}.L0N.rho", V),
                       (3, f"within.{t}.L1.rho", V)])


def _a1_row(model_re: str, t: str) -> Site:
    """附錄 A.1：五種 depth 變體 × rho/p，同一列十個數字。"""
    order = ("kl_auc_norm", "depth_tau_0.05", "depth_tau_0.1",
             "depth_tau_0.5", "depth_argmax")
    slots = []
    for i, dv in enumerate(order):
        slots.append((i * 2, f"grid.{t}.L0.{dv}.rho", V))
        slots.append((i * 2 + 1, f"grid.{t}.L0.{dv}.p",
                      UB if (t == "olmo" and dv in ("depth_tau_0.05", "depth_tau_0.1")) else V))
    # A.1 是唯一「第一個資料格就是粗體正值」的表（kl_auc_norm 五個模型全正）。
    # 用前瞻辨識，不能吃掉數字本身，否則擷取會從小數點後開始。
    return Site(files="WRITEUP_*.md", label=f"A.1 五變體格 {t}",
                anchor=rf"^\|\s*{model_re}\s*\|(?=\s*\*\*\+0\.)",
                slots=slots)


SITES: list = [
    # ---------------- docs/WRITEUP_DRAFT_EN.md
    # v0.2 寫「Spearman …」，v0.3 改寫成「within L0: …」——兩種都掛著，
    # 哪一版的文字在，哪一個 anchor 就生效。
    Site("WRITEUP_DRAFT_EN.md", r"predicts how early in depth the model commits to its "
         r"final answer \(Spearman", label="TL;DR L0 五模型 rho（v0.2 寫法）",
         slots=[(i, f"within.{t}.L0.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"within L0: \*\*", label="TL;DR L0 五模型 rho", span_lines=2,
         slots=[(i, f"within.{t}.L0.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_DRAFT_EN.md", r"\*\*flat in the matched non-memorized control\*\* \(",
         label="TL;DR L0N 五模型 rho（v0.2 寫法）",
         slots=[(i, f"within.{t}.L0N.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"Matched control:", label="TL;DR L0N 五模型 rho", span_lines=2,
         slots=[(i, f"within.{t}.L0N.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"Single-token fact cloze:", label="TL;DR L1 五模型 rho",
         span_lines=2,
         slots=[(i, f"within.{t}.L1.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md",
         r"correlate with the gold-log-prob memory scale in all (?:5 models|five runs)",
         label="TL;DR 外部錨點 rho 範圍與 p 上界", span_lines=2,
         slots=[(0, "anchor.count_gold.rho_min", V),
                (1, "anchor.count_gold.rho_max", V),
                (2, "anchor.count_gold.p_max", UB)]),
    Site("WRITEUP_*.md",
         r"(?:and directly with commitment depth in OLMo-2|"
         r"Counts also correlate directly with depth in OLMo-2-1B)",
         label="TL;DR 錨點對 depth（OLMo）", span_lines=2,
         slots=[(0, "anchor.olmo.count_depth.rho", V),
                (1, "anchor.olmo.count_depth.p", V)]),
    Site("WRITEUP_*.md",
         r"orders conditions retrieval-low to invention-high in both families: L0",
         label="TL;DR 熵階梯（雙旗艦逐值）", span_lines=3,
         slots=[(0, "entropy.1.4b.L0.mean", V), (1, "entropy.olmo.L0.mean", V),
                (2, "entropy.1.4b.L5.mean", V), (3, "entropy.olmo.L5.mean", V),
                (4, "entropy.1.4b.L0N.mean", V), (5, "entropy.olmo.L0N.mean", V)]),
    Site("WRITEUP_DRAFT_EN.md",
         r"orders conditions retrieval-low → invention-high in both families \(L0",
         label="TL;DR 熵階梯範圍（雙旗艦）",
         slots=[(0, "entropy.flagships.L0.min", V), (1, "entropy.flagships.L0.max", V),
                (2, "entropy.flagships.L5.min", V), (3, "entropy.flagships.L5.max", V),
                (4, "entropy.flagships.L0N.min", V), (5, "entropy.flagships.L0N.max", V)]),
    Site("WRITEUP_*.md", r"entropy rank-biserial L0 vs L0N:?",
         label="TL;DR 熵 rank-biserial",
         slots=[(0, "rb.1.4b.L0_vs_L0N.final_entropy", V),
                (1, "rb.olmo.L0_vs_L0N.final_entropy", V)]),
    Site("WRITEUP_DRAFT_EN.md", r"boundary guard for teacher-forced scoring fired 0 times "
         r"in 5 models ×", label="Methods 有 gold 的總題數（v0.2 寫法）",
         slots=[(0, "golditems.total", V)]),
    Site("WRITEUP_*.md", r"\(flagships\): L0",
         label="§1 七層熵階梯（1.4B / OLMo）",
         slots=[(0, "entropy.1.4b.L0.mean", V), (1, "entropy.olmo.L0.mean", V),
                (2, "entropy.1.4b.L1.mean", V), (3, "entropy.olmo.L1.mean", V),
                (4, "entropy.1.4b.L2.mean", V), (5, "entropy.olmo.L2.mean", V),
                (6, "entropy.1.4b.L0N.mean", V), (7, "entropy.olmo.L0N.mean", V),
                (8, "entropy.1.4b.L3.mean", V), (9, "entropy.olmo.L3.mean", V),
                (10, "entropy.1.4b.L4.mean", V), (11, "entropy.olmo.L4.mean", V),
                (12, "entropy.1.4b.L5.mean", V), (13, "entropy.olmo.L5.mean", V)]),
    Site("WRITEUP_*.md", r"Rank-biserial: final(?:-layer)? entropy",
         label="§2 L0 vs L0N 對比效應量",
         slots=[(0, "rb.1.4b.L0_vs_L0N.final_entropy", V),
                (1, "rb.olmo.L0_vs_L0N.final_entropy", V),
                (3, "rb.1.4b.L0_vs_L0N.depth_tau_0.1", V),
                (4, "rb.olmo.L0_vs_L0N.depth_tau_0.1", V)]),
    _row(r"Pythia-410M", "410m"),
    _row(r"Pythia-1B", "1b"),
    _row(r"Pythia-1\.4B", "1.4b"),
    _row(r"Pythia-2\.8B", "2.8b"),
    _row(r"OLMo-2-1B", "olmo"),
    _a1_row(r"Pythia-410M", "410m"),
    _a1_row(r"Pythia-1B", "1b"),
    _a1_row(r"Pythia-1\.4B", "1.4b"),
    _a1_row(r"Pythia-2\.8B", "2.8b"),
    _a1_row(r"OLMo-2-1B", "olmo"),

    # ---------------- v0.3 專有的宣稱（改寫後才出現的句子）
    Site("WRITEUP_*.md", r"on the 17 items shared with the Pythia\s+runs it is \*\*",
         label="TL;DR OLMo 共用 17 題", span_lines=2,
         slots=[(0, "shared17.olmo.L0.rho", V), (1, "shared17.olmo.L0.p", V)]),
    Site("WRITEUP_*.md", r"Restricted to the shared 17, OLMo is \*\*",
         label="§3 OLMo 共用 17 題",
         slots=[(0, "shared17.olmo.L0.rho", V), (1, "shared17.olmo.L0.p", V)]),
    Site("WRITEUP_*.md", r"\*\*OLMo-2-1B\*\* \(n = 20, p =",
         label="TL;DR BH 校正後存活的兩個模型", span_lines=2,
         slots=[(0, "within.olmo.L0.p", V), (1, "bh_q.olmo.L0", UB),
                (3, "within.1.4b.L0.p", V), (4, "bh_q.1.4b.L0", V)]),
    Site("WRITEUP_*.md", r"only OLMo-2-1B separates from both\s+\(p =",
         label="TL;DR Fisher z 兩個對比", span_lines=2,
         slots=[(0, "fisherz.olmo.L0_vs_L0N.p", V),
                (1, "fisherz.olmo.L0_vs_L1.p", UB)]),
    Site("WRITEUP_*.md", r"per-item memory scores correlate across models at rho",
         label="TL;DR 跨模型記憶分數一致性", span_lines=2,
         slots=[(0, "crossmodel.gold.rho_min", V), (1, "crossmodel.gold.rho_max", V)]),
    Site("WRITEUP_*.md", r"no single item moves any of the five\s+correlations by more than",
         label="TL;DR leave-one-out 位移上界", span_lines=2,
         slots=[(0, "loo.max_shift_all", UB)]),
    Site("WRITEUP_*.md", r"no single item moves any rho by more than",
         label="A.2 leave-one-out 位移上界",
         slots=[(0, "loo.max_shift_all", UB)]),
    Site("WRITEUP_*.md", r"OLMo's −0\.86 spans \[",
         label="TL;DR OLMo leave-one-out 區間", span_lines=2,
         slots=[(0, "loo.olmo.L0.min", V), (1, "loo.olmo.L0.max", V)]),
    Site("WRITEUP_*.md", r"significant at p <",
         label="§建構效度 frac_at_cap 的 p 上界",
         slots=[(0, "fraccap.p_max", UB)]),
    Site("WRITEUP_*.md", r"standard-deviation ratios ",
         label="§2 depth 組內離散度比",
         note="本表用 sd(L0)/sd(對照) 與 sd(L0)/sd(填空)、ddof=1、只取有 gold 的題目，"
              "上界落在 OLMo 的 L0/L0N = 7.84；若作者用的是別的配對或 ddof，請說明後改這裡的算法",
         slots=[(0, "sdratio.min", V), (1, "sdratio.max", V)]),
    Site("WRITEUP_*.md", r"fired 0 times across \*\*267\*\* model × gold-item pairs \(",
         label="§Measures 每模型的 gold 題數拆分",
         slots=[(i, f"golditems.{t}", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"prompt length spans 8× across the ladder\s+\(",
         label="§1 階梯 prompt token 中位數範圍", span_lines=2,
         slots=[(0, "ladder.1.4b.tokmed_min", V), (1, "ladder.1.4b.tokmed_max", V)]),
    Site("WRITEUP_*.md", r"correlates with entropy at rho \+",
         label="§1 全題目長度—熵相關",
         slots=[(0, "ladder.1.4b.len_ent_all.rho", V),
                (1, "ladder.olmo.len_ent_all.rho", V)]),
    Site("WRITEUP_*.md", r"L0 and L0N are matched in words \(rank-biserial",
         label="TL;DR 詞數配平（44 詞 gold 續文）", span_lines=2,
         slots=[(0, "words.1.4b.rb", V), (1, "words.1.4b.p", V)]),
    Site("WRITEUP_*.md", r"control fragments into more subword tokens per word\s*"
         r"\(rank-biserial", label="TL;DR 每詞子詞數與其 Mann-Whitney p", span_lines=3,
         slots=[(0, "tokperword.1.4b.rb", V), (1, "tokperword.olmo.rb", V),
                (2, "tokperword.1.4b.p", V), (3, "tokperword.olmo.p", V)]),
    Site("WRITEUP_*.md", r"The superseded control correlation was \*\*",
         label="§Setup v2.1 前的對照組相關（歷史值，非漏改）",
         slots=[(0, "precontrol.1.4b.L0N.rho", V), (1, "precontrol.1.4b.L0N.p", V),
                (2, "precontrol.1.4b.L0N.n", V)]),
    Site("WRITEUP_*.md", r"lowest L0 gold-log-prob SD of the five \(",
         label="§3 L0 gold 標準差（ddof=1，同 analyze_pilot）",
         note="兩個數字用了不同慣例：0.78 是 ddof=1（與 analyze_pilot.summarize 一致），"
              "1.32 只有在 ddof=0 下才對（ddof=1 是 1.36；ddof=0 下 2.8B 會變成 0.76）。"
              "兩個都要用同一個慣例",
         slots=[(0, "goldsd.2.8b.L0", V), (1, "goldsd.410m.L0", V)]),
    Site("WRITEUP_*.md", r"greedy decoding reused corpus phrasing more than a single "
         r"temperature-0\.8 sample on 11 of 12 invention items \(Wilcoxon p =",
         label="§5 L5 溯源 Wilcoxon（OLMo）",
         slots=[(0, "novelty.olmo.wilcoxon_p", V)]),
    Site("WRITEUP_*.md", r"the same comparison on Pythia-1\.4B was not significant \(p =",
         label="§5 L5 溯源 Wilcoxon（Pythia）",
         slots=[(0, "novelty.1.4b.wilcoxon_p", V)]),
    # 天花板之間夾著「24-layer」「(410M, 1.4B)」「16-layer」等數字，所以槽位是 0/4/6
    Site("WRITEUP_*.md", r"structural ceiling of \(L−1\)/L:",
         label="§Measures 三種層數的結構天花板",
         slots=[(0, "fraccap.1.4b.cap", V), (4, "fraccap.olmo.cap", V),
                (6, "fraccap.2.8b.cap", V)]),
    Site("WRITEUP_DRAFT_EN.md", r"it memorizes the canon so well — mean gold",
         label="§3 2.8B 範圍限縮：L0 gold 平均（v0.2 專有）",
         slots=[(0, "goldmean.2.8b.L0", V), (1, "goldmean.410m.L0", V)]),
    Site("WRITEUP_*.md",
         r"median corpus window count per passage correlates with gold log-prob "
         r"in all five (?:models|runs)", label="§4(a) 外部錨點 rho 範圍與 p 上界",
         slots=[(0, "anchor.count_gold.rho_min", V),
                (1, "anchor.count_gold.rho_max", V),
                (2, "anchor.count_gold.p_max", UB)]),
    Site("WRITEUP_*.md", r"and directly with depth in OLMo-2(?:-1B)? \(",
         label="§4(a) 錨點對 depth（OLMo）",
         slots=[(0, "anchor.olmo.count_depth.rho", V),
                (1, "anchor.olmo.count_depth.p", V)]),
    Site("WRITEUP_DRAFT_EN.md", r"leaves every L0 result essentially unchanged "
         r"\(e\.g\., Pythia-1\.4B", label="§4(b) 排除位置 0 後的 L0 rho（v0.3 已撤回）",
         slots=[(0, "pos1plus.1.4b.L0.rho", V), (1, "pos1plus.olmo.L0.rho", V)]),
    Site("WRITEUP_*.md",
         r"partialling final entropy out of the L0 dose-response leaves OLMo-2 significant \(",
         label="§4(c) OLMo 偏相關",
         slots=[(0, "partial.olmo.gold_depth.rho", V),
                (1, "partial.olmo.gold_depth.p", V)]),
    Site("WRITEUP_*.md",
         r"blind-calibrated LLM judging shows OLMo-2-1B genuinely attempts distant blends ~",
         label="§5 盲評 L4 通過率",
         slots=[(0, "judge.olmo.l4_pass_rate", V),
                (1, "judge.1.4b.l4_pass_rate", V)]),
    Site("WRITEUP_*.md", r"robustness plateau \((?:Pile|pile) eligibility",
         label="統計框架 語料資格高原",
         slots=[(0, "plateau.eligible_min", V), (1, "plateau.eligible_max", V),
                (2, "plateau.total", V)]),
    # F19 落地後這三處才會命中；先掛好，數字一寫進去就自動對帳。
    Site("WRITEUP_*.md", r"rho\(depth, frac_at_cap\)\s*=", pending=True,
         label="F19 depth~frac_at_cap 五模型",
         slots=[(i, f"fraccap.{t}.depth.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"rho\(gold, frac_at_cap\)\s*(?:is|=)", pending=True,
         label="F19 gold~frac_at_cap 五模型",
         slots=[(i, f"fraccap.{t}.gold.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"structural ceiling of \(L−1\)/L \(", pending=True,
         label="F18 三種層數的結構天花板",
         slots=[(0, "fraccap.1.4b.cap", V), (1, "fraccap.olmo.cap", V),
                (2, "fraccap.2.8b.cap", V)]),
    Site("WRITEUP_*.md", r"within-L0 `kl_auc_norm` result that runs opposite "
         r"and significant in 5/5 \(", pending=True,
         label="F14 kl_auc_norm 反向組內結果",
         slots=[(i, f"klauc.{t}.L0.rho", V) for i, t in enumerate(ORDER)]),
    Site("WRITEUP_*.md", r"four unreported count-vs-depth cells \(", pending=True,
         label="F14 四個未報告的 count-vs-depth 格",
         slots=[(0, "anchor.410m.count_depth.rho", V), (1, "anchor.1b.count_depth.rho", V),
                (2, "anchor.1.4b.count_depth.rho", V),
                (3, "anchor.2.8b.count_depth.rho", V)]),

    # ---------------- docs/learn/07-entropy.md
    Site("07-entropy.md", r"`levels\.\*\.final_entropy_mean\.mean`：L0",
         label="卡 07 七層熵階梯（1.4B）＋ L0N 與 L3 的相對位置",
         slots=[(0, "entropy.1.4b.L0.mean", V), (1, "entropy.1.4b.L1.mean", V),
                (2, "entropy.1.4b.L2.mean", V), (3, "entropy.1.4b.L3.mean", V),
                (4, "entropy.1.4b.L4.mean", V), (5, "entropy.1.4b.L5.mean", V),
                (6, "entropy.1.4b.L0N.mean", V), (7, "entropy.1.4b.L3.mean", V)]),

    # ---------------- docs/learn/08-kl-depth.md
    Site("08-kl-depth.md", r"`depth\.kl_auc_norm` 效應量（effect size[^）]*）是",
         label="卡 08 L1 vs L4 效應量（AUC）",
         slots=[(0, "rb.1.4b.L1_vs_L4.kl_auc_norm", V)]),
    Site("08-kl-depth.md", r"`depth\.depth_tau_0\.1` 效應量是",
         label="卡 08 L1 vs L4 效應量（tau 0.1）",
         slots=[(0, "rb.1.4b.L1_vs_L4.depth_tau_0.1", V)]),
    Site("08-kl-depth.md", r"合併 L0（",
         label="卡 08 混組樣本數",
         slots=[(0, "within.1.4b.L0.n", V), (1, "within.1.4b.L0N.n", V),
                (2, "within.1.4b.L1.n", V), (3, "pooled.1.4b.n", V)]),
    Site("08-kl-depth.md", r"−1～1）=",
         label="卡 08 混組劑量反應 rho / p",
         slots=[(0, "pooled.1.4b.rho", V), (1, "pooled.1.4b.p", V)]),
    Site("08-kl-depth.md", r"L0 組內 Pythia rho=",
         label="卡 08 L0 組內（1.4B / OLMo）",
         slots=[(0, "within.1.4b.L0.rho", V), (1, "within.1.4b.L0.p", V),
                (2, "within.1.4b.L0.n", V), (3, "within.olmo.L0.rho", V),
                (4, "within.olmo.L0.p", UB), (5, "within.olmo.L0.n", V)]),
    Site("08-kl-depth.md", r"L1 組內則無梯度（",
         label="卡 08 L1 組內（1.4B / OLMo）",
         slots=[(0, "within.1.4b.L1.rho", V), (1, "within.olmo.L1.rho", V)]),

    # ---------------- docs/learn/09-gold-logprob.md
    Site("09-gold-logprob.md", r"例如全樣本 rho =",
         label="卡 09 混組劑量反應",
         slots=[(0, "pooled.1.4b.rho", V), (1, "pooled.1.4b.p", V),
                (2, "pooled.1.4b.n", V)]),

    # ---------------- docs/learn/11-control-confound.md
    Site("11-control-confound.md", r"L0 比「沒背過的」L0N 熵低很多（效應量",
         label="卡 11 L0 vs L0N 熵效應量",
         slots=[(0, "rb.1.4b.L0_vs_L0N.final_entropy", V),
                (1, "rb.olmo.L0_vs_L0N.final_entropy", V)]),
    Site("11-control-confound.md", r"收斂深度的效應則對門檻敏感——tau 0\.1 為",
         label="卡 11 depth 效應量的門檻掃描",
         slots=[(0, "rb.1.4b.L0_vs_L0N.depth_tau_0.1", V),
                (2, "rb.1.4b.L0_vs_L0N.depth_tau_0.05", V),
                (4, "rb.1.4b.L0_vs_L0N.depth_tau_0.5", V),
                (5, "rb.1.4b.L0_vs_L0N.depth_argmax", V)]),
    Site("11-control-confound.md", r"絕對差距同樣不大（tau 0\.1 下",
         label="卡 11 L0 / L0N 的 depth 平均與絕對差距",
         slots=[(0, "depthmean.1.4b.L0.depth_tau_0.1", V),
                (1, "depthmean.1.4b.L0N.depth_tau_0.1", V)]),

    # ---------------- docs/learn/13-memorization-generalization.md
    Site("13-memorization-generalization.md", r"第十二步報告 rho=",
         label="卡 13 混組劑量反應",
         slots=[(0, "pooled.1.4b.rho", V), (1, "pooled.1.4b.p", V),
                (2, "pooled.1.4b.n", V)]),

    # ---------------- docs/learn/15-three-hypotheses.md
    Site("15-three-hypotheses.md",
         r"兩個獨立血統模型的相關性證據（(?:混組版 )?Pythia rho=",
         label="卡 15 兩家族混組相關、組內五值與熵效應量",
         slots=[(0, "pooled.1.4b.rho", V), (1, "pooled.1.4b.p", V),
                (2, "pooled.olmo.rho", V), (3, "pooled.olmo.p", V)]
               + [(5 + i, f"within.{t}.L0.rho", V) for i, t in enumerate(ORDER)]
               + [(10, "rb.1.4b.L0_vs_L0N.final_entropy", V),
                  (11, "rb.olmo.L0_vs_L0N.final_entropy", V)]),

    # ---------------- docs/learn/16-effect-size-dose.md
    Site("16-effect-size-dose.md", r"pilot 一個條件常常只有",
         label="卡 16 條件題數範圍（跨模型 × L0/L0N/L1/L4）",
         slots=[(0, "conditions.n_min", V), (1, "conditions.n_max", V)]),
    Site("16-effect-size-dose.md", r"（L0 只有",
         label="卡 16 L0 / L4 題數",
         slots=[(0, "within.1.4b.L0.n", V)]),
    Site("16-effect-size-dose.md", r"的 `final_entropy` 效應量是 \*\*",
         label="卡 16 L1 vs L4 熵效應量",
         slots=[(0, "rb.1.4b.L1_vs_L4.final_entropy", V)]),
    Site("16-effect-size-dose.md", r"結果寫進 `analysis\[\"dose_response\"\]`：\*\*rho =",
         label="卡 16 混組劑量反應（主敘述）",
         slots=[(0, "pooled.1.4b.rho", V), (1, "pooled.1.4b.p", V),
                (2, "pooled.1.4b.n", V)]),
    Site("16-effect-size-dose.md", r"「混組 rho 只有",
         label="卡 16 常見誤解 1 引用的混組 rho",
         slots=[(0, "pooled.1.4b.rho", V)]),
    Site("16-effect-size-dose.md", r"n=53）的相關從舊版的 −0\.37 弱化到",
         label="卡 16 v2.1 前後對照（現行值）",
         slots=[(0, "pooled.1.4b.rho", V)]),
    Site("16-effect-size-dose.md", r"p 值也從 0\.007 掉到",
         label="卡 16 v2.1 前後對照（現行 p）",
         slots=[(0, "pooled.1.4b.p", V)]),
    Site("16-effect-size-dose.md", r"你們的 dose_response 只有 n=",
         label="卡 16 答辯模擬題樣本數",
         slots=[(0, "pooled.1.4b.n", V)]),
    Site("16-effect-size-dose.md", r"v2\.1 對照組重建後 p 值從 0\.007 掉到",
         label="卡 16 答辯模擬題 p 值",
         slots=[(0, "pooled.1.4b.p", V)]),
    Site("16-effect-size-dose.md", r"對 depth_tau_0\.1 相關，得到",
         label="卡 16 五模型 L0 組內 rho（常見誤解）",
         slots=[(i, f"within.{t}.L0.rho", V) for i, t in enumerate(ORDER)]),
    Site("16-effect-size-dose.md", r"算出 gold_logprob 對 depth_tau_0\.1 的相關，得到",
         label="卡 16 五模型 L0 組內 rho（答辯）",
         slots=[(i, f"within.{t}.L0.rho", V) for i, t in enumerate(ORDER)]),
]


# ---------------------------------------------------------------- 數字比對

# 數字字面值，含科學記號（p = 8.7e-05）。後顧擋掉型號與欄位名裡的數字
# （Pythia-1.4b、OLMo-2-1B、L0N-08、depth_tau_0.1、v2.1）。
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9._\-−])([+\-−]?)(\d+(?:\.\d+)?|\.\d+)(?:[eE]([+\-−]?\d+))?(?![0-9])")


def parse_num(m) -> tuple:
    """回傳 (值, 尾數字面值, 指數)。指數 0 代表沒寫科學記號。"""
    sign, body, exp = m.group(1), m.group(2), m.group(3)
    v = float(body)
    if sign in ("-", "−"):
        v = -v
    e = int(exp.replace("−", "-")) if exp else 0
    return v * 10 ** e, body, e


def decimals(body: str) -> int:
    return len(body.split(".")[1]) if "." in body else 0


def quant(v: float, d: int, mode) -> float:
    return float(Decimal(str(v)).quantize(Decimal(1).scaleb(-d), rounding=mode))


def canon(v: float) -> str:
    """數字的正規寫法：去掉尾隨 0 與正號，−0.530 與 -0.53 視為同一種寫法。"""
    return format(Decimal(str(v)).normalize(), "f")


def classify(cited: float, body: str, exp: int, raw: float):
    """回傳 (status, note)。status: ok / double / wrong。

    科學記號比對尾數：把 raw 換算到同一個指數再比，8.7e-05 vs 8.4046e-05 才判得出來。
    """
    d = decimals(body)
    scale = 10.0 ** exp
    c, r = cited / scale, raw / scale
    eps = 0.5 * 10 ** -(d + 3)
    suffix = f"e{exp}" if exp else ""

    def show(v):
        return f"{v:g}{suffix}"

    for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
        if abs(c - quant(r, d, mode)) < eps:
            return "ok", ""
    # 「p = .0001」對 raw 4.8e-05 是期刊慣例的最小可寫值（等同 ≤ .0001），不是算錯。
    # 只在「引用值正好是該精度下的最小非零刻度，且 raw 比它更小」時才放行。
    if d and abs(c - 10 ** -d) < eps and 0 < r < c:
        return "ok", f"raw {r:.3g} 小於此精度的最小刻度，寫成 {show(c)} 屬慣例"
    for k in range(d + 1, d + 5):
        inter = quant(r, k, ROUND_HALF_UP)
        if abs(c - quant(inter, d, ROUND_HALF_UP)) < eps:
            return "double", (f"二次進位：raw {show(r)} →{k}dp {show(inter)} →{d}dp "
                              f"{show(c)}；從 raw 直接進位應為 "
                              f"{show(quant(r, d, ROUND_HALF_UP))}")
    return "wrong", f"從 raw 直接進位應為 {show(quant(r, d, ROUND_HALF_UP))}"


# 已被 v2.1 取代的舊值。anchor 只看它認得的句子，這張表補的是另一半：
# 不管寫在哪一句，只要這些字面值還在，就值得人眼確認一次是「漏改」還是「刻意講歷史」。
# 只列「現在沒有任何合法用途」的字面值。−0.37（= L0/L0N depth 效應量）與 11
# （= OLMo 的 L0N 題數）現在都仍然是對的，放進來只會製造假警報，交給 anchor 檢查。
STALE = [
    ("混組劑量反應 rho", {"-0.369"}, "pooled.1.4b.rho", ()),
    # .007 有兩個合法用途，都不是舊的混組 p：frac_at_cap 的 5/5 上界（max p=.0067），
    # 以及卡 16 刻意敘述「從 0.007 掉到 0.0355」的歷史。
    ("混組劑量反應 p", {"0.007", ".007"}, "pooled.1.4b.p",
     ("frac_at_cap", "5/5", "掉到", "從 0.007", "weakened", "弱化", "舊版")),
    ("混組劑量反應 n", {"52"}, "pooled.1.4b.n", ()),
    ("1.4B L0N 熵平均", {"3.52"}, "entropy.1.4b.L0N.mean", ()),
    ("1.4B L0 vs L0N 熵效應量", {"-0.60"}, "rb.1.4b.L0_vs_L0N.final_entropy", ()),
    ("OLMo L0 vs L0N 熵效應量", {"-0.93"}, "rb.olmo.L0_vs_L0N.final_entropy", ()),
    ("1.4B L0 vs L0N depth 效應量", {"-0.52"}, "rb.1.4b.L0_vs_L0N.depth_tau_0.1", ()),
    ("OLMo 混組劑量反應 rho", {"-0.517"}, "pooled.olmo.rho", ()),
]
STALE_WINDOW = 40      # 前後各看這麼多字元找豁免關鍵字

# 教學文件有時要「引用一個錯的數字當示範」（例：雙重捨入卡引用 −0.74 說明錯在哪）。
# 豁免必須在文件裡明示、而且只豁免緊貼在標記後面的那一個數字——絕不豁免整行，
# 否則同一行的正確數字會失去看守。寫法：<!--quoted-error-->−0.74
QUOTED_ERROR_MARK = "<!--quoted-error-->"


def _quoted_error(text: str, num_start: int) -> bool:
    """數字正前方（允許少量空白）是否有明示的錯誤示範標記。"""
    lo = max(0, num_start - len(QUOTED_ERROR_MARK) - 8)
    return text[lo:num_start].rstrip().endswith(QUOTED_ERROR_MARK)


@dataclass
class Finding:
    status: str            # OK / WRONG / UNVERIFIABLE / MISSING-SLOT / ANCHOR-MISS
    file: str
    line: int
    label: str
    stat: str
    cited: str
    raw: Optional[float]
    note: str = ""


# 不等號寫法。宣告成 value 的槽位如果實際上寫成「p < .006」，要當界線判，
# 不能當等值判——否則作者換個講法就會被誤報成算錯。
#
# 文字型不等號一律加 \b，否則 recover 的 over、thunder 的 under 會被當成界線標記，
# 把一個等值槽位無聲地降級成界線判（界線比等值寬，等於默默放行）。
# 「no less than」是下界不是上界，所以 less than 前面要擋掉 no/not；
# 「more than」不進 LB：文件裡它幾乎都出現在「moves ... by no more than X」
# 這種上界句型裡，收進 LB 只會把上界翻成下界。
BOUND_WINDOW = 25      # 往數字前面看這麼多字元找不等號

UB_MARK = re.compile(r"(?:<|≤|＜"
                     r"|\b(?:under|below|at most|no more than|fewer than"
                     r"|(?<![Nn]o )(?<![Nn]ot )less than)\b"
                     r"|小於|低於|不超過|以下)\s*[^\w\n]{0,3}$")
LB_MARK = re.compile(r"(?:>|≥|＞"
                     r"|\b(?:over|above|at least|no less than|no fewer than)\b"
                     r"|大於|高於|至少|以上)\s*[^\w\n]{0,3}$")


def detect_bound(before: str) -> Optional[str]:
    """看數字前面 ~25 字有沒有不等號；有的話回傳 'ub' / 'lb'。"""
    tail = before[-BOUND_WINDOW:]
    if UB_MARK.search(tail):
        return UB
    if LB_MARK.search(tail):
        return "lb"
    return None


def bound_ok(kind: str, cited: float, raw: float) -> bool:
    """界線判定：只問 raw 有沒有越界，不問位數對不對。

    容差 1e-12（相對）只擋浮點雜訊——像 −0.846 對上 −0.8460000000000001 這種
    「印出來一模一樣卻不相等」的比較，之前用 1e-15 會判成越界。
    容差刻意遠小於任何列印精度，所以擋不住真正的越界宣稱：
    「p < .005」碰上 raw 0.00521448 仍然是 FAIL，那是貨真價實的錯誤宣稱。
    """
    tol = max(1e-12, abs(raw) * 1e-12)
    return raw <= cited + tol if kind == UB else raw >= cited - tol


def _names(names: list) -> str:
    """一個寫法可能對到多個統計量（例如同一個 rho 在兩處引用）；列前三個就夠判讀。"""
    return names[0] if len(names) == 1 else f"{names[0]} (+{len(names) - 1} 個同值)"


def scan_file(path: Path, stats: dict, index: tuple) -> tuple:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    starts = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1

    def line_of(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    findings: list = []
    covered: set = set()          # 已被檢查覆蓋的字面值位置
    matched_sites = 0
    for site in SITES:
        if not fnmatch.fnmatch(path.name, site.files):
            continue
        hits = list(re.finditer(site.anchor, text, re.MULTILINE))
        if not hits:
            findings.append(Finding(
                "PENDING" if site.pending else "ANCHOR-MISS", path.name, 0, site.label,
                "", "", None,
                "修訂尚未寫到這段；數字一寫進去就會自動對帳" if site.pending
                else "文件已改寫或此處未引用；這個檢查沒有生效"))
            continue
        matched_sites += 1
        need = max(i for i, _, _ in site.slots) + 1
        for hit in hits:
            ln = line_of(hit.start())
            end = starts[min(ln - 1 + site.span_lines, len(starts) - 1)] \
                if ln - 1 + site.span_lines < len(starts) else len(text)
            region = text[hit.end():end]
            found = [(m,) + parse_num(m) for m in NUM_RE.finditer(region)][:need]
            for idx, stat, kind in site.slots:
                if stat not in stats:
                    findings.append(Finding("UNVERIFIABLE", path.name, ln, site.label, stat,
                                            "", None, "檢查表沒有這個統計量的重算函式"))
                    continue
                if idx >= len(found):
                    findings.append(Finding("MISSING-SLOT", path.name, ln, site.label, stat,
                                            "", stats[stat],
                                            f"anchor 後只找到 {len(found)} 個數字，需要第 {idx + 1} 個"))
                    continue
                m, cited, body, exp = found[idx]
                covered.add(hit.end() + m.start())
                raw = stats[stat]
                lit = m.group(0)
                if raw is None:
                    findings.append(Finding("UNVERIFIABLE", path.name, ln, site.label, stat,
                                            lit, None,
                                            "raw 產不出這個量（不在 pilot_*.json / battery.json 裡）"))
                    continue
                # 宣告是等值、但文件實際寫成不等式時，以文件的寫法為準
                kind = detect_bound(region[:m.start()]) or kind
                if kind in (UB, "lb"):
                    if bound_ok(kind, cited, raw):
                        findings.append(Finding("OK", path.name, ln, site.label, stat, lit, raw))
                    else:
                        side, rel = ("上界", "最大值 {} > {}") if kind == UB \
                            else ("下界", "最小值 {} < {}")
                        findings.append(Finding(
                            "WRONG", path.name, ln, site.label, stat, lit, raw,
                            f"宣稱的{side}不成立：raw "
                            + rel.format(f"{raw:.6g}", f"{cited:g}")))
                    continue
                status, note = classify(cited, body, exp, raw)
                if status != "ok" and site.note:
                    note = f"{note}｜{site.note}"
                findings.append(Finding("OK" if status == "ok" else "WRONG",
                                        path.name, ln, site.label, stat, lit, raw, note))

    # --- 值域掃描：anchor 沒吃到的數字，拿去比對整張統計量表的所有合法寫法
    correct_ix, double_ix = index
    uncovered = []
    for m in NUM_RE.finditer(text):
        if m.start() in covered:
            continue
        if _quoted_error(text, m.start()):
            findings.append(Finding(
                "VALUE-OK", path.name, line_of(m.start()), "值域掃描",
                "quoted-error 明示豁免", m.group(0), None,
                "文件以 <!--quoted-error--> 明示此數字為錯誤示範引用，不對帳"))
            continue
        val, body, exp = parse_num(m)
        if val == 0:
            uncovered.append((line_of(m.start()), m.group(0)))
            continue
        key = sci_key(val, decimals(body)) if abs(val) < 0.005 else canon(val)
        if key in correct_ix:
            findings.append(Finding("VALUE-OK", path.name, line_of(m.start()),
                                    "值域掃描", _names(correct_ix[key]), m.group(0), val))
        elif key in double_ix:
            names = double_ix[key]
            raw0 = stats[names[0]]
            # 寫成「p < .006」時作者主張的是界線不是等值，二次進位的指控不適用；
            # 改問界線成不成立，才不會把一句正確的上界宣稱誤報成算錯。
            bk = detect_bound(text[max(0, m.start() - BOUND_WINDOW):m.start()])
            if bk and bound_ok(bk, val, raw0):
                findings.append(Finding(
                    "VALUE-OK", path.name, line_of(m.start()), "值域掃描",
                    _names(names), m.group(0), raw0,
                    f"以界線判定（raw {raw0:.6g} 未越界），不以等值判定"))
            else:
                findings.append(Finding(
                    "VALUE-WRONG", path.name, line_of(m.start()), "值域掃描",
                    _names(names), m.group(0), raw0,
                    f"這個寫法只能由二次進位得到；從 raw 直接進位應為 "
                    f"{quant(raw0, decimals(body), ROUND_HALF_UP):g}"))
        else:
            uncovered.append((line_of(m.start()), m.group(0)))

    stale = []
    for m in NUM_RE.finditer(text):
        lit = f"{'-' if m.group(1) in ('-', '−') else ''}{m.group(2)}"
        ctx = text[max(0, m.start() - STALE_WINDOW): m.end() + STALE_WINDOW]
        for label, olds, stat, allow in STALE:
            if lit not in olds:
                continue
            if any(a in ctx for a in allow):
                continue          # 不是漏改：是別的統計量或刻意講歷史
            stale.append((line_of(m.start()), m.group(0), label, stats.get(stat)))
    return findings, matched_sites, uncovered, stale


# ---------------------------------------------------------------- 自我測試

def _roundtrip(lit: str, raw: float) -> str:
    """走完整條路：文字 → NUM_RE → parse_num → classify。

    直接呼叫 classify 測不到擷取與符號正規化，而那正是這批誤報的病根所在
    （Unicode 負號 U+2212、前導 +、尾隨 0），所以測試一定要從文字進去。
    """
    m = NUM_RE.search(lit)
    assert m is not None, f"NUM_RE 擷取不到 {lit!r}"
    val, body, exp = parse_num(m)
    return classify(val, body, exp, raw)[0]


# 六個歷史誤報：全都是 raw 的正確進位，卻被判成 PRESENT-BUT-WRONG。
# 判準：cited 在 k 位小數下正確 ⟺ round(raw, k) == parsed(cited)。
ROUNDTRIP_OK = [
    ("+0.35",  0.349906),      # 前導 + 要當符號吃掉，不能當成數字的一部分
    ("0.99",   0.987745),
    ("1.88",   1.87596),
    ("0.78",   0.783715),
    (".002",   0.0017552),     # 沒有整數部分的寫法
    ("−0.846", -0.846),   # Unicode 負號對上 ASCII 負值，值其實相等
]

# 反向鎖：真正的二次進位不能因為放寬容差就被放行，否則這支程式就沒用了。
ROUNDTRIP_DOUBLE = [
    ("+0.08", 0.0745356),          # 0.0745356 →3dp 0.075 →2dp 0.08；直接進位是 0.07
    ("−0.74", -0.734632),     # −0.734632 →3dp −0.735 →2dp −0.74；直接進位是 −0.73
]


def selftest() -> int:
    """把三類修正各自釘一組斷言，回歸就炸在這裡而不是炸在論文裡。"""
    fails: list = []

    def check(label: str, got, want) -> None:
        if got != want:
            fails.append(f"{label}：得到 {got!r}，應為 {want!r}")

    print("=== 1. 進位往返（正確進位不得誤判）")
    for lit, raw in ROUNDTRIP_OK:
        got = _roundtrip(lit, raw)
        print(f"   {'OK  ' if got == 'ok' else 'FAIL'}  {lit:>10s}  vs raw {raw:<12g} → {got}")
        check(f"round-trip {lit} vs {raw}", got, "ok")
    for lit, raw in ROUNDTRIP_DOUBLE:
        got = _roundtrip(lit, raw)
        print(f"   {'OK  ' if got == 'double' else 'FAIL'}  {lit:>10s}  vs raw {raw:<12g} "
              f"→ {got}（應為 double）")
        check(f"double-round {lit} vs {raw}", got, "double")
    # 正規化本身：符號與尾隨 0 不能改變數值
    for lit, want in (("−0.846", -0.846), ("-0.846", -0.846),
                      ("+0.350", 0.35), ("−0.8460", -0.846)):
        check(f"parse {lit}", parse_num(NUM_RE.search(lit))[0], want)

    print("=== 1b. quoted-error 明示豁免（只豁免緊貼標記的那一個數字）")
    _qe = "雙重捨入會寫成 <!--quoted-error-->−0.74，而直接進位是 −0.73。"
    hits = [(m.group(0), _quoted_error(_qe, m.start())) for m in NUM_RE.finditer(_qe)]
    for lit, got in hits:
        want = lit.startswith("−0.74")
        print(f"   {'OK  ' if got == want else 'FAIL'}  {lit:>8s} → 豁免={got}（應為 {want}）")
        check(f"quoted-error {lit}", got, want)

    print("=== 2. 不等號界線（界線句不得當等值判）")
    for pre, want in (("p < ", UB), ("p ≤ ", UB), ("no more than ", UB),
                      ("at most ", UB), ("less than ", UB), ("under ", UB),
                      ("below ", UB), ("at least ", "lb"), ("no less than ", "lb"),
                      ("p > ", "lb"), ("recover ", None), ("a thunder ", None)):
        got = detect_bound(pre)
        print(f"   {'OK  ' if got == want else 'FAIL'}  {pre!r:>18s} → {got}（應為 {want}）")
        check(f"detect_bound {pre!r}", got, want)
    for label, kind, cited, raw, want in (
            ("p < .006 vs .00521448", UB, 0.006, 0.00521448, True),
            ("p ≤ .05 vs .0312", UB, 0.05, 0.0312, True),
            ("no more than 0.25 vs 0.24", UB, 0.25, 0.24, True),
            ("−0.846 vs −0.846（浮點雜訊）", UB, -0.846, -0.8460000000000001, True),
            ("p < .005 vs .00521448（真越界）", UB, 0.005, 0.00521448, False),
            ("at least 0.6 vs 0.71", "lb", 0.6, 0.71, True),
            ("at least 0.8 vs 0.71（真越界）", "lb", 0.8, 0.71, False)):
        got = bound_ok(kind, cited, raw)
        print(f"   {'OK  ' if got == want else 'FAIL'}  {label} → {got}")
        check(f"bound_ok {label}", got, want)

    print("=== 3. 六個曾經無法認證的量，現在都從 raw 重算得出")
    try:
        stats = build_stats(Raw())
    except SystemExit as e:
        print(f"   跳過：raw 不齊（{e}）")
        stats = None
    if stats is not None:
        for key, want, tol in (
                ("judge.olmo.l4_pass_rate", 50.0, 0.5),
                ("judge.1.4b.l4_pass_rate", 8.33, 0.5),
                ("plateau.eligible_min", 15, 0),
                ("plateau.eligible_max", 18, 0),
                ("plateau.total", 20, 0),
                ("tokperword.1.4b.rb", -0.775, 0.002),
                ("tokperword.olmo.rb", -0.818, 0.002),
                ("tokperword.1.4b.p", 0.0005, 0.0002),
                ("tokperword.olmo.p", 0.0002, 0.0002),
                ("words.1.4b.p", 1.0, 0.0),
                ("words.olmo.p", 1.0, 0.0)):
            got = stats.get(key)
            good = got is not None and abs(got - want) <= tol
            print(f"   {'OK  ' if good else 'FAIL'}  {key:28s} = {got}（應為 {want}±{tol}）")
            if not good:
                fails.append(f"{key}：得到 {got!r}，應為 {want}±{tol}")

    if fails:
        print(f"\n--- 自我測試未通過（{len(fails)} 筆）：")
        for f in fails:
            print(f"    {f}")
        return 1
    print("\n--- 自我測試通過：進位往返、不等號界線、六個重算來源都正確。")
    return 0


# ---------------------------------------------------------------- CLI

SYMBOL = {"OK": "  OK ", "WRONG": " FAIL", "UNVERIFIABLE": " ????",
          "MISSING-SLOT": " FAIL", "ANCHOR-MISS": " warn", "PENDING": " ... ",
          "VALUE-OK": "  ok·", "VALUE-WRONG": " FAIL"}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows 主控台預設 cp950，撐不住 − 與中文
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="從 raw JSON 重算 Markdown 裡引用的每個統計量")
    ap.add_argument("files", nargs="*", help="要對帳的 Markdown 檔")
    ap.add_argument("--dump-stats", action="store_true", help="印出全部重算值後結束")
    ap.add_argument("--selftest", action="store_true",
                    help="跑對帳器自己的回歸測試（進位往返／不等號界線／重算來源）後結束")
    ap.add_argument("--list-uncovered", action="store_true",
                    help="列出檔案裡沒有被任何檢查覆蓋的數字")
    ap.add_argument("--quiet", action="store_true", help="只印不符的項目")
    ap.add_argument("--allow-unverifiable", action="store_true",
                    help="把 CLAIMED-BUT-UNVERIFIABLE 降級為警告，不影響離開碼")
    args = ap.parse_args()

    if args.selftest:          # 先於 Raw()：純函式的斷言不該被缺檔擋下
        return selftest()

    raw = Raw()
    stats = build_stats(raw)

    if args.dump_stats:
        for k in sorted(stats):
            v = stats[k]
            print(f"{k:52s} {'(unverifiable)' if v is None else v}")
        return 0
    if not args.files:
        ap.error("要對帳的檔案至少給一個（或用 --dump-stats）")

    index = build_value_index(stats)
    n_fail = n_unver = n_ok = n_stale = n_vacuous = 0
    for f in args.files:
        path = Path(f)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            print(f"[reconcile] 找不到檔案：{f}", file=sys.stderr)
            return 2
        findings, n_sites, uncovered, stale = scan_file(path, stats, index)
        n_checked = sum(1 for f in findings
                        if f.status in ("OK", "WRONG", "VALUE-OK", "VALUE-WRONG"))
        print(f"\n=== {path} （anchor 命中 {n_sites} 處；"
              f"對帳到 {n_checked} 個數字，未覆蓋 {len(uncovered)} 個）")
        for fd in findings:
            if fd.status in ("OK", "VALUE-OK"):
                n_ok += 1
                if args.quiet:
                    continue
            elif fd.status == "UNVERIFIABLE":
                n_unver += 1
            elif fd.status in ("WRONG", "MISSING-SLOT", "VALUE-WRONG"):
                n_fail += 1
            elif fd.status in ("ANCHOR-MISS", "PENDING") and args.quiet:
                continue
            rawtxt = "—" if fd.raw is None else f"{fd.raw:.6g}"
            loc = f"{fd.file}:{fd.line}" if fd.line else fd.file
            # 狀態一定要跟數字同一行。之前狀態在上一行、數字在下一行，
            # 只看數字那行時 OK 與 FAIL 長得一模一樣，讀的人會把正確的判成錯的。
            print(f"{SYMBOL[fd.status]}  {loc:38s} {fd.label}")
            print(f"{SYMBOL[fd.status]}    {fd.stat or '-'}  引用 {fd.cited or '—':>10s}  "
                  f"raw {rawtxt:>12s}" + (f"  · {fd.note}" if fd.note else ""))
        if stale:
            n_stale += len(stale)
            print(f"  -- v2.1 前的舊值仍出現 {len(stale)} 處（人眼確認是漏改還是刻意講歷史）：")
            for ln, lit, label, cur in stale:
                cur_txt = "—" if cur is None else f"{cur:.6g}"
                print(f"       {path.name}:{ln}  {lit:>8s}  {label}（現行值 {cur_txt}）")
        if args.list_uncovered:
            print(f"  -- 未覆蓋的數字（{len(uncovered)} 個）：")
            for ln, lit in uncovered:
                print(f"       {path.name}:{ln}  {lit}")
        # 空集合上的綠燈 = 儀器故障（v0.3 就這樣騙過一次：anchor 全部落空還印通過）。
        # 判準用「像統計量的數字」（兩位小數以上），才不會把只有卡號與層數的
        # 概念卡誤判成沒接上——那些卡本來就不引用任何統計量。
        statlike = [x for x in uncovered if len(x[1].split(".")[-1]) >= 2 and "." in x[1]]
        if n_checked == 0 and len(statlike) >= 5:
            n_vacuous += 1
            print(f"  !! 對帳結果為空：這份文件有 {len(statlike)} 個像統計量的數字，"
                  f"一個都沒有被對到。這是儀器沒接上，不是通過。")

    print(f"\n--- 對帳結果：PRESENT-AND-CORRECT {n_ok} · "
          f"PRESENT-BUT-WRONG {n_fail} · CLAIMED-BUT-UNVERIFIABLE {n_unver}"
          + (f" · 舊值待人眼確認 {n_stale}" if n_stale else ""))
    if n_vacuous:
        print(f"--- 未通過：{n_vacuous} 份文件對帳結果為空（見上面的 !!）。"
              f"要嘛替它加 anchor，要嘛確認它真的不引用任何統計量。")
        return 1
    if n_fail or (n_unver and not args.allow_unverifiable):
        print("--- 未通過：上面每一筆 FAIL / ???? 都要改到與 raw 一致，或把該句刪掉。")
        return 1
    print(f"--- 通過：對帳到的 {n_ok} 個數字都能從 raw 重算出來。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
