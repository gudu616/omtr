"""因果階段 patching 核心（器材部）。

依據規格版本：**CAUSAL_PREREG_v1 第一段凍結（2026-08-21）**。

這支只提供「量」的能力，不做任何裁決。裁決/分析（TOST、置換、Stouffer、
聯合表）不在本檔範圍，另有凍結分析腳本流程。

規格對應（每個公開函式的 docstring 再標一次條號）：
  §2  單 donor 腐蝕-恢復；起手窗＝gold 前 2 個 token；
      R =(Y_patched − Y_corrupted)/(Y_clean − Y_corrupted)，分子分母同窗
  R2  腐蝕＝單 token、換真實英文詞（STR 式）＋預登記強度校準
  R4  patch 位置＝被腐蝕 cue token ±鄰域（≤3 位），先單層帶
  M1  per_tok 整條入 record（不只聚合值）
  M6  陽性對照＝同層帶同位置帶；**第 0 層還原式必過版本明文禁用**
  §7.1 G1 散點＝各 patch 的 (Δs, ΔY)；**v2.8 §2：門檻全在分子 ΔY（nats）**
  v2.8 腐蝕選詞＝分位制（q=0.75），廢除一切分母；適足下限＝10×數值底線
  v2.10 適足下限是**對已選介入的事後資格檢查，不回饋選詞**（故無循環）

工程約束：一次只載一個模型；context 不得超過 3200（專案已知斷崖）；
隨機性一律走顯式 seed 參數。

CPU 煙霧測試在 `harness/tests/test_causal_patch_cpu.py`——測試器材不放在
被測的東西裡面（專案鐵則）。
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from run_pilot import load_model  # noqa: E402  （只讀不改；沿用 NeoX embed_out shim）

# ---------------------------------------------------------------- 凍結常數

LAUNCH_K = 2               # §2 起手窗＝gold 前 2 個 token
CUE_RADIUS_MAX = 3         # R4 cue 鄰域 ≤3 位
CONTEXT_CLIFF = 3200       # 專案已知斷崖（2.27s → 90.7s，且不報錯）
DENOM_EPS = 0.05           # |Y_clean − Y_corrupted| 低於此值，R 標為不穩定（nats）

# ── v2.8 分位制（取代 v2.6 相對帶）。依 DESIGN_PROPOSAL_v2.2_rulings.md v2.8
#    ＋ theory_v3_final.md 附錄 C。
CORRUPT_QUANTILE = 0.75    # 選存活候選中 Δs 的 q 分位者（q 明標「慣例」）
MIN_SURVIVORS = 12         # 存活 <12 → pool_too_small 照實報（仍選）
RANK_GUARD = 50            # 腐蝕後 gold 起手窗排名須在 top-N（護欄，v2.6 起不變）
ADEQUACY_MULTIPLE = 10.0   # 適足性下限＝選中 Δs ≥ 10 × 數值底線（nats）

SPEC_VERSION = ("CAUSAL_PREREG_v1 第一段凍結（2026-08-21）"
                "＋v2.6–v2.13 增補裁決（權威＝DESIGN_PROPOSAL_v2.2_rulings.md）")

# 腐蝕替換詞池。**來源：器材部手寫，不是任何語料統計導出的**——這件事必須
# 標明，因為 STR 的賣點正是「in-distribution 反事實」，而 in-distribution 與否
# 這裡沒有被量過，只是「常見英文詞」的直覺。挑選規則另由校準掃描決定（R2）。
CORRUPT_POOL: dict[str, tuple[str, ...]] = {
    "noun": ("house", "river", "table", "letter", "window", "garden", "morning",
             "brother", "village", "matter", "reason", "picture", "shoulder",
             "carriage", "moment", "corner", "silence", "summer", "father"),
    "verb": ("carried", "opened", "walked", "turned", "answered", "remembered",
             "watched", "followed", "entered", "offered", "listened", "returned"),
    "verb_ing": ("carrying", "opening", "walking", "turning", "answering",
                 "watching", "following", "entering", "offering", "listening"),
    "adj": ("quiet", "narrow", "bitter", "gentle", "distant", "solemn",
            "curious", "ancient", "cheerful", "slender", "peculiar"),
    "proper": ("Thomas", "Margaret", "Henry", "Eleanor", "William", "Charlotte",
               "Edward", "Frances", "Arthur", "Beatrice"),
}

# 停用詞：不當 cue 候選（換掉冠詞/介詞多半只是打亂語法，不是拔掉線索）
STOPWORDS = frozenset("""
a an the and or but if of in on at to for with by from as is was are were be been
being it its he she they them his her their this that these those not no nor so
than then there here which who whom what when where while all any both each few
more most other some such only own same too very can will just do does did done
have has had having i you we us our your my me him
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ---------------------------------------------------------------- 題目正規化

def item_prompt_gold(item: dict) -> tuple[str, str]:
    """取出 (prompt, gold)。

    L0/L0N 用 `gold_continuation`，L0P 用 `expected_continuation`（battery 實況）。
    prompt 一律用 `item["prompt"]`——沿用 run_teacher_forced.py 的慣例，因為
    gold 是接在 `prompt` 後面的那一段，`run_prompt` 與它不保證相接。
    """
    raw = item.get("gold_continuation") or item.get("expected_continuation")
    if not raw:
        raise ValueError(f"item {item.get('id')!r} 沒有 gold_continuation／expected_continuation")
    return item["prompt"], " " + raw.strip()


@dataclass
class Tokenized:
    tokens: torch.Tensor      # (1, n_total)
    n_prompt: int
    n_gold: int
    prompt: str
    gold: str

    @property
    def n_total(self) -> int:
        return int(self.tokens.shape[1])


def tokenize_pair(model, prompt: str, gold: str, context_cliff: int = CONTEXT_CLIFF) -> Tokenized:
    """把 prompt+gold 切好，並檢查 token 邊界沒有被合併掉。

    邊界檢查沿用 run_pilot.gold_logprob 的作法（tokenization_boundary_mismatch）。
    另外擋 context 斷崖：超過 3200 不是慢，是掉進 2.27s→90.7s 的坑且不報錯。
    """
    tokens = model.to_tokens(prompt + gold)
    prompt_tokens = model.to_tokens(prompt)
    n_prompt = int(prompt_tokens.shape[1])
    if not torch.equal(tokens[0, :n_prompt], prompt_tokens[0]):
        raise ValueError("tokenization_boundary_mismatch")
    n_total = int(tokens.shape[1])
    if n_total > context_cliff:
        raise ValueError(f"context {n_total} > {context_cliff}（專案已知斷崖，拒跑）")
    n_gold = n_total - n_prompt
    if n_gold < LAUNCH_K:
        raise ValueError(f"gold 只有 {n_gold} 個 token，撐不起 {LAUNCH_K} 個 token 的起手窗")
    return Tokenized(tokens, n_prompt, n_gold, prompt, gold)


# ---------------------------------------------------------------- 行為終點

@dataclass
class GoldReadout:
    """teacher-forced gold log-prob 的一次讀數。

    §2＋M1：per_tok **整條**保留（不是只留聚合值），起手窗與整段各給一個聚合。
    per_tok[i] = log p(gold token i | prompt + gold[:i])，單位 nats。

    v2.6 追加 `rank_per_tok`：每個 gold token 在該位置的排名（1＝最可能）。
    絕對護欄只留上限「腐蝕後 gold 仍須在 top-50」用的就是這個（rank 制）。
    """
    per_tok: list[float]
    launch_mean: float
    launch_sum: float
    segment_mean: float
    n_gold_tokens: int
    launch_k: int = LAUNCH_K
    rank_per_tok: list[int] = field(default_factory=list)
    launch_rank_max: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


def gold_readout_from_logits(logits: torch.Tensor, tokens: torch.Tensor,
                             n_prompt: int, launch_k: int = LAUNCH_K) -> GoldReadout:
    """從一次 forward 的 logits 算出 gold 讀數（不再多跑 forward）。"""
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    targets = tokens[0, 1:]
    idx = torch.arange(targets.shape[0], device=targets.device)
    per_tok = logp[idx, targets]
    gold_tok = per_tok[n_prompt - 1:]
    k = min(launch_k, int(gold_tok.shape[0]))
    launch = gold_tok[:k]
    # rank：有多少個 token 的 logit 嚴格高於 gold，+1。1 = gold 就是最可能的那個。
    gold_slice = logp[n_prompt - 1:]                       # (n_gold, vocab)
    gold_ids = targets[n_prompt - 1:]
    own = gold_slice[torch.arange(gold_slice.shape[0], device=gold_slice.device), gold_ids]
    ranks = (gold_slice > own.unsqueeze(-1)).sum(-1) + 1
    ranks_l = [int(r) for r in ranks.detach().cpu()]
    return GoldReadout(
        per_tok=[float(x) for x in gold_tok.detach().cpu()],
        launch_mean=float(launch.mean()),
        launch_sum=float(launch.sum()),
        segment_mean=float(gold_tok.mean()),
        n_gold_tokens=int(gold_tok.shape[0]),
        launch_k=k,
        rank_per_tok=ranks_l,
        launch_rank_max=max(ranks_l[:k]) if ranks_l else None,
    )


def gold_readout(model, tok: Tokenized, launch_k: int = LAUNCH_K) -> GoldReadout:
    """一次 forward（不存 cache）→ gold 讀數。"""
    with torch.no_grad():
        logits = model(tok.tokens)
    return gold_readout_from_logits(logits, tok.tokens, tok.n_prompt, launch_k)


def gold_readout_with_cache(model, tok: Tokenized, hook_names: Sequence[str],
                            launch_k: int = LAUNCH_K):
    """一次 forward（存指定 hook 的 cache）→ (讀數, cache)。donor 只跑這一次。"""
    wanted = set(hook_names)
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tok.tokens, names_filter=lambda n: n in wanted)
    return gold_readout_from_logits(logits, tok.tokens, tok.n_prompt, launch_k), cache


# ---------------------------------------------------------------- 腐蝕（R2）

def _word_spans(prompt: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(prompt)]


def guess_pool_class(word: str) -> str:
    """替換詞的詞類猜測。

    ⚠ **這是形態學直覺，不是 POS tagger**——本機 venv 沒有 nltk/spacy，
    器材部不自行加裝套件。猜錯的後果是替換詞詞類不合，讓腐蝕帶進語法違例
    （正是 STR 想避開的 OOD）。每個候選的 pool_class 都寫進紀錄，
    覆寫用 item["cue_pool_class"] 或 --pool-class。
    """
    if word[:1].isupper():
        return "proper"
    if word.endswith("ing"):
        return "verb_ing"
    if word.endswith("ed"):
        return "verb"
    if word.endswith(("ous", "ful", "ive", "able", "ible", "ent", "ant")):
        return "adj"
    return "noun"


def cue_word_candidates(prompt: str, n_candidates: int = 6,
                        stopwords: frozenset = STOPWORDS) -> list[tuple[int, int, str]]:
    """cue 候選＝prompt 尾端最靠近 gold 邊界的 n 個實詞（跳過停用詞）。

    R4 的 cue 是「被腐蝕的那個 token」；靠近邊界的實詞最可能是起手窗的線索，
    真正選哪一個由校準掃描決定（R2），不是這裡決定。
    """
    spans = [s for s in _word_spans(prompt) if s[2].lower() not in stopwords and len(s[2]) > 2]
    return spans[-n_candidates:][::-1]   # 由近而遠


@dataclass
class CorruptCandidate:
    cue_char_start: int
    cue_word: str
    replacement: str
    pool_class: str
    corrupted_prompt: str
    diff_token_idx: list[int]          # clean 與 corrupted 相異的 token 位置
    key: str = ""

    def __post_init__(self):
        self.key = f"{self.cue_char_start}:{self.cue_word}->{self.replacement}"


def build_corrupt_candidate(model, tok: Tokenized, cue_span: tuple[int, int, str],
                            replacement: str, pool_class: str) -> CorruptCandidate | None:
    """造一個腐蝕候選；不合格回 None。

    R2 要求「單 token、換真實英文詞、token 數相同」。這裡不假設 tokenizer
    會乾淨地把一個詞切成一個 token，而是**實際重切整段 prompt 再比對**：
      ① corrupted prompt 的 token 數必須與 clean 相同（否則 gold 邊界位移，
         donor/receiver 的位置對不齊，patch 會貼錯格子）
      ② 相異 token 位置必須連續且落在 prompt 內（單一 cue，不是散彈）
      ③ gold 段的 token 必須逐一相同（腐蝕只准動 prompt）
    任何一條不過就丟掉這個候選——**寧可少候選，不要對不齊的候選**。
    """
    s, e, word = cue_span
    if replacement.lower() == word.lower():
        return None
    corr_prompt = tok.prompt[:s] + replacement + tok.prompt[e:]
    corr_tokens = model.to_tokens(corr_prompt + tok.gold)
    if int(corr_tokens.shape[1]) != tok.n_total:
        return None                                             # ①
    clean = tok.tokens[0]
    diff = (corr_tokens[0] != clean).nonzero(as_tuple=True)[0].tolist()
    if not diff:
        return None
    if max(diff) >= tok.n_prompt:
        return None                                             # ③ 動到 gold
    if max(diff) - min(diff) + 1 != len(diff):
        return None                                             # ② 不連續
    return CorruptCandidate(s, word, replacement, pool_class, corr_prompt, diff)


def corrupt_candidates(model, tok: Tokenized, item: dict | None = None,
                       n_cue: int = 6, n_repl_per_cue: int = 4,
                       pool_class: str | None = None,
                       pool: dict[str, tuple[str, ...]] | None = None,
                       ) -> list[CorruptCandidate]:
    """枚舉腐蝕候選（cue 位置 × 替換詞）。順序完全決定於排序，不用亂數。

    item 可帶覆寫欄位：`cue_word`（指定 cue）、`cue_pool_class`、`corrupt_pool`
    （自帶替換詞清單，例如 L0P 想換成另一個自造詞而不是真英文詞）。
    """
    pool = pool or CORRUPT_POOL
    item = item or {}
    spans = cue_word_candidates(tok.prompt, n_cue)
    if item.get("cue_word"):
        want = item["cue_word"]
        spans = [s for s in _word_spans(tok.prompt) if s[2] == want] or spans
    out: list[CorruptCandidate] = []
    for span in spans:
        cls = item.get("cue_pool_class") or pool_class or guess_pool_class(span[2])
        words = tuple(item.get("corrupt_pool") or pool.get(cls, pool["noun"]))
        kept = 0
        for w in words:
            if kept >= n_repl_per_cue:
                break
            cand = build_corrupt_candidate(model, tok, span, w, cls)
            if cand is not None:
                out.append(cand)
                kept += 1
    return out


def calibrate_corruption(model, tok: Tokenized, clean: GoldReadout,
                         candidates: Sequence[CorruptCandidate],
                         launch_k: int = LAUNCH_K,
                         quantile: float = CORRUPT_QUANTILE,
                         min_survivors: int = MIN_SURVIVORS,
                         rank_guard: int | None = RANK_GUARD,
                         numerical_floor_nats: float | None = None,
                         adequacy_multiple: float = ADEQUACY_MULTIPLE,
                         n_cue_budget: int | None = None) -> dict:
    """腐蝕強度校準掃描（R2；冒煙第 ③ 錨＝這條曲線）。

    **v2.8 分位制（取代 v2.6 相對帶，v2.6 的 Δs/Δs_max ∈ [0.30,0.70] 作廢）。**
    理論部自首 B1.3 的 Δs_max 單詞探針是規格缺陷——單一抽樣不是最大值估計量，
    低估 1.1–5.7 倍是預期行為。**v2.8 廢除一切分母**，改用同一批候選自己的
    分位數當尺：

      0. 存活 = 落差為正（drop > 0）**且**過 rank 護欄（腐蝕後 gold 起手窗
         排名仍在 top-`rank_guard`）。
      1. 在存活候選中，選 **Δs 落在 q=0.75 分位**者（q 明標慣例；
         「不要比值」與「同一把尺」是導出的）。
         實作：取 `np.quantile(drops, q)`，選 drop 最接近該值者，平手用 key
         字典序——與呼叫順序無關。
      2. 存活 < `min_survivors`（12）→ **`pool_too_small` 照實報**（仍選、仍回報）。
      3. 適足性下限：**選中的 Δs ≥ `adequacy_multiple` × 數值底線（nats）**；
         不滿足 → 整題出局，獨立錯誤碼 `below_adequacy_floor`。
         `numerical_floor_nats` 為 None 時**不判**（標 `adequacy_checked: false`）
         ——底線來自 numfloor 段，順序問題見報告。

    整條曲線（含負 drop、含被 rank 護欄擋掉的）完整保留在 `curve` 裡，
    只是不給選。**R 不參與這裡任何門檻**（v2.8 §2：R 只作描述量）。
    """
    curve, n_fwd = [], 0
    for cand in candidates:
        corr_tokens = model.to_tokens(cand.corrupted_prompt + tok.gold)
        with torch.no_grad():
            logits = model(corr_tokens)
        n_fwd += 1
        r = gold_readout_from_logits(logits, corr_tokens, tok.n_prompt, launch_k)
        drop = clean.launch_mean - r.launch_mean
        curve.append({
            "key": cand.key,
            "cue_word": cand.cue_word,
            "replacement": cand.replacement,
            "pool_class": cand.pool_class,
            "diff_token_idx": cand.diff_token_idx,
            "y_corr_launch": r.launch_mean,
            "y_corr_segment": r.segment_mean,
            "delta_launch": r.launch_mean - clean.launch_mean,
            "delta_segment": r.segment_mean - clean.segment_mean,
            "drop_launch": drop,
            "launch_rank_max": r.launch_rank_max,
        })

    # v2.14 儀器部補件①：候選耗盡的**類型**——預算耗盡（掃滿 n_cue×n_repl）
    # vs 產生器耗盡（對齊檢查過不了，生不出那麼多）。兩者的補救方向相反：
    # 前者加預算有用，後者加預算沒用（L0P-H-02 五模型全 8、全距 0，疑結構性）。
    budget = n_cue_budget if n_cue_budget else None
    exhaustion = None
    if budget is not None:
        exhaustion = "budget_exhausted" if len(curve) >= budget else "generator_exhausted"
    base = {"curve": curve, "mode": "quantile_v2.8", "quantile": quantile,
            "candidate_budget": budget, "exhaustion_type": exhaustion,
            "min_survivors": min_survivors, "rank_guard": rank_guard,
            "n_candidates": len(curve), "n_forwards": n_fwd,
            "observed_max_drop": max((c["drop_launch"] for c in curve), default=None),
            "adequacy_multiple": adequacy_multiple,
            "numerical_floor_nats": numerical_floor_nats}
    if not curve:
        # v2.13 D：零候選也要**入分母**且算 bad——所以 n_survivors 一定要有值
        # （缺欄位會讓判定器的 `if n_surv is not None` 把整題跳過，靜默縮小分母）。
        # 錯誤碼統一 no_valid_corrupt_candidate（與 run_item 註解對齊）。
        return {**base, "selected": None, "fallback": "no_valid_corrupt_candidate",
                "n_survivors": 0, "n_nonpositive_drop": 0,
                "n_rank_guard_rejected": 0, "pool_too_small": True}

    positive = [c for c in curve if c["drop_launch"] > 0]
    n_neg = len(curve) - len(positive)
    if rank_guard:
        survivors = [c for c in positive
                     if c["launch_rank_max"] is not None
                     and c["launch_rank_max"] <= rank_guard]
    else:
        survivors = positive
    n_rank_out = len(positive) - len(survivors)
    base.update({"n_nonpositive_drop": n_neg, "n_rank_guard_rejected": n_rank_out,
                 "n_survivors": len(survivors),
                 "pool_too_small": bool(len(survivors) < min_survivors)})
    if not survivors:
        code = "no_positive_drop" if not positive else "no_candidate_within_rank_guard"
        # v2.14 儀器部補件②：零存活格輸出**腐蝕後 rank 分布**（僅診斷，不進判定）。
        # 用途：分辨「腐蝕過強把 gold 打飛」與「clean 排名本來就低」——
        # 兩者都會零存活，但補救方向不同。
        ranks = sorted(c["launch_rank_max"] for c in positive
                       if c["launch_rank_max"] is not None)
        base["zero_survivor_rank_diagnostic"] = {
            "n_positive_drop": len(positive), "ranks_after_corruption": ranks,
            "min_rank": (ranks[0] if ranks else None),
            "median_rank": (ranks[len(ranks) // 2] if ranks else None),
            "rank_guard": rank_guard,
            "note": "僅診斷，不進任何判定（v2.14 儀器部補件②）"}
        return {**base, "selected": None, "fallback": code}

    drops = np.array([c["drop_launch"] for c in survivors], float)
    target = float(np.quantile(drops, quantile))
    sel = min(survivors, key=lambda c: (abs(c["drop_launch"] - target), c["key"]))
    emp_q = float(np.mean(drops <= sel["drop_launch"]))
    out = {**base, "selected": sel["key"], "fallback": None,
           "quantile_target_drop": target,
           "selected_drop_launch": sel["drop_launch"],
           "selected_quantile": emp_q,        # 選中者的經驗分位（對帳用）
           "selected_launch_rank_max": sel["launch_rank_max"]}

    # 適足性下限（v2.8 §1）
    if numerical_floor_nats is None:
        out["adequacy_checked"] = False
        out["adequacy_note"] = ("數值底線未注入（來自 numfloor 段），本題不判適足性；"
                                "以 --numerical-floor-nats 注入後才會判")
    else:
        floor = adequacy_multiple * numerical_floor_nats
        out["adequacy_checked"] = True
        out["adequacy_floor_nats"] = floor
        out["meets_adequacy"] = bool(sel["drop_launch"] >= floor)
        # v2.10：適足下限是**對已選介入的事後資格檢查，不回饋到選詞**
        # （選詞只看分位制），所以 selected / quantile_target_drop 一概不動。
        # v2.8 §1：不滿足 → 整題出局，**獨立錯誤碼**（不與選不出詞的三種混格）。
        out["disqualify_item"] = not out["meets_adequacy"]
        if out["disqualify_item"]:
            out["disqualify_code"] = "below_adequacy_floor"
    return out



# ---------------------------------------------------------------- patch（R4/M6）

def layer_band(n_layers: int, lo_frac: float, hi_frac: float,
               hook_kind: str = "resid_pre") -> list[int]:
    """把「相對深度帶」換成具體層號（五個模型層數 16/24/32 不同，用相對深度才可比）。

    M6 硬禁：`resid_pre` 第 0 層＝把 embedding 整個還原，構造上必過，
    **明文禁用**。踩到直接 raise，不給 flag 繞過。
    """
    lo = max(0, int(np.floor(lo_frac * n_layers)))
    hi = min(n_layers - 1, int(np.ceil(hi_frac * n_layers)) - 1)
    layers = list(range(lo, max(lo, hi) + 1))
    if hook_kind == "resid_pre" and 0 in layers:
        raise ValueError(
            f"相對深度帶 [{lo_frac}, {hi_frac}) 在 n_layers={n_layers} 上對到層 {layers}，"
            "含 resid_pre 第 0 層＝還原式必過的陽性對照，CAUSAL_PREREG_v1 §6 明文禁用。"
            "**不自動夾到第 1 層**——那等於偷改要求的層帶；請呼叫端改帶或改 hook_kind。")
    return layers


def cue_neighborhood(diff_token_idx: Sequence[int], radius: int, n_prompt: int) -> list[int]:
    """cue 鄰域（R4：±3 位為上限）。夾在 prompt 範圍內，不許溢出到 gold。"""
    if radius > CUE_RADIUS_MAX:
        raise ValueError(f"radius {radius} > R4 上限 {CUE_RADIUS_MAX}")
    lo = max(0, min(diff_token_idx) - radius)
    hi = min(n_prompt - 1, max(diff_token_idx) + radius)
    return list(range(lo, hi + 1))


def hook_names_for(layers: Iterable[int], hook_kind: str = "resid_pre") -> list[str]:
    return [f"blocks.{l}.hook_{hook_kind}" for l in layers]


def make_patch_hooks(clean_cache, layers: Sequence[int], positions: Sequence[int],
                     hook_kind: str = "resid_pre",
                     allow_trivial: bool = False) -> list[tuple[str, Callable]]:
    """單 donor 貼回：把 clean run 在指定層帶、指定位置的殘差流貼進 receiver。

    多層帶時每一層都貼；深層的貼入會蓋掉模型自己算出來的東西——這是這個
    設計的性質（層帶越寬干預越強），紀錄裡照實留下 band_layers。

    `allow_trivial` 只給 CPU 管路測試用（貼滿＝必回到 clean，用來驗證管路
    有沒有接對）。M6 的禁令在 layer_band() 與這裡各擋一次，避免有人用
    「直接給層號清單」繞過 layer_band。**跑陽性對照時永遠不准開。**
    """
    if not allow_trivial and hook_kind == "resid_pre" and 0 in list(layers):
        raise ValueError(
            "resid_pre 第 0 層＝還原式必過的陽性對照，CAUSAL_PREREG_v1 §6 明文禁用"
            "（管路測試請顯式傳 allow_trivial=True）")
    pos = torch.tensor(sorted(set(int(p) for p in positions)), dtype=torch.long)
    hooks = []
    for name in hook_names_for(layers, hook_kind):
        donor = clean_cache[name]

        def _fn(resid, hook, donor=donor, pos=pos):
            if int(resid.shape[1]) <= int(pos[-1]):
                raise RuntimeError(
                    f"{hook.name}: seq {int(resid.shape[1])} 容不下 patch 位置 {int(pos[-1])}")
            out = resid.clone()
            p = pos.to(out.device)
            out[:, p, :] = donor[:, p, :].to(out.device, out.dtype)
            return out

        hooks.append((name, _fn))
    return hooks


def patched_readout(model, corr_tokens: torch.Tensor, n_prompt: int,
                    clean_cache, layers: Sequence[int], positions: Sequence[int],
                    hook_kind: str = "resid_pre", launch_k: int = LAUNCH_K,
                    allow_trivial: bool = False, batch: int = 1) -> GoldReadout:
    """一次帶 hook 的 forward → patched 的 gold 讀數。

    `batch > 1`：把同一列 token 複製成一個 batch 再跑，只讀第 0 列。
    數學上答案該完全一樣，實際上 batch 維會改變 kernel 選擇與規約順序，
    差多少就是數值底線的一條路徑（v2.6 §2 的「改 batch 組成/size」）。
    donor cache 的 batch 維是 1，貼回時靠 broadcast 覆蓋整個 batch。
    """
    hooks = make_patch_hooks(clean_cache, layers, positions, hook_kind, allow_trivial)
    toks = corr_tokens if batch == 1 else corr_tokens.repeat(batch, 1)
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=hooks)
    return gold_readout_from_logits(logits[0:1], corr_tokens, n_prompt, launch_k)


def recovery_ratio(y_clean: float, y_corr: float, y_patch: float,
                   eps: float = DENOM_EPS) -> dict:
    """§2 的 R =(Y_patched − Y_corrupted)/(Y_clean − Y_corrupted)，分子分母同窗。

    分母（＝腐蝕落差 Δs）永遠一起回報：它是 G1-gold 散點的橫軸（§7.1），
    也是 R 可不可信的唯一判斷依據。|分母| < eps 時 R 標成不穩定但**照樣回報
    數值**——不自行丟資料，要不要濾由分析部依凍結判準決定。
    """
    denom = y_clean - y_corr
    r = (y_patch - y_corr) / denom if abs(denom) >= 1e-12 else float("nan")
    return {"R": r, "denom": denom, "numer": y_patch - y_corr,
            "denom_unstable": bool(abs(denom) < eps)}


# ------------------------------------------------------- 自由續寫（溯源軌素材）

def free_continuation(model, prompt_tokens: torch.Tensor, max_new_tokens: int = 60,
                      seed: int | None = None, do_sample: bool = False,
                      temperature: float = 0.8,
                      fwd_hooks: Sequence[tuple[str, Callable]] | None = None) -> str:
    """自由續寫（§2 雙軌的第二軌：要它自己寫出來，不是把答案遞給它問信不信）。

    **use_past_kv_cache=False 是刻意的**：patch hook 是按「絕對位置」貼的，
    開 KV cache 時每一步只前傳新 token，hook 拿到的 seq 維只有 1，位置索引
    對不上。關掉 cache 每一步重跑整段，位置索引才恆定。代價是慢，換的是
    「貼在該貼的格子上」。

    seed 為 None 且 do_sample=True 會 raise——取樣不准沒有種子（顯式 seed 原則）。
    """
    if do_sample and seed is None:
        raise ValueError("do_sample=True 必須給 seed")
    if do_sample:
        torch.manual_seed(int(seed))
    n_prompt = int(prompt_tokens.shape[1])
    ctx = model.hooks(fwd_hooks=list(fwd_hooks or []))
    with torch.no_grad(), ctx:
        out = model.generate(prompt_tokens, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature,
                             use_past_kv_cache=False, verbose=False)
    return model.to_string(out[0, n_prompt:])


# ---------------------------------------------------------------- 單題編排

@dataclass
class PatchConfig:
    """一題要跑什麼。全部顯式，沒有隱藏預設。"""
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"mid": (0.40, 0.60)})
    hook_kind: str = "resid_pre"
    radius: int = 2
    launch_k: int = LAUNCH_K
    # v2.8 分位制：不再有任何分母
    quantile: float = CORRUPT_QUANTILE
    min_survivors: int = MIN_SURVIVORS
    rank_guard: int | None = RANK_GUARD
    adequacy_multiple: float = ADEQUACY_MULTIPLE
    numerical_floor_nats: float | None = None       # 由 numfloor 段注入；None=不判適足性
    expand_pool: bool = True                        # v2.14 G3 擴池輪
    expand_max_candidates: int = 48                 # 每題候選上限（凍結）
    denom_eps: float = DENOM_EPS                    # 只影響 R 的 unstable 標記（描述量）
    seed: int = 20260822
    n_cue: int = 6
    n_repl_per_cue: int = 4
    pool_class: str | None = None
    gen_tokens: int = 0          # >0 才做自由續寫（溯源軌素材）
    gen_seed: int = 20260822
    gen_sample: bool = False


def run_item(model, item: dict, cfg: PatchConfig) -> dict:
    """單題完整流程：clean → 校準掃描 → 腐蝕 → 逐層帶 patch → R。

    forward 帳（每題）：
      1（clean，帶 cache）＋ N_cal（校準掃描）＋ 1（選定腐蝕的重量，可省但
      為了讓紀錄自洽而重跑）＋ len(bands)（每個層帶一次）
      ＋ 3×(gen_tokens 步)（若開自由續寫：clean / corrupted / patched）
    每一項都寫進 n_forwards，不用估的。
    """
    t0 = time.time()
    prompt, gold = item_prompt_gold(item)
    tok = tokenize_pair(model, prompt, gold)
    n_layers = int(model.cfg.n_layers)
    bands = {name: layer_band(n_layers, lo, hi, cfg.hook_kind)
             for name, (lo, hi) in cfg.bands.items()}
    all_layers = sorted({l for ls in bands.values() for l in ls})
    clean, cache = gold_readout_with_cache(
        model, tok, hook_names_for(all_layers, cfg.hook_kind), cfg.launch_k)
    n_fwd = 1

    cands = corrupt_candidates(model, tok, item, cfg.n_cue, cfg.n_repl_per_cue,
                               cfg.pool_class)
    # v2.8：沒有 Δs_max 探針了（分母全廢），直接掃候選再取分位
    cal = calibrate_corruption(
        model, tok, clean, cands, cfg.launch_k,
        quantile=cfg.quantile, min_survivors=cfg.min_survivors,
        rank_guard=cfg.rank_guard,
        numerical_floor_nats=cfg.numerical_floor_nats,
        adequacy_multiple=cfg.adequacy_multiple,
        n_cue_budget=cfg.n_cue * cfg.n_repl_per_cue)
    n_fwd += cal["n_forwards"]

    # ── 擴池輪（**v2.15 §2 取代 v2.14 的存活數切法**）─────────────────
    # v2.15：**擴池資格＝耗盡類型，不設任何存活數門檻**。
    #   `budget_exhausted`    → **進擴池**（含零存活格——它們只是還沒掃夠）
    #   `generator_exhausted` → **排除**（對齊檢查生不出更多候選，加預算無用）
    # v2.14 的「1≤存活<12 才擴、零存活不擴」作廢。理由：零存活但預算耗盡的格
    # 只是掃得不夠多，不是本質上救不了；而產生器耗盡的格加多少預算都一樣。
    # L0M-11（generator_exhausted、min rank=51 只差一名）依此排除——
    # **「近失是誘惑不是論據」，top-50 護欄不動，51 照實報。**
    # 其餘凍結不變：至多一輪、每題至多 48 候選、仍<12 標 pool_too_small_final。
    exh = cal.get("exhaustion_type")
    if cfg.expand_pool and exh == "budget_exhausted" and len(cands) < cfg.expand_max_candidates:
        per_cue = max(1, -(-cfg.expand_max_candidates // max(1, cfg.n_cue)))
        cands2 = corrupt_candidates(model, tok, item, cfg.n_cue, per_cue, cfg.pool_class)
        if len(cands2) > len(cands):
            cal2 = calibrate_corruption(
                model, tok, clean, cands2, cfg.launch_k,
                quantile=cfg.quantile, min_survivors=cfg.min_survivors,
                rank_guard=cfg.rank_guard,
                numerical_floor_nats=cfg.numerical_floor_nats,
                adequacy_multiple=cfg.adequacy_multiple,
                n_cue_budget=cfg.n_cue * per_cue)
            n_fwd += cal2["n_forwards"]
            n_after = cal2.get("n_survivors") or 0
            cal2["expansion"] = {
                "applied": True, "round": 1, "eligibility": "budget_exhausted",
                "n_candidates_before": len(cands), "n_candidates_after": len(cands2),
                "n_survivors_before": cal.get("n_survivors"),
                "n_survivors_after": n_after,
                "cap": cfg.expand_max_candidates,
                "still_too_small": bool(n_after < cfg.min_survivors),
                "still_zero": bool(n_after == 0),
                "rule": "v2.15 §2：資格＝耗盡類型；至多一輪；每題至多 48 候選",
            }
            if cal2["expansion"]["still_too_small"]:
                cal2["pool_too_small_final"] = True
            cal, cands = cal2, cands2
        else:
            cal["expansion"] = {"applied": False, "eligibility": "budget_exhausted",
                                "reason": "重掃後候選數未增加（實際已達產生器上限）"}
    elif cfg.expand_pool and exh == "generator_exhausted":
        cal["expansion"] = {
            "applied": False, "eligibility": "generator_exhausted",
            "reason": ("產生器耗盡＝對齊檢查生不出更多候選，加預算無用；"
                       "依 v2.15 §2 排除不擴")}

    rec = {
        "id": item["id"], "level": item.get("level"),
        "n_prompt_tokens": tok.n_prompt, "n_gold_tokens": tok.n_gold,
        "n_layers": n_layers,
        "bands": {k: v for k, v in bands.items()},
        "hook_kind": cfg.hook_kind, "radius": cfg.radius, "launch_k": cfg.launch_k,
        "clean": clean.to_json(),
        "calibration": cal,
        "n_candidates": len(cands),
    }
    if cal["selected"] is None:
        # 三種都是「這題沒有可用的腐蝕」，但原因不同，不能混成一格：
        #   no_valid_corrupt_candidate      → 對齊檢查全數不過（tokenizer 層面）
        #   no_positive_drop                → 造得出候選，但沒有一個真的弄壞 gold
        #   no_candidate_within_rank_guard  → 有落差但全把 gold 打出 top-N
        rec["error"] = cal.get("fallback") or "no_valid_corrupt_candidate"
        rec["n_forwards"] = n_fwd
        rec["seconds"] = time.time() - t0
        return rec

    if cal.get("disqualify_item"):
        # 選詞本身成功了（紀錄完整留著供對帳），但這題不夠格進主量測。
        # 錯誤碼與「選不出腐蝕」那三種分開，v2.7 ③「不靜默歸類」的延伸。
        rec["error"] = cal["disqualify_code"]
        rec["n_forwards"] = n_fwd
        rec["seconds"] = time.time() - t0
        return rec

    chosen = next(c for c in cands if c.key == cal["selected"])
    corr_tokens = model.to_tokens(chosen.corrupted_prompt + tok.gold)
    with torch.no_grad():
        corr_logits = model(corr_tokens)
    n_fwd += 1
    corrupted = gold_readout_from_logits(corr_logits, corr_tokens,
                                         tok.n_prompt, cfg.launch_k)
    positions = cue_neighborhood(chosen.diff_token_idx, cfg.radius, tok.n_prompt)
    rec["corruption"] = {
        "key": chosen.key, "cue_word": chosen.cue_word,
        "replacement": chosen.replacement, "pool_class": chosen.pool_class,
        "diff_token_idx": chosen.diff_token_idx, "positions": positions,
        "corrupted_prompt": chosen.corrupted_prompt,
        "fallback": cal["fallback"],
    }
    rec["corrupted"] = corrupted.to_json()

    patches = []
    for name, layers in bands.items():
        pr = patched_readout(model, corr_tokens, tok.n_prompt, cache, layers,
                             positions, cfg.hook_kind, cfg.launch_k)
        n_fwd += 1
        launch = recovery_ratio(clean.launch_mean, corrupted.launch_mean,
                                pr.launch_mean, cfg.denom_eps)
        segment = recovery_ratio(clean.segment_mean, corrupted.segment_mean,
                                 pr.segment_mean, cfg.denom_eps)
        patches.append({
            "band": name, "band_layers": layers, "positions": positions,
            "patched": pr.to_json(),
            "launch": launch, "segment": segment,
            # §7.1 G1 散點的一個點。**v2.8 §2：門檻全部在分子 ΔY（nats）上，
            # R = ΔY/Δs 僅作描述量、不得進入任何門檻**（分母隨機且近 0 有密度
            # ⇒ R 無有限變異數；截斷位置一動 SD 動 4.2 倍而分子紋風不動）。
            #   delta_s = Y_clean − Y_corrupted（正值＝腐蝕把 gold 弄差了）
            #   delta_y = Y_patched − Y_corrupted（分子；恢復了多少 nats）
            "g1_point": {"delta_s": launch["denom"], "delta_y": launch["numer"],
                         "R": launch["R"]},
        })
    rec["patches"] = patches

    if cfg.gen_tokens > 0:
        prompt_only = tok.tokens[:, :tok.n_prompt]
        corr_prompt_only = corr_tokens[:, :tok.n_prompt]
        gens = {"clean": free_continuation(model, prompt_only, cfg.gen_tokens,
                                          cfg.gen_seed, cfg.gen_sample),
                "corrupted": free_continuation(model, corr_prompt_only, cfg.gen_tokens,
                                               cfg.gen_seed, cfg.gen_sample)}
        for name, layers in bands.items():
            hooks = make_patch_hooks(cache, layers, positions, cfg.hook_kind)
            gens[f"patched::{name}"] = free_continuation(
                model, corr_prompt_only, cfg.gen_tokens, cfg.gen_seed,
                cfg.gen_sample, fwd_hooks=hooks)
        rec["generations"] = gens
        n_fwd += cfg.gen_tokens * (2 + len(bands))

    rec["n_forwards"] = n_fwd
    rec["seconds"] = time.time() - t0
    return rec


# ---------------------------------------------------------------- 前向計時

def time_forward(model, tok: Tokenized, n: int = 20, warmup: int = 3) -> dict:
    """量真實的單次前向秒數（rig_feasibility §3 列為硬前置：樂觀 0.15 vs 保守 1.0
    差 7 倍，開跑前必須先把這個「算的」換成「量的」）。"""
    with torch.no_grad():
        for _ in range(warmup):
            model(tok.tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(tok.tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
    return {"n": n, "warmup": warmup, "seconds_total": dt, "seconds_per_forward": dt / n,
            "n_tokens": tok.n_total,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype)}


def rebuild_selected(model, item: dict, cfg: PatchConfig, rec: dict):
    """從一筆已跑好的紀錄，把「同一個介入」原封不動重建出來。

    數值底線（v2.6 §2）要的是**同介入同輸入**只換 dtype／batch，所以不能重跑
    校準（那會換介入）。回傳 (tok, cache, corr_tokens, layers_by_band, positions)。
    """
    prompt, gold = item_prompt_gold(item)
    tok = tokenize_pair(model, prompt, gold)
    n_layers = int(model.cfg.n_layers)
    bands = {name: layer_band(n_layers, lo, hi, cfg.hook_kind)
             for name, (lo, hi) in cfg.bands.items()}
    all_layers = sorted({l for ls in bands.values() for l in ls})
    clean, cache = gold_readout_with_cache(
        model, tok, hook_names_for(all_layers, cfg.hook_kind), cfg.launch_k)
    corr_tokens = model.to_tokens(rec["corruption"]["corrupted_prompt"] + tok.gold)
    if int(corr_tokens.shape[1]) != tok.n_total:
        raise ValueError("重建的腐蝕 prompt token 數對不上原紀錄")
    with torch.no_grad():
        corrupted = gold_readout_from_logits(model(corr_tokens), corr_tokens,
                                             tok.n_prompt, cfg.launch_k)
    return tok, clean, cache, corr_tokens, corrupted, bands, rec["corruption"]["positions"]


__all__ = [
    "SPEC_VERSION", "LAUNCH_K", "CUE_RADIUS_MAX", "CONTEXT_CLIFF", "CORRUPT_POOL",
    "DENOM_EPS", "PatchConfig", "GoldReadout", "Tokenized",
    "load_model", "item_prompt_gold", "tokenize_pair",
    "gold_readout", "gold_readout_with_cache", "gold_readout_from_logits",
    "corrupt_candidates", "calibrate_corruption", "build_corrupt_candidate",
    "cue_word_candidates", "guess_pool_class",
    "CORRUPT_QUANTILE", "MIN_SURVIVORS", "RANK_GUARD", "ADEQUACY_MULTIPLE",
    "layer_band", "cue_neighborhood", "make_patch_hooks", "patched_readout",
    "hook_names_for", "recovery_ratio", "free_continuation", "run_item",
    "time_forward", "rebuild_selected",
]
