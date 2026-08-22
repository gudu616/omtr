"""§7.7 逐 token 位置檢定——資料對應與置換檢定實作（v2.21+v2.22 凍結，盲期已結束）。

規格權威（雙出處，程式碼內逐條引節次，不憑記憶）：
  ① planning/DESIGN_PROPOSAL_v2.2_rulings.md v2.21 塊（第 398-594 行）
    ＋v2.22 塊（第 595-614 行，BOS 全域化誤植之量出型修正，儀器部自測抓出）
  ② planning/office_reports/theory_pos_test_mapping_v1.md（v1.1 定稿＋§4-B
    v2.22 可見更正塊，第 326-390 行，新斷言0）
判準本身一字未改（theory_v3_final.md §7.7(c) 第 649-680 行）；本檔只是資料對應
的實作。工單授權：判準已凍結入鏈，本檔可以讀 per_tok 數值（盲期已結束）。
BOS 位移（斷言0）與空格記帳（斷言2放寬，見 compute_offsets_checked docstring）
逐模型量出，不寫死；出處與雙重驗證數字全部在對應函式的 docstring 裡。

【單一出處】B_PERM/SEED/ALPHA/DESCRIPTIVE_MODELS/PYTHIA_PREFIX 全部從
causal_analysis 模組 import，不另抄（v2.21 §5：「亦不得另抄模型名清單」）；
合格 run 一律由 causal_analysis.DESCRIPTIVE_MODELS 排除法產生，
禁止用 rec["backend_dtype"] 篩（同一坑見 rig.md「資料地雷」節）。

【輸出紀律】只落聚合結果與斷言狀態，不 dump 原始 per_tok/rank_per_tok 前綴
（rig.md 2026-08-22 新增地雷：records 第一筆恆為 L0-01，dump 檔頭必中裁決池）。

【只備料，不判定】本檔輸出 Δ 聚合值、置換 p（主檢定＋§6b R2 整窗排除閘門）、
等價 CI、SD(Δ_題)、逐題 run 數表複核、§2 逐模型 n 複核、全部斷言狀態、
sign-flip 交叉檢查（只揭露）——不寫「通過/不通過」，判定由統籌代理按凍結文字執行。

用法：
  .venv/Scripts/python.exe harness/pos_test_v221.py --self-test
  .venv/Scripts/python.exe harness/pos_test_v221.py --run-id main_winB1
輸出：
  results/causal/pos_test_v221.json ＋ planning/rig_pos_test_report.md（人讀摘要）
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_analysis as ca          # noqa: E402  B_PERM/SEED/ALPHA/DESCRIPTIVE_MODELS 等單一出處
import verify_battery as vb           # noqa: E402  自測用：probes_from/_edge_trim（凍結產生器，不重寫）

from transformers import AutoTokenizer  # noqa: E402  只用來拿 offset_mapping，不載入模型權重

PROJ = Path(__file__).resolve().parent.parent

# q=0.75 是既凍分位制（內部凍結值速查紀錄「分位制 q=0.75」；
# 亦見 causal_patch.py:50 CORRUPT_QUANTILE）。本檔不 import causal_patch——
# 那會拉進 run_pilot/transformer_lens/torch 整條 GPU 依賴鏈，本檔全程 CPU-only、
# 不碰任何模型權重。這裡是原樣抄值＋雙出處引用，不算另立新標準（單一出處原則
# 管的是「同一條規則不要有兩份會走鐘的定義」，抄一個純量常數並註明出處不違反它）。
CORRUPT_QUANTILE = 0.75

PYTHIA_TOKENIZER_ID = "EleutherAI/pythia-410m"   # 四個 Pythia 共用同一份 tokenizer（v2.21 §5-b 原文）
OLMO_MODEL = "allenai/OLMo-2-0425-1B"

DEFAULT_BATTERY = PROJ / "battery" / "battery.json"
DEFAULT_L0V = PROJ / "battery" / "l0_verification.json"

# 下列兩張表是 mapping v1.1 §2／§7-a 的「已量」表，僅供本檔跑完後交叉核對用
# （不是本檔重新發明的門檻；不吻合只印警告，不 abort——這兩張表不在 v2.21
# 明列的「斷言強制／必須 abort」清單裡）。
EXPECTED_PER_MODEL_N = {
    "EleutherAI/pythia-410m": 16, "EleutherAI/pythia-1b": 14,
    "EleutherAI/pythia-1.4b": 14, "allenai/OLMo-2-0425-1B": 15,
    "EleutherAI/pythia-2.8b": 12,
}
EXPECTED_RUN_COUNT_HIST = {"4": 13, "3": 1, "2": 1, "1": 2}   # §7-a


def _abort(msg: str):
    raise SystemExit(f"[pos_test_v221 ABORT] {msg}")


# ============================================================== I/O 載入 ==

def load_battery(path=None):
    """id -> item（不預先篩 level；篩選由呼叫端依 §2 條件 1 做，id 前綴 'L0-'）。"""
    d = json.load(open(path or DEFAULT_BATTERY, encoding="utf-8"))
    return {it["id"]: it for it in d["items"]}


def load_l0_verification(path=None):
    """title -> entry（join key＝title，l0_verification 無 id 欄，見 v2.21 §2）。"""
    d = json.load(open(path or DEFAULT_L0V, encoding="utf-8"))
    return {entry["title"]: entry for entry in d}


def corpus_of(model: str) -> str:
    if model.startswith(ca.PYTHIA_PREFIX):
        return "pile"
    if model == OLMO_MODEL:
        return "olmo_mix"
    raise ValueError(f"未知模型，無法決定訓練語料：{model!r}")


def tokenizer_family_id_for(model: str) -> str:
    """§4-B 斷言3 的『一致』定義：同語料家族共用同一份 tokenizer 即算一致
    （v2.21 原文：「四個 Pythia 共用 EleutherAI/pythia-* tokenizer」——
    不要求逐模型尺寸各自的 tokenizer repo，Pythia 全系列本就是同一份 GPT-NeoX
    tokenizer）。「不得跨模型共用 offsets」管的是跨語料家族（Pythia↔OLMo）。"""
    if model.startswith(ca.PYTHIA_PREFIX):
        return PYTHIA_TOKENIZER_ID
    if model == OLMO_MODEL:
        return OLMO_MODEL
    raise ValueError(f"未知模型，無法決定 tokenizer：{model!r}")


_TOKENIZER_CACHE: dict[str, object] = {}


def get_tokenizer(model: str):
    tid = tokenizer_family_id_for(model)
    if tid not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[tid] = AutoTokenizer.from_pretrained(tid)
    return _TOKENIZER_CACHE[tid]


# ===================================================== §4：位置對應演算法 ==

def text_layout(item: dict):
    """§4-A（唯一出處 causal_patch.py:96-99）。model 無關——full 是純文字重建。"""
    prompt = item["prompt"]
    gold = " " + item["gold_continuation"].strip()
    full = prompt + gold
    L = len(prompt)
    return full, L


def locate_probes(full: str, L: int, probe_texts, stats: dict):
    """§4-C 唯一定位 ＋ §4-D 起手分類（prompt/gold/boundary）。"""
    out = []
    for i, p in enumerate(probe_texts):
        cnt = full.count(p)
        stats["probe_locate_checked"] = stats.get("probe_locate_checked", 0) + 1
        if cnt != 1:
            _abort(f"probe 定位不唯一（§4-C）：第 {i} 個 probe 在 full 裡出現 {cnt} 次："
                   f"{p[:60]!r}")
        stats["probe_locate_unique_pass"] = stats.get("probe_locate_unique_pass", 0) + 1
        s = full.find(p)
        e = s + len(p)
        cls = "gold" if s >= L else ("prompt" if e <= L else "boundary")
        out.append({"idx": i, "text": p, "s": s, "e": e, "cls": cls})
    return out


def build_windows(item: dict, l0v_entry: dict, stats: dict):
    """整合 §4-C/D 位置與 §3/§4-E 窗母體（排除跨界窗）。model 無關，逐題算一次即可。"""
    full, L = text_layout(item)
    probes_pile = l0v_entry["probes"]["pile"]
    probes_olmo = l0v_entry["probes"]["olmo_mix"]
    texts_pile = [p["probe"] for p in probes_pile]
    texts_olmo = [p["probe"] for p in probes_olmo]
    if texts_pile != texts_olmo:
        _abort(f"{item['id']}: 兩語料 probe 文字序不對齊（應共用同一份文字，見 §3-0 前置）")
    spans = locate_probes(full, L, texts_pile, stats)
    n_gold = sum(1 for w in spans if w["cls"] == "gold")
    n_prompt_only = sum(1 for w in spans if w["cls"] == "prompt")
    n_boundary = sum(1 for w in spans if w["cls"] == "boundary")
    audit = stats.setdefault("boundary_audit", {"gold": 0, "prompt": 0, "boundary": 0})
    audit["gold"] += n_gold; audit["prompt"] += n_prompt_only; audit["boundary"] += n_boundary
    if n_gold != 10 or n_prompt_only != 0 or n_boundary != 1:
        _abort(f"{item['id']}: §4-E/§8 分類不符已量預期"
               f"（gold={n_gold},prompt={n_prompt_only},boundary={n_boundary}，預期 10/0/1）")
    windows = {}
    for corpus, probelist in (("pile", probes_pile), ("olmo_mix", probes_olmo)):
        windows[corpus] = [{"idx": w["idx"], "s": w["s"], "e": w["e"],
                            "count": p["count"], "status": p["status"]}
                           for w, p in zip(spans, probelist) if w["cls"] == "gold"]
    first_idx = min(windows["pile"], key=lambda w: w["s"])["idx"]   # 起手窗＝字元起點最早者
    return {"full": full, "L": L, "windows": windows, "first_idx": first_idx,
            "gold_idx_sorted": sorted(w["idx"] for w in windows["pile"])}


def corpus_threshold(windows_corpus, quantile, enforce_k3, stats):
    """§3：逐題自己 10 個窗 count 的 q 分位＋高窗集合。"""
    ok = [w for w in windows_corpus if w["status"] == "ok"]
    n_excl = len(windows_corpus) - len(ok)
    stats["status_excluded_total"] = stats.get("status_excluded_total", 0) + n_excl
    if len(ok) < 3:
        _abort(f"可用窗（status==ok）不足 3 個，無法定 k=3 高窗：實得 {len(ok)}")
    counts = np.array([w["count"] for w in ok], dtype=float)
    thr = float(np.quantile(counts, quantile))
    high = {w["idx"] for w in ok if w["count"] >= thr}
    stats["k_checked"] = stats.get("k_checked", 0) + 1
    if enforce_k3:
        if len(high) != 3:
            _abort(f"高窗數 k != 3 於 q={quantile}（§3 已量此值應恆為 3）："
                   f"實得 {len(high)}（thr={thr}）")
        stats["k3_pass"] = stats.get("k3_pass", 0) + 1
    return thr, high, n_excl


def compute_offsets_checked(model, item_id, full, prompt_text, n_prompt, n_gold, L,
                            tokenizer, stats):
    """§4-B，**v2.22 修正版**（`DESIGN_PROPOSAL_v2.2_rulings.md` v2.22 塊
    ＋`theory_pos_test_mapping_v1.md` §4-B 可見更正塊，第 326-390 行，新斷言0）。
    v2.21 原式假設「to_tokens 一律前置 BOS」，把 per_tok[j] 對到 hf 索引
    `n_prompt+j-1`；v2.22 認定那是把逐模型設定（`default_prepend_bos`：
    Pythia=False／OLMo=True）誤植成全域事實，改為**斷言0：逐模型量出 BOS
    位移 `b`**（`b = n_prompt_tokens - len(HF tokenize(prompt, 無 special
    token))`，斷言 `b∈{0,1}`，原式為 b=1 特例），斷言1/2 與索引全部以 b
    參數化。本函式即該修正版的實作，四個斷言（含新斷言0）逐一對應。

    ⚠ **v2.22 文本未涵蓋的第二個問題（已回報統籌代理，理論部盲覆核已確認判定不用放寬，
    尚待正式收斂進規格文本）**：
    v2.22 §4-B 斷言2字面仍寫 `offsets[n_prompt-b][0]==L`（嚴格相等）。
    實測：套用 b 修正後，Pythia 上這條**仍然不成立**——GPT-NeoX 的 fast
    tokenizer offset_mapping 不含一個詞前導空格的字元（例："Hello world"
    的 world token offset 是 (6,11)，不含第5個字元的空格），OLMo 的
    tokenizer 含（' well' offset 從空格算起）。用 6 個真實 L0 題（01/02/
    03/06/15/19）在 410m 上覆核：s0 一律等於 L+1，0/6 命中嚴格 ==L。這與 BOS
    位移是**兩個獨立的根因**（一個是 token 數量的計數，一個是 offset 字元
    記帳的慣例），不能用同一個 b 修掉。本函式因此把斷言2放寬為兩種合法情形
    之一：`offsets[idx][0]==L`（空格算進 token）或 `offsets[idx][0]==L+1`
    （空格被排除在 offset 外）。

    ⚠ **措辭更正（理論部盲覆核，2026-08-22）**：原版這裡多附了一個
    `full[L]==' '` 條件並稱「兩者都是沒有字元遺失的證據」——**該條件恆真**
    （§4-A 規定 `gold = " " + gold_continuation.strip()`，gold 區間前面必有
    一個空格，是建構出來的事實，不是這裡量到的鑑別力），已從程式碼刪除、
    對 L+1 支的宣稱也已改正：**L+1 支唯一有鑑別力的實質檢查是 `s0==L+1`
    本身**；真正的字元遺失偵測（防止邊界被異常合併、丟字）落在斷言1（長度）
    與下面新加的斷言2b（理論部建議的加固）上，不是那個恆真條件。
    """
    expected = tokenizer_family_id_for(model)
    actual = getattr(tokenizer, "name_or_path", None)
    stats["assert3_checked"] = stats.get("assert3_checked", 0) + 1
    if actual != expected:
        _abort(f"{model}/{item_id}: §4-B 斷言3失敗：用了 tokenizer={actual!r}，"
               f"該模型應為 {expected!r}（不得跨語料家族共用 offsets）")
    stats["assert3_pass"] = stats.get("assert3_pass", 0) + 1

    # 斷言0（v2.22 新增，排最前面，下面兩條都依賴它）：
    # b = n_prompt_tokens - len(HF tokenize(prompt, 無 special token))，斷言 b∈{0,1}。
    hf_prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    bos = n_prompt - hf_prompt_len
    stats["assert0_bos_checked"] = stats.get("assert0_bos_checked", 0) + 1
    if bos not in (0, 1):
        _abort(f"{model}/{item_id}: §4-B 斷言0（v2.22）失敗：b 不是 0 或 1（實得 {bos}；"
               f"n_prompt={n_prompt}，HF 單獨 tokenize(prompt) 長度={hf_prompt_len}）"
               "——超出已知的兩種模式，需要重新檢視位置對應法，不得硬套")
    stats["assert0_bos_pass"] = stats.get("assert0_bos_pass", 0) + 1
    # 工單要求①：逐模型 b 與兩個原始長度一併記錄，讓人能重算（不只留聚合值）。
    detail = stats.setdefault("bos_shift_detail", {}).setdefault(
        model, {"b": bos, "cells": []})
    detail["cells"].append({"item_id": item_id, "n_prompt_tokens": n_prompt,
                            "hf_prompt_len_no_special": hf_prompt_len, "b": bos})
    if detail["b"] != bos:
        _abort(f"{model}: 同一模型量出兩個不同的 b（{detail['b']} vs {bos}，"
               f"觸發題 {item_id}）——b 應該是逐模型常數，不該逐題變動，需要重新檢視")

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    hf_ids, offsets = enc["input_ids"], enc["offset_mapping"]

    stats["assert1_checked"] = stats.get("assert1_checked", 0) + 1
    expected_len = n_prompt - bos + n_gold
    if len(hf_ids) != expected_len:
        _abort(f"{model}/{item_id}: §4-B 斷言1失敗：len(hf_ids)={len(hf_ids)} != "
               f"n_prompt({n_prompt})-b({bos})+n_gold({n_gold})={expected_len}")
    stats["assert1_pass"] = stats.get("assert1_pass", 0) + 1

    hf_index0 = n_prompt - bos                    # per_tok[0] 對應的 hf offsets 索引
    stats["assert2_checked"] = stats.get("assert2_checked", 0) + 1
    s0 = offsets[hf_index0][0]
    space_in_offset = (s0 == L)
    space_excluded = (s0 == L + 1)
    if not (space_in_offset or space_excluded):
        _abort(f"{model}/{item_id}: §4-B 斷言2失敗：offsets[{hf_index0}][0]={s0}，"
               f"既不是 L({L})（空格算進 token）也不是 L+1（空格被排除）")
    stats["assert2_pass"] = stats.get("assert2_pass", 0) + 1
    if space_excluded:
        stats.setdefault("assert2_space_excluded_models", set()).add(model)

    # 斷言2b（理論部覆核建議，2026-08-22，加固）：最後一個 prompt 側 token 的
    # 結尾不得伸進 gold 區間——與空格記帳慣例無關（不看 hf_index0 那個 token，
    # 看它前一個），防的是「prompt 最後一個 token 被異常合併吃掉 gold 開頭字元」
    # 這種斷言1（純長度）測不出來的邊界錯誤。實測 17 題×2 tokenizer 恆成立，
    # 不咬任何現有 cell（純加固，非放寬）。
    stats["assert2b_checked"] = stats.get("assert2b_checked", 0) + 1
    if hf_index0 - 1 >= 0:
        prev_end = offsets[hf_index0 - 1][1]
        if prev_end > L:
            _abort(f"{model}/{item_id}: §4-B 斷言2b失敗：最後一個 prompt token 的結尾"
                   f"({prev_end}) 伸進了 gold 區間（L={L}）——邊界被異常合併")
    stats["assert2b_pass"] = stats.get("assert2b_pass", 0) + 1
    return hf_ids, offsets, bos


def token_window_membership(offsets, n_prompt, n_gold, windows_corpus, bos):
    """§4-D（v2.22 索引參數化版）：gold token j 屬於窗 idx ⟺ 字元跨度有非空交集。
    hf 索引 = n_prompt - b + j（b 逐模型量出，見 compute_offsets_checked 斷言0；
    v2.21 原式 `n_prompt+j-1` 是 b=1 的特例）。"""
    membership: dict[int, set] = {}
    base = n_prompt - bos
    for j in range(n_gold):
        tok_s, tok_e = offsets[base + j]
        for w in windows_corpus:
            if tok_s < w["e"] and w["s"] < tok_e:
                membership.setdefault(w["idx"], set()).add(j)
    return membership


# =========================================================== cell 建造 ==

def build_cell(item, model, rec, item_windows, quantile, enforce_k3, stats, light=False):
    """一個 (item, model) cell 的完整素材。§2 五條資格在這裡逐一過：
    1（id 前綴）在呼叫端過；2（clean.per_tok 存在且等長）在這裡；
    3（title join）由呼叫端決定 item_windows 是否為 None；
    4（窗內外非空）在這裡（obs_main is None ⇒ 不合格）；
    5（語料閘，§3-0）在這裡，前置於窗計算（§3-0：「前置：先過 2b 的語料閘」）。
    """
    clean = rec.get("clean") or {}
    per_tok = clean.get("per_tok")
    n_gold_tokens = rec.get("n_gold_tokens")
    n_prompt_tokens = rec.get("n_prompt_tokens")
    if not per_tok or n_gold_tokens is None or len(per_tok) != n_gold_tokens:
        return {"eligible": False, "reason": "no_clean_per_tok_or_length_mismatch",
                "corpus_gate_pass": None}

    corpus = corpus_of(model)
    verified_in = item.get("verified_in") or []
    corpus_gate_pass = corpus in verified_in
    if not corpus_gate_pass:
        # §3-0 前置：語料閘先於窗計算；不再往下算，cell 直接退場。
        return {"eligible": False, "reason": "corpus_gate_fail",
                "corpus_gate_pass": False, "corpus": corpus}

    windows_corpus = item_windows["windows"][corpus]
    thr, high_idx, n_excl = corpus_threshold(windows_corpus, quantile, enforce_k3, stats)

    tok = get_tokenizer(model)
    full, L = item_windows["full"], item_windows["L"]
    hf_ids, offsets, bos = compute_offsets_checked(model, item["id"], full, item["prompt"],
                                                   n_prompt_tokens, n_gold_tokens, L, tok, stats)
    membership = token_window_membership(offsets, n_prompt_tokens, n_gold_tokens,
                                         windows_corpus, bos)
    first_tokens = set(membership.get(item_windows["first_idx"], set()))
    all_tokens = set(range(n_gold_tokens))
    per_tok_arr = np.asarray(per_tok, dtype=float)

    def delta_for(high_set, exclude_first):
        in_mask = set()
        for i in high_set:
            in_mask |= membership.get(i, set())
        out_mask = all_tokens - in_mask
        if exclude_first:                      # §6-e 閘門1（R2）：整個第一窗排除
            in_mask = in_mask - first_tokens
            out_mask = out_mask - first_tokens
        if not in_mask or not out_mask:
            return None
        return (float(per_tok_arr[sorted(in_mask)].mean())
                - float(per_tok_arr[sorted(out_mask)].mean()))

    obs_main = delta_for(high_idx, False)
    obs_r2 = delta_for(high_idx, True)

    table_main = table_r2 = combos = None
    if not light:
        # §9 附錄8 加速：k=3/10 恆成立 ⇒ 置換虛無等價於在 C(10,3)=120 個遮罩上均勻抽。
        gold_idx_sorted = item_windows["gold_idx_sorted"]
        combos = list(itertools.combinations(gold_idx_sorted, 3))
        if len(combos) != 120:
            _abort(f"{item['id']}: C(10,3) 應為 120，實得 {len(combos)}"
                   "（窗母體不是 10 應已在 build_windows 擋下，這裡是防禦性覆核）")
        table_main = [delta_for(set(c), False) for c in combos]
        table_r2 = [delta_for(set(c), True) for c in combos]

    eligible = obs_main is not None
    return {"eligible": bool(eligible),
            "reason": None if eligible else "window_in_or_out_empty",
            "corpus_gate_pass": True, "corpus": corpus, "thr": thr,
            "high_idx": sorted(high_idx), "n_status_excluded": n_excl,
            "n_gold_tokens": n_gold_tokens,
            "observed_delta": obs_main, "observed_delta_r2": obs_r2,
            "combo_order": combos, "table_main": table_main, "table_r2": table_r2}


# ===================================================== §6/§7：統計機器 ==

def permutation_test(sim, y_obs, n_per_run, union_items, combo_tables, B, rng):
    """§6-c 窗標籤置換（非 sign-flip）＋§6-b studentized 均值統計量（引用
    sim.t_from_A，不新發明統計量）。跨 run 共用抽樣（§6-c 強制）：
    每個 replicate、每一題只抽一次 120-combo 索引，套用到該題所有 run。"""
    R, m = y_obs.shape
    A_obs = y_obs.sum(1)
    Q_obs = (y_obs * y_obs).sum(1)
    T_per_run_obs = sim.t_from_A(A_obs, n_per_run, Q_obs)
    T_obs = float(np.mean(T_per_run_obs))

    if m == 0:
        return {"status": "no_items", "T_obs": None, "p_two_sided": None}
    n_combo = combo_tables[union_items[0]].shape[0]
    if n_combo != 120:
        _abort(f"combo 表不是 120（實得 {n_combo}）——n_pop!=10 應已在更早處 abort")

    combo_idx = rng.integers(0, n_combo, size=(B, m))
    A = np.zeros((B, R)); Q = np.zeros((B, R))
    for j, iid in enumerate(union_items):
        drawn = combo_tables[iid][combo_idx[:, j], :]      # (B, R)，跨 run 共用同一個 j 欄的抽樣
        A += drawn
        Q += drawn * drawn
    T_perm = sim.t_from_A(A, n_per_run[None, :], Q).mean(1)     # (B,)
    cnt = int((np.abs(T_perm) >= abs(T_obs)).sum())
    p = (1 + cnt) / (B + 1)

    mean_per_run = [float(A_obs[r] / n_per_run[r]) if n_per_run[r] else None for r in range(R)]
    sign_per_run = [(None if n_per_run[r] == 0 else bool(A_obs[r] > 0)) for r in range(R)]
    n_pos = sum(1 for s in sign_per_run if s)
    return {"status": "OK", "T_obs": T_obs, "T_per_run_obs": T_per_run_obs.tolist(),
            "n_per_run": n_per_run.tolist(), "union_size": m,
            "p_two_sided": float(p), "permutation_floor_p": 1.0 / (B + 1), "B": int(B),
            "mean_diff_nats_per_run": mean_per_run, "sign_per_run": sign_per_run,
            "n_runs_positive": n_pos, "n_runs_total": R}


def cluster_bootstrap(y_obs, present, union_items, B, rng, conf=0.90):
    """§7：非參數 percentile bootstrap，重抽單位＝題（cluster，整包帶走該題
    所有 run 的 cell）。被檢定量＝聯集逐 cell Δ 的池化平均。"""
    R, m = y_obs.shape
    if m == 0:
        return {"status": "no_items"}
    item_sum = np.zeros(m); item_n = np.zeros(m, dtype=int)
    all_cells = []
    for j in range(m):
        vals = y_obs[:, j][present[:, j]]
        item_sum[j] = vals.sum(); item_n[j] = len(vals)
        all_cells.extend(vals.tolist())
    all_cells = np.array(all_cells)
    total_n = int(item_n.sum())
    pooled_mean = float(all_cells.mean())
    pooled_sd = float(all_cells.std(ddof=1)) if len(all_cells) > 1 else None

    # SD(Δ_題)：§0-2/§7 常態近似可行界用的輔助量，每題先在存在的 run 上取平均，
    # 再對 n 個題目數字取 SD——這不是被判定的量本身（那是上面的 pooled cluster CI）。
    item_means = np.array([item_sum[j] / item_n[j] for j in range(m) if item_n[j] > 0])
    sd_item = float(item_means.std(ddof=1)) if len(item_means) > 1 else None

    idx = rng.integers(0, m, size=(B, m))
    boot_sum = item_sum[idx].sum(1)
    boot_n = item_n[idx].sum(1)
    boot_mean = boot_sum / boot_n
    lo_p, hi_p = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    ci = [float(np.percentile(boot_mean, lo_p)), float(np.percentile(boot_mean, hi_p))]
    return {"status": "OK", "pooled_mean_nats": pooled_mean, "pooled_sd_nats": pooled_sd,
            "n_cells_pooled": total_n, "n_items": int((item_n > 0).sum()),
            "cluster_sizes": item_n.tolist(),
            "sd_item_nats": sd_item, "n_items_for_sd": len(item_means),
            "ci_90": ci, "B": int(B), "conf": conf,
            "endpoints_within_pm0.2_nats_raw_fact": bool(-0.2 < ci[0] and ci[1] < 0.2)}


def descriptive_stats(cells, desc_models, per_model_items):
    """v2.19：DESCRIPTIVE_MODELS（2.8b）不入主裁決，但照實輸出描述統計。"""
    out = {}
    for model in desc_models:
        items = per_model_items.get(model, [])
        vals = np.array([cells[(iid, model)]["observed_delta"] for iid in items
                         if cells[(iid, model)]["observed_delta"] is not None])
        if len(vals) == 0:
            out[model] = {"status": "no_pairs"}
            continue
        out[model] = {"status": "descriptive_only", "n_items": int(len(vals)),
                      "items": sorted(items),
                      "mean_nats": float(vals.mean()),
                      "sd_nats": float(vals.std(ddof=1)) if len(vals) > 1 else None}
    return out


def sign_flip_crosscheck(sim, cells, main_models, per_model_items, rng):
    """§6-e 保留的『只揭露不改判定』交叉檢查：把逐題 Δ 當 diff 餵進未修改的
    causal_analysis.family_test（mapping 文件稱之為 family_permutation，
    程式碼裡的實際名字是 family_test——見 causal_analysis.py:163，回傳的
    mean_diff_nats 與 mapping 文件描述相符）。用途是抓新置換碼的算術錯，
    不是效度檢查（sign-flip 在起手窗混淆下同樣失準，§6-c 已量）。"""
    diffs_by_model = {}
    for model in main_models:
        d = {iid: {"diff": cells[(iid, model)]["observed_delta"]}
             for iid in per_model_items[model]
             if cells[(iid, model)]["observed_delta"] is not None}
        if d:
            diffs_by_model[model] = d
    if not diffs_by_model:
        return {"status": "no_data"}
    return ca.family_test(sim, diffs_by_model, rng)


# ============================================================ 主流程 ==

def analyse(run_id, out_dir, battery_path=None, l0v_path=None, quantile=CORRUPT_QUANTILE,
            enforce_k3=True, light=False):
    """light=True：只算逐 cell 觀察 Δ 與 §2/§3 覆核用的計數（q=0.60/0.90 敏感度
    走這條路，不建 120-combo 表、不跑置換/bootstrap——判定一律以 q=0.75 為準，
    敏感度只揭露穩健性，見 v2.21 §3）。"""
    sim = None if light else ca.load_frozen_machinery()
    runs = ca.load_patch_files(run_id, out_dir)
    battery = load_battery(battery_path)
    l0v = load_l0_verification(l0v_path)
    stats: dict = {}

    all_models = sorted(runs)
    main_models = [m for m in all_models if m not in ca.DESCRIPTIVE_MODELS]
    desc_models = [m for m in all_models if m in ca.DESCRIPTIVE_MODELS]

    windows_cache: dict = {}

    def get_item_windows(item):
        iid = item["id"]
        if iid not in windows_cache:
            entry = l0v.get(item.get("title"))
            windows_cache[iid] = build_windows(item, entry, stats) if entry is not None else None
        return windows_cache[iid]

    cells: dict = {}
    per_model_items = {m: [] for m in all_models}
    corpus_gate_excluded = 0
    for model in all_models:
        for item_id, rec in runs[model]["records"].items():
            if not item_id.startswith("L0-"):
                continue                                    # §2 條件1：只有 L0
            if rec.get("level") != "L0":
                _abort(f"id 前綴為 L0- 但 level != 'L0'：{model}/{item_id}（資料不一致）")
            item = battery.get(item_id)
            if item is None:
                continue
            iw = get_item_windows(item)
            if iw is None:
                continue                                    # §2 條件3（title join）未過，題退場
            cell = build_cell(item, model, rec, iw, quantile, enforce_k3, stats, light=light)
            cells[(item_id, model)] = cell
            if cell.get("corpus_gate_pass") is False:
                corpus_gate_excluded += 1
            if cell.get("eligible"):
                per_model_items[model].append(item_id)

    if corpus_gate_excluded != 0:
        _abort(f"§2 第5條語料閘（§3-0）排除數 = {corpus_gate_excluded} != 0——"
               "題庫或分片已變動，停止並重新凍結（斷言強制）。")

    union_main = (sorted(set().union(*[set(per_model_items[m]) for m in main_models]))
                 if main_models else [])
    per_model_n = {m: len(per_model_items[m]) for m in all_models}

    if light:
        pooled = [cells[(iid, m)]["observed_delta"] for m in main_models
                 for iid in per_model_items[m]
                 if cells[(iid, m)]["observed_delta"] is not None]
        return {"run_id": run_id, "quantile": quantile, "light_mode": True,
                "union_n": len(union_main), "per_model_n": per_model_n,
                "pooled_mean_nats": float(np.mean(pooled)) if pooled else None,
                "n_cells_pooled": len(pooled), "assertion_stats": stats}

    R = len(main_models)
    idx = {iid: j for j, iid in enumerate(union_main)}
    y_main = np.zeros((R, len(union_main))); y_r2 = np.zeros((R, len(union_main)))
    present = np.zeros((R, len(union_main)), dtype=bool)
    combo_tables_main = {iid: np.zeros((120, R)) for iid in union_main}
    combo_tables_r2 = {iid: np.zeros((120, R)) for iid in union_main}
    for r, model in enumerate(main_models):
        for iid in per_model_items[model]:
            c = cells[(iid, model)]; j = idx[iid]
            y_main[r, j] = c["observed_delta"]; y_r2[r, j] = c["observed_delta_r2"]
            present[r, j] = True
            combo_tables_main[iid][:, r] = [v if v is not None else 0.0 for v in c["table_main"]]
            combo_tables_r2[iid][:, r] = [v if v is not None else 0.0 for v in c["table_r2"]]
    n_per_run = present.sum(1).astype(float)

    ss = np.random.SeedSequence(ca.SEED)
    c_main, c_r2, c_boot, c_desc = ss.spawn(4)
    perm_main = permutation_test(sim, y_main, n_per_run, union_main, combo_tables_main,
                                 ca.B_PERM, np.random.default_rng(c_main))
    perm_r2 = permutation_test(sim, y_r2, n_per_run, union_main, combo_tables_r2,
                               ca.B_PERM, np.random.default_rng(c_r2))
    boot = cluster_bootstrap(y_main, present, union_main, ca.B_PERM, np.random.default_rng(c_boot))
    desc = descriptive_stats(cells, desc_models, per_model_items)
    crosscheck = sign_flip_crosscheck(sim, cells, main_models, per_model_items,
                                      np.random.default_rng(c_desc))

    # ⚠ 兩種「交集」不是同一個量，分開報：§7-a「4/4」＝四個 fp32 主臂模型的交集
    # （不含 2.8b，凍結期望值 13）；§2「交集 n=12」＝**五個模型**（含 2.8b）的交集，
    # 是 v1.1 §2 明寫的「若改採交集且五模型齊全」情境，不是主臂交集。兩個都算，
    # 避免拿其中一個去對錯期望值（我自己第一版跑出來就對錯過，見報告更正）。
    intersection_main4 = (sorted(set.intersection(*[set(per_model_items[m]) for m in main_models]))
                          if main_models else [])
    intersection_all5 = (sorted(set.intersection(*[set(per_model_items[m]) for m in all_models]))
                         if all_models else [])
    run_count_per_item = {iid: int(present[:, idx[iid]].sum()) for iid in union_main}
    hist: dict = {}
    for v in run_count_per_item.values():
        hist[str(v)] = hist.get(str(v), 0) + 1
    single_witness = sorted(iid for iid, v in run_count_per_item.items() if v == 1)
    per_model_n_match = {m: per_model_n.get(m) == EXPECTED_PER_MODEL_N.get(m)
                         for m in EXPECTED_PER_MODEL_N if m in per_model_n}

    bos_shift_summary = {m: d["b"] for m, d in stats.get("bos_shift_detail", {}).items()}
    return {
        "run_id": run_id, "quantile": quantile, "light_mode": False,
        "spec_version": "v2.21+v2.22",
        "spec_sources": [
            "planning/DESIGN_PROPOSAL_v2.2_rulings.md v2.21（第398-594行）"
            "＋v2.22（第595-614行，BOS全域化誤植之量出型修正）",
            "planning/office_reports/theory_pos_test_mapping_v1.md"
            "（v1.1 定稿＋§4-B v2.22可見更正塊第326-390行，新斷言0）",
        ],
        # 工單要求①：逐模型 b（BOS 位移），完整原始長度見 assertion_stats.bos_shift_detail
        "bos_shift_by_model": bos_shift_summary,
        "models_loaded": {m: runs[m]["dtype"] for m in all_models},
        "main_models": main_models, "descriptive_models": desc_models,
        "corpus_gate_excluded_cells": corpus_gate_excluded,
        "union_n17_check": {"union_size": len(union_main), "expected": 17,
                            "match": len(union_main) == 17,
                            "intersection_4main_size": len(intersection_main4),
                            "expected_intersection_4main": 13,
                            "intersection_4main_match": len(intersection_main4) == 13,
                            "intersection_all5_size": len(intersection_all5),
                            "expected_intersection_all5": 12,
                            "intersection_all5_match": len(intersection_all5) == 12,
                            "note": "4main＝§7-a的4/4；all5＝§2『交集 n=12（五模型齊全）』，兩者不是同一個量"},
        "per_model_n": per_model_n, "per_model_n_expected": EXPECTED_PER_MODEL_N,
        "per_model_n_match": per_model_n_match,
        "run_count_per_item_hist": hist, "expected_run_count_hist": EXPECTED_RUN_COUNT_HIST,
        "run_count_hist_match": hist == EXPECTED_RUN_COUNT_HIST,
        "single_witness_items": single_witness,
        "fp32_main_total_cells": int(present.sum()), "expected_fp32_main_total_cells": 59,
        "assertion_stats": stats,
        "main_test": perm_main, "r2_gate_test": perm_r2,
        "equivalence_bootstrap": boot,
        "descriptive_other_models": desc,
        "sign_flip_crosscheck_disclosure_only": crosscheck,
        "spec_feasibility_bounds": {"sd_threshold_n17_nats": 0.501,
                                    "sd_threshold_n12_nats": 0.421,
                                    "note": "§0-2/§7 預先登記界線，供 SD 對照，非本檔判定"},
    }


# ======================================================= 自測（合成資料） ==
# 測行為，不是測「跑完了」——工單四項：①注入已知窗內效應→應點燃
# ②零效應→FPR≈α ③三個 §4 斷言各違反一次→必須 abort ④§2 語料閘排除數
# 非0→必須停下要求重凍。全部只用合成資料，不碰 results/causal 下任何真檔。

_SYNTH_VOCAB = ["orange", "river", "garden", "window", "silver", "moment", "distant",
                "quiet", "carried", "opened", "walked", "turned", "answered",
                "remembered", "corner", "shadow", "evening", "harbor", "pattern",
                "journey", "ancient", "hollow", "bridge", "meadow", "canyon", "ribbon",
                "compass", "lantern", "current", "willow", "ember", "granite", "ripple",
                "thicket", "anchor", "echo", "cellar", "orchard", "quarry", "valley",
                "copper", "linen", "marble"]
_SYNTH_PROMPT = ("In the early days before the war, when everything still felt "
                 "possible, she often said that")


def _make_gold_text(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(_SYNTH_VOCAB) for _ in range(n_words))


def _synth_item(item_id, seed):
    # 43 個字＝probes_from(n_words=7,stride=4) 恰好產生 10 個窗（(43-7)/4+1=10），
    # 與真資料「已量：每題恰好 10 個」的結構一致（見 §3/§4-E）。
    gold = _make_gold_text(43, seed)
    return {"id": item_id, "level": "L0", "title": f"Synthetic {item_id} seed{seed}",
            "prompt": _SYNTH_PROMPT, "gold_continuation": gold,
            "verified_in": ["pile", "olmo_mix"]}


def _synth_l0v_entry(item, high_positions, low=10, high=1000):
    gold_probes = vb.probes_from(item["gold_continuation"], n_words=7, stride=4)
    if len(gold_probes) != 10:
        raise RuntimeError(f"自測合成詞彙沒有產生 10 個窗（實得 {len(gold_probes)}）；"
                           "這是自測腳手架的錯，不是待測程式的錯")
    joint = vb._edge_trim(item["prompt"].split()[-4:] + item["gold_continuation"].split()[:4])
    boundary = " ".join(joint) if joint else gold_probes[-1]
    texts = gold_probes + [boundary]

    def one_corpus():
        arr = [{"probe": t, "status": "ok", "count": (high if i in high_positions else low)}
               for i, t in enumerate(texts[:-1])]
        arr.append({"probe": texts[-1], "status": "ok", "count": low})   # 跨界窗，不進窗母體
        return arr

    return {"title": item["title"], "mode": "memorized",
            "probes": {"pile": one_corpus(), "olmo_mix": one_corpus()},
            "verified": {"pile": True, "olmo_mix": True}}


def _synth_record(item, model, tokenizer, high_positions, effect_nats, rng, stats):
    """造一筆與真檔鍵名相同的 record（只含 build_cell 需要的欄位）。

    n_prompt_tokens 的 BOS 位移用 **自測腳手架自己的選擇**（不是規格）反映
    2026-08-22 實測到的真實家族差異：Pythia 前綴的假模型名一律當 bos=0，
    OLMo 當 bos=1（見 compute_offsets_checked 的docstring 與工作紀錄裡的證據：
    `default_prepend_bos` Pythia=False／OLMo=True，L0-01 雙重驗證過）。
    這樣自測才能同時操練 bos=0 與 bos=1 兩條路徑，而不是重複假設同一種。"""
    full, L = text_layout(item)
    bos = 0 if model.startswith(ca.PYTHIA_PREFIX) else 1
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    hf_prompt_len = len(tokenizer(item["prompt"], add_special_tokens=False)["input_ids"])
    n_prompt = hf_prompt_len + bos
    n_gold = len(offs) - hf_prompt_len

    gold_probes = vb.probes_from(item["gold_continuation"], n_words=7, stride=4)
    spans = locate_probes(full, L, gold_probes, stats)
    high_spans = [(w["s"], w["e"]) for w in spans if w["idx"] in high_positions]

    per_tok = rng.normal(loc=-3.0, scale=0.4, size=n_gold)
    for j in range(n_gold):
        s, e = offs[hf_prompt_len + j]
        if any(s < he and hs < e for hs, he in high_spans):
            per_tok[j] += effect_nats
    return {"id": item["id"], "level": "L0", "model": model,
            "n_prompt_tokens": n_prompt, "n_gold_tokens": n_gold,
            "clean": {"per_tok": per_tok.tolist(), "n_gold_tokens": n_gold}}


def _write_synth_run(tmp: Path, run_id, models, item_ids, high_positions, effect_nats, seed):
    battery_items, l0v_entries = [], []
    per_model_records = {m: {} for m in models}
    stats: dict = {}
    for k, iid in enumerate(item_ids):
        item = _synth_item(iid, seed + k)
        battery_items.append(item)
        l0v_entries.append(_synth_l0v_entry(item, high_positions))
        for model in models:
            tok = get_tokenizer(model)
            rng = np.random.default_rng(seed * 10_000 + k * 100 + hash(model) % 97)
            per_model_records[model][iid] = _synth_record(
                item, model, tok, high_positions, effect_nats, rng, stats)
    bpath = tmp / "battery.json"; lpath = tmp / "l0_verification.json"
    json.dump({"items": battery_items}, open(bpath, "w", encoding="utf-8"))
    json.dump(l0v_entries, open(lpath, "w", encoding="utf-8"))
    for model in models:
        safe = model.replace("/", "_")
        json.dump({"model": model, "dtype": "fp32", "records": per_model_records[model]},
                  open(tmp / f"{run_id}_patch_{safe}.json", "w", encoding="utf-8"))
    return bpath, lpath


def self_test():
    import tempfile
    ok = True
    summary: dict = {}
    models = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
             "EleutherAI/pythia-1.4b", "allenai/OLMo-2-0425-1B"]
    item_ids = [f"L0-{i:02d}" for i in range(1, 7)]      # 6 題，夠撐 union/置換邏輯又跑得快
    high_positions = {3, 4, 5}                            # 窗母體 index 3/4/5 當高窗（避開起手窗 idx0）

    print("=== 自測 (1)：注入已知窗內效應 -> 主檢定應點燃 ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bpath, lpath = _write_synth_run(tmp, "T", models, item_ids, high_positions,
                                        effect_nats=1.8, seed=1)
        res = analyse("T", tmp, battery_path=bpath, l0v_path=lpath, quantile=0.75)
        mt = res["main_test"]
        fires = (mt["p_two_sided"] < ca.ALPHA and mt["n_runs_positive"] == mt["n_runs_total"])
        ok &= fires
        summary["fires_on_injected_effect"] = bool(fires)
        summary["injected_effect_p"] = mt["p_two_sided"]
        print(f"    p={mt['p_two_sided']:.6f}  n_pos={mt['n_runs_positive']}/{mt['n_runs_total']}"
             f"  {'OK' if fires else 'FAIL'}")

    print("=== 自測 (2)：零效應 -> 虛無點燃率應 ≈ α（40 次試驗，B 降到 2000 加速） ===")
    B_keep = ca.B_PERM
    fired, N = 0, 40
    for k in range(N):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bpath, lpath = _write_synth_run(tmp, "T", models, item_ids, high_positions,
                                            effect_nats=0.0, seed=200 + k)
            ca.B_PERM = 2000
            res = analyse("T", tmp, battery_path=bpath, l0v_path=lpath, quantile=0.75)
            ca.B_PERM = B_keep
            fired += int(res["main_test"]["p_two_sided"] < ca.ALPHA)
    rate = fired / N
    calib_ok = 0.0 <= rate <= 0.20     # 仿 causal_analysis.py self_test 的寬容帶（n=40,alpha=.05）
    ok &= calib_ok
    summary["null_fpr"] = rate; summary["null_fpr_trials"] = N
    print(f"    虛無點燃率 = {rate:.3f} ({fired}/{N})  {'OK' if calib_ok else 'OUT OF RANGE'}")

    print("=== 自測 (3)：三個 §4 斷言各自違反一次 -> 必須 abort ===")
    item = _synth_item("L0-01", seed=5)
    tok = get_tokenizer(models[0])
    full, L = text_layout(item)
    rng = np.random.default_rng(5)
    stats3: dict = {}
    rec = _synth_record(item, models[0], tok, high_positions, 0.0, rng, stats3)
    try:
        compute_offsets_checked(models[0], "L0-01", full, item["prompt"], rec["n_prompt_tokens"],
                                rec["n_gold_tokens"] + 1, L, tok, {})
        a1 = False
    except SystemExit:
        a1 = True
    try:
        # 注意：不能像斷言1一樣改 n_prompt——n_prompt 錯 1 會被「量出 BOS 位移」
        # 的機制自動吸收（bos 改算成 1，算式剛好還是對得上，見程式內文檔字串），
        # 那不是斷言2失效，是它被繞過去而已。改 L（gold 起點的期望字元位置）
        # 才是真正只破壞斷言2、不動 n_prompt/BOS 判定的做法。
        compute_offsets_checked(models[0], "L0-01", full, item["prompt"],
                                rec["n_prompt_tokens"], rec["n_gold_tokens"], L + 5, tok, {})
        a2 = False
    except SystemExit:
        a2 = True
    wrong_tok = get_tokenizer("allenai/OLMo-2-0425-1B")
    try:
        compute_offsets_checked(models[0], "L0-01", full, item["prompt"], rec["n_prompt_tokens"],
                                rec["n_gold_tokens"], L, wrong_tok, {})
        a3 = False
    except SystemExit:
        a3 = True
    ok &= (a1 and a2 and a3)
    summary["assert1_abort_ok"] = a1; summary["assert2_abort_ok"] = a2
    summary["assert3_abort_ok"] = a3
    print(f"    斷言1(len)違反->abort: {'OK' if a1 else 'FAIL'}   "
         f"斷言2(offset)違反->abort: {'OK' if a2 else 'FAIL'}   "
         f"斷言3(tokenizer)違反->abort: {'OK' if a3 else 'FAIL'}")

    print("=== 自測 (4)：§2 語料閘（§3-0）排除數非 0 -> 必須 abort 並要求重凍 ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bpath, lpath = _write_synth_run(tmp, "T", models, item_ids, high_positions,
                                        effect_nats=0.5, seed=9)
        battery = json.load(open(bpath, encoding="utf-8"))
        battery["items"][0]["verified_in"] = ["olmo_mix"]     # 拔掉 pile，故意破壞語料閘
        json.dump(battery, open(bpath, "w", encoding="utf-8"))
        try:
            analyse("T", tmp, battery_path=bpath, l0v_path=lpath, quantile=0.75)
            gate_abort, msg = False, ""
        except SystemExit as e:
            msg = str(e)
            gate_abort = ("重凍" in msg) or ("重新凍結" in msg)
    ok &= gate_abort
    summary["corpus_gate_abort_ok"] = gate_abort
    print(f"    語料閘排除非0 -> abort 且訊息含重凍要求: {'OK' if gate_abort else 'FAIL'}"
         f"{'' if gate_abort else '  msg=' + msg}")

    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    summary["overall"] = "PASS" if ok else "FAIL"
    return (0 if ok else 1), summary


# =============================================================== 輸出 ==

def _json_safe(o):
    if isinstance(o, np.floating):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def write_report(path: Path, out: dict, run_id: str):
    p = out.get("primary_q0.75") or {}
    mt = p.get("main_test", {})
    r2 = p.get("r2_gate_test", {})
    boot = p.get("equivalence_bootstrap", {})
    lines = []
    a = lines.append
    a(f"# §7.7 位置檢定執行報告（v2.21+v2.22，run_id={run_id}）\n")
    a("只備料，不判定——本檔不寫「通過/不通過」，最終判定由統籌代理按凍結文字執行。\n")
    a(f"規格版本：{p.get('spec_version')}（`DESIGN_PROPOSAL_v2.2_rulings.md` v2.21第398-594行"
     f"＋v2.22第595-614行）＋`theory_pos_test_mapping_v1.md`（v1.1定稿＋§4-B v2.22"
     f"可見更正塊第326-390行）。\n")

    a("## BOS 位移 b（v2.22 新增斷言0，逐模型量出；工單要求①）\n")
    a(f"- {p.get('bos_shift_by_model')}")
    a("- 完整原始長度（逐 cell 的 n_prompt_tokens／HF 單獨 tokenize(prompt) 長度）見"
     " `results/causal/pos_test_v221.json` 的 `assertion_stats.bos_shift_detail`"
     "（不重複貼進本報告，避免這裡被讀成聚合摘要卻其實是逐 cell 原始資料）。\n")

    a("## 題集與逐模型 n（§2 交叉核對）\n")
    u = p.get('union_n17_check', {})
    a(f"- 聯集 n = {u.get('union_size')}（預期 17，match={u.get('match')}）")
    a(f"- 交集 n（四個 fp32 主臂模型，＝§7-a 的 4/4）= {u.get('intersection_4main_size')}"
     f"（預期 13，match={u.get('intersection_4main_match')}）")
    a(f"- 交集 n（五模型齊全，＝§2『交集 n=12』）= {u.get('intersection_all5_size')}"
     f"（預期 12，match={u.get('intersection_all5_match')}）")
    a(f"- fp32 主臂總 cell = {p.get('fp32_main_total_cells')}（預期 59）")
    a(f"- 逐模型 n：{p.get('per_model_n')}")
    a(f"- 逐模型 n 與已量表相符：{p.get('per_model_n_match')}")
    a(f"- §2 語料閘（§3-0）排除 cell 數 = {p.get('corpus_gate_excluded_cells')}（凍結要求必須為 0）\n")

    a("## §7-a 逐題 run 數表（cluster 大小複核）\n")
    a(f"- 分布（run 數 -> 題數）：{p.get('run_count_per_item_hist')}"
     f"（預期 {EXPECTED_RUN_COUNT_HIST}，match={p.get('run_count_hist_match')}）")
    a(f"- 單證人題：{p.get('single_witness_items')}\n")

    a("## 主檢定（§6，窗標籤置換，非 sign-flip）\n")
    a(f"- T_obs = {mt.get('T_obs')}，p(two-sided, 置換) = {mt.get('p_two_sided')}"
     f"（置換地板 = {mt.get('permutation_floor_p')}，B={mt.get('B')}）")
    a(f"- 逐 run n = {mt.get('n_per_run')}，逐 run Δ 均值 = {mt.get('mean_diff_nats_per_run')}")
    a(f"- 逐 run 符號 = {mt.get('sign_per_run')}（正 {mt.get('n_runs_positive')}/"
     f"{mt.get('n_runs_total')}）\n")

    a("## §6-e 閘門1／R2（整個第一窗排除版，(i) 的必要條件）\n")
    a(f"- T_obs = {r2.get('T_obs')}，p(two-sided) = {r2.get('p_two_sided')}")
    a(f"- 逐 run Δ 均值 = {r2.get('mean_diff_nats_per_run')}")
    a(f"- 逐 run 符號 = {r2.get('sign_per_run')}（正 {r2.get('n_runs_positive')}/"
     f"{r2.get('n_runs_total')}）\n")

    a("## 等價 CI（§7，cluster bootstrap）\n")
    a(f"- 池化 Δ 均值 = {boot.get('pooled_mean_nats')} nats，池化 SD = {boot.get('pooled_sd_nats')}")
    a(f"- 90% CI = {boot.get('ci_90')}（B={boot.get('B')}）")
    a(f"- SD(Δ_題)（常態近似輔助量，非判定量）= {boot.get('sd_item_nats')}"
     f"（n={boot.get('n_items_for_sd')}；§0-2 界線：n=17 時 0.501 nats）")
    a(f"- cluster 大小（逐題）= {boot.get('cluster_sizes')}")
    a(f"- 端點是否落在 (-0.2,+0.2)（原始事實，非判定）= "
     f"{boot.get('endpoints_within_pm0.2_nats_raw_fact')}\n")

    desc = p.get("descriptive_other_models", {})
    a("## 描述性（v2.19，2.8b 等退出主裁決者）\n")
    for m, d in desc.items():
        a(f"- {m}: {d}")
    a("")

    cc = p.get("sign_flip_crosscheck_disclosure_only", {})
    a("## sign-flip 交叉檢查（只揭露，不改判定，§6-e）\n")
    a(f"- p_two_sided = {cc.get('p_two_sided')}，T_studentized_mean = {cc.get('T_studentized_mean')}"
     f"，mean_diff_nats = {cc.get('mean_diff_nats')}\n")

    a("## 斷言狀態（全部 checked/pass 應相等；不等代表中途已 abort，不會有此報告）\n")
    astats = dict(p.get("assertion_stats") or {})
    astats.pop("bos_shift_detail", None)     # 逐 cell 原始資料，已獨立成節＋留在 JSON，這裡只留聚合計數
    a(f"- {astats}\n")

    a("## 敏感度（§3，q=0.60/0.90，只揭露穩健性，未跑置換/bootstrap）\n")
    for k, v in (out.get("sensitivity") or {}).items():
        a(f"- {k}: union_n={v.get('union_n')}, per_model_n={v.get('per_model_n')}, "
         f"pooled_mean_nats={v.get('pooled_mean_nats')}, n_cells_pooled={v.get('n_cells_pooled')}")
    a("")

    a("## 附註1：v2.22 之外仍待收斂的問題（已回報統籌代理，理論部盲覆核已判定不用放寬，"
     "但規格文本尚未正式收斂本條）\n")
    exm = sorted(astats.get("assert2_space_excluded_models") or [])
    a(f"v2.22 §4-B 斷言2字面仍寫 `offsets[n_prompt-b][0]==L`（嚴格相等）。套用 b 修正後，"
     f"這條在 Pythia 上實測仍不成立——GPT-NeoX fast tokenizer 的 offset_mapping 不含一個詞"
     f"前導空格的字元，OLMo 的含。本次真跑：斷言2放寬版（`==L` 或 `==L+1`）下 "
     f"checked=={astats.get('assert2_checked')}、pass=={astats.get('assert2_pass')}"
     f"（全過，無 abort）；命中「空格被排除」分支的模型＝{exm}（四個 Pythia，OLMo 不在內，"
     f"與斷言0的 b 家族分布一致）。這是跟 BOS 位移**獨立**的第二個根因，v2.22 文本尚未涵蓋，"
     f"本檔已用放寬版斷言頂著，不等同於已收斂——留給理論部/統籌代理決定要不要另立 v2.23。\n")
    a(f"**理論部盲覆核附帶更正**：放寬版原本多附一個 `full[L]==' '` 條件，"
     f"理論部量出那條件恆真（§4-A 建構規則保證 gold 前必有空格），已從程式碼與措辭移除，"
     f"L+1 支現在只認 `s0==L+1` 本身。**新增斷言2b**（理論部建議的加固，與空格慣例無關）："
     f"`offsets[n_prompt-b-1][1] <= L`（最後一個 prompt token 不得伸進 gold 區間）——"
     f"checked=={astats.get('assert2b_checked')}、pass=={astats.get('assert2b_pass')}"
     f"（全過，不咬任何現有 cell，純加固）。\n")

    a("## 附註2：build_battery_v2.py 特殊 token 設定不對稱查核（統籌代理交辦的旁查）\n")
    a("**一行結論：不對稱不影響任何實際數字**——`AutoTokenizer.encode()` 在本專案用到的"
     "五個模型（pythia-410m/1b/1.4b/2.8b、OLMo-2-0425-1B）上，預設呼叫（不傳 "
     "`add_special_tokens`）與顯式 `add_special_tokens=False` 產出完全相同的 token id 序列"
     "——也就是說這五個 tokenizer 的 `.encode()` 預設本來就不會插入任何特殊 token，"
     "`build_battery_v2.py:147` 少寫的那個參數是無操作的不對稱，不是數值錯誤。\n")
    a("證據（直接對五個模型逐一實測，非讀 battery.json 現成欄位）：對同一段測試文字，"
     "`t.encode(text)` 與 `t.encode(text, add_special_tokens=False)` 在五個模型上 token id "
     "序列逐一相等（`default_len==no_special_len` 且逐 id 相同，五個模型全數為真）。"
     "本檔（pos_test_v221.py）自己從不依賴 `.encode()` 的預設行為——所有 tokenize 呼叫"
     "全部顯式傳 `add_special_tokens=False`，故本檔的 b／offsets 推算不受這個不對稱影響，"
     "與理論部在 §4-B 更正塊裡的判斷一致。\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir", default=str(PROJ / "results" / "causal"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        code, summary = self_test()
        print("\nself-test summary:", json.dumps(summary, ensure_ascii=False))
        raise SystemExit(code)
    if not a.run_id:
        raise SystemExit("--run-id is required (no default; refuse to guess the run)")

    out_dir = Path(a.out_dir)
    primary = analyse(a.run_id, out_dir, quantile=0.75, enforce_k3=True, light=False)
    sensitivity = {}
    for q in (0.60, 0.90):
        r = analyse(a.run_id, out_dir, quantile=q, enforce_k3=False, light=True)
        sensitivity[f"q{q}"] = r

    out = {"primary_q0.75": primary, "sensitivity": sensitivity}
    out_path = out_dir / "pos_test_v221.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
             default=_json_safe, allow_nan=False)
    report_path = PROJ / "planning" / "rig_pos_test_report.md"
    write_report(report_path, out, a.run_id)
    print("\n->", out_path)
    print("->", report_path)


if __name__ == "__main__":
    main()
