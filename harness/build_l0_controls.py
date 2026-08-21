"""L0N 對照組建構：冷門公版書深處取段，與 L0 同構（88 詞、44 切分）但語料近乎查無。

科學目的：L0（背過的名段）vs L0N（沒背過的同文體段落）——長度、文體、年代、
題材全部匹配，唯一差異是「在不在訓練語料裡」。L0 的內部效應必須贏過 L0N
才能歸因於記憶，而不是歸因於「古典長文」。

用法： .venv/Scripts/python.exe harness/build_l0_controls.py
輸出： battery/l0_control_candidates.json（verify_battery.py --mode control 的輸入）
"""
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from build_l0_gutenberg import fetch_book, shift_split  # noqa: E402  (含 normalize、退避與鏡像)

OUT = PROJ / "battery" / "l0_control_candidates.json"

# 高編號 = 近年才收錄的冷門作品；部分會 404 或非英文，抓得到幾本算幾本
CANDIDATE_IDS = [62012, 62345, 63111, 63888, 64502, 65123, 65789, 66234,
                 66890, 67345, 67901, 68456, 69012, 69567, 70123, 70678,
                 71234, 71889]
TOTAL_WORDS = 88
SPLIT_AT = 44
WORD_OFFSET = 1500  # 從書的深處取，避開開頭（開頭最常被引用）
TARGET_N = 12


def looks_english_prose(words) -> bool:
    text = " " + " ".join(w.lower() for w in words) + " "
    stop_hits = sum(text.count(f" {w} ") for w in
                    ("the", "and", "of", "to", "a", "in", "was", "he", "she", "it"))
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return stop_hits >= 8 and caps <= len(words) * 0.08


def main():
    cache: dict = {}
    candidates = []
    for gid in CANDIDATE_IDS:
        if len(candidates) >= TARGET_N:
            break
        try:
            text = fetch_book(gid, cache)
        except Exception as e:
            print(f"BOOK {gid} failed: {e!r}")
            continue
        body_start = text.find("*** START OF")
        if body_start >= 0:
            nl = text.find("***", body_start + 12)
            text = text[nl + 3:] if nl >= 0 else text[body_start:]
        words = text.split()
        if len(words) < WORD_OFFSET + TOTAL_WORDS + 500:
            print(f"BOOK {gid}: too short, skip")
            continue
        # 從 offset 起找第一個以句號結尾之詞的下一位置，讓段落從句首開始
        start = WORD_OFFSET
        for j in range(WORD_OFFSET, min(WORD_OFFSET + 200, len(words) - TOTAL_WORDS)):
            if words[j].endswith((".", "!", "?")):
                start = j + 1
                break
        span = words[start:start + TOTAL_WORDS + 6]
        if not looks_english_prose(span[:TOTAL_WORDS]):
            print(f"BOOK {gid}: not english prose at offset, skip")
            continue
        # 與 L0 配平：邊界必須落在實詞（事實查核抓到的未配平差異——
        # L0 有 shift_split、對照組沒有，會讓 L0N 首 token 更受句法決定）
        split = shift_split(span, base=SPLIT_AT)
        candidates.append({
            "title": f"control gutenberg:{gid} @w{start}",
            "source_work": f"gutenberg:{gid}",
            "year": 0,
            "prompt_text": " ".join(span[:split]),
            "continuation_text": " ".join(span[split:TOTAL_WORDS + (split - SPLIT_AT)]),
            "split_at": split,
            "boundary_word": span[split],
            "why_famous": "OBSCURE control passage (expected near-zero corpus count)",
        })
        print(f"OK   gutenberg:{gid} @w{start}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"l0_candidates": candidates}, f, ensure_ascii=False, indent=2)
    print(f"done: {len(candidates)} control candidates -> {OUT}")
    # 需要 re.escape 之類時再引入；保留 import 對齊風格
    _ = re


if __name__ == "__main__":
    main()
