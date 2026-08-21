"""L0 素材建構：從 Project Gutenberg 抓公版原文、機械切分成 prompt/continuation。

設計原因：叫 LLM 逐字回憶名段會 (a) 記錯字、(b) 觸發訓練資料萃取類安全機制。
改為程式抓取 → 字句 100% 依 Gutenberg 版本 → 之後由 verify_battery.py 用
infini-gram 判定各語料是否真的大量收錄（記憶 ground truth）。

用法： .venv/Scripts/python.exe harness/build_l0_gutenberg.py
輸出： battery/l0_candidates.json（verify_battery.py 的輸入格式）
"""
import json
import re
import time
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "battery" / "l0_candidates.json"
UA = {"User-Agent": "OMTR-research/0.1 (public-domain corpus builder)"}

# (gutenberg_id, title, start_phrase, year)
# QC v2 調整：撤詩體（Sonnet 18、The Raven——換行被壓平就不是模型背過的形式）、
# 撤 Declaration opening（其 gold 與 self-evident 段的 prompt 逐字重疊）、
# 補 5 段高重複散文候選以擴大 Pythia/OLMo 皆可用的交集。
PASSAGES = [
    (1342, "Pride and Prejudice opening", "It is a truth universally acknowledged", 1813),
    (2701, "Moby-Dick opening", "Call me Ishmael", 1851),
    (98,   "A Tale of Two Cities opening", "It was the best of times", 1859),
    (11,   "Alice in Wonderland opening", "Alice was beginning to get very tired", 1865),
    (10,   "KJV Genesis 1", "In the beginning God created the heaven and the earth", 1611),
    (10,   "KJV Psalm 23", "The LORD is my shepherd; I shall not want", 1611),
    (10,   "KJV John 3:16", "For God so loved the world, that he gave", 1611),
    (10,   "KJV John 1:1", "In the beginning was the Word, and the Word was with God", 1611),
    (10,   "KJV 1 Corinthians 13", "Though I speak with the tongues of men and of angels", 1611),
    (10,   "KJV Beatitudes", "Blessed are the poor in spirit", 1611),
    (1,    "Declaration self-evident truths", "We hold these truths to be self-evident", 1776),
    (5,    "US Constitution preamble", "We the People of the United States", 1787),
    (4,    "Gettysburg Address", "Four score and seven years ago", 1863),
    (100,  "Hamlet soliloquy", "To be, or not to be", 1603),
    (100,  "Julius Caesar funeral speech", "Friends, Romans, countrymen, lend me your ears", 1599),
    (76,   "Huckleberry Finn opening", "You don't know about me without you have read", 1884),
    (84,   "Frankenstein creation scene", "It was on a dreary night of November", 1818),
    (1661, "A Scandal in Bohemia opening", "To Sherlock Holmes she is always", 1891),
    (1080, "A Modest Proposal opening", "It is a melancholy object to those who walk", 1729),
    (120,  "Treasure Island opening", "Squire Trelawney, Dr. Livesey, and the rest of these gentlemen", 1883),
    (36,   "War of the Worlds opening", "No one would have believed in the last years of the nineteenth century", 1898),
    (16,   "Peter Pan opening", "All children, except one, grow up", 1911),
]

TOTAL_WORDS = 88
SPLIT_AT = 44
# 邊界詞若是語法必然詞（is/the/of...），下一 token 靠句法就能猜，量不到記憶——
# 把切分點往後挪到第一個實詞（QC issue: L0-syntax-forced-boundaries）
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "he", "she", "they", "we", "you", "i", "his", "her", "its", "their",
    "that", "this", "which", "who", "whom", "had", "has", "have", "will",
    "shall", "would", "should", "may", "might", "not", "no", "so", "than",
}


def shift_split(words, base=SPLIT_AT, max_shift=6):
    """回傳切分點：讓 continuation 首詞是實詞。挪不動就用 base。"""
    for s in range(base, min(base + max_shift, len(words) - 20)):
        w = words[s].strip(".,;:!?\"'()[]—–-").lower()
        if w and w not in FUNCTION_WORDS and not w.isdigit():
            return s
    return base


def normalize(text: str) -> str:
    """QC 修正：保留破折號與其他標點（洗掉會讓 gold 偏離模型背過的原文）。
    只做：直引號化、去底線（Gutenberg 斜體記號）、KJV 行內節號剝除、空白摺疊。"""
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("_", ""))
    text = re.sub(r"\b\d+:\d+\b", " ", text)  # KJV 行內節號（1:1 等）
    return re.sub(r"\s+", " ", text)


MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{gid}/pg{gid}.txt",
    "https://aleph.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
]


def fetch_book(gid: int, cache: dict) -> str:
    """指數退避 + 鏡像後備；Gutenberg 主站對連續抓取會間歇性 503/504。"""
    if gid in cache:
        return cache[gid]
    last_err: Exception = RuntimeError("no attempt")
    for round_i in range(4):
        for url_t in MIRRORS:
            try:
                req = urllib.request.Request(url_t.format(gid=gid), headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                cache[gid] = normalize(raw)
                time.sleep(3.0)  # 禮貌速率
                return cache[gid]
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        time.sleep(4.0 * (round_i + 1))
    raise last_err


def main():
    cache: dict = {}
    candidates, failures = [], []
    # 先抓不重複書目，再統一抽段（避免同書多段時重複觸發節流）
    for gid in dict.fromkeys(gid for gid, *_ in PASSAGES):
        try:
            fetch_book(gid, cache)
            print(f"BOOK {gid} fetched ({len(cache[gid])} chars)")
        except Exception as e:
            print(f"BOOK {gid} failed: {e!r}")
    for gid, title, phrase, year in PASSAGES:
        if gid not in cache:
            failures.append({"title": title, "reason": "book download failed"})
            continue
        text = cache[gid]
        phrase_n = normalize(phrase)
        idx = text.find(phrase_n)
        if idx < 0:
            failures.append({"title": title, "reason": "start phrase not found"})
            continue
        words = text[idx:].split()[:TOTAL_WORDS + 6]
        if len(words) < TOTAL_WORDS:
            failures.append({"title": title, "reason": "passage too short"})
            continue
        split = shift_split(words)
        candidates.append({
            "title": title,
            "source_work": f"gutenberg:{gid}",
            "year": year,
            "prompt_text": " ".join(words[:split]),
            "continuation_text": " ".join(words[split:TOTAL_WORDS + (split - SPLIT_AT)]),
            "split_at": split,
            "boundary_word": words[split],
            "why_famous": "curated public-domain canon; wording taken verbatim from Project Gutenberg",
        })
        print(f"OK   {title} (split@{split} -> '{words[split]}')")
    for f in failures:
        print(f"FAIL {f['title']}: {f['reason']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"l0_candidates": candidates, "failures": failures}, fh,
                  ensure_ascii=False, indent=2)
    print(f"done: {len(candidates)} candidates, {len(failures)} failures -> {OUT}")


if __name__ == "__main__":
    main()
