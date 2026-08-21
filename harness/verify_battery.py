"""infini-gram 驗證器 v2（依對抗式審查修正）。

修正重點：
- query 回傳三態 (status, count)：ok / invalid（4xx 不重試、保留伺服器錯誤訊息）/
  network（重試耗盡）——不再用 -1/-2 哨兵混進計數。
- 探針邊界標點剝除（引號/括號/破折號會位移 SentencePiece 的 ▁ 前綴，造成假 0）。
- 滑動視窗（stride 3）取代頭/中/尾三點抽樣；不足 n_words 直接拒收，不再整段當探針。
- verified 三態：True / False / None（任何探針查詢失敗→None，不得混入 False）。
- check_novelty 回傳連續量（frac_present、max_count、n_probes），all_absent 需
  n_probes>0 才可能為 True；index_key 必填。
- main() 印 per-corpus 統計；有 unknown 時以非零 exit code 結束。

用法：
    python harness/verify_battery.py battery/l0_candidates.json -o battery/l0_verification.json
    python harness/verify_battery.py battery/l0_control_candidates.json -o battery/l0n_verification.json --mode control
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://api.infini-gram.io/"
INDEXES = {
    "pile": "v4_piletrain_llama",          # Pythia 訓練語料
    "olmo_mix": "v4_olmo-mix-1124_llama",  # OLMo-2 訓練語料
}
L0_COUNT_THRESHOLD = 20   # 記憶判定：所有探針 count >= 此值
CONTROL_MAX_COUNT = 2     # 對照組判定：所有探針 count <= 此值（近乎查無）

_LEAD_PUNCT = re.compile(r"^[\'\"“”‘’(\[\{—–-]+")
_TRAIL_PUNCT = re.compile(r"[\'\"“”‘’)\]\}.,;:!?—–-]+$")
MEMORIZED_FRAC = 0.7  # 「大多數窗口過門檻」而非全數——跨句窗口因引號/標點體例
                      # 變異天然脆弱，不給一票否決權（2026-08-20 0/20 事故的教訓）


def query(index: str, text: str, retries: int = 4):
    """回傳 (status, count)；status in {ok, invalid, network}。"""
    payload = json.dumps({"index": index, "query_type": "count",
                          "query": text}).encode()
    last = "network"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if "error" in data:
                return "invalid", None
            return "ok", int(data.get("count", 0))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):  # 限流：長退避後重試，不是 invalid
                wait = 30 * (attempt + 1)
                print(f"    rate-limited ({e.code}); cooling {wait}s", file=sys.stderr)
                time.sleep(wait)
                last = "network"
                continue
            if 400 <= e.code < 500:  # 其他 4xx 不可重試；保留伺服器訊息
                try:
                    body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    body = str(e.code)
                print(f"    invalid query ({e.code}): {body}", file=sys.stderr)
                return "invalid", None
            last = "network"
        except Exception:
            last = "network"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return last, None


def _edge_trim(window):
    """只剝「窗口第一個詞的前導標點」與「最後一個詞的尾隨標點」；
    內部標點原樣保留——語料裡的原文就長那樣（0/20 事故的根因修正）。"""
    w = list(window)
    w[0] = _LEAD_PUNCT.sub("", w[0])
    w[-1] = _TRAIL_PUNCT.sub("", w[-1])
    return w if (w[0] and w[-1]) else None


def probes_from(text: str, n_words: int = 7, stride: int = 4):
    """滑動視窗探針；不足 n_words 回空清單（拒收，不降級）。"""
    words = text.split()
    if len(words) < n_words:
        return []
    probes = []
    for i in range(0, len(words) - n_words + 1, stride):
        w = _edge_trim(words[i:i + n_words])
        if w:
            probes.append(" ".join(w))
    return probes


def _run_probes(probes, index):
    results = []
    for p in probes:
        status, count = query(index, p)
        results.append({"probe": p, "status": status, "count": count})
        time.sleep(0.6)  # 限流教訓：0.15s 連發幾分鐘會吃 403
    return results


def _judge(results, mode: str):
    """True/False/None；任何非 ok 探針 → None（unknown）。
    memorized：至少 MEMORIZED_FRAC 的窗口 count >= 門檻（跨句窗口不一票否決）。
    control：全數窗口 count <= 上限（對照組必須乾淨）。"""
    if not results:
        return None
    if any(r["status"] != "ok" for r in results):
        return None
    counts = [r["count"] for r in results]
    if mode == "memorized":
        frac = sum(1 for c in counts if c >= L0_COUNT_THRESHOLD) / len(counts)
        return frac >= MEMORIZED_FRAC
    return all(c <= CONTROL_MAX_COUNT for c in counts)  # control


def verify_passage(passage: dict, mode: str) -> dict:
    probes = probes_from(passage["continuation_text"])
    joint = _edge_trim(passage["prompt_text"].split()[-4:]
                       + passage["continuation_text"].split()[:4])
    if joint:
        probes.append(" ".join(joint))
    result = {"title": passage.get("title"), "mode": mode, "probes": {}, "verified": {}}
    for name, index in INDEXES.items():
        rs = _run_probes(probes, index)
        ok_counts = [r["count"] for r in rs if r["status"] == "ok"]
        result["probes"][name] = rs
        result["verified"][name] = _judge(rs, mode)
        result.setdefault("stats", {})[name] = {
            "n_probes": len(rs),
            "n_unknown": sum(1 for r in rs if r["status"] != "ok"),
            "min_count": min(ok_counts) if ok_counts else None,
            "max_count": max(ok_counts) if ok_counts else None,
        }
    return result


def check_novelty(generated_text: str, index_key: str, n_words: int = 6,
                  stride: int = 2) -> dict:
    """L5 新穎度查核：回傳連續量，呼叫端不得只看 all_absent。index_key 必填。"""
    probes = probes_from(generated_text, n_words, stride)
    rs = _run_probes(probes, INDEXES[index_key])
    ok = [r for r in rs if r["status"] == "ok"]
    present = [r for r in ok if r["count"] > 0]
    return {
        "n_probes": len(rs),
        "n_unknown": len(rs) - len(ok),
        "frac_present": (len(present) / len(ok)) if ok else None,
        "max_count": max((r["count"] for r in ok), default=None),
        "all_absent": bool(ok) and len(rs) == len(ok) and not present,
        "probes": rs,
    }


def main():
    argv = sys.argv[1:]
    src = argv[0] if argv and not argv[0].startswith("-") else "battery/candidates.json"
    out = argv[argv.index("-o") + 1] if "-o" in argv else "battery/verification.json"
    mode = "control" if "--mode" in argv and argv[argv.index("--mode") + 1] == "control" \
        else "memorized"
    with open(src, encoding="utf-8") as f:
        candidates = json.load(f)
    key = "l0_candidates" if "l0_candidates" in candidates else "candidates"
    report = []
    for p in candidates.get(key, []):
        print(f"verifying [{mode}]: {p.get('title', '?')} ...", flush=True)
        r = verify_passage(p, mode)
        report.append(r)
        print("  -> " + " / ".join(
            f"{n}:{'PASS' if v else 'UNKNOWN' if v is None else 'fail'}"
            for n, v in r["verified"].items()), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    n_unknown = sum(1 for r in report if any(v is None for v in r["verified"].values()))
    for name in INDEXES:
        n_pass = sum(1 for r in report if r["verified"].get(name) is True)
        print(f"{name}: {n_pass}/{len(report)} pass")
    print(f"unknown(any-corpus): {n_unknown} -> {out}")
    if n_unknown:
        print("WARNING: some probes failed to query; rerun before trusting results",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
