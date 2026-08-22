"""出貨關卡：把包解到跟專案無關的資料夾，把 README 說的每一道指令真的敲一次。

為什麼需要這支（2026-08-22 立，見 docs/CORRECTION_20260822_REPRODUCIBILITY.md §5）
------------------------------------------------------------------------------
v1.0 出貨後，外部審閱者 clone 下來照 README 敲指令，發現跑不動。事後查，
壞掉的每一條在檢查清單上**都有寫**：

  · PRE_UPLOAD_CHECKLIST.md:27 要求掃 `C:\\Users\\...` 機器路徑——掃了，
    抓到 .md 裡的，漏掉 .py 裡的（掃的人想的是「文件」，不是「程式」）。
  · RELEASE CHECK ④ 查「README 提到的路徑存不存在」——19 條全在。
    但沒有人把那些指令敲下去。壞掉的相依寫在程式裡，不在 README 裡。

所以漏網的東西沒有掉在任何一項檢查**裡面**，全部掉在檢查與檢查**之間**。
再加第九項檢查只會多出第九道縫。這支腳本是唯一新增的規矩，它的設計要點是
**指令清單從 README 現場解析出來，不另外維護一份**——另外維護就會漂走，
那正是這次出事的機制。

用法
----
    python harness/verify_shipping.py                     # 驗 publish/repo_bundle
    python harness/verify_shipping.py --bundle <path>
    python harness/verify_shipping.py --skip-slow         # 跳過標記為慢的指令（僅供開發回圈）
    python harness/verify_shipping.py --no-git            # 跳過 git 換行正規化測試

退出碼：0＝全過可上傳；1＝有指令壞掉；2＝MANIFEST 在 clone 後對不上；3＝設定錯誤。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJ / "publish" / "repo_bundle"

# README 裡放指令的那一節的標題。改標題要一起改這裡——
# 找不到就 FATAL，不會靜默地「零道指令全過」（那正是這次的病）。
SECTION_RE = re.compile(r"^#+\s*Verify the numbers yourself\s*$", re.M)
FENCE_RE = re.compile(r"```(?:bash|sh|console)?\n(.*?)```", re.S)
# 預期會跑很久的（載模型／大量置換）。--skip-slow 才會跳，預設一律跑。
SLOW_HINTS = ("pos_test", "calibration")


def parse_commands(readme: Path) -> list[str]:
    """從 README 的驗證節解析出 python 指令，順序照原文。"""
    text = readme.read_text(encoding="utf-8", errors="replace")
    m = SECTION_RE.search(text)
    if not m:
        raise SystemExit(
            f"[FATAL] {readme} 找不到「Verify the numbers yourself」節。\n"
            "        標題改過就要同步改 SECTION_RE——不然這支會靜默地驗零道指令，\n"
            "        而那正是本規矩要防的東西。"
        )
    tail = text[m.end():]
    nxt = re.search(r"^#+\s", tail, re.M)
    body = tail[: nxt.start()] if nxt else tail

    cmds: list[str] = []
    for block in FENCE_RE.findall(body):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("python "):
                cmds.append(line[len("python "):].strip())
    return cmds


def run_all(bundle: Path, cmds: list[str], skip_slow: bool, timeout: int) -> list[tuple]:
    """把包複製到暫存區（跟專案無關的路徑）再跑，避免相對路徑意外接到倉庫。"""
    results = []
    with tempfile.TemporaryDirectory(prefix="omtr_shipcheck_") as td:
        room = Path(td) / "cleanroom"
        shutil.copytree(bundle, room)
        print(f"[room] {room}\n")
        for cmd in cmds:
            if skip_slow and any(h in cmd for h in SLOW_HINTS):
                print(f"SKIP  | {cmd}")
                results.append(("SKIP", cmd, ""))
                continue
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, *cmd.split()], cwd=room,
                                   capture_output=True, text=True, errors="replace",
                                   timeout=timeout)
                code, out = p.returncode, (p.stdout + p.stderr)
            except subprocess.TimeoutExpired:
                code, out = -1, f"逾時（>{timeout}s）"
            dt = time.time() - t0
            if code == 0:
                print(f"PASS  | {cmd}  ({dt:.0f}s)")
                results.append(("PASS", cmd, ""))
            else:
                last = (out.strip().splitlines() or [""])[-1][:200]
                print(f"FAIL  | {cmd}  ({dt:.0f}s)\n        {last}")
                results.append(("FAIL", cmd, last))
    return results


def check_manifest_after_clone(bundle: Path) -> tuple[int, int, list[str]]:
    """真的 git init + clone 一次（core.autocrlf=input）再驗 MANIFEST。

    本機直接驗雜湊永遠會過——包就是在本機生的。會壞的是**旅行之後**：
    git 對換行做正規化，內容沒變、指紋卻變了。所以這裡一定要走真的 clone。
    """
    with tempfile.TemporaryDirectory(prefix="omtr_gitcheck_") as td:
        src, dst = Path(td) / "pkg", Path(td) / "clone"
        shutil.copytree(bundle, src)
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        q = dict(cwd=src, capture_output=True, text=True, env=env)
        subprocess.run(["git", "init", "-q", "."], **q)
        subprocess.run(["git", "config", "core.autocrlf", "input"], **q)
        subprocess.run(["git", "config", "user.email", "noreply@example.com"], **q)
        subprocess.run(["git", "config", "user.name", "shipcheck"], **q)
        subprocess.run(["git", "add", "-A", "-f"], **q)
        subprocess.run(["git", "commit", "-qm", "shipcheck"], **q)
        r = subprocess.run(["git", "clone", "-q", str(src), str(dst)],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0 or not (dst / "MANIFEST.json").exists():
            return (0, 0, [f"clone 失敗：{r.stderr.strip()[:200]}"])

        man = json.loads((dst / "MANIFEST.json").read_text(encoding="utf-8"))
        bad = []
        for e in man["files"]:
            f = dst / e["path"]
            if not f.exists():
                bad.append(f"{e['path']} (缺檔)")
            elif hashlib.sha256(f.read_bytes()).hexdigest() != e["sha256"]:
                bad.append(e["path"])
        return (len(man["files"]) - len(bad), len(man["files"]), bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    bundle = a.bundle.resolve()
    readme = bundle / "README.md"
    if not readme.is_file():
        print(f"[FATAL] 找不到 {readme}")
        return 3

    cmds = parse_commands(readme)
    if not cmds:
        print("[FATAL] 驗證節裡解析到 0 道指令——這是設定壞了，不是全過。")
        return 3

    print(f"=== 出貨關卡 ===\n包：{bundle}\nREADME 列了 {len(cmds)} 道指令\n")
    results = run_all(bundle, cmds, a.skip_slow, a.timeout)

    rc = 0
    fails = [c for s, c, _ in results if s == "FAIL"]
    skips = [c for s, c, _ in results if s == "SKIP"]

    if not a.no_git:
        print("\n=== MANIFEST 在真的 clone 之後 ===")
        ok, total, bad = check_manifest_after_clone(bundle)
        print(f"對得上 {ok} / {total}")
        if bad:
            print("對不上：")
            for b in bad[:10]:
                print(f"  - {b}")
            if len(bad) > 10:
                print(f"  …另外 {len(bad) - 10} 個")
            rc = 2
        else:
            print("（.gitattributes 生效，換行沒被動過）")

    print("\n=== 結論 ===")
    print(f"指令：{len(results) - len(fails) - len(skips)} 過 / {len(fails)} 壞 / {len(skips)} 跳過")
    if fails:
        for c in fails:
            print(f"  FAIL: {c}")
        rc = 1
    if skips:
        print("  ⚠ 有指令被跳過——--skip-slow 只供開發回圈，上傳前必須不帶這個旗標跑一次。")
    print("可以上傳。" if rc == 0 and not skips else "不可上傳。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
