# AI Collaboration Disclosure / AI 協作揭露聲明

## English

This research was conceived, directed, and decided by the author (a high-school student working
as an independent researcher). All substantive choices — the research question, scope decisions, resource
trade-offs, acceptance or rejection of every reviewer finding, and the decision to publish —
were made by the author.

AI agents (Anthropic Claude) carried out execution under the author's direction:
literature search, engineering, experiment runs, statistical analysis, review,
verification, adversarial red-teaming, and the drafting of this text. At scale: a
21-agent adversarial review team made 263 tool calls against the measurement harness
alone and surfaced 16 confirmed defects, all fixed before any GPU run (process log,
step 9). The author reviewed the content via a bilingual (Chinese/English) mirror and
takes full responsibility for it.

Verification discipline: a dedicated verification agent independently reproduced every
claimed defect before the team accepted it (16 confirmed findings, all fixed pre-run);
we regenerate every number in the text from raw result files; the author's agents
resolved every citation to its source before submission. The repository's process log
publishes the full audit trail, including the mistakes the team made and corrected
along the way.

Text for this paper was drafted by Claude (Anthropic Claude 5-family models) during 2026-08-20 to 2026-08-21. Claude models
launched on or after 2 August 2026 embed a statistical text watermark (Anthropic's
implementation of the SynthID-Text approach published by Google DeepMind). We note
this only for completeness: by Anthropic's own account a mark indicates only that
Claude was likely involved at some point, and cannot distinguish text Claude wrote
from text Claude edited, translated or proofread. At the time of writing no public
detector is available; Anthropic has announced a forthcoming detection API. Nothing
in this disclosure rests on it — no detector score, in either direction, is evidence
about this work. The scope of AI involvement is what this statement says it is, and
the process log is where you check it.

## 中文

本研究由作者（高中生獨立研究者）發想、主持與決策。所有實質選擇——研究問題、範圍、
資源取捨、對每一條審查發現的採納與否、以及發表決定——皆由作者本人做出。

AI 代理（Anthropic Claude）在作者指揮下完成執行：文獻檢索、工程、實驗、統計分析、
審查、驗證、對抗式紅隊測試，以及本文起草。規模如實揭露：一支由 21 個代理組成的
對抗式審查團隊，光是對量測程式就呼叫了 263 次工具，找出 16 項確認屬實的缺陷，
全數在燒 GPU 前修復（製作歷程日誌第九步）。作者透過中英對照版逐段審閱內容，並對
全文負完全責任。

驗證紀律：獨立的查核代理在團隊採納每一條缺陷主張前，先實際重現它（16 項確認缺陷，
開跑前全數修復）；我們將文中所有數字回溯至原始結果檔重新產生；作者的代理團隊在
投稿前逐一核實每一條引用的來源。repo 的製作歷程日誌公開完整稽核軌跡，包含團隊在
過程中犯過並修正的錯誤。

本文文字由 Claude（Anthropic Claude 5 家族模型）於 2026-08-20 至 08-21 間起草。2026 年 8 月 2 日以後推出的
Claude 模型嵌有統計式文字浮水印（Anthropic 對 Google DeepMind 發表之 SynthID-Text
方法的實作）。我們僅為求完整而提及此事：依 Anthropic 官方說法，浮水印只能顯示
Claude 在某個時間點很可能參與過，無法區分文字是 Claude 所寫、還是 Claude 編輯、
翻譯或校對過的文字。截至本文撰寫時尚無公開偵測工具；Anthropic 已宣布即將推出
偵測 API。本聲明的任何內容都不依賴這一點——無論偵測分數指向哪個方向，都不構成
本研究的證據。AI 參與的範圍以本聲明的文字為準，查核請見製作歷程日誌。
