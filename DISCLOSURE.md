# AI Collaboration Disclosure / AI 協作揭露聲明

## English

This research was conceived, directed, and decided by the author (an independent researcher). All substantive choices — the research question, scope decisions, resource
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

**This disclosure in three parts.** *What the AI did:* literature search, engineering,
experiment runs, statistical analysis, review, verification, adversarial red-teaming,
and the drafting of the papers' text, written by Claude (Anthropic Claude 5-family
models) between 2026-08-20 and 2026-08-22. *What the author did:* set the research
question, made every scope and resource decision, accepted or rejected each reviewer
finding, reviewed the content through a bilingual mirror, and decided what to publish.
*Who is responsible for what:* the author is responsible for all of it — every claim,
every number, and every decision to release. The scope of AI involvement is what this
statement says it is, and the process log is where you check it.

*On watermarking:* Claude models launched on or after 2 August 2026 embed a statistical
text watermark. **No detector score, in either direction, is evidence about this work** —
nothing in this disclosure rests on it.

### Division-of-labor amendment (dated 2026-08-21, evening)

Effective with the work that follows draft v0.7 (the causal-phase design and
onward), the first-pass literature search is conducted **by the author
personally**: choosing search axes and query strings, screening abstracts, and
deciding what gets a full-text read. AI agents assist with full-text retrieval,
local-snapshot verbatim verification of quotations, and — where the author
delegates it — supplementary searches. The statements above remain the accurate
record for v0.5–v0.7, whose literature work was AI-executed; this amendment
changes the process going forward, not the history. The change was requested by
the author.

## 中文

本研究由作者（獨立研究者）發想、主持與決策。所有實質選擇——研究問題、範圍、
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

**本揭露的三個要素。** *AI 做了什麼*：文獻檢索、工程、實驗、統計分析、審查、驗證、
對抗式紅隊測試，以及論文文字的起草——由 Claude（Anthropic Claude 5 家族模型）
於 2026-08-20 至 08-22 間寫成。*作者做了什麼*：定研究問題、做每一個範圍與資源決策、
決定採納或駁回每一條審查發現、透過中英對照版逐段審閱、並決定發表什麼。
*誰對什麼負責*：作者對全部負責——每一條主張、每一個數字、每一次發表決定。
AI 參與的範圍以本聲明的文字為準，查核請見製作歷程日誌。

*關於浮水印*：2026 年 8 月 2 日以後推出的 Claude 模型嵌有統計式文字浮水印。
**無論偵測分數指向哪個方向，都不構成本研究的證據**——本聲明不依賴這一點。

### 分工變更附註（2026-08-21 晚定）

自 v0.7 之後的工作（因果階段設計起）生效：**文獻檢索的第一輪由作者本人
執行**——選定檢索軸與查詢字串、以摘要篩選、決定哪些論文進入全文精讀。
AI 代理協助全文取得、引文的本地快照逐字核對，以及（作者委派時的）補充
檢索。上文各段對 v0.5–v0.7 的描述維持不變——那些版本的文獻工作確實由
AI 執行；本附註改變的是此後的流程，不是歷史。此變更由作者本人提出。
