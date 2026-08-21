# Correction (2026-08-21): what the "corpus duplication counts" anchor actually measures
# 更正（2026-08-21）：「語料重複次數」這個錨實際量的是什麼

**Status: verdicts unchanged. This corrects the *name and interpretation* of one
published measurement, not any registered decision.**
**狀態：所有已登記的裁決不變。本更正針對的是一個已發表量測的*名稱與解讀*，
不是任何裁決。**

---

## English

### What was published

v0.5–v0.7 present the model-external anchor under the headline name **"corpus
duplication counts from infini-gram"** (anchor correlations rho 0.64–0.73 vs the
memory scale, all five runs), and describe L0 items as corpus-verified passages
(gate: window counts ≥ 20 in ≥ 70% of probe windows). The methods sections
describe the window-level mechanism correctly, and the appendix uses the accurate
name "median corpus window count per passage"; **this correction targets the
headline-level name and the reading it invites** — that the corpus contains many
copies of *the passage*.

### What was actually measured

Operationally, the anchor is the **median count over a passage's eleven probe
windows (ten 7-word windows at stride 4, plus one window spanning the
prompt/continuation boundary), taken over ok probes** (`pooled_anchor_analysis.py`,
replicating `T2_dupcount`). New probes (2026-08-21, script and raw JSON shipped
with this correction) show that this quantity is not a passage copy count:

- **Within one passage, window counts are far from uniform.** The ratio between a
  passage's highest- and lowest-count probe window has median **16.2× (Pile,
  17 gate-eligible items)** and **42.3× (olmo-mix, 20 items)**; zero-count floors
  are clamped to 1 (2 such items per corpus). A true copy count would make every
  window of the same passage count roughly equally.
- **In Pile, complete copies of the full ~44-word passage are much rarer than the
  gate's threshold suggests.** Extending the probe window stepwise
  (7→12→20→30→44 words, same start), the count at 44 words — the passage's
  actual copy count — has median **18** over the 17 Pile-eligible items
  (10 of 17 below the gate's 20; 2 items at 0). Three further items (Hamlet,
  Julius Caesar, Treasure Island) never passed the Pile gate and are excluded
  from all published Pile-side analyses; the corresponding olmo-mix figures
  (all 20 items eligible) are median **64**, 6 of 20 below 20, 3 at 0 —
  so the shortfall against the threshold is a Pile-side finding, while the
  *peak-vs-copies gap* holds in both corpora.
- Concrete example: the *Pride and Prejudice* item's first probed 7-gram —
  **"well fixed in the minds of the"**, a mid-sentence fragment of the gold
  continuation, not the famous opening line (which sits on the prompt side and
  is not probed by this curve) — occurs **171 times** in Pile, while the
  complete 44-word passage occurs **0 times** at our probe's tokenization.
  (A zero at 44 words is individually ambiguous — punctuation variants could
  hide a copy — but several items retain positive counts at 44 words, so the
  probe does resolve full-length matches.)

### What this changes

1. **Rename.** The anchor should be read as **"median probe-window frequency —
   in practice dominated by the passage's most-repeated short phrases"**, not as
   a count of passage duplicates. All headline occurrences of "corpus duplication
   counts" in v0.5–v0.7 should be read under this label. The correlations
   themselves are unchanged; what they anchor the memory scale to is
   short-phrase repetition frequency.
2. **Dose locality.** The dose–response finding survives — a phrase repeated 171
   times was seen 171 times — but the dose attaches to **short phrases**, while
   gold log-probability is measured over the **whole ~44-word passage**. The
   published unit of dose was overstated; this is a precision problem, not an
   existence problem.
3. **Selection pressure (interpretation, flagged as such).** A gate keyed to
   short-window frequency structurally favours passages containing
   often-repeated phrases, and often-quoted text tends to be predictable text.
   This corpus-side observation is **consistent with** the predictability
   confound that the preregistered L0P control had already surfaced model-side
   in v0.7 — two independent routes, one of which requires no model at all,
   pointing at the same suspect. (Registered analyses separating this account
   from localized exposure are at present underpowered; neither is established.)

### What this does not change

No registered decision rule fired or un-fires: the v0.7 verdict record ("none")
and all correlation values stand. The L0 items remain corpus-attested text; what
is corrected is how much of each passage the counts attest, and what the
anchor's headline name claims.

### Reproduce

`harness/gate_decay_probe.py` (window-length curves, 200 infini-gram queries) and
`results/gate/gate_decay.json` (raw counts) ship with this correction; the
medians above recompute from the JSON in a few lines (clamp zero floors to 1
before taking ratios, or they divide by zero). Peak/floor ratios recompute from
the already-shipped `battery/l0_verification.json` alone. Two technical notes:
the decay probe anchors at the continuation's first words and does not apply the
gate's edge-trim, so its 7-word counts are not identical to the gate's own first
window (observed difference direction makes the curves *under*-state amplification,
i.e. is conservative for this correction); and Pile-side statistics above use the
17 gate-eligible items, matching every published Pile-side analysis.

---

## 中文

### 原本怎麼寫

v0.5–v0.7 用「infini-gram 查到的**語料重複次數**」作為模型外部錨的標題名稱
（對記憶量尺 rho 0.64–0.73，五次跑），並把 L0 題描述為語料實證段落
（閘門：≥70% 的探測窗 count ≥ 20）。方法段對窗級機制的描述是準確的，
附錄用的也是準確名稱「每段語料窗 count 中位數」；**本更正針對的是標題層級
的名稱與它邀請的讀法**——「語料裡有很多份*這個段落*」。

### 實際量到的是什麼

發表版的錨在操作上是**該段落 11 個探測窗（10 個 stride 4 的 7 詞窗＋1 個跨
prompt/續寫邊界的窗）中 ok 探測的 count 中位數**。2026-08-21 的新探測
（腳本與原始 JSON 隨本更正出貨）顯示它不是段落副本數：

- **同一段內部，各窗的 count 差很多**：最高頻窗對最低頻窗的比值，中位數是
  **16.2×（Pile，過閘的 17 題）／42.3×（olmo-mix，20 題）**；底值為 0 時
  以 1 代入（每語料各 2 題）。真的副本數會讓同段每個窗差不多。
- **在 Pile 裡，整段 44 詞的完整副本比閘門門檻暗示的少得多**：同起點逐步
  加長探測窗（7→12→20→30→44 詞），44 詞處的 count——段落真正的副本數
  ——在過閘的 17 題中位是 **18**（10 題低於閘門要求的 20；2 題是 0）。
  另外 3 題（哈姆雷特／凱撒／金銀島）本來就沒過 Pile 閘門，所有已發表的
  Pile 側分析都排除它們；olmo-mix 側（20 題全數過閘）中位是 **64**、
  6 題低於 20、3 題是 0——所以「不到門檻」是 Pile 側的發現，
  而「峰值遠高於副本數」在兩個語料都成立。
- 具體例子：《傲慢與偏見》那題被探測的第一個 7 連詞——
  **"well fixed in the minds of the"**，gold 續寫裡的句中片段，
  **不是**那句開場名言（名言在 prompt 側，這條曲線根本沒量到它）——
  在 Pile 出現 **171 次**，而完整 44 詞段落在本探測的斷詞下是 **0 次**。
  （單題的 44 詞 0 有歧義——標點差異可能藏住副本——但有些題在 44 詞
  仍是正數，證明探測抓得到全長命中。）

### 因此要改什麼

1. **改名**：這個錨應讀作「**探測窗頻率中位數——實務上由段落裡最常被重複
   的短句主導**」，不是段落副本數。v0.5–v0.7 標題層級所有「語料重複次數」
   的出現處都應以此標籤解讀。相關數字本身不變；變的是記憶量尺被錨到
   什麼東西上。
2. **劑量的位置**：劑量反應本身站得住——一句被重複 171 次就是被看過
   171 次——但劑量下在**短句**上，而 gold log-prob 量的是**整段約 44 詞**。
   發表版把劑量的單位寫大了。這是精度問題，不是存廢問題。
3. **選題壓力（解讀，明標為解讀）**：以短窗頻率為閘門，結構上偏向收進
   「含有常被重複短句」的段落，而常被引用的文字傾向於好預測。這個語料側
   的觀察**與** v0.7 預註冊 L0P 對照在模型側量到的可預測性混淆**一致**
   ——兩條獨立的路（其中一條完全不用跑模型）指向同一個嫌疑犯。
   （目前已登記的判別分析檢定力不足以把這個帳戶與「局部化曝光」分開；
   兩者都未被證實。）

### 不變的部分

沒有任何已登記裁決因此點燃或熄滅：v0.7 的裁決紀錄（none）與所有相關值照舊。
L0 題仍是語料實證文本；被更正的是「count 實證了段落的多少」以及錨的標題
名稱所宣稱的東西。

### 重算

`harness/gate_decay_probe.py`（窗長曲線，200 次 infini-gram 查詢）與
`results/gate/gate_decay.json`（原始 count）隨本更正出貨；上述中位數幾行就能
從 JSON 重算（取比值前先把 0 底值以 1 代入，否則除以零）。峰/底比只需既有的
`battery/l0_verification.json` 即可重算。兩個技術註記：衰減探測從續寫開頭
取窗、未套閘門的 edge-trim，所以它的 7 詞 count 與閘門自己的第一窗不完全同物
（觀察到的差異方向使曲線**低估**放大倍率，對本更正而言是保守方向）；
上述 Pile 側統計用過閘的 17 題，與所有已發表 Pile 側分析的分母一致。
