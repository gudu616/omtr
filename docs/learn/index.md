# OMTR 學習材料索引

> 目的：讓任何想加入這個研究的人（包括未來的自己）能從零看懂每個決策。
> 每張卡片固定格式：專業詞彙 → 一句話白話版 → 為什麼本研究用得到 →
> 再深一點 → 在我們的程式碼/數據裡的位置 → 常見誤解 → 答辯模擬題。
> 規則：**允許專業詞彙，但每個詞第一次出現必附白話解釋**。

## 建議閱讀順序

### 第一部：模型是怎麼運作的（地基）
- [01 下一個詞預測](01-next-token.md) — LLM 唯一真正在做的事
- [02 Token 與 Tokenizer](02-tokenizer.md) — 模型眼中的「字」和你想的不一樣
- [03 Base 模型 vs Instruct 模型](03-base-vs-instruct.md) — 為什麼我們的模型「不聽話」
- [04 殘差流與層](04-residual-stream.md) — 24 層流水線上的那條輸送帶
- [05 Cloze 與 Few-shot 鷹架](05-cloze-scaffold.md) — 跟 base 模型說話的正確姿勢

### 第二部：我們的量測儀器
- [06 Logit Lens](06-logit-lens.md) — 中途攔稿：「現在交卷你會寫什麼？」★範例卡
- [07 熵：猶豫程度計](07-entropy.md)
- [08 KL 散度與收斂深度](08-kl-depth.md) — 答案在第幾層定案
- [09 Gold Logprob：記憶強度尺](09-gold-logprob.md)
- [10 表徵分離度](10-separation.md) — 哪一層「分得出」回憶與創作

### 第三部：實驗怎麼設計才算數
- [11 對照組與混淆變因](11-control-confound.md) — 為什麼要有 L0N
- [12 任務階梯 L0–L5](12-task-ladder.md) — 從純回憶爬到純創造
- [13 記憶化 vs 泛化](13-memorization-generalization.md)
- [14 語料溯源與 infini-gram](14-provenance.md) — 「它真的背過嗎」怎麼證明
- [15 三個假說：閘門／工作點／雙迴路](15-three-hypotheses.md) — 本研究到底想裁決什麼

### 第四部：統計與研究實務
- [16 效應量與劑量反應](16-effect-size-dose.md) — 小樣本時代的正確統計姿勢
- [17 操縱檢查](17-manipulation-check.md) — 先確認模型真的在做你以為的事
- [18 對抗式審查](18-adversarial-review.md) — 先審查、後燒卡
- [19 用 AI 團隊做研究](19-ai-team-research.md) — 分工、分級、打撈與 QC
- [20 下一步的工具預告：Patching 與 SAE](20-patching-sae.md)
- [21 數字對帳與雙重捨入](21-double-rounding.md) — 308 個數字怎麼被機器看守
- [22 同分天花板](22-tie-ceiling.md) — 相關係數的可達上限，與對自己開刀的誠實
- [23 置換檢定與尺的校準](23-permutation-calibration.md) — 量真資料之前，先用假資料驗尺
- [24 裁決規則的可點燃性](24-decision-rule-reachability.md) — 預註冊的門要先確認打得開
- [25 文獻檢索的紀律](25-literature-search.md) — 摘要只能篩選，判定必須全文；查詢字串要留檔

## 維護規則

研究每進一個新階段，用到新概念就補卡；卡片內容與程式碼衝突時，以程式碼為準並回頭修卡。
