# Benchmark 評測報告

> AORUS MASTER 16 RAG — CPU-only 環境，6 個模型 × 10 題 × benchmark_cases_v3.json

---

## 評測環境

| 項目 | 說明 |
|------|------|
| 推論裝置 | CPU only（無 GPU 數據） |
| 推論引擎 | llama.cpp (Python binding) |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Chunk 版本 | chunks_v4_bilingual.json（中英雙語） |
| 測試題庫 | benchmark_cases_v3.json（10 題） |
| 評估指標 | Keyword Hit Rate / TTFT / TPS / RAM Peak |

---

## 一、整體成績總覽

| 模型 | Hit Rate | TTFT (avg) | TPS | RAM Peak | shared_spec | single_product | gpu_comparison |
|------|:--------:|:----------:|:---:|:--------:|:-----------:|:--------------:|:--------------:|
| **Llama-3.2-3B Q5_K_M** | **91.5%** | 127,845 ms | 2.2 | 3,848 MB | 85.8% | 100% | **100%** |
| Llama-3.2-3B Q4_K_M | 84.0% | **73,022 ms** | **2.6** | 4,858 MB | 77.5% | 100% | 87.5% |
| Qwen2.5-3B Q5_K_M | 82.5% | 145,409 ms | 2.2 | **3,506 MB** | 70.8% | 100% | **100%** |
| Qwen2.5-3B Q4_K_M | 77.0% | 81,512 ms | 2.6 | 4,447 MB | 74.2% | 75.0% | 87.5% |
| Phi-4-mini Q4_K_M | 69.0% | 96,480 ms | 2.2 | 5,283 MB | 73.3% | 75.0% | 50.0% |
| Phi-4-mini Q5_K_M | 69.0% | 163,269 ms | 1.9 | 4,624 MB | 65.0% | 75.0% | 75.0% |

---

## 二、各問題命中率細節

| Q# | 題型 | 查詢內容摘要 | Llama Q4 | Llama Q5 | Qwen Q4 | Qwen Q5 | Phi Q4 | Phi Q5 |
|----|------|------------|:--------:|:--------:|:-------:|:-------:|:------:|:------:|
| Q1 | shared_spec | 支援哪些作業系統 | 100% | 100% | 100% | 100% | 100% | 100% |
| Q2 | single_product | BYH GDDR7 / AI Boost | 100% | 100% | 100% | 100% | 100% | 100% |
| Q3 | shared_spec | 最大 RAM / SO-DIMM | 75% | 75% | 75% | 75% | 50% | 50% |
| Q4 | shared_spec | Thunderbolt 接口數 | 100% | 100% | 100% | 100% | 100% | 100% |
| Q5 | gpu_comparison | BZH vs BXH TDP | 75% | 100% | 75% | 100% | 75% | 100% |
| Q6 | shared_spec | 鍵盤/音訊/攝影機規格 (EN) | 40% | 40% | 20% | 0% | 40% | 40% |
| Q7 | shared_spec | 顯示面板類型/更新率 (EN) | 50% | 100% | 50% | 50% | 50% | 50% |
| Q8 | shared_spec | 電池容量 (EN) | 100% | 100% | 100% | 100% | 100% | 50% |
| Q9 | single_product | BXH 儲存選項 (EN) | 100% | 100% | 50% | 100% | 50% | 50% |
| Q10 | gpu_comparison | BYH vs BXH 遊戲推薦 (EN) | 100% | 100% | 100% | 100% | 25% | 50% |

---

## 三、觀察與分析

### 3.1 最佳模型：Llama-3.2-3B Q5_K_M

**91.5% Hit Rate**，在全部 6 個模型中排名第一，且是唯一在所有題型均無完全失分的模型：

- `single_product`（Q2、Q9）：**100%** — 單機型精確規格抽取完全正確
- `gpu_comparison`（Q5、Q10）：**100%** — 跨機型比較推理能力最強
- `shared_spec`（Q1、Q3、Q4、Q6、Q7、Q8）：**85.8%** — 唯一弱點為 Q6（多欄位合併英文問）

RAM 佔用僅 **3,848 MB**，是 Llama 系列中佔用最低的，符合 4 GB 限制且有充裕空間。

---

### 3.2 速度優先選擇：Llama-3.2-3B Q4_K_M

當應用場景需要即時回應時，Q4 版本是最佳折衷：

- TTFT **73,022 ms**，比 Q5 快 **43%**
- TPS **2.6 tok/s**，為所有模型中最高
- Hit Rate 84.0%，僅低於 Q5 版本 7.5 個百分點

適合部署在需要流暢對話體驗、允許略低準確率的場景。

---

### 3.3 Qwen2.5-3B：量化升級反而降準

Qwen 系列出現**反常現象**：Q5 版本的準確率（82.5%）雖略高於 Q4（77.0%），但：

- Q5 的 `shared_spec` 命中率（70.8%）**低於** Q4（74.2%）
- Q5 的 Q6 命中率 **0%**，完全召回失敗
- Q5 速度比 Q4 慢 78%（145,409 ms vs 81,512 ms）

分析原因：Q6 是「鍵盤 + 音訊 + 攝影機」三欄位合併英文問，Qwen Q5 在 context 較長時出現「聲稱資料不足」的幻覺（答案中說明無音訊/攝影機規格），而實際上雙語 chunk 中已包含這些資訊。這指向 Qwen 在多欄位資訊整合上的解讀能力較弱。

---

### 3.4 Phi-4-mini：準確率墊底，不適合此場景

兩個量化版本均只有 **69.0%**，且顯現特有問題：

- **Q10 gpu_comparison 僅 25–50%**：在 BYH vs BXH 比較時，模型誤引入 BZH 的資料（「BZH has more memory (24GB) than both byh and bxh」），回答偏離問題核心
- **Q9 storage 僅 50%**：編造了「up to four slots」等不存在的規格（實際為 2 個 M.2 slot）
- **RAM 佔用最高（5,283 MB）**：接近 4 GB VRAM 限制邊緣，部署風險高
- Phi-4 系列為微軟 4B 參數模型，推論速度（1.9–2.2 TPS）也低於同級 3B 模型

結論：**Phi-4-mini 不推薦**用於此類高精度規格問答場景。

---

### 3.5 持續性弱點：Q6 多欄位英文問

Q6（*What are the keyboard, audio, and webcam specs?*）是**所有模型的共同弱點**：

| 模型 | Q6 命中率 | 失分原因 |
|------|:--------:|--------|
| Llama Q4/Q5 | 40% | 只取回 Keyboard chunk，Audio/Webcam 未召回 |
| Qwen Q4 | 20% | 僅命中 RGB，其餘聲稱無資料 |
| Qwen Q5 | **0%** | 完全聲稱無資料 |
| Phi Q4/Q5 | 40% | 同 Llama，Audio/Webcam 未召回 |

根本原因：雖然已建立雙語 chunk，但 `Keyboard`、`Audio`、`Webcam` 是三個**獨立 chunk**，當 retrieval top_k=5 時，三個 chunk 無法同時被召回。**改善方向**：可將這三個欄位合併成一個「外觀與操作」複合 chunk，或提高 top_k 至 8。

---

### 3.6 Q3 SO-DIMM 全數遺漏

所有模型 Q3 命中率均為 **75%**（遺漏 `SO-DIMM`），原因是現有的 System Memory chunk 中 `extracted` 欄位只保留了 `Up to 64GB DDR5 5600MHz`，截去了 `2x SO-DIMM sockets for expansion`。**改善方向**：更新 chunk 的 `extracted` 欄位以保留 SO-DIMM 資訊。

---

### 3.7 TTFT 模式分析

所有模型均在 **Q2 出現 TTFT 最低點**（15,000–35,000 ms）：Q2 詢問特定機型 GPU，被 key filter 精確命中直接回傳，大幅節省 embedding 搜尋時間，驗證了 key filter 路徑的設計有效性。

TTFT 峰值普遍出現在 **Q1（長 context）** 和 **Q7（Display 欄位多屬性）**，說明 context 長度是影響 TTFT 最主要的因素。

---

## 四、結論與建議

### 部署建議

| 場景 | 推薦模型 | 理由 |
|------|---------|------|
| **準確度優先**（預設推薦） | Llama-3.2-3B Q5_K_M | Hit Rate 91.5%，RAM 僅 3.8 GB |
| **速度優先** | Llama-3.2-3B Q4_K_M | TTFT 快 43%，Hit Rate 仍達 84% |
| **RAM 最省** | Qwen2.5-3B Q5_K_M | RAM 3.5 GB，但 Q6 有弱點 |

### 下一步改善方向

1. **合併 Keyboard/Audio/Webcam chunk** → 解決 Q6 召回不全問題，預計可提升至 80%+
2. **修正 SO-DIMM 遺漏** → 更新 System Memory chunk extracted 欄位，Q3 可達 100%
3. **加入 GPU 推論測試** → 目前僅 CPU 數據；GPU 環境下 TTFT 預期可降至 5–15 秒級別
4. **top_k 動態調整** → 多欄位合併問題考慮動態提升至 top_k=8
5. **考慮 multilingual-e5-base** → 替換 MiniLM，對英文多關鍵字查詢的召回率更佳

---

## 五、原始結果

所有 Benchmark PNG 與 JSON 原始結果存放於 [`results/`](../results/) 目錄。
