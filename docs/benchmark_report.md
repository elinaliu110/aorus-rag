# Benchmark Evaluation Report (vLLM GPU Edition)

> GPU & Colab environment · **6 models** × 10 queries · benchmark_cases.json
> Covers: Llama-3.2-3B AWQ / Llama-3.2-1B AWQ / Qwen2.5-3B AWQ / Qwen2.5-1.5B AWQ

---

## Evaluation Environment

| Item | Details |
|------|---------|
| Inference Engine | vLLM (AWQ INT4 quantization) |
| GPU | Tesla T4 (Google Colab) |
| Total VRAM | 15,360 MB |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Chunk Version | chunks.json (bilingual, 55 chunks) |
| Benchmark Cases | benchmark_cases.json (10 queries) |
| Metrics | Keyword Hit Rate / TTFT / TPS / VRAM |

---

## Models Tested

| Model Family | Parameters | Quantization | HuggingFace ID | 4GB Ready |
|-------------|:----------:|:------------:|----------------|:---------:|
| Llama-3.2-Instruct (AMead10) | **3B** | AWQ INT4 | `AMead10/Llama-3.2-3B-Instruct-AWQ` | ✅ |
| Llama-3.2-Instruct (casper) | **3B** | AWQ INT4 | `casperhansen/llama-3.2-3b-instruct-awq` | ❌ |
| Qwen2.5-Instruct | **3B** | AWQ INT4 | `Qwen/Qwen2.5-3B-Instruct-AWQ` | ✅ |
| Llama-3.2-Instruct (AMead10) | **1B** | AWQ INT4 | `AMead10/Llama-3.2-1B-Instruct-AWQ` | ✅ |
| Llama-3.2-Instruct (casper) | **1B** | AWQ INT4 | `casperhansen/llama-3.2-1b-instruct-awq` | ✅ |
| Qwen2.5-Instruct | **1.5B** | AWQ INT4 | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | ✅ |

> casperhansen 3B: Model VRAM Δ = 3,540 MB; total VRAM during inference = 4,107 MB, exceeding the 4,096 MB threshold by 11 MB.

---

## 1. Overall Results (All 6 Models)

| Model | Params | Hit Rate | TTFT (avg) | TPS | Model VRAM Δ | 4GB | shared_spec | single_product | gpu_comparison |
|-------|:------:|:--------:|:----------:|:---:|:------------:|:---:|:-----------:|:--------------:|:--------------:|
| **AMead10 Llama AWQ** | **3B** | **84.0%** | 269 ms | 36.1 | 3,492 MB | ✅ | 77.5% | **100%** | **87.5%** |
| casperhansen Llama AWQ | 3B | 81.5% | 241 ms | 37.9 | 3,540 MB | ❌ | 73.3% | **100%** | **87.5%** |
| Qwen2.5 AWQ | 3B | 58.2% | 274 ms | 31.0 | 3,468 MB | ✅ | 51.1% | 75.0% | 62.5% |
| Qwen2.5 AWQ | 1.5B | 53.7% | 202 ms | 36.5 | 3,462 MB | ✅ | 60.3% | 62.5% | 25.0% |
| casperhansen Llama AWQ | 1B | 40.3% | **132 ms** | 56.6 | **3,398 MB** | ✅ | 39.4% | 70.8% | 12.5% |
| AMead10 Llama AWQ | 1B | 32.0% | 133 ms | **57.4** | 3,446 MB | ✅ | 39.4% | 29.2% | 12.5% |

---

## 2. Per-Query Hit Rate Breakdown

| Q# | Type | Query Summary | AMead10 3B | casper 3B | Qwen 3B | Qwen 1.5B | casper 1B | AMead10 1B |
|----|------|--------------|:----------:|:---------:|:-------:|:---------:|:---------:|:---------:|
| Q1 | shared_spec | Supported OS (ZH) | 100% | 100% | 100% | 100% | 100% | 100% |
| Q2 | single_product | BYH GDDR7 / AI Boost (ZH) | 100% | 100% | 100% | 100% | 67% | 33% |
| Q3 | shared_spec | Max RAM / SO-DIMM (ZH) | 75% | 75% | 75% | **100%** | 25% | 25% |
| Q4 | shared_spec | Thunderbolt ports (ZH) | 100% | 100% | 67% | 67% | 67% | 67% |
| Q5 | gpu_comparison | BZH vs BXH TDP (ZH) | 75% | 75% | 75% | 50% | 25% | 25% |
| Q6 | shared_spec | Keyboard / Audio / Webcam (EN) | 40% | 40% | 40% | 20% | 20% | 20% |
| Q7 | shared_spec | Display panel & refresh rate (EN) | 50% | 25% | 25% | 25% | 25% | 25% |
| Q8 | shared_spec | Battery capacity (EN) | 100% | 100% | **0%** | 50% | **0%** | **0%** |
| Q9 | single_product | BXH storage options (EN) | 100% | 100% | 50% | 25% | 75% | 25% |
| Q10 | gpu_comparison | BYH vs BXH gaming recommendation (EN) | 100% | 100% | 50% | **0%** | **0%** | **0%** |

---

## 3. Observations & Analysis

### 3.1 Best Model: AMead10/Llama-3.2-3B-Instruct-AWQ (84.0%)

Hit Rate **84.0%** — highest among all six models, and the only one satisfying both "accuracy ≥ 80%" and "4GB VRAM compliant":

- `single_product` (Q2, Q9): **100%** — single-model spec extraction fully correct
- `gpu_comparison` (Q5, Q10): **87.5%** — Q5 misses `140W` keyword (model outputs `140 W` with a space; keyword format issue)
- `shared_spec` (Q1–Q8): **77.5%** — weaknesses concentrated on Q6 (multi-field EN) and Q7 (display resolution numbers)

Q4 Thunderbolt achieves **TTFT 128 ms** — the lowest single-query TTFT across all 60 measurements (6 models × 10 queries), confirming the key filter exact-retrieval path is fully effective.

---

### 3.2 Fastest but VRAM Non-Compliant: casperhansen 3B (81.5%)

Average TTFT **241 ms**, TPS **37.9 tok/s** — fastest in the 3B group. However:

- Model VRAM Δ **3,540 MB**; total VRAM during inference **4,107 MB**, exceeding limit by **11 MB**
- Marked 4GB non-compliant
- Hit Rate 81.5% — 2.5 pp below AMead10 3B
- Q7 display only hits OLED (25%), missing 240Hz, 2560, 1600

The margin is very small (11 MB). On GPUs with ≥ 6 GB VRAM (e.g. RTX 4060), this model remains a valid speed-priority option.

---

### 3.3 Qwen Scaling: 3B vs 1.5B Trade-offs

| Metric | Qwen 3B | Qwen 1.5B | Delta |
|--------|:-------:|:---------:|:-----:|
| Hit Rate | 58.2% | 53.7% | −4.5% |
| TTFT | 274 ms | 202 ms | −27% (faster) |
| TPS | 31.0 | 36.5 | +18% (faster) |
| gpu_comparison | 62.5% | **25.0%** | **−37.5%** |
| Q10 Gaming Rec. | 50% | **0%** | Complete failure |
| Q3 SO-DIMM (ZH) | 75% | **100%** | Only 100% in Q3 across all models |

Qwen 1.5B is meaningfully faster, but `gpu_comparison` degrades dramatically. Q10 completely fails: the model describes "RTX GeForce GTX 30 series (discontinued in 2012)" — a full hallucination with no grounding in the retrieved context, demonstrating that 1.5B models lack the capacity for complex multi-model comparison reasoning.

Q3 RAM is a highlight for Qwen 1.5B (**100%** — the only model to fully hit SO-DIMM across all six), showing that small models can still perform well on straightforward Chinese spec queries.

---

### 3.4 Llama-3.2-1B AWQ Models: Extremely Fast, Low Accuracy

Both 1B models exceed **56 tok/s** TPS with avg TTFT ~**132 ms** — approximately 2× faster than the 3B AMead10 in TTFT. However, accuracy is severely inadequate:

**Universally failed queries (both 1B models scored 0%):**

| Query | casperhansen 1B failure | AMead10 1B failure |
|-------|------------------------|-------------------|
| Q8 Battery | Hallucinates "98V ±10% battery voltage formula" | Generates power consumption formula derivation, entirely detached from context |
| Q10 Gaming Rec. | Describes "GTX 30 series (discontinued June 2012)" | Compares "cooling system fan configurations" between models |

**casperhansen 1B vs AMead10 1B differences:**
- casperhansen 1B `single_product`: **70.8%** vs AMead10 1B: 29.2% — substantially better
- AMead10 1B `gpu_comparison`: **12.5%** — lowest across all six models
- Both score identically on `shared_spec` (39.4%); divergence is in English instruction-following stability

---

### 3.5 Universal Weakness: Q6 Multi-Field English (All Models ≤ 40%)

Q6 (*"What are the keyboard, audio, and webcam specs of the AORUS MASTER 16?"*) is the **common weak point across all models**:

All six models score at most 40% on Q6. Root cause is the **retrieval architecture**: three separate chunks for three separate fields, and top_k=5 cannot retrieve all three simultaneously. This is independent of model size.

1B models score lower (20%) because even when the Keyboard chunk is retrieved, they cannot reliably extract the N-Key keyword from English context.

---

### 3.6 TTFT Patterns Across Models

| Pattern | Observation |
|---------|------------|
| 1B models are consistently fastest | avg TTFT ~132 ms, ~2× faster than 3B AMead10 (269 ms) |
| Qwen 1.5B TTFT highest in small model group | 202 ms, Q1 peaks at 434 ms — Qwen tokenizer less efficient |
| Q4 Thunderbolt — minimum TTFT per model | 1B models: 83–127 ms; key filter bypass clearly effective |
| Q8 Battery — near-minimum TTFT per model | 59–88 ms; short questions produce fast prefill |
| Q1 OS — maximum TTFT per model | 204–462 ms; long context causes longer prefill |

---


---

## 4. Conclusions & Recommendations

### Model Selection Guide

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| **GPU accuracy-first** (default) | AMead10 Llama-3.2-3B AWQ | 84% hit rate, 4GB compliant, 100% on single_product |
| **GPU speed-first (VRAM headroom)** | casperhansen Llama-3.2-3B AWQ | TTFT 241ms / TPS 37.9, 11MB over VRAM limit |
| **Minimum VRAM (lowest acceptable accuracy)** | Qwen2.5-1.5B AWQ | 3,462 MB|
| ❌ **Not for production** | Any 1B model | 32–40% accuracy|

### Next Steps for Improvement

1. **Data Architecture**: Optimize chunk.json schemas for enhanced bilingual (CN/EN) semantic alignment.

2. **Model Precision**: Upgrade to AWQ 8-bit/GPTQ quantization to target a 5–10% accuracy gain.

3. **Retrieval Tuning**: Implement Hybrid Search and re-ranking to maximize document relevance.

4. **Query Adaptation**: Deploy Dynamic top_k=8 logic for complex, multi-field English queries.

5. **Embedding Evaluation**: Benchmark MTEB-leaderboard models to identify the optimal semantic fit.

---

## 5. Raw Data

All benchmark JSON files and PNG charts are available in the [`results/vllm-gpu/`](../results/vllm-gpu/) directory.
