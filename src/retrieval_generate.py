"""
retrieval_generate.py 
═══════════════════════════════════════════════════════
Includes:
  - Stage C-1: extract_product_filter / extract_key_filter
  - Stage C-2: retrieve / build_context
  - Stage C-3: load_llm
  - Stage C-4: generate_stream（streaming + TTFT/TPS evaluation）
"""

import json
import time
from typing import Generator
from vector_index import VectorIndex

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

# Change this to your actual GGUF model path
DEFAULT_MODEL_PATH = "models/your-model.gguf"   

PRODUCTS = {
    "BZH": "AORUS MASTER 16 BZH",
    "BYH": "AORUS MASTER 16 BYH",
    "BXH": "AORUS MASTER 16 BXH",
}

# Comparison keywords, do not apply product filter when these are present
COMPARISON_KEYWORDS = {
    "比較", "差異", "差別", "哪個", "哪款", "推薦", "建議",
    "vs", "versus", "compare", "difference", "better", "best",
    "which", "recommend", "between", "should i", "choose",
}

# Query keyword to chunk key mapping
# When matched, retrieve directly by key (exact match), skipping vector search
KEY_ALIASES: dict[str, str] = {
    "gpu": "Video Graphics", "顯卡": "Video Graphics", "顯示卡": "Video Graphics",
    "vram": "Video Graphics", "顯示記憶體": "Video Graphics",
    "cpu": "CPU", "處理器": "CPU", "晶片": "CPU",
    "螢幕": "Display", "display": "Display", "oled": "Display",
    "解析度": "Display", "dolby vision": "Display", "g-sync": "Display",
    "gsync": "Display", "hdr": "Display", "hz": "Display",
    "記憶體": "System Memory", "ram": "System Memory", "ddr": "System Memory",
    "儲存": "Storage", "ssd": "Storage", "m.2": "Storage", "nvme": "Storage",
    "電池": "Battery", "battery": "Battery", "續航": "Battery",
    "重量": "Weight", "weight": "Weight", "幾公斤": "Weight",
    "尺寸": "Dimensions (W x D x H)", "dimensions": "Dimensions (W x D x H)",
    "連接埠": "I/O Port", "port": "I/O Port", "usb": "I/O Port",
    "thunderbolt": "I/O Port", "hdmi": "I/O Port", "type-c": "I/O Port",
    "鍵盤": "Keyboard Type", "keyboard": "Keyboard Type",
    "音訊": "Audio", "speaker": "Audio", "喇叭": "Audio",
    "wifi": "Communications", "藍牙": "Communications", "bluetooth": "Communications",
    "wireless": "Communications", "connectivity": "Communications",
    "lan": "Communications", "network": "Communications", "無線": "Communications",
    "攝影機": "Webcam", "webcam": "Webcam", "camera": "Webcam",
    "os": "OS", "作業系統": "OS", "windows": "OS",
    "安全": "Security", "tpm": "Security",
    "變壓器": "Adapter", "adapter": "Adapter", "充電": "Adapter",
    "顏色": "Color", "color": "Color",
}


# ══════════════════════════════════════════════════════════
# STAGE C-1 — FILTER EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_product_filter(query: str) -> str | None:
    """
    Extracts product filter from the query.
    Returns None (no filter) in the following cases:
      - Comparison keywords detected (e.g., "vs", "compare")
      - Query mentions more than one model simultaneously
    """
    q_upper = query.upper()
    q_lower = query.lower()

    if any(kw in q_lower for kw in COMPARISON_KEYWORDS):
        return None

    matched = [sid for sid in PRODUCTS if sid in q_upper]
    if len(matched) > 1:
        return None

    return matched[0] if matched else None


def extract_key_filter(query: str) -> str | None:
    """
    Extracts specification key filter from the query.
    If matched, allows direct retrieval by key, skipping vector search.
    """
    q_lower = query.lower()
    for alias, key in KEY_ALIASES.items():
        if alias in q_lower:
            return key
    return None


# ══════════════════════════════════════════════════════════
# STAGE C-2 — RETRIEVAL
# ══════════════════════════════════════════════════════════

def retrieve(index: VectorIndex, query: str, top_k: int = 5) -> list[dict]:
    """
    Optimized retrieval logic:
      1. Extract product_filter and key_filter.
      2. If key_filter hits -> Direct exact retrieval (skips encoding/vector search).
      3. Perform vector search with metadata filtering.
      4. Supplement with "ALL" summary chunks if no product_filter is present.
      5. Deduplicate and return results.
    """
    product_filter = extract_product_filter(query)
    key_filter     = extract_key_filter(query)

    # -- Key filter exact match path --------------------------
    if key_filter:
        exact = index.get_by_key(key_filter, short_id=product_filter)
        if exact:
            # Supplement with "ALL" summary if no specific product is filtered
            supplement = index.get_by_short_id("ALL") if product_filter is None else []
            return _merge_unique(exact, supplement, top_k)

    # -- Vector search path -----------------------------------
    # If no product filter, triple top_k to ensure chunks for all three models are captured
    search_k = top_k if product_filter else top_k * 3
    results  = index.search(query, top_k=search_k, product_filter=product_filter)

    # Supplement with "ALL" summary if no specific product is filtered
    if product_filter is None:
        all_summary = index.get_by_short_id("ALL")
        results     = _merge_unique(results, all_summary, search_k)

    return _dedup(results)[:top_k]


def build_context(chunks: list[dict], max_tokens: int = 1400) -> str:
    """
    Combines chunks into a context string with token budget control to manage TTFT.
    Note: Bilingual chunks (TW/EN) are roughly twice as long as pure Chinese.
    Max_tokens default is adjusted from 800 to 1400.
    Estimation: Mixed CH/EN ~0.7 tokens per character.
    """
    lines, total = [], 0
    for c in chunks:
        est = int(len(c["text"]) * 0.7)
        if total + est > max_tokens:
            break
        lines.append(c["text"])
        total += est
    return "\n".join(lines)


def _dedup(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    return unique


def _merge_unique(
    primary:   list[dict],
    secondary: list[dict],
    limit:     int,
) -> list[dict]:
    return _dedup(primary + secondary)[:limit]


# ══════════════════════════════════════════════════════════
# STAGE C-3 — LLM LOADING
# ══════════════════════════════════════════════════════════

def load_llm(model_path: str = DEFAULT_MODEL_PATH):
    from llama_cpp import Llama
    print(f"[LLM] Loading llama.cpp model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("[LLM] Model loaded successfully.")
    return llm


# ══════════════════════════════════════════════════════════
# STAGE C-4 — STREAMING GENERATION + TTFT/TPS METRICS
# ══════════════════════════════════════════════════════════

def generate_stream(
    llm,
    query:          str,
    context:        str,
    max_new_tokens: int = 200,
) -> Generator[tuple[str, dict], None, None]:
    """
    Yields (token_text, metrics_dict).
    metrics_dict is only populated on the final yield:
      {"ttft_ms": float, "tps": float, "total_tokens": int, "total_ms": float}
    """
    prompt = (
        "<|system|>\n"
        "You are a professional specifications expert for GIGABYTE AORUS MASTER 16 AM6H. "
        "Answer questions based on the provided specification data.\n"
        "If the question is in Chinese, answer in Traditional Chinese; "
        "if in English, answer in English.\n"
        "Only answer based on the provided data. If the answer is not in the data, state so clearly.\n<|end|>\n"
        f"<|user|>\nSpecification Data:\n{context}\n\nQuestion: {query}<|end|>\n"
        "<|assistant|>\n"
    )

    t_start      = time.perf_counter()
    t_first      = None
    total_tokens = 0

    stream = llm(
        prompt,
        max_tokens=max_new_tokens,
        stream=True,
        temperature=0.0,
        repeat_penalty=1.3,
        stop=["<|end|>", "<|user|>", "<|system|>", "\n\n\n"],  # Immediate stop triggers
    )

    for chunk in stream:
        token_text = chunk["choices"][0]["text"]
        if not token_text:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        total_tokens += 1
        yield token_text, {}

    t_end    = time.perf_counter()
    ttft_ms  = (t_first - t_start) * 1000 if t_first else 0.0
    total_ms = (t_end - t_start) * 1000
    gen_ms   = (t_end - t_first) * 1000 if t_first else 1.0
    tps      = total_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

    yield "", {
        "ttft_ms":      round(ttft_ms, 1),
        "tps":          round(tps, 2),
        "total_tokens": total_tokens,
        "total_ms":     round(total_ms, 1),
    }