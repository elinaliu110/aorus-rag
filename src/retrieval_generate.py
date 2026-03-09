"""
retrieval_generate_vllm.py
═══════════════════════════════════════════════════════
vLLM edition — replaces llama.cpp with the OpenAI-compatible HTTP API.
Includes:
  - Stage C-1: extract_product_filter / extract_key_filter
  - Stage C-2: retrieve / build_context
  - Stage C-3: load_llm  (creates a vLLM client instead of loading weights)
  - Stage C-4: generate_stream (streaming generation + TTFT/TPS evaluation)

Prerequisites:
  1. Start the vLLM server:
       vllm serve <your-model> --host 0.0.0.0 --port 8000
     Or with tensor parallelism (multi-GPU):
       vllm serve <your-model> --tensor-parallel-size 2 --port 8000

  2. Install the OpenAI SDK:
       pip install openai
"""

import time
from typing import Generator
from openai import OpenAI
from vector_index import VectorIndex

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

# Address of the vLLM server (default: localhost port 8000)
VLLM_BASE_URL = "http://localhost:8000/v1"

# Model name / HuggingFace model ID passed to `vllm serve`
DEFAULT_MODEL_NAME = "your-model-name"

# Mapping of short product IDs to full product names
PRODUCTS = {
    "BZH": "AORUS MASTER 16 BZH",
    "BYH": "AORUS MASTER 16 BYH",
    "BXH": "AORUS MASTER 16 BXH",
}

# When any of these keywords appear in the query, skip the product filter
# so that the retrieval covers all products (e.g. for comparison questions)
COMPARISON_KEYWORDS = {
    "比較", "差異", "差別", "哪個", "哪款", "推薦", "建議",
    "vs", "versus", "compare", "difference", "better", "best",
    "which", "recommend", "between", "should i", "choose",
}

# Maps query keywords (lowercase) to canonical spec-chunk section keys.
# Used to perform exact key-based retrieval when the user asks about a
# specific hardware attribute (e.g. "GPU", "battery", "SSD").
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
    Return a single product short ID (e.g. "BZH") if the query targets
    exactly one product; otherwise return None.

    Returns None when:
      - The query contains comparison keywords (user wants all products).
      - More than one product ID is found in the query.
      - No product ID is found (query is product-agnostic).
    """
    q_upper = query.upper()
    q_lower = query.lower()

    # Comparison queries should not be filtered to a single product
    if any(kw in q_lower for kw in COMPARISON_KEYWORDS):
        return None

    matched = [sid for sid in PRODUCTS if sid in q_upper]
    if len(matched) > 1:
        return None  # Multiple products mentioned — no filter

    return matched[0] if matched else None


def extract_key_filter(query: str) -> str | None:
    """
    Return a canonical spec-section key (e.g. "Video Graphics") if a
    recognised hardware keyword is present in the query; else return None.

    This enables exact key-based chunk lookup before falling back to
    vector similarity search.
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
    Retrieve the most relevant spec chunks for *query*.

    Strategy:
      1. Detect product and section-key filters from the query text.
      2. If a key filter matches, perform an exact key lookup first, then
         supplement with ALL-product summary chunks if no product filter.
      3. Otherwise, run vector similarity search (widening the search when
         no product filter is set, to cover all three SKUs).
      4. Always append ALL-product summary chunks when no product is
         targeted, so the LLM has cross-SKU context.
      5. Deduplicate and return the top *top_k* chunks.
    """
    product_filter = extract_product_filter(query)
    key_filter     = extract_key_filter(query)

    if key_filter:
        # Exact section match — fast path
        exact = index.get_by_key(key_filter, short_id=product_filter)
        if exact:
            supplement = index.get_by_short_id("ALL") if product_filter is None else []
            return _merge_unique(exact, supplement, top_k)

    # Widen search when no product is pinpointed (covers all SKUs)
    search_k = top_k if product_filter else top_k * 3
    results  = index.search(query, top_k=search_k, product_filter=product_filter)

    if product_filter is None:
        # Add summary chunks that span all products
        all_summary = index.get_by_short_id("ALL")
        results     = _merge_unique(results, all_summary, search_k)

    return _dedup(results)[:top_k]


def build_context(chunks: list[dict], max_tokens: int = 600) -> str:
    """
    Concatenate chunk texts into a single context string, stopping before
    the estimated token budget (*max_tokens*) is exceeded.

    Token count is estimated as len(text) × 0.7 (rough approximation for
    mixed Chinese/English content).
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
    """Remove duplicate chunks, preserving original order."""
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
    """
    Merge two chunk lists, deduplicate, and return at most *limit* items.
    Primary chunks are preferred (appear first after dedup).
    """
    return _dedup(primary + secondary)[:limit]


# ══════════════════════════════════════════════════════════
# STAGE C-3 — vLLM CLIENT SETUP
# ══════════════════════════════════════════════════════════

def load_llm(
    base_url:   str = VLLM_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """
    Create an OpenAI-compatible client pointed at the running vLLM server.

    Returns a dict with keys:
      - "client":     the OpenAI client instance
      - "model_name": the model identifier to pass in API calls

    Note: Unlike llama.cpp, no model weights are loaded in Python here.
    The model was already loaded when the vLLM server started.
    A lightweight ping (list models) is performed to confirm the server
    is reachable and the requested model is available.
    """
    print(f"[vLLM] Connecting to vLLM server at: {base_url}")
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",  # vLLM local server does not validate the API key
    )

    # Quick connectivity check: list available models
    try:
        available = [m.id for m in client.models.list().data]
        print(f"[vLLM] Server ready. Available models: {available}")
        if model_name not in available:
            print(f"[vLLM] WARNING: '{model_name}' not found in server model list.")
    except Exception as e:
        print(f"[vLLM] WARNING: Could not reach server — {e}")

    return {"client": client, "model_name": model_name}


# ══════════════════════════════════════════════════════════
# STAGE C-4 — STREAMING GENERATION + TTFT/TPS METRICS
# ══════════════════════════════════════════════════════════

def generate_stream(
    llm:            dict,           # dict returned by load_llm()
    query:          str,
    context:        str,
    max_new_tokens: int = 200,
) -> Generator[tuple[str, dict], None, None]:
    """
    Stream generated tokens from the vLLM server and yield latency metrics.

    Yields:
      (token_text, metrics_dict)

    For every token except the last:
      token_text is the decoded text fragment, metrics_dict is empty ({}).

    On the final yield:
      token_text is "" and metrics_dict contains:
        - ttft_ms      : Time To First Token in milliseconds
        - tps          : Tokens Per Second (generation throughput)
        - total_tokens : Total tokens generated
        - total_ms     : Wall-clock time for the full request in ms

    Differences from the llama.cpp version:
      - Uses the OpenAI Chat Completions API (supported by vLLM).
      - System/user roles are passed via the `messages` list instead of
        manually concatenated prompt strings.
      - Token text is read from chunk.choices[0].delta.content.
    """
    client:     OpenAI = llm["client"]
    model_name: str    = llm["model_name"]

    # Build the chat message list (system prompt + user query with context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional specifications expert for GIGABYTE AORUS MASTER 16 AM6H. "
                "Answer questions based on the provided specification data.\n"
                "If the question is in Chinese, answer in Traditional Chinese; "
                "if in English, answer in English.\n"
                "Only answer based on the provided data. If the answer is not in the data, state so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Specification Data:\n{context}\n\nQuestion: {query}",
        },
    ]

    t_start      = time.perf_counter()
    t_first      = None   # Timestamp of the first token received
    total_tokens = 0

    # Open a streaming chat completion request
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,          # Deterministic output
        stream=True,
        extra_body={
            # vLLM-specific parameter equivalent to llama.cpp's repeat_penalty
            "repetition_penalty": 1.3,
            # Stop sequences to prevent the model from generating role markers
            "stop": ["<|end|>", "<|user|>", "<|system|>"],
        },
    )

    # Yield each token as it arrives
    for chunk in stream:
        delta = chunk.choices[0].delta
        token_text = delta.content or ""
        if not token_text:
            continue  # Skip keep-alive or empty delta chunks
        if t_first is None:
            t_first = time.perf_counter()  # Record time of first token
        total_tokens += 1
        yield token_text, {}

    # Compute final performance metrics after the stream ends
    t_end    = time.perf_counter()
    ttft_ms  = (t_first - t_start) * 1000 if t_first else 0.0
    total_ms = (t_end - t_start) * 1000
    gen_ms   = (t_end - t_first) * 1000 if t_first else 1.0
    tps      = total_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

    # Final yield: empty token text + populated metrics dict
    yield "", {
        "ttft_ms":      round(ttft_ms, 1),
        "tps":          round(tps, 2),
        "total_tokens": total_tokens,
        "total_ms":     round(total_ms, 1),
    }
