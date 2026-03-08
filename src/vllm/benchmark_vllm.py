"""
benchmark_vllm.py
══════════════════════════════════════════════════
vLLM edition — replaces llama.cpp with the OpenAI-compatible HTTP API.
Supports benchmarking a single model or multiple models sequentially,
recording VRAM usage for each model run.

Quantitative Metrics:
  - Keyword Hit Rate
  - TTFT (Time To First Token)
  - TPS (Tokens Per Second)
  - [GPU] VRAM Load Delta / VRAM Peak per query
  - [CPU] Peak RAM Delta / CPU Avg % / CPU Max %

Prerequisites:
  pip install openai psutil matplotlib requests

Note: This script automatically starts and stops the vLLM server —
      no manual server management is required.

Execution:
    # Single model
    python benchmark_vllm.py \\
        --models Qwen/Qwen2.5-1.5B-Instruct

    # Multiple models in one run (comma-separated)
    python benchmark_vllm.py \\
        --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,meta-llama/Llama-3.2-1B-Instruct

    # Full parameter specification
    python benchmark_vllm.py \\
        --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct \\
        --base-url http://localhost:8000/v1 \\
        --cases data/benchmark_cases.json \\
        --chunks data/chunks.json \\
        --emb data/embeddings.npy \\
        --out-dir results \\
        --gpu-util 0.85 \\
        --max-model-len 4096

    # Simulate a 4 GB VRAM environment (T4 15 GB × 0.26 ≈ 4 GB)
    python benchmark_vllm.py \\
        --models Qwen/Qwen2.5-3B-Instruct \\
        --gpu-util 0.26
"""

import json
import os

# ── Colab matplotlib backend fix ──────────────────────────────
# Colab sets MPLBACKEND to matplotlib_inline.backend_inline by default,
# which is incompatible with subprocess / script mode.
# Override it before importing matplotlib.
os.environ["MPLBACKEND"] = "Agg"

import argparse
import subprocess
import threading
import time
import psutil

from vector_index import VectorIndex
from retrieval_generate_vllm import (
    retrieve,
    build_context,
    generate_stream,
    load_llm,
    VLLM_BASE_URL,
    DEFAULT_MODEL_NAME,
)

DEFAULT_CASES_PATH = "data/benchmark_cases.json"
DEFAULT_OUT_DIR    = "results"
VLLM_PORT          = 8000


# ══════════════════════════════════════════════════════════
# VRAM UTILS
# ══════════════════════════════════════════════════════════

def get_vram_usage() -> dict:
    """
    Query current GPU VRAM usage via nvidia-smi.
    Returns a dict with keys: gpu_name, used_mb, total_mb, free_mb.
    Returns an empty dict if nvidia-smi is unavailable or fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return {}
        return {
            "gpu_name": parts[0],
            "used_mb":  int(parts[1]),
            "total_mb": int(parts[2]),
            "free_mb":  int(parts[3]),
        }
    except Exception:
        return {}


def detect_device(vram_info: dict) -> dict:
    """
    Determine whether the model is running on GPU or CPU based on VRAM
    usage reported after the model has been loaded.

    Args:
        vram_info: Output of get_vram_usage() taken after model load.

    Returns a dict describing the active device and its memory stats.
    The "within_4gb" flag indicates whether VRAM usage fits within 4 GB,
    which is relevant for consumer-grade GPU compatibility checks.
    """
    if vram_info and vram_info.get("used_mb", 0) > 100:
        return {
            "device":        "GPU",
            "gpu_name":      vram_info["gpu_name"],
            "vram_total_mb": vram_info["total_mb"],
            "vram_used_mb":  vram_info["used_mb"],
            "within_4gb":    vram_info["used_mb"] <= 4096,
        }
    return {"device": "CPU"}


# ══════════════════════════════════════════════════════════
# vLLM SERVER LIFECYCLE
# ══════════════════════════════════════════════════════════

def start_vllm_server(
    model_id:      str,
    port:          int   = VLLM_PORT,
    gpu_util:      float = 0.85,
    max_model_len: int   = 4096,
    quantization:  str | None = None,
    enforce_eager: bool  = False,
) -> subprocess.Popen | None:
    """
    Launch a vLLM OpenAI-compatible API server and block until it is ready.

    The function reads server stdout line by line and returns only after
    "Application startup complete" is detected.  If a fatal error keyword
    is found, the process is killed and None is returned.

    Args:
        model_id:      HuggingFace model ID or local model path.
        port:          TCP port for the server to listen on.
        gpu_util:      Fraction of GPU VRAM vLLM is allowed to use (0–1).
        max_model_len: Maximum sequence length (context window) in tokens.
        quantization:  Optional quantization method (e.g. "awq", "gptq").
        enforce_eager: Disable FlashAttention2 and CUDA Graphs.
                       Required for T4 GPUs (compute capability 7.5);
                       FA2 requires compute capability ≥ 8.0 (A100/H100).

    Returns:
        A Popen handle on success, or None if startup failed.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_id,
        "--host",                   "0.0.0.0",
        "--port",                   str(port),
        "--max-model-len",          str(max_model_len),
        "--dtype",                  "half",
        "--gpu-memory-utilization", str(gpu_util),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if enforce_eager:
        cmd += ["--enforce-eager"]

    # T4 (compute capability 7.5) does not support FA2;
    # fall back to the xformers attention backend instead.
    env = os.environ.copy()
    if enforce_eager:
        env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
        print(f"\n[vLLM] Starting server: {model_id}")
        print(f"[vLLM] T4 mode: enforce_eager=True, VLLM_ATTENTION_BACKEND=XFORMERS")
    else:
        print(f"\n[vLLM] Starting server: {model_id}")

    # Route stdout to PIPE only for the startup phase.
    # The pipe is closed once the ready signal is received to avoid
    # blocking the parent process when the server eventually exits.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # FA2-incompatibility warnings are non-fatal; only watch for truly
    # unrecoverable errors that prevent the server from starting.
    FATAL_KEYWORDS = ("killed", "cuda out of memory", "runtimeerror",
                      "no space left", "segmentation fault")

    server_ready = False
    try:
        for line in proc.stdout:
            print(f"  [log] {line}", end="")
            if "Application startup complete" in line:
                server_ready = True
                break  # Stop reading; server continues running in background
            if any(kw in line.lower() for kw in FATAL_KEYWORDS):
                print(f"[vLLM] ❌ Startup failed: {model_id}\n")
                proc.kill()
                return None
    finally:
        # Close the pipe so the parent is not blocked when the server exits
        proc.stdout.close()

    if server_ready:
        print(f"[vLLM] ✅ Server ready: {model_id}\n")
        return proc

    print(f"[vLLM] ❌ Process exited unexpectedly: {model_id}\n")
    proc.kill()
    return None


def stop_vllm_server(proc: subprocess.Popen, wait_sec: int = 8) -> None:
    """
    Gracefully terminate the vLLM server and wait for VRAM to be released.

    Sends SIGTERM first to allow vLLM to shut down cleanly; escalates to
    SIGKILL if the process does not exit within 10 seconds.
    An additional sleep of *wait_sec* seconds is added afterwards to ensure
    the GPU memory is fully freed before the next model is loaded.
    """
    if proc and proc.poll() is None:
        proc.terminate()               # Graceful shutdown via SIGTERM
        try:
            proc.wait(timeout=10)      # Wait up to 10 s for clean exit
        except subprocess.TimeoutExpired:
            proc.kill()                # Force kill if still running
            proc.wait()
    time.sleep(wait_sec)               # Allow VRAM to be reclaimed by the OS
    print("[vLLM] 🛑 Server stopped, VRAM released.\n")


# ══════════════════════════════════════════════════════════
# CPU RESOURCE MONITOR
# ══════════════════════════════════════════════════════════

class CPUMonitor:
    """
    Background thread that samples CPU utilisation and process RSS memory
    at regular intervals while used as a context manager.

    Usage:
        with CPUMonitor() as monitor:
            do_work()
        stats = monitor.stats()

    stats() returns:
        ram_delta_mb : RSS increase from entry to peak (MB)
        ram_peak_mb  : Peak RSS observed during the window (MB)
        cpu_avg_pct  : Average system-wide CPU utilisation (%)
        cpu_max_pct  : Maximum system-wide CPU utilisation (%)
    """

    def __init__(self, interval: float = 0.1):
        self._interval  = interval          # Sampling period in seconds
        self._process   = psutil.Process(os.getpid())
        self._cpu_samples: list[float] = []
        self._ram_before  = 0.0             # RSS at context entry (MB)
        self._ram_peak    = 0.0             # Highest RSS seen (MB)
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        """Sampling loop executed on the background daemon thread."""
        while not self._stop.is_set():
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            rss = self._process.memory_info().rss / 1024 ** 2
            if rss > self._ram_peak:
                self._ram_peak = rss
            time.sleep(self._interval)

    def __enter__(self):
        self._ram_before = self._process.memory_info().rss / 1024 ** 2
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)

    def stats(self) -> dict:
        samples = self._cpu_samples or [0.0]
        return {
            "ram_delta_mb": round(self._ram_peak - self._ram_before, 1),
            "ram_peak_mb":  round(self._ram_peak, 1),
            "cpu_avg_pct":  round(sum(samples) / len(samples), 1),
            "cpu_max_pct":  round(max(samples), 1),
        }


# ══════════════════════════════════════════════════════════
# CASES LOADER
# ══════════════════════════════════════════════════════════

def load_cases(path: str) -> list[tuple[str, list[str], str]]:
    """
    Load benchmark test cases from a JSON file.

    Expected JSON schema (list of objects):
        [{"query": str, "keywords": [str, ...], "type": str}, ...]

    Returns a list of (query, keywords, type) tuples.
    Entries with empty queries are skipped with a warning.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] Benchmark cases file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for i, item in enumerate(raw):
        query    = item.get("query", "").strip()
        keywords = item.get("keywords", [])
        qtype    = item.get("type", "unknown")
        if not query:
            print(f"[Warning] Question {i+1} has empty query, skipping")
            continue
        cases.append((query, keywords, qtype))
    print(f"[✓] Loaded {len(cases)} cases from {path}")
    return cases


# ══════════════════════════════════════════════════════════
# BENCHMARK RUNNER (single model)
# ══════════════════════════════════════════════════════════

def run_benchmark(
    index:        VectorIndex,
    llm:          dict,
    cases:        list[tuple[str, list[str], str]],
    device_info:  dict,
    model_name:   str,
    vram_idle_mb: int,
    save_path:    str,
) -> dict:
    """
    Run a full RAG benchmark for a single model and save results to disk.

    For each test case:
      1. Retrieve relevant chunks from the vector index.
      2. Build a context string and call the LLM via generate_stream().
      3. Measure keyword hit rate, TTFT, TPS, and resource usage.

    After all cases:
      - Print a summary table (overall + per query-type averages).
      - Save detailed results to *save_path* (JSON).
      - Generate and save performance charts (PNG).

    Returns:
        A summary dict suitable for multi-model comparison aggregation.
    """
    is_gpu = device_info.get("device") == "GPU"

    print("\n" + "═" * 65)
    print("  RAG BENCHMARK  [vLLM]")
    print("═" * 65)
    print(f"  Model          : {model_name}")
    print(f"  Device         : {device_info.get('device', '?')}")

    if is_gpu:
        # Subtract idle VRAM to isolate the model's actual memory footprint
        model_vram_mb = device_info.get("vram_used_mb", 0) - vram_idle_mb
        print(f"  GPU            : {device_info.get('gpu_name', '?')}")
        print(f"  VRAM (idle)    : {vram_idle_mb} MB")
        print(f"  VRAM (loaded)  : {device_info.get('vram_used_mb', '?')} MB")
        print(f"  VRAM (model Δ) : {model_vram_mb} MB")
        print(f"  VRAM Total     : {device_info.get('vram_total_mb', '?')} MB")
        print(f"  4GB Ready      : {'✅ Yes' if device_info.get('within_4gb') else '❌ Exceeds 4GB'}")
    else:
        proc = psutil.Process(os.getpid())
        model_vram_mb = 0
        print(f"  RAM Usage      : {proc.memory_info().rss / 1024**2:.0f} MB")
        print(f"  CPU Cores      : {psutil.cpu_count(logical=True)} logical cores")

    results = []

    for i, (query, expected_kw, qtype) in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] [{qtype}] {query}")
        print("─" * 55)

        # RAG: retrieve context and stream the generated answer
        retrieved = retrieve(index, query)
        context   = build_context(retrieved)

        answer_parts: list[str] = []
        gen_metrics:  dict      = {}

        print("Response: ", end="", flush=True)

        if is_gpu:
            # Stream generation; snapshot VRAM after each query
            for token, m in generate_stream(llm, query, context):
                if token:
                    print(token, end="", flush=True)
                    answer_parts.append(token)
                if m:
                    gen_metrics = m
            vram_now         = get_vram_usage()
            resource_metrics: dict = {
                "vram_used_mb":  vram_now.get("used_mb", 0),
                "vram_model_mb": model_vram_mb,
            }
        else:
            # Stream generation while monitoring CPU and RAM
            with CPUMonitor() as monitor:
                for token, m in generate_stream(llm, query, context):
                    if token:
                        print(token, end="", flush=True)
                        answer_parts.append(token)
                    if m:
                        gen_metrics = m
            resource_metrics = monitor.stats()

        print()

        # Evaluate keyword hit rate for this query
        answer   = "".join(answer_parts)
        hit_kws  = [kw for kw in expected_kw if kw.lower() in answer.lower()]
        hit_rate = len(hit_kws) / len(expected_kw) if expected_kw else 0.0

        print(f"  Keywords : {hit_kws} / {expected_kw}  hit={hit_rate:.0%}")
        print(f"  TTFT={gen_metrics.get('ttft_ms','?')} ms  "
              f"TPS={gen_metrics.get('tps','?')} tok/s  "
              f"Total={gen_metrics.get('total_ms','?')} ms  "
              f"Tokens={gen_metrics.get('total_tokens','?')}")

        if is_gpu:
            print(f"  VRAM (query peak) : {resource_metrics.get('vram_used_mb','?')} MB  "
                  f"| Model Δ : {model_vram_mb} MB")
        else:
            print(f"  RAM Δ={resource_metrics.get('ram_delta_mb','?')} MB  "
                  f"RAM Peak={resource_metrics.get('ram_peak_mb','?')} MB  "
                  f"CPU avg={resource_metrics.get('cpu_avg_pct','?')}%  "
                  f"CPU max={resource_metrics.get('cpu_max_pct','?')}%")

        results.append({
            "id":           i,
            "query":        query,
            "query_type":   qtype,
            "answer":       answer,
            "expected_kw":  expected_kw,
            "hit_keywords": hit_kws,
            "hit_rate":     hit_rate,
            **gen_metrics,
            **resource_metrics,
        })

    # ── Aggregate summary ─────────────────────────────────
    avg_hit  = sum(r["hit_rate"] for r in results) / len(results)
    avg_ttft = sum(r.get("ttft_ms", 0) for r in results) / len(results)
    avg_tps  = sum(r.get("tps", 0) for r in results) / len(results)

    print("\n" + "═" * 65)
    print("  BENCHMARK SUMMARY  [vLLM]")
    print("═" * 65)
    print(f"  Model                : {model_name}")
    print(f"  Device               : {device_info.get('device')}")
    print(f"  Avg Keyword Hit Rate : {avg_hit:.1%}")
    print(f"  Avg TTFT             : {avg_ttft:.0f} ms")
    print(f"  Avg TPS              : {avg_tps:.1f} tok/s")

    if is_gpu:
        avg_vram = sum(r.get("vram_used_mb", 0) for r in results) / len(results)
        print(f"  Model VRAM Δ         : {model_vram_mb} MB")
        print(f"  Avg Query VRAM       : {avg_vram:.0f} MB")
        print(f"  4GB Ready            : {'✅ Yes' if device_info.get('within_4gb') else '❌ Exceeds 4GB'}")
    else:
        avg_ram = sum(r.get("ram_delta_mb", 0) for r in results) / len(results)
        avg_cpu = sum(r.get("cpu_avg_pct", 0) for r in results) / len(results)
        print(f"  Avg RAM Delta        : {avg_ram:.0f} MB")
        print(f"  Avg CPU Utilization  : {avg_cpu:.1f} %")

    # Break down hit rate by query type
    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r["query_type"], []).append(r["hit_rate"])
    print()
    for qtype, rates in by_type.items():
        print(f"  [{qtype:18s}]  hit={sum(rates)/len(rates):.1%}  n={len(rates)}")
    print("═" * 65)

    summary = {
        "model_name":    model_name,
        "device":        device_info.get("device"),
        "avg_hit_rate":  round(avg_hit, 4),
        "avg_ttft_ms":   round(avg_ttft, 1),
        "avg_tps":       round(avg_tps, 2),
        "vram_model_mb": model_vram_mb,
        "within_4gb":    device_info.get("within_4gb", False),
        **device_info,
    }

    # Persist detailed results as JSON
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"[Benchmark] Results saved → {save_path}")

    # Generate and save per-model performance charts
    chart_path = save_path.replace(".json", ".png")
    _save_charts(results, avg_hit, avg_ttft, avg_tps, model_name, device_info, chart_path)

    return summary


# ══════════════════════════════════════════════════════════
# MULTI-MODEL COMPARISON CHART
# ══════════════════════════════════════════════════════════

def _save_comparison_chart(summaries: list[dict], save_path: str) -> None:
    """
    Generate a side-by-side bar chart comparing all benchmarked models.

    Panels (top to bottom):
      1. Avg Keyword Hit Rate (%)
      2. Avg TTFT (ms)
      3. Avg TPS (tok/s)
      4. Model VRAM Usage (MB) — GPU runs only, with a 4 GB reference line
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Chart] matplotlib not installed, skipping.")
        return

    # Extract values for each model
    labels = [s["model_name"].split("/")[-1] for s in summaries]
    hits   = [s["avg_hit_rate"] * 100 for s in summaries]
    ttfts  = [s["avg_ttft_ms"] for s in summaries]
    tpss   = [s["avg_tps"] for s in summaries]
    vrams  = [s.get("vram_model_mb", 0) for s in summaries]
    is_gpu = any(s.get("device") == "GPU" for s in summaries)

    nrows  = 4 if is_gpu else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(max(8, len(labels) * 2), 4 * nrows))
    fig.suptitle("Multi-Model Benchmark Comparison [vLLM]",
                 fontsize=13, fontweight="bold")

    palette = ["#4C9BE8", "#E8834C", "#5DBE7A", "#9B59B6", "#E84C4C"]

    def bar_chart(ax, values, title, ylabel, hline=None, hline_label=None):
        """Helper: draw a labelled bar chart with an optional horizontal reference line."""
        bars = ax.bar(labels, values,
                      color=[palette[i % len(palette)] for i in range(len(labels))],
                      edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", linewidth=1.2,
                       label=hline_label or str(hline))
            ax.legend(fontsize=8)
        # Annotate each bar with its numeric value
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    bar_chart(axes[0], hits,  "Avg Keyword Hit Rate (%)", "Hit Rate (%)")
    bar_chart(axes[1], ttfts, "Avg TTFT (ms)",            "TTFT (ms)")
    bar_chart(axes[2], tpss,  "Avg TPS (tok/s)",          "TPS")
    if is_gpu:
        bar_chart(axes[3], vrams, "Model VRAM Usage (MB)", "VRAM (MB)",
                  hline=4096, hline_label="4GB limit")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Benchmark] Comparison chart saved → {save_path}")


# ══════════════════════════════════════════════════════════
# PER-MODEL CHART
# ══════════════════════════════════════════════════════════

def _save_charts(
    results:     list[dict],
    avg_hit:     float,
    avg_ttft:    float,
    avg_tps:     float,
    model_name:  str,
    device_info: dict,
    save_path:   str,
) -> None:
    """
    Generate a per-model result chart with the following panels:

    Panel 1 — Keyword Hit Rate per query (colour-coded by query type).
    Panel 2 — TTFT per query as a line chart, with average TPS in title.
    Panel 3 — Resource usage per query:
                GPU mode: VRAM used (MB) bar chart + 4 GB limit line.
                CPU mode: RAM peak (MB) bars + CPU avg % line (dual Y-axis).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    is_gpu = device_info.get("device") == "GPU"

    # Colour scheme for query categories
    TYPE_COLORS = {
        "single_product": "#4C9BE8",
        "gpu_comparison":  "#E8834C",
        "shared_spec":     "#5DBE7A",
    }

    labels    = [f"Q{r['id']}" for r in results]
    hit_rates = [r["hit_rate"] * 100 for r in results]
    ttfts     = [r.get("ttft_ms", 0) for r in results]
    colors    = [TYPE_COLORS.get(r["query_type"], "#aaaaaa") for r in results]

    # Only show the resource panel if data is actually available
    has_resource = (
        is_gpu and any(r.get("vram_used_mb") for r in results)
    ) or (
        not is_gpu and any(r.get("ram_peak_mb") for r in results)
    )

    nrows = 3 if has_resource else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4 * nrows))
    ax1, ax2  = axes[0], axes[1]
    ax3       = axes[2] if has_resource else None

    device_label = device_info.get("device", "Unknown")
    if is_gpu:
        device_label += f" ({device_info.get('gpu_name', '')})"
    fig.suptitle(
        f"AORUS MASTER 16 RAG — Benchmark Results [vLLM]\n"
        f"Model: {model_name}  |  Device: {device_label}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    # ── Panel 1: Keyword Hit Rate ──────────────────────────
    bars = ax1.bar(labels, hit_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(avg_hit * 100, color="#E84C4C", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Keyword Hit Rate (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("Keyword Hit Rate per Query")
    for bar, val in zip(bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax1.legend(handles=patches + [
        plt.Line2D([0], [0], color="#E84C4C", linestyle="--", label=f"avg {avg_hit:.1%}")
    ], fontsize=8, loc="lower right")

    # ── Panel 2: TTFT per query ────────────────────────────
    ax2.plot(labels, ttfts, marker="o", color="#4C9BE8",
             linewidth=1.8, markersize=6, label="TTFT (ms)")
    ax2.axhline(avg_ttft, color="#E84C4C", linestyle="--", linewidth=1.2,
                label=f"avg {avg_ttft:.0f} ms")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title(f"Time To First Token per Query   |   avg TPS = {avg_tps:.1f} tok/s")
    ax2.legend(fontsize=9)
    max_ttft = max(ttfts) if ttfts else 1
    for x, y in zip(labels, ttfts):
        ax2.text(x, y + max_ttft * 0.02, f"{y:.0f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 3: Resource usage ────────────────────────────
    if ax3 is not None:
        if is_gpu:
            # GPU: bar chart of VRAM used per query
            vrams = [r.get("vram_used_mb", 0) for r in results]
            ax3.bar(labels, vrams, color="#9B59B6", edgecolor="white", linewidth=0.5)
            ax3.axhline(4096, color="#E84C4C", linestyle="--", linewidth=1.2,
                        label="4GB limit (4096 MB)")
            ax3.set_ylabel("VRAM Used (MB)")
            ax3.set_title("GPU VRAM Usage per Query")
            ax3.legend(fontsize=9)
            for bar, val in zip(ax3.patches, vrams):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                         f"{val}", ha="center", va="bottom", fontsize=8)
        else:
            # CPU: RAM peak bars (left Y) + CPU avg % line (right Y)
            ram_peaks = [r.get("ram_peak_mb", 0) for r in results]
            cpu_avgs  = [r.get("cpu_avg_pct", 0) for r in results]
            color_ram, color_cpu = "#9B59B6", "#E8834C"

            bars3 = ax3.bar(labels, ram_peaks, color=color_ram,
                            edgecolor="white", linewidth=0.5, label="RAM Peak (MB)")
            ax3.set_ylabel("RAM Peak (MB)", color=color_ram)
            ax3.tick_params(axis="y", labelcolor=color_ram)
            ax3.set_title("CPU Mode: RAM Usage & CPU Utilization per Query")

            # Overlay CPU utilisation on a secondary Y-axis
            ax3b = ax3.twinx()
            ax3b.plot(labels, cpu_avgs, marker="s", color=color_cpu,
                      linewidth=1.8, markersize=6, label="CPU Avg %")
            ax3b.set_ylabel("CPU Avg (%)", color=color_cpu)
            ax3b.tick_params(axis="y", labelcolor=color_cpu)
            ax3b.set_ylim(0, 110)

            # Merge legends from both axes
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

            for bar, val in zip(bars3, ram_peaks):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                         f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Benchmark] Chart saved → {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark with vLLM backend",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Comma-separated list of model IDs, e.g.:\n"
            "  Qwen/Qwen2.5-1.5B-Instruct\n"
            "  Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct"
        ),
    )
    parser.add_argument("--base-url",      default=VLLM_BASE_URL,
                        help="vLLM server base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--cases",         default=DEFAULT_CASES_PATH,
                        help="Path to benchmark cases JSON")
    parser.add_argument("--chunks",        default="data/chunks.json",
                        help="Path to chunks JSON")
    parser.add_argument("--emb",           default="data/embeddings.npy",
                        help="Path to embeddings cache (.npy)")
    parser.add_argument("--out-dir",       default=DEFAULT_OUT_DIR,
                        help="Directory to save results (default: results/)")
    parser.add_argument("--gpu-util",      default=0.85, type=float,
                        help="vLLM gpu-memory-utilization (default: 0.85)\n"
                             "Use 0.26 to simulate 4 GB VRAM on a T4 (15 GB × 0.26 ≈ 4 GB)")
    parser.add_argument("--max-model-len", default=4096, type=int,
                        help="vLLM max-model-len / context window in tokens (default: 4096)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable FlashAttention2 and CUDA Graphs.\n"
                             "Required for T4 GPUs (compute capability 7.5; FA2 needs >= 8.0).")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load shared resources ──────────────────────────────
    if not os.path.exists(args.chunks):
        print(f"[Error] {args.chunks} not found. Please run chunk_create.py first.")
        exit(1)
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    cases = load_cases(args.cases)

    index = VectorIndex()
    index.build(chunks_data, emb_cache=args.emb)

    # Record idle VRAM before any model is loaded (shared baseline for all models)
    vram_idle    = get_vram_usage()
    vram_idle_mb = vram_idle.get("used_mb", 0)
    print(f"[VRAM] Idle baseline: {vram_idle_mb} MB")

    all_summaries: list[dict] = []

    # ── Run benchmark for each model sequentially ──────────
    for idx, model_id in enumerate(model_list, 1):
        print(f"\n{'█' * 65}")
        print(f"  MODEL {idx}/{len(model_list)}: {model_id}")
        print(f"{'█' * 65}")

        # Launch the vLLM server for this model
        server_proc = start_vllm_server(
            model_id      = model_id,
            port          = VLLM_PORT,
            gpu_util      = args.gpu_util,
            max_model_len = args.max_model_len,
            enforce_eager = args.enforce_eager,
        )
        if server_proc is None:
            print(f"[Skip] {model_id} failed to start, skipping.\n")
            all_summaries.append({"model_name": model_id, "status": "FAILED"})
            continue

        # Measure VRAM after the model weights have been loaded
        vram_loaded = get_vram_usage()
        device_info = detect_device(vram_loaded)

        # Create the OpenAI-compatible client
        llm = load_llm(base_url=args.base_url, model_name=model_id)

        # Sanitise model name for use in file paths (replace "/" with "_")
        safe_name = model_id.replace("/", "_")
        save_path = os.path.join(args.out_dir, f"benchmark_{safe_name}.json")

        # Execute the benchmark
        summary = run_benchmark(
            index        = index,
            llm          = llm,
            cases        = cases,
            device_info  = device_info,
            model_name   = model_id,
            vram_idle_mb = vram_idle_mb,
            save_path    = save_path,
        )
        summary["status"] = "OK"
        all_summaries.append(summary)

        # Shut down the server to release VRAM before loading the next model
        stop_vllm_server(server_proc)

    # ── Multi-model aggregation ────────────────────────────
    if len(all_summaries) > 1:
        ok_summaries = [s for s in all_summaries if s.get("status") == "OK"]

        # Save aggregated comparison JSON
        comparison_json = os.path.join(args.out_dir, "benchmark_comparison.json")
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n[Benchmark] Comparison JSON saved → {comparison_json}")

        # Generate the cross-model comparison chart
        if ok_summaries:
            comparison_chart = os.path.join(args.out_dir, "benchmark_comparison.png")
            _save_comparison_chart(ok_summaries, comparison_chart)

        # Print final comparison table
        print("\n" + "═" * 75)
        print(f"  {'Model':<35} {'HitRate':>8} {'TTFT':>8} {'TPS':>7} {'VRAM Δ':>8}  4GB")
        print("═" * 75)
        for s in all_summaries:
            if s.get("status") != "OK":
                print(f"  {s['model_name']:<35}  {'FAILED':>8}")
                continue
            within = "✅" if s.get("within_4gb") else "❌"
            print(
                f"  {s['model_name']:<35}"
                f"  {s['avg_hit_rate']*100:>6.1f}%"
                f"  {s['avg_ttft_ms']:>6.0f}ms"
                f"  {s['avg_tps']:>5.1f}"
                f"  {s.get('vram_model_mb', 0):>6}MB"
                f"  {within}"
            )
        print("═" * 75)
