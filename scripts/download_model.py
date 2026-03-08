"""
scripts/download_model.py
═══════════════════════════════════════════════
Downloads specified GGUF quantized models from Hugging Face to the models/ directory.

Execution:
    python scripts/download_model.py --model llama-3.2-3b-q5   (Recommended)
    python scripts/download_model.py --model llama-3.2-3b-q4
    python scripts/download_model.py --model qwen2.5-3b-q5
    python scripts/download_model.py --model qwen2.5-3b-q4
    python scripts/download_model.py --model phi4-mini-q4
    python scripts/download_model.py --list               (List all available models)
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# Direct download URLs from HuggingFace (GGUF format)
MODEL_URLS: dict[str, tuple[str, str]] = {
    # alias → (filename, url)
    "llama-3.2-3b-q5": (
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
    ),
    "llama-3.2-3b-q4": (
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    "qwen2.5-3b-q5": (
        "Qwen2.5-3B-Instruct-Q5_K_M.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
        "qwen2.5-3b-instruct-q5_k_m.gguf",
    ),
    "qwen2.5-3b-q4": (
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
        "qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    "phi4-mini-q4": (
        "Phi-4-mini-instruct-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/phi-4-mini-instruct-GGUF/resolve/main/"
        "phi-4-mini-instruct-Q4_K_M.gguf",
    ),
    "phi4-mini-q5": (
        "Phi-4-mini-instruct-Q5_K_M.gguf",
        "https://huggingface.co/bartowski/phi-4-mini-instruct-GGUF/resolve/main/"
        "phi-4-mini-instruct-Q5_K_M.gguf",
    ),
}


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 ** 2
        total_mb = total_size / 1024 ** 2
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)


def download(alias: str) -> None:
    if alias not in MODEL_URLS:
        print(f"[Error] Unknown model alias: '{alias}'")
        print(f" Available aliases: {', '.join(MODEL_URLS)}")
        sys.exit(1)

    filename, url = MODEL_URLS[alias]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename

    if dest.exists():
        print(f"[✓] File exists, skipping download:{dest}")
        return

    print(f"[↓] Downloading {filename}")
    print(f"    Source: {url}")
    print(f"    Target: {dest}")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print(f"\n[✓] Download complete: {dest}  ({dest.stat().st_size / 1024**2:.1f} MB)")
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF models from HuggingFace")
    parser.add_argument("--model", help="Model alias to download (see --list)")
    parser.add_argument("--list",  action="store_true", help="List all available model aliases")
    args = parser.parse_args()

    if args.list or not args.model:
        print("Available model aliases:")
        for alias, (fname, _) in MODEL_URLS.items():
            dest = MODELS_DIR / fname
        status = "✓ Downloaded" if dest.exists() else "  Not Downloaded"
        print(f"  {alias:<22} {status}  →  {fname}")
        return

    download(args.model)


if __name__ == "__main__":
    main()
