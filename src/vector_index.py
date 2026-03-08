"""
vector_index.py
═══════════════════════════════════════════════
Supported Bilingual Embedding Models (Chinese/English):
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (Default, lightweight & fast)
  - intfloat/multilingual-e5-base                                (Higher accuracy)
  - intfloat/multilingual-e5-large                               (Highest accuracy, slower)

Execution:
# Use default model
python vector_index.py

# Specify model and paths
python vector_index.py \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy \
    --model  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import json
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

# List of available bilingual embedding models
SUPPORTED_MODELS = {
    "minilm":   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5-base":  "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
}

DEFAULT_MODEL = SUPPORTED_MODELS["minilm"]


class VectorIndex:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        print(f"[Index] Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model      = SentenceTransformer(model_name)
        self.chunks:     list[dict]        = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[dict], emb_cache: str = "data/embeddings.npy") -> None:
        self.chunks = chunks
        if os.path.exists(emb_cache):
            print(f"[Index] Embedding cache found, skipping encoding -> {emb_cache}")
            self.embeddings = np.load(emb_cache)
            return
        print(f"[Index] Encoding {len(chunks)} chunks（model: {self.model_name}）...")
        # Use the bilingual 'text' field (CH / EN) to ensure cross-lingual semantic coverage
        texts = [c["text"] for c in chunks]

        # Multilingual-E5 models require 'passage: ' prefix for optimal retrieval performance
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        embs  = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self.embeddings = embs / np.maximum(norms, 1e-9)
        os.makedirs(os.path.dirname(emb_cache) or ".", exist_ok=True)
        np.save(emb_cache, self.embeddings)
        print(f"[Index] Embeddings saved -> {emb_cache}")

    def search(
        self,
        query:          str,
        top_k:          int        = 5,
        product_filter: str | None = None,
    ) -> list[dict]:
        # Multilingual-E5 models require 'query: ' prefix
        q_text = f"query: {query}" if "e5" in self.model_name.lower() else query
        q_emb  = self.model.encode([q_text])
        q_emb /= np.maximum(np.linalg.norm(q_emb), 1e-9)
        scores = (self.embeddings @ q_emb.T)[:, 0].copy()

        if product_filter:
            for i, c in enumerate(self.chunks):
                if c.get("short_id") != product_filter:
                    scores[i] = -1.0

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "_score": float(scores[i])} for i in top_idx]

    def get_by_key(
        self,
        key:      str,
        short_id: str | None = None,
    ) -> list[dict]:
        # Exact retrieval by specification key, with optional short_id filter.
        return [
            c for c in self.chunks
            if c.get("key") == key
            and (short_id is None or c.get("short_id") == short_id)
        ]

    def get_by_short_id(self, short_id: str) -> list[dict]:
        # Retrieves all chunks belonging to a specific short_id.
        return [c for c in self.chunks if c.get("short_id") == short_id]


# ══════════════════════════════════════════════════════════
# MAIN — CLI SUPPORT FOR MODEL / CHUNKS / EMB PATHS
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vector Index and Test Search")
    parser.add_argument(
        "--chunks", default="data/chunks.json",
        help="Path to chunks JSON (default: data/chunks.json)",
    )
    parser.add_argument(
        "--emb", default="data/embeddings.npy",
        help="Path to output npy file (default: data/embeddings.npy)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "Embedding model name (supports aliases or HuggingFace IDs)\n"
            f"  Aliases:{SUPPORTED_MODELS}\n"
            f"  Default:{DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-encoding and ignore existing cache",
    )
    args = parser.parse_args()

    # Support for aliases (e.g., --model e5-base)
    model_name = SUPPORTED_MODELS.get(args.model, args.model)

    # Remove existing cache if --force is used
    if args.force and os.path.exists(args.emb):
        os.remove(args.emb)
        print(f"[Index] Force rebuild: deleted cache -> {args.emb}")

    if not os.path.exists(args.chunks):
        print(f"[Error] {args.chunks} not found. Please run chunk_create.py first.")
        exit(1)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    print(f"[Main] Successfully loaded {len(chunks_data)} chunks from {args.chunks}")

    # Bilingual detection warning
    if any("text_zh" in c for c in chunks_data):
        print("[Info] Bilingual chunks detected. Embeddings will use merged CH/EN text.")

    # Build index
    index = VectorIndex(model_name=model_name)
    index.build(chunks_data, emb_cache=args.emb)

    # Test queries (including English/mixed to verify bilingual effectiveness)
    test_queries = [
        "AORUS MASTER 16 BZH 的 GPU 是什麼型號？",
        "AORUS MASTER 16 BXH 支援最大多少 GB RAM？",
        "What is AORUS MASTER 16 BYH Adapter？",
        "What is AORUS MASTER 16 OS？",
        "筆電電池容量多少？"
    ]
    print("\n" + "═" * 50)
    print(f"  SEARCH TEST  |  model: {model_name}")
    print("═" * 50)
    for q in test_queries:
        results = index.search(q, top_k=3)
        print(f"\nQuery: {q}")
        for i, res in enumerate(results):
            print(f"  {i+1}. [{res['short_id']:4s}] score={res['_score']:.4f}  {res['text'][:80]}...")
    print("═" * 50)