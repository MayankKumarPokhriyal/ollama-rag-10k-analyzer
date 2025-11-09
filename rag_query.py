#!/usr/bin/env python3
"""
CLI for querying the FAISS-backed RAG index built from a 10-K PDF using Ollama (Llama3).

Usage:
    python rag_query.py "What are Tesla's major risks?"

Requirements:
- `store/faiss.index` and `store/chunks.pkl` should exist (built by the notebook).
- SentenceTransformer model 'all-MiniLM-L6-v2' available (auto-downloads if missing).
- Ollama CLI installed with `llama3` model pulled.
"""

import sys
import os
import subprocess
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------
STORE_DIR = Path("store")
INDEX_PATH = STORE_DIR / "faiss.index"
CHUNKS_PATH = STORE_DIR / "chunks.pkl"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def call_ollama_cli(prompt: str, temperature: float = 0.2, timeout: int = 120) -> str:
    """Call Ollama locally via subprocess."""
    try:
        cmd = ["ollama", "run", "llama3", "--temperature", str(temperature)]
        print("🔹 Calling Ollama locally (Llama3)...")
        result = subprocess.run(
            cmd, input=prompt, text=True, capture_output=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Make sure it's installed and in your PATH.")

def retrieve_context(query: str, index, chunks, embedder, k: int = 5) -> str:
    """Retrieve top-k relevant chunks for a query."""
    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = normalize_embeddings(q_vec)
    D, I = index.search(q_vec, k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    return "\n\n---\n\n".join(retrieved)

# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"<your question>\"")
        sys.exit(1)

    question = sys.argv[1]

    # Verify files
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("Missing FAISS index or chunks.pkl. Please run the notebook first.")
        sys.exit(2)

    print(f"Query: {question}")
    print(f"Loading FAISS index from {INDEX_PATH}")
    print(f"Loading chunks from {CHUNKS_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    context = retrieve_context(question, index, chunks, embedder, k=5)

    SYSTEM_PROMPT = """You are a precise financial analyst.
Use ONLY the provided company 10-K context below to answer the question.
If the answer is not found in the context, reply 'Not available in the 10-K report.'
Always cite [source: company 10-K]."""

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        answer = call_ollama_cli(prompt)
        print("\nAnswer:\n", answer)
        print("\n[source: company 10-K]")
    except Exception as e:
        print(f"Error during Ollama call: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
