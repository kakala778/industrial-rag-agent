"""A small, inspectable Markdown retrieval demo for M1."""

import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "examples" / "docs"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


def load_documents(docs_dir=None):
    """Load top-level Markdown files, keeping their text and basename."""
    docs_dir = Path(docs_dir) if docs_dir is not None else DOCS_DIR
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Markdown document directory not found: {docs_dir}\n"
            "Create examples/docs/ and add your own .md test documents."
        )

    markdown_files = sorted(docs_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(
            f"No .md files found in: {docs_dir}\n"
            "Add your own Markdown test documents and try again."
        )

    documents = []
    for path in markdown_files:
        documents.append({"text": path.read_text(encoding="utf-8"), "source": path.name})
    return documents


def split_documents(documents):
    """Split documents on blank lines and keep non-empty paragraphs."""
    chunks = []
    for document in documents:
        chunk_id = 0
        for paragraph in re.split(r"\n\s*\n", document["text"]):
            text = paragraph.strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "source": document["source"],
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1
    return chunks


def load_model():
    """Load the requested Sentence Transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks, model):
    """Embed chunk text in order; embedding row i belongs to chunks[i]."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(query_embedding, chunk_embeddings):
    """Return the cosine score between one query vector and each chunk vector."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    vectors = np.asarray(chunk_embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("chunk_embeddings must be a 2D array")
    if vectors.shape[1] != query.size:
        raise ValueError("query and chunk embeddings must have the same dimension")

    scores = np.zeros(vectors.shape[0], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    nonzero_vectors = vector_norms > 0
    if query_norm > 0 and np.any(nonzero_vectors):
        scores[nonzero_vectors] = (
            vectors[nonzero_vectors] @ query
        ) / (vector_norms[nonzero_vectors] * query_norm)
    return scores


def retrieve(query_embedding, chunks, embeddings, top_k=3):
    """Rank chunks by explicit NumPy cosine similarity and return the top-k."""
    if len(chunks) != len(embeddings):
        raise ValueError("each chunk must have exactly one embedding")
    if top_k <= 0 or not chunks:
        return []

    scores = cosine_similarity(query_embedding, embeddings)
    best_indices = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "score": float(scores[index]),
            "source": chunks[index]["source"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
        }
        for index in best_indices
    ]


def main():
    try:
        documents = load_documents()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    chunks = split_documents(documents)
    if not chunks:
        print("Error: no non-empty Markdown paragraphs were found.", file=sys.stderr)
        return 1

    try:
        model = load_model()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Keep each embedding row aligned with the chunk at the same index.
    embeddings = embed_chunks(chunks, model)
    print(f"Loaded {len(documents)} documents into {len(chunks)} chunks.")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        query_embedding = model.encode(
            question,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results = retrieve(query_embedding, chunks, embeddings, top_k=TOP_K)
        for rank, result in enumerate(results, start=1):
            print(f"Top {rank}")
            print(f"score: {result['score']:.4f}")
            print(f"source: {result['source']}")
            print(f"chunk_id: {result['chunk_id']}")
            print()
            print(result["text"])
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

