"""A small, inspectable Markdown retrieval demo for M1."""

import sys

if __package__:
    from .retrieval import (
        DOCS_DIR,
        TOP_K,
        cosine_similarity,
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )
else:
    from retrieval import (
        DOCS_DIR,
        TOP_K,
        cosine_similarity,
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )


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
