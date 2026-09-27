"""Run the existing local RAG flow with documents loaded from one PDF."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .document_loader import load_pdf
    from .rag_demo import build_context, build_prompt, format_sources, generate_answer
    from .retrieval import (
        TOP_K,
        embed_chunks,
        load_model,
        retrieve,
        split_documents,
    )
else:
    from document_loader import load_pdf
    from rag_demo import build_context, build_prompt, format_sources, generate_answer
    from retrieval import TOP_K, embed_chunks, load_model, retrieve, split_documents


def initialize_pdf_retriever(pdf_path):
    """Load, chunk, and embed a PDF once for this demo session."""
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No non-empty text chunks were found in the PDF.")
    model = load_model()
    embeddings = embed_chunks(chunks, model)
    return documents, chunks, model, embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions against one PDF using the local RAG pipeline."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF knowledge source")
    args = parser.parse_args()

    try:
        documents, chunks, model, embeddings = initialize_pdf_retriever(args.pdf_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(documents)} PDF pages into {len(chunks)} chunks.")

    while True:
        try:
            question = input("问题: ").strip()
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
        if not results:
            print("没有检索到相关资料。\n")
            continue

        print("\nRetrieved Context:")
        for rank, result in enumerate(results, start=1):
            print(f"\nTop {rank}:")
            print(f"source: {result['source']}")
            page = result.get("metadata", {}).get("page")
            if page is not None:
                print(f"page: {page}")
            print(f"score: {result['score']:.4f}")
            print(f"chunk_id: {result['chunk_id']}")

        prompt = build_prompt(question, build_context(results))
        print("\nGenerating answer...")
        try:
            answer = generate_answer(prompt)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"\n回答：\n\n{answer}\n")
        sources = format_sources(results)
        if sources:
            print(f"Sources:\n\n{sources}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
