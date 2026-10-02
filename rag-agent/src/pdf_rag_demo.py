"""Run the existing local RAG flow with documents loaded from one PDF."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .document_loader import load_pdf
    from .hybrid_retrieval import HYBRID_CANDIDATE_K, retrieve_hybrid
    from .lexical_retrieval import BM25Index
    from .rag_demo import build_context, build_prompt, format_sources, generate_answer
    from .retrieval import (
        TOP_K,
        embed_chunks,
        load_model,
        retrieve,
        split_documents,
    )
    from .reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank
else:
    from document_loader import load_pdf
    from hybrid_retrieval import HYBRID_CANDIDATE_K, retrieve_hybrid
    from lexical_retrieval import BM25Index
    from rag_demo import build_context, build_prompt, format_sources, generate_answer
    from retrieval import TOP_K, embed_chunks, load_model, retrieve, split_documents
    from reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank


RERANK_CANDIDATE_K = HYBRID_CANDIDATE_K


def initialize_pdf_retriever(
    pdf_path,
    parser="pymupdf",
    *,
    force_parse=False,
    mineru_runner=None,
    representation="structured",
):
    """Load, chunk, and embed a PDF once for this demo session."""
    documents = load_pdf(
        pdf_path,
        parser=parser,
        force=force_parse,
        runner_path=mineru_runner,
        representation=representation,
    )
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No non-empty text chunks were found in the PDF.")
    model = load_model()
    embeddings = embed_chunks(chunks, model)
    return documents, chunks, model, embeddings


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Ask questions against one PDF using the local RAG pipeline."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF knowledge source")
    parser.add_argument(
        "--mode",
        choices=("dense", "reranker", "compare"),
        default=None,
        help="Legacy Dense/reranker mode; cannot be combined with --retriever/--rerank",
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default=None,
        help="Retriever to use (default: dense)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the selected retriever's Top-20 candidates with BGE",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Hugging Face cross-encoder model (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--parser",
        choices=("pymupdf", "mineru"),
        default="pymupdf",
        help="PDF parser (default: pymupdf baseline)",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Re-run MinerU instead of using its local parse cache",
    )
    parser.add_argument(
        "--mineru-runner",
        type=Path,
        help="Path to run-mineru.ps1 (otherwise use MINERU_RUNNER_PATH/default)",
    )
    parser.add_argument(
        "--representation",
        choices=("structured", "flat"),
        default="structured",
        help="MinerU document representation (default: structured; flat is for comparison)",
    )
    return parser


def resolve_retrieval_options(*, mode, retriever, rerank):
    """Resolve legacy modes or orthogonal retriever/reranker selectors."""
    if mode is not None and (retriever is not None or rerank):
        raise ValueError("--mode cannot be combined with --retriever or --rerank")

    if mode == "dense":
        return "dense", False, False
    if mode == "reranker":
        return "dense", True, False
    if mode == "compare":
        return "dense", True, True
    if mode is not None:
        raise ValueError(f"unsupported legacy retrieval mode: {mode}")

    selected_retriever = retriever or "dense"
    if selected_retriever not in ("dense", "hybrid"):
        raise ValueError(f"unsupported retriever: {selected_retriever}")
    return selected_retriever, bool(rerank), False


def main():
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args()

    try:
        selected_retriever, use_reranker, compare_mode = resolve_retrieval_options(
            mode=args.mode,
            retriever=args.retriever,
            rerank=args.rerank,
        )
    except ValueError as exc:
        argument_parser.error(str(exc))

    try:
        documents, chunks, model, embeddings = initialize_pdf_retriever(
            args.pdf_path,
            parser=args.parser,
            force_parse=args.force_parse,
            mineru_runner=args.mineru_runner,
            representation=args.representation,
        )
        bm25_index = BM25Index(chunks) if selected_retriever == "hybrid" else None
        reranker_model = load_reranker(args.reranker_model) if use_reranker else None
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Parser: {args.parser}")
    if args.parser == "mineru":
        print(f"Representation: {args.representation}")
    represented_pages = {
        (
            (document.get("metadata") or {}).get("source"),
            (document.get("metadata") or {}).get("page"),
        )
        for document in documents
        if (document.get("metadata") or {}).get("page") is not None
    }
    print(
        f"Loaded {len(documents)} documents across {len(represented_pages)} "
        f"represented PDF pages into {len(chunks)} chunks."
    )
    if selected_retriever == "hybrid":
        print("Retriever: hybrid")

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
        if compare_mode:
            dense_candidates = retrieve(
                query_embedding,
                chunks,
                embeddings,
                top_k=RERANK_CANDIDATE_K,
            )
            reranked_results = rerank(
                question,
                dense_candidates,
                reranker_model,
                top_k=TOP_K,
            )
            _print_retrieved_results("Dense results", dense_candidates[:TOP_K])
            _print_retrieved_results("Reranker results", reranked_results)
            results = reranked_results
        elif selected_retriever == "dense" and not use_reranker:
            results = retrieve(query_embedding, chunks, embeddings, top_k=TOP_K)
        elif selected_retriever == "dense":
            candidates = retrieve(
                query_embedding,
                chunks,
                embeddings,
                top_k=RERANK_CANDIDATE_K,
            )
            results = rerank(question, candidates, reranker_model, top_k=TOP_K)
        else:
            candidates = retrieve_hybrid(
                question,
                query_embedding,
                chunks,
                embeddings,
                bm25_index,
            )
            results = (
                rerank(question, candidates, reranker_model, top_k=TOP_K)
                if use_reranker
                else candidates[:TOP_K]
            )
        if not results:
            print("没有检索到相关资料。\n")
            continue

        if not compare_mode:
            _print_retrieved_results(
                "Retrieved Context",
                results,
                show_scores=(selected_retriever != "hybrid"),
            )

        prompt = build_prompt(question, build_context(results))
        if compare_mode:
            print("\nGenerating answer from reranker Top-3 context...")
        else:
            print("\nGenerating answer...")
        try:
            answer = generate_answer(prompt)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"\n回答：\n\n{answer}\n")
        sources = format_sources(
            results,
            include_score=(selected_retriever != "hybrid"),
        )
        if sources:
            print(f"Sources:\n\n{sources}\n")

    return 0


def _print_retrieved_results(title, results, *, show_scores=True):
    print(f"\n{title}:")
    for rank, result in enumerate(results, start=1):
        print(f"\nTop {rank}:")
        print(f"source: {result['source']}")
        page = result.get("metadata", {}).get("page")
        if page is not None:
            print(f"page: {page}")
        if show_scores:
            print(f"score: {result['score']:.4f}")
            if "reranker_score" in result:
                print(f"reranker_score: {result['reranker_score']:.4f}")
        print(f"chunk_id: {result['chunk_id']}")


if __name__ == "__main__":
    raise SystemExit(main())
