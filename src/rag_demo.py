"""Minimal Retrieval-Augmented Generation demo using local Ollama."""

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .retrieval import (
        TOP_K,
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )
else:
    from retrieval import (
        TOP_K,
        embed_chunks,
        load_documents,
        load_model,
        retrieve,
        split_documents,
    )


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"
SYSTEM_PROMPT = (
    "你是一个知识库问答助手。\n\n"
    "请严格根据提供的资料回答问题。\n"
    "如果资料中没有答案，请明确说明无法从资料中确定。\n"
    "不要编造不存在的信息。"
)


def build_context(results):
    """Format retrieved chunks with their source names for the prompt."""
    sections = []
    for result in results:
        sections.append(
            f"Source:\n{result['source']}\n\n"
            f"Content:\n{result['text']}"
        )
    return "\n\n---\n\n".join(sections)


def build_prompt(question, context):
    """Build the user message containing the retrieved context and question."""
    return f"参考资料：\n\n{context}\n\n问题：\n\n{question}"


def generate_answer(prompt):
    """Send one non-streaming chat request to the local Ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
    }
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"Ollama returned HTTP {exc.code}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_CHAT_URL}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out after 180 seconds.") from exc

    try:
        result = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Ollama returned an invalid JSON response.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned a JSON response that is not an object.")

    message = result.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response is missing the 'message' object.")
    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama response is missing 'message.content'.")
    answer = answer.strip()
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1].strip()
    return answer


def initialize_retriever():
    """Load documents and create chunk embeddings once for this CLI session."""
    documents = load_documents()
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No non-empty Markdown chunks were found.")
    model = load_model()
    embeddings = embed_chunks(chunks, model)
    return documents, chunks, model, embeddings


def main():
    try:
        documents, chunks, model, embeddings = initialize_retriever()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(documents)} documents into {len(chunks)} chunks.")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
