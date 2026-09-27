"""Minimal Retrieval-Augmented Generation demo using local Ollama."""

import json
import re
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


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"
PROMPT_TEMPLATE = """你是一个基于知识库的问答助手。

请严格根据提供的资料回答问题。

规则：
1. 只能使用资料中的信息回答。
2. 如果资料中没有明确答案，请回答“根据现有资料无法确定”，不要自行推测。
3. 回答时尽量简洁准确。
4. 最后列出参考来源，包括文件名和章节信息。

资料：
{context}

问题：
{question}

请回答："""


def get_section(result, section_by_chunk=None):
    """Use available metadata or the heading associated with this chunk."""
    section = result.get("section")
    metadata = result.get("metadata") or {}
    if not section:
        section = metadata.get("section")
    if not section and section_by_chunk is not None:
        section = section_by_chunk.get((result["source"], result.get("chunk_id")))
    if section:
        return str(section)

    for line in result["text"].splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line.strip())
        if match:
            return match.group(1).strip()
    return "未标注"


def build_section_map(chunks):
    """Carry each Markdown heading forward to its continuation chunks."""
    current_sections = {}
    section_by_chunk = {}
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    for chunk in chunks:
        source = chunk["source"]
        section = current_sections.get(source, "未标注")
        for line in chunk["text"].splitlines():
            match = heading_pattern.match(line.strip())
            if match:
                section = match.group(2).strip()
        current_sections[source] = section
        section_by_chunk[(source, chunk["chunk_id"])] = section

    return section_by_chunk


def build_context(results, section_by_chunk=None):
    """Format retrieved chunks with source, section, and page metadata."""
    sections = []
    for result in results:
        metadata = result.get("metadata") or {}
        page = metadata.get("page")
        page_line = f"\n[页码: {page}]" if page is not None else ""
        sections.append(
            f"[来源: {result['source']}]\n"
            f"[章节: {get_section(result, section_by_chunk)}]{page_line}\n\n"
            f"内容:\n{result['text']}"
        )
    return "\n\n---\n\n".join(sections)


def build_prompt(question, context):
    """Build the knowledge-grounded prompt for the local generation model."""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def format_sources(results, section_by_chunk=None):
    """Format unique retrieved sources with section and retrieval metadata."""
    citations = []
    seen = set()

    for result in results:
        metadata = result.get("metadata", {})
        page = metadata.get("page")
        section = get_section(result, section_by_chunk)
        citation_key = (result["source"], section, page, result["chunk_id"])
        if citation_key in seen:
            continue
        seen.add(citation_key)
        page_line = f"Page:\n{page}\n\n" if page is not None else ""
        citations.append(
            f"[{len(citations) + 1}]\n"
            f"File:\n{result['source']}\n\n"
            f"Section:\n{section}\n\n"
            f"{page_line}"
            f"Chunk:\n{result['chunk_id']}\n\n"
            f"Score:\n{result['score']:.4f}"
        )

    return "\n\n".join(citations)


def call_ollama(prompt):
    """Send one non-streaming generation request to local Ollama."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    request = Request(
        OLLAMA_GENERATE_URL,
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
            f"Could not connect to Ollama at {OLLAMA_GENERATE_URL}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out after 180 seconds.") from exc

    try:
        result = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Ollama returned an invalid JSON response.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned a JSON response that is not an object.")

    answer = result.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama response is missing the 'response' text.")
    answer = answer.strip()
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1].strip()
    return answer


def generate_answer(prompt):
    """Keep the existing PDF demo interface while using Ollama generation."""
    return call_ollama(prompt)


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
        section_by_chunk = build_section_map(chunks)
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
            print(f"section: {get_section(result, section_by_chunk)}")
            print(f"score: {result['score']:.4f}")
            print(f"chunk_id: {result['chunk_id']}")

        prompt = build_prompt(question, build_context(results, section_by_chunk))
        print("\nGenerating answer...")
        try:
            answer = generate_answer(prompt)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"\n回答：\n\n{answer}\n")
        sources = format_sources(results, section_by_chunk)
        if sources:
            print(f"来源：\n\n{sources}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
