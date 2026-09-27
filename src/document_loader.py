"""Load PDF pages as simple text documents with source metadata."""

import argparse
import sys
from pathlib import Path

import pymupdf

if __package__:
    from . import mineru_loader
else:
    import mineru_loader


def load_pdf(
    path,
    parser="pymupdf",
    *,
    force=False,
    runner_path=None,
    cache_root=None,
    powershell_executable=None,
):
    """Return one document per PDF page, with a 1-based page number."""
    if parser == "mineru":
        return mineru_loader.load_pdf_with_mineru(
            path,
            force=force,
            runner_path=runner_path,
            cache_root=cache_root,
            powershell_executable=powershell_executable,
        )
    if parser != "pymupdf":
        raise ValueError(
            f"Unsupported PDF parser: {parser!r}. Choose 'pymupdf' or 'mineru'."
        )

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            documents.append(
                {
                    "text": page.get_text("text"),
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                }
            )
    return documents


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF by page.")
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF file")
    args = parser.parse_args()

    try:
        documents = load_pdf(args.pdf_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"pages: {len(documents)}")
    print(f"documents: {len(documents)}")
    print("metadata:")
    for document in documents:
        metadata = document["metadata"]
        print(f"- source: {metadata['source']}, page: {metadata['page']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
