"""Thin adapter from one uploaded PDF to the existing M11 research APIs."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.agent.harness import AgentHarness
from src.agent.io import load_corpus
from src.agent.markdown_report import render_research_report
from src.agent.selector import OllamaActionSelector
from src.agent.tools import KnowledgeBaseSession


class DocumentLoadError(RuntimeError):
    """The uploaded PDF could not be parsed into the existing corpus format."""


class AgentExecutionError(RuntimeError):
    """The existing session, Agent loop, or host report renderer failed."""

    def __init__(self, message, *, page_count):
        super().__init__(message)
        self.page_count = page_count


@dataclass(frozen=True)
class ResearchRun:
    state: object
    report: str
    page_count: int


def run_uploaded_pdf_research(pdf_bytes, task):
    """Parse one in-memory upload and run the unchanged evidence-reference loop."""
    if not isinstance(task, str) or not task.strip() or len(task) > 4000:
        raise ValueError("task must be nonempty and at most 4000 characters")
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("uploaded PDF must contain data")

    try:
        with TemporaryDirectory(prefix="industrial-rag-agent-upload-") as temporary:
            pdf_path = Path(temporary) / "uploaded.pdf"
            pdf_path.write_bytes(pdf_bytes)
            try:
                corpus = load_corpus([f"S1={pdf_path}"], parser="pymupdf")
            except Exception as exc:
                detail = str(exc).replace(str(pdf_path), "uploaded PDF")
                raise DocumentLoadError(
                    f"{type(exc).__name__}: {detail or 'PDF parsing failed.'}"
                ) from None
            pages = corpus.get("S1") if isinstance(corpus, dict) else None
            if not isinstance(pages, list) or not pages:
                raise DocumentLoadError("No pages could be extracted from the uploaded PDF.")
            page_count = len(pages)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"{type(exc).__name__}: {exc}") from None

    try:
        session = KnowledgeBaseSession(corpus)
        selector = OllamaActionSelector(action_contract="evidence_reference")
        state = AgentHarness(session, selector).run(task, ["S1"])
        report = render_research_report(state, session)
    except Exception as exc:
        raise AgentExecutionError(
            f"{type(exc).__name__}: {exc}", page_count=page_count
        ) from None

    return ResearchRun(state=state, report=report, page_count=page_count)
