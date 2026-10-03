"""Session-local evidence tools; retrieval algorithms remain in existing modules."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re

import numpy as np

from ..hybrid_retrieval import retrieve_hybrid
from ..lexical_retrieval import BM25Index, chunk_identity
from ..retrieval import embed_chunks, load_model, retrieve, split_documents
from ..reranker import load_reranker, rerank as rerank_candidates


SAFE_ALIAS = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
PROVENANCE_KEYS = ("source", "page", "block_type", "block_index", "chunk_id")
TOOL_STATUSES = {"ok", "no_evidence", "invalid_scope", "invalid_evidence_id",
                 "timeout", "error"}
REFERENCE_MAX_CHARS = 1200
TEXT_CONTEXT_CHARS = 200


@dataclass
class ToolResult:
    status: str
    results: list = field(default_factory=list)
    message: str = ""

    def __post_init__(self):
        if self.status not in TOOL_STATUSES:
            raise ValueError("unknown tool status")


class KnowledgeBaseSession:
    """Load one corpus once; reuse resources across scoped searches in one task.

    Aliases are caller-defined public identifiers, never paths. Empty aliases
    remain valid scopes. Only documented provenance is exposed to the Agent.
    """

    def __init__(self, corpus, *, model=None, reranker_model=None):
        if not isinstance(corpus, dict) or not corpus:
            raise ValueError("corpus must map safe aliases to document lists")
        self.scopes = tuple(corpus)
        self.documents, self.chunks, self.registry = [], [], {}
        self._id_by_identity, self._parents = {}, {}
        self.scope_indexes = {}
        self.model, self.reranker_model = model, reranker_model
        self.embeddings = None
        for alias, documents in corpus.items():
            if not isinstance(alias, str) or not SAFE_ALIAS.fullmatch(alias):
                raise ValueError("source aliases must be safe identifiers")
            if not isinstance(documents, list):
                raise ValueError("scope documents must be a list")
            for ordinal, original in enumerate(documents):
                if not isinstance(original, dict) or not isinstance(original.get("text"), str):
                    raise ValueError("Document requires string text")
                meta = original.get("metadata") or {}
                page = meta.get("page")
                block_type = meta.get("block_type", "document")
                block_index = meta.get("block_index", ordinal)
                if page is not None and (type(page) is not int or page < 1):
                    raise ValueError("page must be null or a positive integer")
                if not isinstance(block_type, str) or not SAFE_ALIAS.fullmatch(block_type):
                    raise ValueError("block type must be a safe identifier")
                if type(block_index) is not int or block_index < 0:
                    raise ValueError("block index must be a nonnegative integer")
                metadata = dict(source=alias, page=page, block_type=block_type,
                                block_index=block_index)
                if "table_header_rows" in meta:
                    header_rows = meta["table_header_rows"]
                    row_count = len(original["text"].splitlines())
                    if (block_type != "table" or not isinstance(header_rows, (list, tuple))
                            or len(header_rows) > 3
                            or any(type(i) is not int or i < 0 or i >= row_count for i in header_rows)
                            or len(set(header_rows)) != len(header_rows)):
                        raise ValueError("invalid structural table header rows")
                    metadata["table_header_rows"] = tuple(header_rows)
                parent_key = (alias, page, block_type, block_index)
                if parent_key in self._parents:
                    raise ValueError("ambiguous parent identity")
                parent = {"text": original["text"], "metadata": metadata}
                self._parents[parent_key] = parent
                self.documents.append(parent)
                for chunk in split_documents([parent]):
                    identity = chunk_identity(chunk)
                    encoded = json.dumps(identity, ensure_ascii=True,
                                         separators=(",", ":")).encode("utf-8")
                    evidence_id = "ev_" + hashlib.sha256(encoded).hexdigest()
                    if evidence_id in self.registry:
                        raise ValueError("evidence identity collision")
                    row = self._public(chunk, evidence_id)
                    self.registry[evidence_id] = row
                    self._id_by_identity[identity] = evidence_id
                    self.chunks.append(chunk)

    @staticmethod
    def _public(chunk, evidence_id):
        metadata = chunk["metadata"]
        return dict(evidence_id=evidence_id, text=chunk["text"],
                    source=chunk["source"], page=metadata["page"],
                    block_type=metadata["block_type"],
                    block_index=metadata["block_index"], chunk_id=chunk["chunk_id"])

    def resolve_scopes(self, scopes):
        if scopes is None:
            return self.scopes
        if (not isinstance(scopes, (list, tuple)) or not scopes
                or any(not isinstance(s, str) or s not in self.scopes for s in scopes)
                or len(set(scopes)) != len(scopes)):
            return None
        # Corpus order makes equivalent subsets deterministic and cacheable.
        return tuple(s for s in self.scopes if s in scopes)

    def search_knowledge(self, query, scopes=None, retriever="hybrid", rerank=True,
                         top_k=3):
        resolved = self.resolve_scopes(scopes)
        if resolved is None:
            return ToolResult("invalid_scope", message="Unknown or invalid scope")
        if (not isinstance(query, str) or not query.strip() or len(query) > 4000
                or retriever not in ("dense", "hybrid") or type(rerank) is not bool
                or type(top_k) is not int or not 1 <= top_k <= 3):
            return ToolResult("error", message="Invalid search arguments")
        indices = [i for i, c in enumerate(self.chunks) if c["source"] in resolved]
        if not indices:
            return ToolResult("no_evidence", message="Scope contains no searchable text")
        try:
            if self.model is None:
                self.model = load_model()
            if self.embeddings is None:
                self.embeddings = embed_chunks(self.chunks, self.model)
            chunks = [self.chunks[i] for i in indices]
            embeddings = np.asarray(self.embeddings)[indices]
            query_vector = self.model.encode(query, convert_to_numpy=True,
                                             show_progress_bar=False)
            if retriever == "hybrid":
                if resolved not in self.scope_indexes:
                    self.scope_indexes[resolved] = BM25Index(chunks)
                candidates = retrieve_hybrid(query, query_vector, chunks, embeddings,
                                             self.scope_indexes[resolved])
            else:
                candidates = retrieve(query_vector, chunks, embeddings,
                                      top_k=20 if rerank else top_k)
            if rerank and candidates:
                if self.reranker_model is None:
                    self.reranker_model = load_reranker()
                candidates = rerank_candidates(query, candidates, self.reranker_model,
                                               top_k=top_k)
            rows = [deepcopy(self.registry[self._id_by_identity[chunk_identity(c)]])
                    for c in candidates[:top_k]]
            return ToolResult("ok" if rows else "no_evidence", rows)
        except TimeoutError:
            return ToolResult("timeout", message="Search execution timed out")
        except Exception:
            return ToolResult("error", message="Search execution failed")

    def lookup_evidence(self, evidence_id):
        if not isinstance(evidence_id, str) or evidence_id not in self.registry:
            return ToolResult("invalid_evidence_id", message="Unknown session evidence ID")
        row = deepcopy(self.registry[evidence_id])
        parent_key = tuple(row[key] for key in PROVENANCE_KEYS[:-1])
        parent = self._parents[parent_key]["text"]
        child = row["text"]
        start = parent.find(child)
        unique = start >= 0 and parent.find(child, start + 1) < 0
        # Ambiguous occurrence: return exact child, never guess an original offset.
        if not unique:
            row.update(context_kind="child_only", offset=None, truncated=True)
        else:
            offset = max(0, start - 200)
            offset = min(offset, max(0, len(parent) - 2000))
            row.update(text=parent[offset:offset + 2000], offset=offset,
                       context_kind="original_block" if len(parent) <= 2000
                       else "parent_neighborhood", truncated=len(parent) > 2000)
        row["provenance"] = {key: row[key] for key in PROVENANCE_KEYS}
        return ToolResult("ok", [row])

    def render_evidence_reference(self, evidence_id):
        """Return bounded original text and active-session provenance for one ID.

        This is an authenticity renderer only. It uses structural offsets and
        explicit table-header metadata, never query labels or relevance data.
        """
        if not isinstance(evidence_id, str) or evidence_id not in self.registry:
            raise ValueError("unknown session evidence ID")
        row = self.registry[evidence_id]
        parent_record = self._parents[tuple(row[key] for key in PROVENANCE_KEYS[:-1])]
        parent, child = parent_record["text"], row["text"]
        if len(child) > REFERENCE_MAX_CHARS:
            raise ValueError("evidence child exceeds reference bound")
        positions = []
        start = 0
        while True:
            found = parent.find(child, start)
            if found < 0:
                break
            positions.append(found)
            start = found + 1
        unique = len(positions) == 1
        ranges = None
        kind = "child_only_ambiguous"
        child_start = child_end = None
        if unique:
            child_start, child_end = positions[0], positions[0] + len(child)
            if row["block_type"] == "table":
                line_ranges = []
                offset = 0
                for line in parent.splitlines(keepends=True):
                    line_ranges.append((offset, offset + len(line)))
                    offset += len(line)
                if offset < len(parent):
                    line_ranges.append((offset, len(parent)))
                target_rows = [i for i, (a, b) in enumerate(line_ranges)
                               if a < child_end and b > child_start]
                header_rows = parent_record["metadata"].get("table_header_rows", ())
                if target_rows:
                    selected = sorted(set(target_rows) | set(header_rows))
                    proposed = [{"start": line_ranges[i][0], "end": line_ranges[i][1]}
                                for i in selected]
                    if sum(r["end"] - r["start"] for r in proposed) <= REFERENCE_MAX_CHARS:
                        ranges = proposed
                        kind = "table_rows_with_header" if set(header_rows) - set(target_rows) else "table_rows"
                    else:
                        kind = "child_only_row_too_long"
                else:
                    kind = "child_only_row_unmapped"
            else:
                lower = max(0, child_start - TEXT_CONTEXT_CHARS)
                upper = min(len(parent), child_end + TEXT_CONTEXT_CHARS)
                boundaries = list(re.finditer(r"(?:[。！？!?；;]\s*|\n+|\.(?=\s))", parent))
                before = [m.end() for m in boundaries if m.end() <= child_start]
                after = [m.end() for m in boundaries if child_end <= m.start() < upper]
                span_start = before[-2] if len(before) > 1 and before[-2] >= lower else lower
                span_end = after[0] if after else upper
                if span_start <= child_start and span_end >= child_end:
                    ranges = [{"start": span_start, "end": span_end}]
                    kind = "text_context"
        if ranges is None:
            excerpt = child
            span = dict(kind=kind, coordinate="child", ranges=None, child_start=None,
                        child_end=None, length=len(child), max_chars=REFERENCE_MAX_CHARS)
        else:
            excerpt = "".join(parent[r["start"]:r["end"]] for r in ranges)
            if not excerpt or len(excerpt) > REFERENCE_MAX_CHARS:
                raise ValueError("host reference exceeds bound")
            span = dict(kind=kind, coordinate="parent", ranges=ranges,
                        child_start=child_start, child_end=child_end,
                        length=len(excerpt), max_chars=REFERENCE_MAX_CHARS)
        result = {"evidence_id": evidence_id,
                  **{key: row[key] for key in PROVENANCE_KEYS}}
        result.update(excerpt=excerpt, span=span,
                      excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                      context_chars=max(0, len(excerpt) - len(child)))
        return result
