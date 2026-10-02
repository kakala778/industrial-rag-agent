"""Fixed in-memory BM25 and rank fusion for controlled retrieval experiments."""

from collections import Counter, defaultdict
from copy import deepcopy
import math
import re
import unicodedata

BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/.][a-z0-9]+)*|[\u3400-\u9fff]+")
DASH_TRANSLATION = str.maketrans({char: "-" for char in "‐‑‒–—−"})


def tokenize(text):
    """Keep identifiers intact; emit Chinese characters and adjacent bigrams."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold().translate(DASH_TRANSLATION)
    tokens = []
    for match in TOKEN_PATTERN.finditer(normalized):
        token = match.group()
        if "\u3400" <= token[0] <= "\u9fff":
            tokens.extend(token)
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def chunk_identity(chunk):
    """chunk_id is local to each Document, so location is part of the key."""
    metadata = chunk.get("metadata") or {}
    source = chunk.get("source", metadata.get("source"))
    chunk_id = chunk.get("chunk_id")
    if not isinstance(source, str) or not isinstance(chunk_id, int) or isinstance(chunk_id, bool):
        raise ValueError("candidate requires source and integer chunk_id")
    return (source, metadata.get("page"), metadata.get("block_type"),
            metadata.get("block_index"), chunk_id)


def _candidate(chunk, score, corpus_index):
    return {"text": chunk["text"], "source": chunk["source"],
            "chunk_id": chunk["chunk_id"], "metadata": deepcopy(chunk.get("metadata") or {}),
            "score": float(score), "corpus_index": corpus_index}


class BM25Index:
    """Postings-based BM25; zero-overlap chunks are never returned."""

    def __init__(self, chunks, *, k1=BM25_K1, b=BM25_B):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        if len({chunk_identity(chunk) for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("corpus has duplicate structural chunk identities")
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        for index, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(chunk["text"]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.idf = {term: math.log1p((len(self.chunks) - len(entries) + 0.5) /
                                    (len(entries) + 0.5))
                    for term, entries in self.postings.items()}

    def search(self, query, top_k=3):
        terms = dict.fromkeys(tokenize(query))
        if top_k <= 0 or not terms or not self.average_length:
            return []
        scores = defaultdict(float)
        for term in terms:
            for index, frequency in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += self.idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
        return [_candidate(self.chunks[index], scores[index], index) for index in ranked]


def rrf_fuse(dense_results, bm25_results, *, top_k=20, k=RRF_K):
    """Equal-weight reciprocal ranks with one vote per unique chunk per list."""
    if not math.isfinite(k) or k < 0:
        raise ValueError("RRF k must be finite and nonnegative")
    if top_k <= 0:
        return []
    fused = {}
    for label, results in (("dense", dense_results), ("bm25", bm25_results)):
        seen = set()
        unique_rank = 0
        for candidate in results:
            identity = chunk_identity(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            unique_rank += 1
            if identity not in fused:
                fused[identity] = dict(candidate, metadata=deepcopy(candidate.get("metadata") or {}),
                                       rrf_score=0.0, dense_rank=None, bm25_rank=None)
            row = fused[identity]
            row[f"{label}_rank"] = unique_rank
            row[f"{label}_score"] = candidate.get("score")
            row["rrf_score"] += 1 / (k + unique_rank)
    # Python's stable sort retains dense-first discovery order for equal sums.
    ranked = sorted(fused.values(), key=lambda row: -row["rrf_score"])[:top_k]
    for row in ranked:
        row["score"] = row["rrf_score"]
    return ranked
