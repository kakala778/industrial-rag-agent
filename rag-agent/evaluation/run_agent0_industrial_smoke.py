"""Private cached-PDF wiring smoke; no industrial task-accuracy claim.

Freeze a local manifest before inference. Existing Q01/Q06 select two different
document scopes, not hidden retrieval filters. Only Q01 has a pre-existing
one-sided keyword/page diagnostic; there is no cross-document semantic GT.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.harness import AgentHarness
from src.agent.io import APP_ROOT, load_corpus, write_private_result
from src.agent.policy import DeterministicPolicy
from src.agent.selector import OllamaActionSelector
from src.agent.tools import KnowledgeBaseSession


def inventory(paths):
    result = {}
    for path in sorted(set(Path(p).resolve() for p in paths)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result


def freeze_manifest(data_root, cache_root):
    root, cache = Path(data_root).resolve(), Path(cache_root).resolve()
    qa_files = list((root / "qa").rglob("m6_final_qa.local.json"))
    if len(qa_files) != 1:
        raise ValueError("one matching private QA file required")
    rows = json.loads(qa_files[0].read_text(encoding="utf-8"))
    pdfs = sorted((root / "input").glob("*.pdf"))
    sources = []
    for index in (0, 5):
        matches = [p for p in pdfs if p.name == Path(rows[index]["expected_source"]).name]
        if len(matches) != 1:
            raise ValueError("private PDF source is missing or ambiguous")
        sources.append(matches[0])
    if sources[0] == sources[1]:
        raise ValueError("smoke requires distinct real document sources")
    protected = pdfs + [p for p in cache.rglob("*") if p.is_file()] + [qa_files[0]]
    # Freeze query/aliases/oracle before any model call, never pass oracle to selector.
    manifest = {"version": "agent0-industrial-wiring-v1", "cache_root": str(cache),
                "documents": [f"A={sources[0]}", f"B={sources[1]}"],
                "query": "比较 A 与 B 中关于以下问题的证据，缺少依据时说明不足：" + rows[0]["question"],
                "scopes": ["A", "B"], "input_hashes": inventory(protected),
                "one_sided_diagnostic": {"scope": "A", "page": rows[0]["expected_page"],
                                         "keywords": rows[0]["expected_keywords"]},
                "semantic_cross_document_ground_truth": None}
    return write_private_result(manifest, name="industrial_manifest")


def run_smoke(manifest_path, *, policy="both", progress=print):
    path = Path(manifest_path).resolve()
    if not path.is_relative_to((APP_ROOT / "outputs" / "agent0").resolve()):
        raise ValueError("smoke manifest must be in ignored outputs/agent0")
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    before = inventory(manifest["input_hashes"])
    if before != manifest["input_hashes"]:
        raise ValueError("frozen smoke input hashes changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    calls = []
    def forbid_mineru(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("MinerU execution forbidden during smoke")
    start = time.monotonic()
    results, states = {}, {}
    with patch("src.mineru_loader._run_mineru", side_effect=forbid_mineru):
        corpus = load_corpus(manifest["documents"], parser="mineru",
                             cache_root=manifest["cache_root"])
        session = KnowledgeBaseSession(corpus)
        progress(f"Cached industrial corpus loaded: scopes=2, chunks={len(session.chunks)}")
        policies = ("deterministic", "qwen") if policy == "both" else (policy,)
        for selected in policies:
            selector = OllamaActionSelector() if selected == "qwen" else DeterministicPolicy()
            started = time.monotonic()
            state = AgentHarness(session, selector).run(manifest["query"], manifest["scopes"])
            grounded = all(row["quote"] in state.looked_up_evidence[row["evidence_id"]]["text"]
                           and row["scope"] == state.looked_up_evidence[row["evidence_id"]]["source"]
                           for row in state.findings)
            oracle = manifest["one_sided_diagnostic"]
            # Historical matcher diagnostic, not a new answer or field-relation judge.
            from evaluation.evaluate_pdf_retrieval import _contains_keywords_in_texts
            a_results = [r for h in state.search_history for r in h["results"]
                         if r["source"] == oracle["scope"] and r["page"] == oracle["page"]]
            result = {"status": state.status, "comparison": state.comparison,
                      "steps": state.step_count, "search_calls": state.search_calls,
                      "lookup_calls": state.lookup_calls, "finding_count": len(state.findings),
                      "scoped_search": all(h["scopes"][0] in manifest["scopes"] for h in state.search_history),
                      "citation_grounded": grounded if state.findings else None,
                      "one_sided_Q01_page_keyword_hit": bool(a_results) and _contains_keywords_in_texts(
                          [r["text"] for r in a_results], oracle["keywords"]),
                      "semantic_task_success": None,
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            results[selected], states[selected] = result, asdict(state)
            progress(json.dumps({"policy": selected, **result}, ensure_ascii=True))
    after = inventory(before)
    unchanged = before == after and path.read_bytes() == manifest_bytes
    output = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "protected_file_count": len(before), "protected_inputs_unchanged": unchanged,
              "mineru_calls": len(calls), "corpus_chunks": len(session.chunks),
              "total_seconds": round(time.monotonic() - start, 3), "results": results,
              "states": states}
    target = write_private_result(output, name="industrial_smoke")
    progress(f"Private smoke result: {target}")
    if not unchanged or calls:
        raise RuntimeError("smoke violated frozen-input or MinerU boundary")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--manifest")
    parser.add_argument("--policy", choices=("both", "deterministic", "qwen"), default="both")
    args = parser.parse_args(argv)
    if args.manifest:
        manifest = Path(args.manifest)
    else:
        if not args.data_root or not args.cache_root:
            parser.error("provide --manifest, or both --data-root and --cache-root")
        manifest = freeze_manifest(args.data_root, args.cache_root)
        print(f"Frozen private manifest: {manifest}", flush=True)
    result = run_smoke(manifest, policy=args.policy, progress=lambda message: print(message, flush=True))
    return 0 if all(r["status"] in ("finished", "incomplete") for r in result["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
