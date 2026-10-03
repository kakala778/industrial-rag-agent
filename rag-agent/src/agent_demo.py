"""Run: python -m src.agent_demo --query ... --document A=... --document B=... --scopes A B"""

import argparse
import json
import os

from .agent.harness import AgentHarness
from .agent.io import load_corpus, write_private_result
from .agent.policy import DeterministicPolicy
from .agent.selector import OllamaActionSelector
from .agent.deepseek_selector import DeepSeekActionSelector
from .agent.tools import KnowledgeBaseSession


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agent 0: scoped evidence-text investigation")
    parser.add_argument("--query", required=True)
    parser.add_argument("--document", action="append", default=[], metavar="ALIAS=PATH")
    parser.add_argument("--scopes", nargs=2)
    parser.add_argument("--parser", choices=("pymupdf", "mineru"), default="pymupdf")
    parser.add_argument("--cache-root")
    parser.add_argument("--policy", choices=("deterministic", "qwen", "deepseek"), default="deterministic",
                        help="deepseek sends query/evidence excerpts to the paid official API")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-search-calls", type=int, default=4)
    parser.add_argument("--max-lookup-calls", type=int, default=6)
    parser.add_argument("--save-local", action="store_true",
                        help="Save private state/trace only to ignored outputs/agent0")
    args = parser.parse_args(argv)
    if not args.scopes or len(set(args.scopes)) != 2:
        print(json.dumps({"status": "clarify", "question": "请指定两份不同的文档别名。"}, ensure_ascii=False))
        return 2
    aliases = [s.partition("=")[0] for s in args.document]
    if any(s not in aliases for s in args.scopes):
        print(json.dumps({"status": "invalid_scope"}))
        return 2
    if args.policy == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print(json.dumps({"status": "error", "message": "deepseek:missing_key"}))
        return 1
    try:
        corpus = load_corpus(args.document, parser=args.parser, cache_root=args.cache_root)
        session = KnowledgeBaseSession(corpus)
        policy = (DeepSeekActionSelector() if args.policy == "deepseek" else
                  OllamaActionSelector() if args.policy == "qwen" else DeterministicPolicy())
        state = AgentHarness(session, policy, max_steps=args.max_steps,
                             max_search_calls=args.max_search_calls,
                             max_lookup_calls=args.max_lookup_calls).run(args.query, args.scopes)
    except (OSError, ValueError, RuntimeError):
        print(json.dumps({"status": "error", "message": "Session initialization failed; check local inputs/resources."}))
        return 1
    print(json.dumps({"status": state.status, "comparison": state.comparison,
                      "answer": state.answer, "steps": state.step_count,
                      "search_calls": state.search_calls, "lookup_calls": state.lookup_calls},
                     ensure_ascii=False, indent=2))
    if args.save_local:
        print(f"Private result: {write_private_result(state)}")
    return 0 if state.status == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
