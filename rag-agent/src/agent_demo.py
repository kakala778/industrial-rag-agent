"""Run M11: python -m src.agent_demo --task ... --document A=file.pdf --document B=manual.pdf.

The previous --query/--scopes invocation remains as an Agent 0 compatibility path.
"""

import argparse
import json
import os

from .agent.deepseek_selector import DeepSeekActionSelector
from .agent.harness import AgentHarness
from .agent.io import load_corpus, write_private_result, write_research_report
from .agent.policy import DeterministicPolicy, DeterministicReferencePolicy
from .agent.selector import OllamaActionSelector
from .agent.tools import KnowledgeBaseSession, SAFE_ALIAS


def _parser():
    parser = argparse.ArgumentParser(description="Bounded local industrial document research demo")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="M11 research task; scopes come from --document aliases")
    task_group.add_argument("--query", help="Legacy Agent 0 task (requires --scopes)")
    parser.add_argument("--document", action="append", default=[], metavar="ALIAS=PATH")
    parser.add_argument("--scopes", nargs=2, help="Legacy Agent 0: exactly two document aliases")
    parser.add_argument("--parser", choices=("pymupdf", "mineru"), default="pymupdf")
    parser.add_argument("--cache-root")
    parser.add_argument("--policy", choices=("deterministic", "qwen", "deepseek",
                                                "deepseek-reference"), default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-search-calls", type=int, default=4)
    parser.add_argument("--max-lookup-calls", type=int, default=6)
    parser.add_argument("--save-local", action="store_true",
                        help="Legacy Agent 0 only: save private state/trace under ignored outputs/agent0")
    return parser


def _document_aliases(specifications, *, minimum, maximum):
    if len(specifications) < minimum or (maximum is not None and len(specifications) > maximum):
        return None
    aliases = []
    for specification in specifications:
        if not isinstance(specification, str):
            return None
        alias, separator, path = specification.partition("=")
        if (not separator or not SAFE_ALIAS.fullmatch(alias) or not path.strip()
                or alias in aliases):
            return None
        aliases.append(alias)
    return aliases


def _m11_main(args):
    if (args.scopes is not None or args.save_local
            or not isinstance(args.task, str) or not args.task.strip()
            or len(args.task) > 4000):
        print("Error: M11 requires a bounded --task and two to four --document ALIAS=PATH arguments.")
        return 2
    aliases = _document_aliases(args.document, minimum=2, maximum=4)
    if aliases is None:
        print("Error: M11 requires two to four documents with unique safe aliases.")
        return 2

    policy_name = args.policy or "qwen"
    if policy_name == "deepseek":
        print("Error: M11 supports the DeepSeek evidence-reference selector via --policy deepseek-reference.")
        return 2
    if policy_name == "deepseek-reference" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print("Error: DeepSeek API key is unavailable; no documents were loaded and no report was written.")
        return 1

    try:
        corpus = load_corpus(args.document, parser=args.parser, cache_root=args.cache_root)
        session = KnowledgeBaseSession(corpus)
        if policy_name == "deterministic":
            policy = DeterministicReferencePolicy()
        elif policy_name == "qwen":
            policy = OllamaActionSelector(action_contract="evidence_reference")
        else:
            policy = DeepSeekActionSelector(action_contract="evidence_reference")

        if policy_name == "deepseek-reference":
            print("Privacy disclosure: the task and selected evidence excerpts leave this machine and are sent to the DeepSeek API.")
        state = AgentHarness(session, policy, max_steps=args.max_steps,
                             max_search_calls=args.max_search_calls,
                             max_lookup_calls=args.max_lookup_calls).run(args.task, aliases)
        report_path = write_research_report(state, session)
    except (OSError, ValueError, RuntimeError):
        print("Error: research session or report creation failed; no report path is available.")
        return 1

    print(f"Status: {state.status}")
    print(f"Report: {report_path.as_posix()}")
    return 0 if state.status == "finished" else 2


def _legacy_agent0_main(args):
    if not args.scopes or len(set(args.scopes)) != 2:
        print(json.dumps({"status": "clarify", "question": "请指定两份不同的文档别名。"}, ensure_ascii=False))
        return 2
    aliases = _document_aliases(args.document, minimum=2, maximum=None)
    if aliases is None or any(scope not in aliases for scope in args.scopes):
        print(json.dumps({"status": "invalid_scope"}, ensure_ascii=False))
        return 2

    policy_name = args.policy or "deterministic"
    if policy_name in ("deepseek", "deepseek-reference") and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print(json.dumps({"status": "error", "message": "deepseek:missing_key"}, ensure_ascii=False))
        return 1
    try:
        corpus = load_corpus(args.document, parser=args.parser, cache_root=args.cache_root)
        session = KnowledgeBaseSession(corpus)
        policy = (DeepSeekActionSelector(action_contract="evidence_reference")
                  if policy_name == "deepseek-reference" else
                  DeepSeekActionSelector() if policy_name == "deepseek" else
                  OllamaActionSelector() if policy_name == "qwen" else DeterministicPolicy())
        if policy_name in ("deepseek", "deepseek-reference"):
            print("Privacy disclosure: the task and selected evidence excerpts leave this machine and are sent to the DeepSeek API.")
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


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.task is not None:
        return _m11_main(args)
    if args.query is not None:
        return _legacy_agent0_main(args)
    print("Error: provide --task for M11 research reports or --query for the legacy Agent 0 path.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
