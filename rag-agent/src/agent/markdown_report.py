"""Host-rendered Markdown reports for the private M11 research demo."""

from html import escape as escape_html
import re

from .tools import REFERENCE_MAX_CHARS, SAFE_ALIAS


_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?:[^\\/\s<>:\"|?*]+[\\/])*[^\\/\s<>:\"|?*]*)"
)
_UNC_PATH = re.compile(r"(?<![A-Za-z0-9])\\\\[^\\/\s]+\\[^\\/\s]+(?:\\[^\\/\s]+)*")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:[^/\s<>:\"|?*]+/)+[^/\s<>:\"|?*]*")
_MARKDOWN_PUNCTUATION = re.compile(r"([\\`*_{}\[\]()#+\-.!_|>])")
_KNOWN_TERMINAL_STATUSES = {
    "finished", "clarify", "incomplete", "budget_exceeded", "tool_error",
    "timeout", "invalid_scope", "invalid_action",
}


def _redact_paths(value):
    value = _UNC_PATH.sub("[path omitted]", value)
    value = _WINDOWS_PATH.sub("[path omitted]", value)
    return _POSIX_PATH.sub("[path omitted]", value)


def _safe_text(value, *, maximum):
    if not isinstance(value, str):
        return ""
    value = _redact_paths(value.replace("\r\n", "\n").replace("\r", "\n"))
    value = "".join(character for character in value
                    if character in "\n\t" or ord(character) >= 32)
    return value[:maximum]


def _escape_line(value):
    return _MARKDOWN_PUNCTUATION.sub(r"\\\1", escape_html(value, quote=False))


def _blockquote(value):
    text = _safe_text(value, maximum=4000)
    lines = text.split("\n") if text else ["(empty)"]
    return "\n".join("> " + _escape_line(line) for line in lines)


def _safe_alias(value):
    return value if isinstance(value, str) and SAFE_ALIAS.fullmatch(value) else "invalid-alias"


def _host_reference(evidence_id, scope, state, session):
    if (not isinstance(evidence_id, str) or evidence_id not in state.evidence_ids
            or evidence_id not in state.looked_up_evidence):
        raise ValueError("report evidence must be observed and looked up")
    lookup = state.looked_up_evidence[evidence_id]
    if lookup.get("source") != scope:
        raise ValueError("report evidence scope does not match its alias")
    observed = any(
        history.get("scopes") == [scope] and history.get("status") == "ok"
        and any(row.get("evidence_id") == evidence_id for row in history.get("results", []))
        for history in state.search_history
    )
    if not observed:
        raise ValueError("report evidence was not observed in a scoped search")
    renderer = getattr(session, "render_evidence_reference", None)
    if not callable(renderer):
        raise ValueError("active session cannot validate report evidence")
    reference = renderer(evidence_id)
    if (not isinstance(reference, dict) or reference.get("evidence_id") != evidence_id
            or reference.get("source") != scope
            or not isinstance(reference.get("excerpt"), str)
            or not reference["excerpt"]
            or len(reference["excerpt"]) > REFERENCE_MAX_CHARS
            or (reference.get("page") is not None
                and (type(reference["page"]) is not int or reference["page"] < 1))
            or not isinstance(reference.get("block_type"), str)
            or not SAFE_ALIAS.fullmatch(reference["block_type"])
            or type(reference.get("block_index")) is not int
            or reference["block_index"] < 0):
        raise ValueError("active session returned an invalid host reference")
    return reference


def _evidence_sections(state, session, *, preliminary):
    rows = []
    if preliminary:
        for evidence_id in state.evidence_ids:
            lookup = state.looked_up_evidence.get(evidence_id)
            if not isinstance(lookup, dict):
                continue
            scope = _safe_alias(lookup.get("source"))
            if scope not in {_safe_alias(item) for item in state.requested_scopes}:
                continue
            rows.append((scope, _host_reference(evidence_id, scope, state, session)))
    else:
        for finding in state.findings:
            scope = _safe_alias(finding.get("scope"))
            status = finding.get("status")
            if status != "evidence_found":
                continue
            evidence = finding.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("finished evidence outcome has no host references")
            for item in evidence:
                if not isinstance(item, dict):
                    raise ValueError("finished evidence reference is malformed")
                rows.append((scope, _host_reference(item.get("evidence_id"), scope, state, session)))

    if not rows:
        return ["No host-validated looked-up evidence is available."]
    sections = []
    for scope, reference in rows:
        page = "page unavailable" if reference["page"] is None else f"page {reference['page']}"
        location = f"{page}; block {reference['block_type']}:{reference['block_index']}"
        sections.extend([
            f"### Source `{scope}`",
            f"- Citation: `{reference['evidence_id']}`",
            f"- Location: {location}",
            _blockquote(reference["excerpt"]),
            "",
        ])
    return sections[:-1] if sections and sections[-1] == "" else sections


def render_research_report(state, session):
    """Render task-local state and revalidated active-session evidence only."""
    status = state.status if state.status in _KNOWN_TERMINAL_STATUSES else "unknown"
    task = _safe_text(state.original_query, maximum=4000)
    scopes = state.resolved_scopes or state.requested_scopes
    safe_scopes = [_safe_alias(scope) for scope in scopes]
    lines = [
        "# Industrial Research Report",
        "",
        "## Task",
        _blockquote(task),
        "",
        "## Sources",
    ]
    lines.extend(f"- `{scope}`" for scope in safe_scopes)
    if not safe_scopes:
        lines.append("- No valid document alias was resolved.")

    lines.extend(["", "## Evidence"])
    preliminary = status != "finished"
    if preliminary:
        lines.extend([
            "> Preliminary only: the Agent did not complete a FINISH action.",
            "",
        ])
    lines.extend(_evidence_sections(state, session, preliminary=preliminary))

    lines.extend(["", "## Findings", f"Terminal status: `{status}`"])
    if status == "finished":
        explanations = {
            "evidence_found": "selected host-validated source records; this does not establish relevance or correctness",
            "no_evidence_found": "the configured search returned no candidates; this does not establish that the document contains no relevant information",
            "insufficient_scope": "candidates were returned and looked up, but no evidence IDs were selected",
        }
        observed_scopes = set()
        for finding in state.findings:
            if not isinstance(finding, dict):
                raise ValueError("finished finding is malformed")
            scope = _safe_alias(finding.get("scope"))
            outcome = finding.get("status")
            if scope not in safe_scopes or outcome not in explanations or scope in observed_scopes:
                raise ValueError("finished findings contain an invalid scope status")
            observed_scopes.add(scope)
            history = [row for row in state.search_history
                       if row.get("scopes") == [scope] and row.get("status") in ("ok", "no_evidence")]
            if not history:
                raise ValueError("finished scope has no successful search record")
            candidates = {item.get("evidence_id") for row in history
                          if row.get("status") == "ok" for item in row.get("results", [])}
            looked_candidates = candidates.intersection(state.looked_up_evidence)
            if outcome == "no_evidence_found" and candidates:
                raise ValueError("no_evidence_found contradicts observed candidates")
            if outcome == "insufficient_scope" and (not candidates or not looked_candidates):
                raise ValueError("insufficient_scope requires a looked-up candidate")
            line = f"- Source `{scope}`: `{outcome}` — {explanations[outcome]}"
            if outcome == "evidence_found":
                references = _finished_reference_pairs(finding, scope, state, session)
                ids = [item["evidence_id"] for row_scope, item in references]
                if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
                    raise ValueError("evidence_found requires one to three unique references")
                if any(evidence_id not in candidates for evidence_id in ids):
                    raise ValueError("report evidence was not returned by a scoped search")
                line += "; citations: " + ", ".join(f"`{evidence_id}`" for evidence_id in ids)
            lines.append(line)
        if observed_scopes != set(safe_scopes):
            raise ValueError("finished report requires one outcome per source alias")
    else:
        lines.append("The Agent did not produce final per-source findings.")
        if status == "clarify":
            lines.extend(["", "## Clarification", _blockquote(state.pending_clarification)])

    lines.extend([
        "",
        "## Limitations",
        "- Citations and provenance identify source locations; they do not prove parser fidelity, relevance, support, correctness, applicability, equivalence, or compliance.",
        "- This report provides no engineering or compliance verdict.",
    ])
    if preliminary:
        lines.append("- Any displayed evidence is preliminary and was not finalized by FINISH.")
    return "\n".join(lines).rstrip() + "\n"


def _finished_reference_pairs(finding, scope, state, session):
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("finished evidence outcome has no host references")
    if any(not isinstance(item, dict) for item in evidence):
        raise ValueError("finished evidence reference is malformed")
    return [(scope, _host_reference(item.get("evidence_id"), scope, state, session))
            for item in evidence]
