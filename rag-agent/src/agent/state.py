"""Explicit task-local state. Raw hidden reasoning is never a state field."""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    original_query: str
    requested_scopes: list
    resolved_scopes: list = field(default_factory=list)
    search_history: list = field(default_factory=list)
    evidence_ids: list = field(default_factory=list)
    lookup_history: list = field(default_factory=list)
    looked_up_evidence: dict = field(default_factory=dict)
    pending_clarification: str = ""
    step_count: int = 0
    search_calls: int = 0
    lookup_calls: int = 0
    remaining_budget: dict = field(default_factory=dict)
    status: str = "running"
    findings: list = field(default_factory=list)
    comparison: str = "insufficient_evidence"
    answer: str = ""
    errors: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    clarification_required: str = ""
