# Agent 1.3 — Semantic Comparator ROI Review and Demo Readiness

**Review date:** 2026-10-03

**Decision:** `AGENT1_FROZEN_FOR_DEMO`

**Scope:** bounded evidence-first demonstrator; not engineering decision support.

## Decision

Freeze Agent 1. Do not spend another milestone tuning semantic scores on the
current 17 tasks. The project can move toward an end-to-end demonstration built
around scoped evidence research, host-rendered references, explicit coverage,
and honest incomplete outcomes. The semantic comparator remains experimental,
is not the application default, and must not be presented as a reliable
engineering-equivalence or compliance judge.

This is Decision A only for the stated demo goal. It does **not** mean the
comparator is ready to make technical decisions. If a demo depends on asserting
that two engineering requirements are equivalent or identifying their
engineering difference, the current comparator is a blocker for that claim.
The next milestone may assemble the bounded evidence workflow, but this review
does not start Agent 2 or implement demo changes.

No new model pass or bounded experiment was run. Agent 1.2 already isolated the
largest transport change on the unchanged cohort. The remaining aggregate
results establish a clear limitation, but do not identify a reliable small
patch: the prompt already instructs the model to classify dimensions
independently, and the current metrics do not establish that unit conversion is
the cause of the unit errors. Running another prompt variant against the same
small, AI-reviewed set would tune on the evaluation cohort and provide weak
generalization evidence.

## What the current system can demonstrate

- Agent 0 can run a bounded, scoped evidence investigation and distinguish
  completed, incomplete, clarification, invalid-action, timeout, and budget
  outcomes. Its ordinary comparison is explicitly a comparison of quoted text,
  not semantic engineering equivalence.
- The opt-in Agent 0.3 reference contract validates observed evidence IDs,
  scopes, successful lookups, and session ownership, then renders text and
  provenance from the active parsed-document registry. In its small frozen
  cohort, 11/11 selected references passed host authenticity checks. This means
  the rendered text and provenance matched the current parsed registry; it
  does not establish original-PDF fidelity, semantic relevance, or support for
  every claim. The relevance labels were provisional and AI-assisted.
- Agent 1.2's semantic model receives bounded evidence excerpts, but its output
  does not carry evidence IDs or citations. It is an isolated evaluation path,
  not an integrated/default application behavior. A demo that displays a
  semantic assessment would need to pair it with host-rendered source evidence
  and label the assessment experimental.
- The useful user-visible value today is finding, scoping, and inspecting
  evidence while surfacing missing or incomplete support. It is not a dependable
  automatic comparison of engineering conditions and units.

## Agent 1.2 evidence

Agent 1.2 used the same frozen 17-task cohort as Agent 1.1, with 5
`EQUIVALENT`, 6 `DIFFERENT`, and 6 `NOT_COMPARABLE` tasks. The semantic GT is
AI-reviewed against PDF evidence, not independently human-adjudicated.

| Measure | Result |
| --- | ---: |
| Provider responses completed | 17/17 |
| Full Host contract accepted | 15/17 (88.2%) |
| Strict verdict accuracy, invalid responses counted as failures | 14/17 (82.4%) |
| Verdict accuracy among accepted responses | 14/15 (93.3%) |
| Accepted dimension accuracy: object / value / unit / condition | 15/15 / 13/15 / 8/15 / 7/15 |
| Host-rejected consistency errors | 2/17 |
| Known-dimension `NOT_APPLICABLE` violations | 2/8 applicable value/unit checks |

There was one accepted wrong verdict. The two consistency failures were
rejected by the unchanged Host validator, so they were not silently turned
into accepted findings. Unit and condition/applicability errors remain common
even though the prompt tells the model to assess each dimension first and not
to use an object mismatch as a shortcut for `not_applicable`.

The one Agent 1.2 pass made 17 requests with no API errors or retries. Its
estimated usage cost was ¥0.03651888. Raw responses remain in the existing
ignored local output directory; this review neither reads nor modifies those
raw records.

## Readiness by concern

| Concern | Finding | Demo consequence |
| --- | --- | --- |
| Citation integrity | Host authenticates the selected ID, scope, current parsed text, and provenance; 11/11 passed in the Agent 0.3 cohort. | Suitable for a bounded provenance demonstration, with “current parsed source” stated accurately. |
| Citation support / relevance | Authenticity is not semantic relevance or claim support; labels were provisional and adjacent context was not independently reviewed word by word. | Show evidence beside claims and retain explicit unsupported/incomplete outcomes. Do not advertise a citation as proof. |
| Evidence grounding | Agent 1.2 compares supplied excerpts and cannot invent citation fields, but its final semantic dimensions are not host-verified against evidence. | Grounding is inspectable, not guaranteed. A user must be able to inspect the cited passage. |
| Obvious comparison mistakes | Object/field labels were 15/15, but unit 8/15 and condition/applicability 7/15; one accepted verdict was wrong and two outputs were rejected. | Critical blocker for authoritative equivalence, discrepancy, or compliance claims. Acceptable only as a clearly experimental overlay in a learning demo. |
| User-visible value | Scoped search, coverage, evidence lookup, provenance, and honest incompleteness are demonstrable without asserting semantic correctness. | Sufficient to continue an evidence-first demo; semantic decision quality is not yet the product value. |

## Failure classification

**Critical blockers for an authoritative technical-comparison feature:**

- Unit accuracy of 8/15 and condition/applicability accuracy of 7/15 among
  accepted outputs. A user could miss a relevant constraint or misunderstand
  whether two values are comparable.
- One accepted incorrect verdict (14/15 accepted verdict accuracy), plus two
  consistency errors safely rejected by Host validation.
- The semantic result does not include a host-validated claim-to-citation
  mapping. Provenance validation alone cannot establish that a cited span
  supports the model's interpretation.
- The evaluation oracle is AI-reviewed, not independently human-adjudicated;
  the 17-task cohort cannot establish stable field accuracy.

**Acceptable limitations for the bounded demo, if stated in the interface and
demo script:**

- Agent 1 remains optional and experimental; the deterministic evidence-first
  path can report excerpts and provenance without an engineering verdict.
- Agent 0.3 citations are authentic to the active parsed registry, not a proof
  of perfect PDF parsing or semantic support.
- Fixed-cohort results are small and local. Unsupported or incomplete evidence
  must remain visible rather than being converted into agreement.
- Qwen action selection and semantic comparison are experimental; neither is a
  production decision path.

**Low-ROI targets before the demo milestone:**

- More Agent planning. It does not address the observed dimension-label errors,
  and the Agent 1.2 runner does not exercise Agent planning.
- Expanding the same AI-reviewed benchmark before independent adjudication.
  More labels with the same authority would add volume without resolving the
  main uncertainty.
- A new model pass on the same 17 tasks. Without a separately adjudicated
  validation set, its apparent gain could reflect cohort tuning rather than
  general improvement.

## Expected ROI ranking

Ranks reflect expected engineering return **before** a bounded demo, combining
likely user impact, implementation cost, and strength of current evidence.

| Rank | Investment | Expected ROI now | Reason |
| ---: | --- | --- | --- |
| 1 | D. Better prompt / contract | Low to medium | Lowest implementation cost, but the current v2 prompt already states independent dimension rules and errors persist. A further rewrite needs new independently adjudicated evidence to avoid tuning the frozen cohort. |
| 2 | B. Condition/applicability reasoning | Medium potential, low near-term certainty | This is the weakest dimension (7/15) and important to the domain, but resolving applicability and comparability needs clear human adjudication and semantic evidence; a rule engine is not justified. |
| 3 | E. Stronger model | Low to medium, unmeasured | Could improve semantic judgments, but no controlled model comparison exists on a trustworthy validation set; model aliases/revisions may also change. |
| 4 | A. Unit normalization | Low now | The prompt explicitly does not convert units and says differing expressions require review. Current aggregates do not show that conversion errors dominate; adding normalization could silently equate different conditions or quantities. Revisit only after reviewing the concrete error types. |
| 5 | C. More benchmark cases | Low now; medium after adjudication | Independent human review and a held-out set would improve decision quality. Merely adding more AI-reviewed cases would not settle whether labels or model semantics are wrong. |
| 6 | F. More Agent planning | Very low for this bottleneck | Planning cannot correct unit or applicability interpretation, and Agent 1.2 is an isolated comparator. |

The highest next evidence value is not a larger model or another prompt pass. It
is an independent, targeted adjudication of existing semantic labels and
failure types, followed only if needed by a small held-out test. That is future
research, not a prerequisite for an evidence-first prototype demonstration.

## Recommended next milestone

Prepare an end-to-end demo milestone around the bounded evidence workflow:
explicit document scopes, host-validated evidence references, visible
provenance, and clear incomplete/unsupported results. Keep RAG and Agent 0
frozen. Do not make Agent 1.2 the default or present its verdict as authoritative.
If a semantic comparison is included as a preview, show the source evidence
beside every assessment and label unit/condition judgments as experimental.

This review changed documentation only. It did not modify source code, RAG,
Agent 0, benchmark or GT files, private PDFs/QA, or ignored outputs. No commit,
push, merge, or tag was made.
