import json
from typing import Any

from crewai.tools import tool

from retrieval.retriever import retrieve_issues, retrieve_safety_notes

# Alternate key names the LLM sometimes uses → canonical name expected by retriever
_KEY_ALIASES = {
    "vehicle_make": "make",
    "vehicle_model": "model",
    "primary_symptoms": "symptoms",
    "symptom": "symptoms",
    "free_text": "symptoms",
    "suspected_subsystem": "subsystem",
    "vehicle_year": "year",
    "usage_hours": "mileage",
}


def _normalise_context(raw: Any) -> dict[str, Any]:
    """Accept dict or JSON string; normalise aliased keys."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {"free_text": raw}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        canonical = _KEY_ALIASES.get(k.lower().strip(), k.lower().strip())
        out[canonical] = v
    # merge symptom-like fields into a single "symptoms" string
    parts = [str(out.get("symptoms") or ""), str(out.get("free_text") or "")]
    merged = " ".join(p for p in parts if p).strip()
    if merged:
        out["symptoms"] = merged
    return out


@tool("retrieve_issue_info_tool")
def retrieve_issue_info_tool(query_context: dict[str, Any]) -> str:
    """
    Tool: Retrieve known vehicle maintenance rows from the fleet maintenance CSV.

    Pass keys like: make, model, subsystem, symptoms/primary_symptoms/free_text.

    Tool output is verbatim row fields plus an internal similarity score column
    `_retrieval_similarity` when matches are statistically strong enough.
    """
    ctx = _normalise_context(query_context)
    issues = retrieve_issues(ctx, top_k=5)
    sim = float(ctx.get("retrieval_best_similarity", 0.0) or 0.0)
    method = ctx.get("retrieval_method", "tfidf")
    debug_line = (
        f"[DEBUG make={ctx.get('make','?')} model={ctx.get('model','?')} "
        f"symptoms_len={len(str(ctx.get('symptoms','')))} "
        f"sim={sim:.4f} method={method} note={ctx.get('retrieval_note','')}]"
    )
    if not issues:
        return f"{debug_line}\nNO_MATCHES_FOUND"

    lines: list[str] = [debug_line, "CANDIDATE_ISSUES_START"]
    for i, issue in enumerate(issues, start=1):

        lines.append(f"- ISSUE #{i}")

        keys = sorted(issue.keys())

        for key in keys:

            val = issue[key]

            lines.append(f"  {key}: {val}")

    lines.append(
        "\nRULE: Probable Causes must explicitly tie KB rows back to the operator's wording; "
        "discard any row whose vehicle/details conflict with STRUCTURED CONTEXT."
    )
    lines.append("CANDIDATE_ISSUES_END")
    return "\n".join(lines)


@tool("build_diagnostic_plan_tool")
def build_diagnostic_plan_tool(candidate_issues_block: str, context: dict[str, Any]) -> str:
    """
    Take candidate issues plus UI context dict and produce draft planning scaffolding.
    """
    return (
        "DIAGNOSTIC_PLAN_DRAFT_START\n"
        "Treat the retrieved maintenance rows as SOFT evidence anchors for {user_message}.\n"
        "If candidate block is NO_MATCHES_FOUND or rows clearly mismatch the customer's truck/\n"
        "symptoms, write a GENERAL diagnostic playbook only (verification steps first), cite NO\n"
        "specific phantom vehicle from the KB.\n"
        f"{candidate_issues_block}\n\n"
        "Context from UI/forms + structured summary:\n"
        f"{context}\n"
        "DIAGNOSTIC_PLAN_DRAFT_END"
    )


@tool("safety_check_tool")
def safety_check_tool(diagnostic_plan_text: str, subsystem: str | None = None) -> str:
    """
    Attach safety bulletins to a drafted plan.
    """
    notes = retrieve_safety_notes(subsystem)
    lines = ["SAFETY_REVIEW_START", "Original plan:", diagnostic_plan_text, "\nRelevant safety bulletins:"]
    for n in notes:

        lines.append(f"- [{n['subsystem']}] {n['hazard']}: {n['note']}")

    lines.append("SAFETY_REVIEW_END")
    return "\n".join(lines)
