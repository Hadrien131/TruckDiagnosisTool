import json
import re
from typing import Any

import numpy as np
from crewai.tools import tool

from retrieval.retriever import retrieve_issues, retrieve_safety_notes, get_session_context

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


def _numeric_override(ctx: dict, issues: list, top_k: int = 5) -> list:
    """
    If retrieve_issues used TF-IDF (evidenced by no _retrieval_method on any row),
    re-run using explicit numeric symptom scoring directly in this layer.
    This is a belt-and-suspenders guard that operates independently of retriever.py
    version — the fix lives here so it can't be missed by a stale module cache.
    Only fires when both make and model are known (in-KB vehicle).
    """
    make = str(ctx.get("make") or "").strip()
    model = str(ctx.get("model") or "").strip()
    if not (make and model):
        return issues  # cross-fleet path; let retriever handle it

    # If numeric scoring already ran, nothing to do
    if any(i.get("_retrieval_method") == "numeric_symptom_match" for i in issues):
        return issues

    # TF-IDF path detected — apply numeric override
    try:
        from retrieval.retriever import _build_issues_search_index, _compute_numeric_scores

        df, _, _, _ = _build_issues_search_index(force=False)

        # Locate make/model column
        mm_col = next(
            (c for c in df.columns if "make_and_model" in c.lower()
             or ("make" in c.lower() and "model" in c.lower())),
            None,
        )
        if mm_col is None:
            return issues

        lowered = df[mm_col].astype(str).str.lower()
        mm_mask = lowered.str.contains(re.escape(make.lower()), na=False) & \
                  lowered.str.contains(re.escape(model.lower()), na=False)
        eligible = mm_mask.to_numpy(dtype=bool)

        if int(eligible.sum()) < 3:
            return issues  # too few rows — keep whatever retriever returned

        # Build symptom query from all available context fields + session
        query_hint = " ".join(filter(None, [
            str(ctx.get("symptoms") or ""),
            str(ctx.get("user_message") or ""),
            str(ctx.get("primary_symptoms") or ""),
            str(ctx.get("free_text") or ""),
        ]))

        eligible_df = df[eligible_mask := eligible].reset_index(drop=False)
        scores, n_signals = _compute_numeric_scores(eligible_df, query_hint)

        # Predictive score tiebreaker
        if "Predictive_Score" in eligible_df.columns:
            import pandas as pd
            pred = pd.to_numeric(eligible_df["Predictive_Score"], errors="coerce").fillna(0.0)
            pmax = float(pred.max())
            if pmax > 0:
                scores += 0.1 * (pred.values / pmax)

        score_max = float(scores.max()) if scores.size and scores.max() > 0 else 1.0
        norm_scores = scores / score_max
        top_idx = np.argsort(norm_scores)[::-1][:top_k]

        out = []
        for i in top_idx:
            row = eligible_df.iloc[int(i)].to_dict()
            row.pop("index", None)
            row["_retrieval_similarity"] = round(float(norm_scores[i]), 4)
            row["_retrieval_method"] = "numeric_symptom_match"
            out.append(row)

        if out:
            subset_k = int(eligible.sum())
            ctx["retrieval_note"] = (
                f"{make} {model} found in KB ({subset_k} rows); "
                f"numeric override applied ({n_signals} signal(s); "
                f"query={query_hint[:60]!r})."
            )
            ctx["make_model_not_in_kb"] = False
            return out

    except Exception as exc:  # noqa: BLE001
        ctx["retrieval_note"] = f"numeric_override_error: {exc}"

    return issues


@tool("retrieve_issue_info_tool")
def retrieve_issue_info_tool(query_context: dict[str, Any]) -> str:
    """
    Tool: Retrieve known vehicle maintenance rows from the fleet maintenance CSV.

    Pass keys like: make, model, subsystem, symptoms/primary_symptoms/free_text.

    Tool output is verbatim row fields plus an internal similarity score column
    `_retrieval_similarity` when matches are statistically strong enough.
    """
    ctx = _normalise_context(query_context)
    # Merge session context here (before debug line) so make/model/symptoms are
    # visible even when the LLM passes an empty dict. Global dict works across threads;
    # threading.local did not (CrewAI tool calls run in a different thread).
    for key, val in get_session_context().items():
        if key not in ctx or not ctx[key]:
            ctx[key] = val
    issues = retrieve_issues(ctx, top_k=5)
    # Belt-and-suspenders: if retriever used TF-IDF, force numeric scoring here
    issues = _numeric_override(ctx, issues)
    if not issues:
        note = ctx.get("retrieval_note", "")
        return f"NO_MATCHES_FOUND\n{note}" if note else "NO_MATCHES_FOUND"

    note = ctx.get("retrieval_note", "")
    cross_fleet = ctx.get("make_model_not_in_kb", False)
    header = (
        f"KB_NOTE: {note}\n"
        f"CROSS_FLEET_FALLBACK: {'true' if cross_fleet else 'false'}\n"
    ) if note else ""
    lines: list[str] = [header + "CANDIDATE_ISSUES_START"] if header else ["CANDIDATE_ISSUES_START"]
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
