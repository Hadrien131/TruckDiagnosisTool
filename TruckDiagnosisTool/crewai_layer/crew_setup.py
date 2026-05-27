from __future__ import annotations

import json
from typing import Any, Dict, List

from crewai import Crew, Process

from crewai_layer.agents import create_all_agents
from crewai_layer.tasks import (
    create_analyze_symptom_task,
    create_plan_diagnostics_task,
    create_retrieve_issues_task,
    create_review_safety_task,
    create_synthesize_response_task,
)
from retrieval.retriever import prime_truck_issues_index, set_session_context
from utils.logging_utils import get_project_logger

logger = get_project_logger()


def build_tasks(agents) -> List:
    supervisor = [a for a in agents if "Supervisor" in a.role][0]
    retriever = [a for a in agents if "Retrieval" in a.role][0]
    planner = [a for a in agents if "Planner" in a.role][0]
    safety = [a for a in agents if "Safety" in a.role][0]

    t1 = create_analyze_symptom_task(supervisor)
    t2 = create_retrieve_issues_task(retriever)
    t3 = create_plan_diagnostics_task(planner)
    t4 = create_review_safety_task(safety)
    t5 = create_synthesize_response_task(supervisor)

    t2.context = [t1]
    t3.context = [t1, t2]
    t4.context = [t1, t3]
    t5.context = [t1, t2, t3, t4]

    return [t1, t2, t3, t4, t5]


def _format_conversation(history: List[Dict[str, str]], max_turns: int = 14, max_chars: int = 6000) -> str:
    snippets: List[str] = []

    tail = history[-max_turns:] if history else []

    for turn in tail:
        role = str(turn.get("role", "user"))
        msg = str(turn.get("content", "")).strip()
        if not msg:
            continue
        snippets.append(f"{role.upper()}: {msg}")

    merged = "\n".join(snippets)
    if len(merged) <= max_chars:
        return merged
    return "...[conversation truncated]\n" + merged[-max_chars:]


def run_diagnostic_crew(
    conversation_history: List[Dict[str, str]],
    user_message: str,
    ui_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Invoke the sequential Crew with explicit template inputs CrewAI substitutes into Task strings.
    """
    prime_truck_issues_index()

    ui_context = dict(ui_context or {})
    formatted_history = _format_conversation(conversation_history)
    serialized_ui = json.dumps(ui_context, indent=2, default=str)

    # Inject full context as server-side fallback so the retriever has user_message
    # and all sidebar fields even when the LLM passes a sparse query_context dict.
    # 'symptoms' is set to user_message so the retriever uses actual symptom text as the
    # TF-IDF query, not just "Volvo FH" which produces near-identical scores for all rows.
    set_session_context({
        "make": ui_context.get("make", ""),
        "model": ui_context.get("model", ""),
        "year": ui_context.get("year", ""),
        "mileage": ui_context.get("mileage", ""),
        "symptoms": user_message.strip(),
        "user_message": user_message.strip(),
        "recent_maintenance": ui_context.get("recent_maintenance", ""),
        "ambient_notes": ui_context.get("ambient_notes", ""),
        "load_condition": ui_context.get("load_condition", ""),
    })

    agents = create_all_agents()

    tasks = build_tasks(agents)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    inputs = {
        "conversation_history": formatted_history,
        "user_message": user_message.strip(),
        "ui_context": serialized_ui,
    }

    logger.info("Starting diagnostic crew for new message")

    crew_output = crew.kickoff(inputs=inputs)

    raw = crew_output.raw

    tasks_output_texts: List[str] = []

    try:

        for to in crew_output.tasks_output:  # type: ignore[attr-defined]

            tasks_output_texts.append(getattr(to, "raw", str(to)))

    except Exception:

        pass

    return {
        "final_markdown": raw,
        "raw_output": raw,
        "tasks_output": tasks_output_texts,
    }


def warm_kb_for_streamlit() -> None:
    """
    Lightweight hook referenced from Streamlit cache layer to front-load embeddings.
    """
    prime_truck_issues_index()
