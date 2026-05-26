from typing import List

from crewai import Agent

from crewai_layer.tools import (
    build_diagnostic_plan_tool,
    retrieve_issue_info_tool,
    safety_check_tool,
)
from models.llm_client import get_planner_safety_llm, get_supervisor_llm


def create_supervisor_agent() -> Agent:
    return Agent(
        role="Truck Diagnostic Supervisor Agent",
        goal=(
            "Coordinate truck diagnostics safely; final answers MUST reflect ONLY the user's stated "
            "vehicle + symptoms ({user_message}) and sidebar fields ({ui_context})."
        ),
        backstory=(
            "Workshop foreman: structured diagnostics, refusal to retrofit unrelated KB rows into "
            "the customer's story, and concise Markdown output."
        ),
        llm=get_supervisor_llm(),
        verbose=False,
        allow_delegation=False,
        tools=[],
    )


def create_retriever_agent() -> Agent:
    return Agent(
        role="Truck Diagnostic Retrieval Agent",
        goal=(
            "Call retrieve_issue_info_tool exactly once using symptoms + STRUCTURED CONTEXT from "
            "Task 1 merged with {ui_context}; never hallucinate unseen rows."
        ),
        backstory=(
            "You only summarize tool output verbatim; tool already enforces similarity gating."
        ),
        llm=get_planner_safety_llm(),
        verbose=False,
        allow_delegation=False,
        tools=[retrieve_issue_info_tool],
    )


def create_planner_agent() -> Agent:
    return Agent(
        role="Diagnostic Planner Agent",
        goal=(
            "Turn KB candidates + customer's words into actionable steps WITHOUT inventing mismatched trucks."
        ),
        backstory="Senior diagnostics engineer disciplined about evidence-vs-speculation labeling.",
        llm=get_planner_safety_llm(),
        verbose=False,
        allow_delegation=False,
        tools=[build_diagnostic_plan_tool],
    )


def create_safety_agent() -> Agent:
    return Agent(
        role="Safety Review Agent",
        goal="Layer PPE/Lock-out guidance onto the drafted plan strictly from safety bullets + plan text.",
        backstory=(
            "Fleet safety officer insisting on verbatim hazard language when uncertain and blocking "
            "reckless troubleshooting steps."
        ),
        llm=get_planner_safety_llm(),
        verbose=False,
        allow_delegation=False,
        tools=[safety_check_tool],
    )


def create_all_agents() -> List[Agent]:
    return [
        create_supervisor_agent(),
        create_retriever_agent(),
        create_planner_agent(),
        create_safety_agent(),
    ]
