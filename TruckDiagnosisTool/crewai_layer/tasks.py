from crewai import Agent, Task


def create_analyze_symptom_task(supervisor: Agent) -> Task:
    return Task(
        description=(
            "CONVERSATION (last turns):\n{conversation_history}\n\n"
            "LATEST USER MESSAGE:\n{user_message}\n\n"
            "SIDEBAR / FORM CONTEXT (truth for vehicle meta):\n{ui_context}\n\n"
            "Extract BOTH the user's verbatim complaint AND sidebar vehicle facts.\n"
            "Return ONE short paragraph recap plus a STRUCTURED_JSON block formatted EXACTLY as:\n"
            "STRUCTURED_CONTEXT_START\n"
            '{"vehicle_make":"","vehicle_model":"","year":"","mileage":"","load_condition":"","'
            'ambient_temperature":"","recent_maintenance":"","primary_symptoms":"","operating_conditions":"",'
            '"suspected_subsystem":"","maintenance_history_signals":"","free_text":"<full user wording>"}'
            "\nSTRUCTURED_CONTEXT_END\n"
            "If information is absent, leave empty strings—do NOT guess vehicle identity."
        ),
        expected_output="Paragraph + fenced STRUCTURED_CONTEXT JSON block usable by retrieval tool.",
        agent=supervisor,
    )


def create_retrieve_issues_task(retriever: Agent) -> Task:
    return Task(
        description=(
            "Structured context sits in outputs of Task 1; DO NOT overwrite it.\n\n"
            "TASK: Immediately call retrieve_issue_info_tool once with Python dict literal keys matching "
            '{"make":"vehicle_make","model":"vehicle_model","symptoms":"<primary_symptoms + free_text + '
            '{user_message} keywords>","subsystem":"suspected_subsystem","mileage":"<from JSON>",'
            '"year":"<from JSON>"}'
            "; inject missing blanks if necessary.\n"
            "Return ONLY the verbatim tool blob (CANDIDATE_ISSUES_*) plus a ONE-line note PASS/FAIL on "
            "whether rows appear relevant versus {user_message}."
        ),
        expected_output="Tool output verbatim + one-line PASS/FAIL relevance note.",
        agent=retriever,
        tools=retriever.tools,
    )


def create_plan_diagnostics_task(planner: Agent) -> Task:
    return Task(
        description=(
            "Inputs: Structured context from Task 1, retrieval blob from Task 2, ORIGINAL SYMPTOMS {user_message}.\n\n"
            "CALL build_diagnostic_plan_tool(candidate_issues_block=<Task2 output>, context=<parsed JSON dict from "
            "Task1 STRUCTURED CONTEXT>, ... ) EXACT signatures per tool docs.\n"
            "Then polish into DIAGNOSTIC_PLAN_REFINED_START / END emphasizing:\n"
            "- Separate **Customer vehicle facts** vs **Historical KB rows**\n"
            "- If retrieval was NO_MATCHES or FAIL, give generic verification-first workflow & stop inventing fleets."
        ),
        expected_output="DIAGNOSTIC_PLAN draft wrapper + DIAGNOSTIC_PLAN_REFINED block.",
        agent=planner,
        tools=planner.tools,
    )


def create_review_safety_task(safety_agent: Agent) -> Task:
    return Task(
        description=(
            "Given refined diagnostic plan from Task 3 and suspected_subsystem from structured JSON, invoke "
            "safety_check_tool with that subsystem string plus the plan body.\n"
            "Preserve SAFETY markers from tool.\n\n"
            "Customer symptoms reference: {user_message}"
        ),
        expected_output="SAFETY_REVIEW block referencing original plan bullets.",
        agent=safety_agent,
        tools=safety_agent.tools,
    )


def create_synthesize_response_task(supervisor: Agent) -> Task:
    return Task(
        description=(
            "Supervisor FINAL answer for {user_message} using Tasks 1-4 ONLY.\n\n"
            "**GROUNDING CONTRACT**\n"
            "- Summary mirrors sidebar + STRUCTURED CONTEXT + user wording; NEVER swap in KB vehicle IDs.\n"
            "- Probable Causes: if retrieval weak/empty, headline that KB lacked matches and speak hypothetically.\n"
            "- When citing KB rows, start bullets with similarity score `_retrieval_similarity` echo if present "
            "or label as 'Possible parallel case:'.\n"
            "- Forbidden: pretending a Volvo case is the user's Peterbilt (example pattern).\n\n"
            "## Summary of the Situation\n"
            "## Evidence from Fleet KB (explicitly labeled, optional)\n"
            "## Probable Causes\n"
            "## Diagnostic Workflow (Step-by-Step)\n"
            "## Required Tools & Parts\n"
            "## Estimated Time\n"
            "## Safety Considerations\n"
            "## Notes & Uncertainties\n"
            "Close with reassurance to verify on-vehicle readings/codes.\n\n"
            "Operational context recap: sidebar JSON {ui_context}\n"
            "Conversation tail: {conversation_history}"
        ),
        expected_output="Markdown adhering to headings above; truthful about retrieval strength.",
        agent=supervisor,
    )
