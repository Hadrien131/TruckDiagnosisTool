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
            "Structured context is in Task 1 output. DO NOT overwrite it.\n\n"
            "Call retrieve_issue_info_tool ONCE with this exact dict (fill values from Task 1 JSON):\n"
            "  make      → the vehicle_make value  (e.g. 'Volvo')\n"
            "  model     → the vehicle_model value  (e.g. 'FH')\n"
            "  symptoms  → concatenate ALL of: primary_symptoms field + free_text field + the key symptom "
            "words extracted from the user message: {user_message}\n"
            "  subsystem → suspected_subsystem value\n"
            "  mileage   → mileage value\n"
            "  year      → year value\n"
            "  recent_maintenance → recent_maintenance value from Task 1 JSON\n\n"
            "IMPORTANT: symptoms MUST be a non-empty string combining Task 1 fields with the user wording above.\n"
            "Return ONLY the verbatim tool output plus a ONE-line PASS/FAIL relevance note."
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
            "- Evidence section: if Task 2 returned CANDIDATE_ISSUES rows (any `_retrieval_method` value), "
            "these ARE real KB records — list key fields (Make_and_Model, Engine_Temperature, "
            "Vibration_Levels, Fuel_Consumption, Brake_Condition, Failure_History, Maintenance_Type). "
            "Label as 'KB parallel case'. Do NOT say 'no matches' when rows are present.\n"
            "- `_retrieval_method: numeric_symptom_match` means rows were scored by numeric symptom columns "
            "— treat them as valid evidence, same weight as text matches.\n"
            "- Only write 'No KB matches' when Task 2 output literally contains NO_MATCHES_FOUND.\n"
            "- When Task 2 contains CROSS_FLEET_FALLBACK: true, the user's vehicle is NOT in the KB. "
            "Open the Evidence section with: 'Your specific vehicle ([make] [model]) is not in our fleet KB. "
            "The closest symptom-matched cases from other vehicles are shown below.' "
            "Then list the rows as 'Cross-fleet parallel case:' and use them to inform probable causes.\n"
            "- Probable Causes: if retrieval truly empty, speak hypothetically; if rows present, tie causes "
            "to the actual numeric column values seen in those rows.\n"
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
