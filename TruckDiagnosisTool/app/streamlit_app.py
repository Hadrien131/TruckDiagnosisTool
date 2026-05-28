"""Streamlit facade for the CrewAI-assisted truck diagnostic workflow."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path so sibling packages (crewai_layer, utils, etc.)
# are importable when Streamlit Cloud runs the app from the app/ subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
# Disable OpenTelemetry before crewai imports — without a collector the GRPC
# connection attempt hangs silently on Streamlit Cloud.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import streamlit as st

from utils.config import ensure_openai_api_key
from utils.security import sanitize_user_input

_BASE = Path(__file__).resolve().parent.parent

_CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}
.hero {
    background: linear-gradient(118deg,#0f2027,#203a43,#2c5364);
    padding: 1.6rem 1.5rem;
    border-radius: 16px;
    color: #eef5ff;
    margin-bottom: 1.35rem;
    box-shadow: 0 15px 40px rgba(4,29,72,0.35);
}
.hero h1 { margin: 0; font-size: 1.8rem; }
.hero p {
    margin: .45rem 0 0;
    color: #cddcf5;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(15,63,134,0.12);
}
div[data-testid="stMarkdownContainer"] p { line-height: 1.52; }
.stSpinner > div { border-top-color:#4fc3ff !important; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _strip_code_fence(text: str) -> str:
    """Remove ``` fences the LLM sometimes wraps its markdown output in."""
    text = text.strip()
    # e.g. ```markdown\n...\n``` or ```\n...\n```
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


@st.cache_resource(show_spinner=False)
def _get_crew_runner():
    from crewai_layer.crew_setup import run_diagnostic_crew
    return run_diagnostic_crew


def init_session_state() -> None:
    _load_streamlit_secret_keys()
    ensure_openai_api_key()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi — I'm your diagnostics copilot.\n\n"
                    "Fill in the sidebar, then describe what the vehicle is actually doing "
                    "(symptoms first, fault codes optional). Always verify readings on the truck."
                ),
            },
        ]
    if "_kb_ready" not in st.session_state:
        # Build index lazily so the page renders before the heavy TF-IDF fit.
        st.session_state["_kb_ready"] = False


def _safe_default_load(selection: list[str], value: str) -> str:
    return value if value in selection else selection[0]


def side_vehicle_form() -> Dict[str, Any]:
    load_opts = ["Empty", "Half", "Full"]
    default_load = getattr(st.session_state, "load", "Full")
    with st.sidebar:
        st.markdown("### Vehicle dossier")
        make = st.text_input("Make", value=st.session_state.get("make", ""))
        model = st.text_input("Model", value=st.session_state.get("model", ""))
        year = st.text_input("Year / series", value=st.session_state.get("year", ""))
        mileage = st.text_input("Mileage / hours", value=st.session_state.get("mileage", ""))
        ambient_temp = st.text_input("Ambient / intake temp notes", value=st.session_state.get("temp", ""))
        load_condition = st.selectbox(
            "Load Condition",
            load_opts,
            index=load_opts.index(_safe_default_load(load_opts, str(default_load))),
        )
        recent_maintenance = st.text_area("Recent Maintenance", value=st.session_state.get("recent_maint", ""))

        st.session_state.make = make.strip()
        st.session_state.model = model.strip()
        st.session_state.year = year.strip()
        st.session_state.mileage = mileage.strip()
        st.session_state.temp = ambient_temp.strip()
        st.session_state.load = load_condition
        st.session_state.recent_maint = recent_maintenance.strip()

        st.caption("Sidebar values are injected verbatim into the crew prompts.")

        return {
            "make": make.strip(),
            "model": model.strip(),
            "year": year.strip(),
            "mileage": mileage.strip(),
            "ambient_notes": ambient_temp.strip(),
            "load_condition": load_condition,
            "recent_maintenance": recent_maintenance.strip(),
        }


def _load_streamlit_secret_keys() -> None:
    for key_name in ("OPENAI_API_KEY", "OPENAI_API_KEY2", "openai_api_key2", "openai_api_key"):
        try:
            value = st.secrets.get(key_name)
        except Exception:
            continue
        if value:
            import os

            os.environ.setdefault(key_name, str(value))


def main() -> None:
    st.set_page_config(page_title="Truck Diagnostics Copilot", page_icon="🚛", layout="wide", initial_sidebar_state="expanded")

    init_session_state()
    _friendly_env_hint()
    _inject_css()

    st.markdown(
        "<div class='hero'><h1>🚛 Fleet Diagnostics Copilot</h1>"
        "<p>Grounded retrieval + procedural planning with explicit evidence separation. "
        "<small style='opacity:.5'>v7.0</small></p></div>",
        unsafe_allow_html=True,
    )

    ui_context = side_vehicle_form()

    if not ui_context.get("make") and not ui_context.get("model"):
        st.info(
            "👈 **Fill in the vehicle details in the sidebar first** — make, model, year and mileage "
            "are used to search the fleet knowledge base. Once filled, describe the symptoms here."
        )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe symptoms & fault codes — fill the sidebar with vehicle details first")
    if not prompt:
        return

    allowed, sanitized = sanitize_user_input(prompt)
    if not allowed:
        st.session_state["messages"].append({"role": "assistant", "content": sanitized})
        return

    st.session_state["messages"].append({"role": "user", "content": sanitized})
    with st.chat_message("user"):
        st.markdown(sanitized)

    with st.chat_message("assistant"):
        first_run = not st.session_state.get("_kb_ready", False)
        spinner_label = (
            "⏳ First query after restart — building index & warming up (allow up to 10 min)…"
            if first_run
            else "Running retrieval · planner · safety review…"
        )
        with st.status(spinner_label, expanded=False) as status:
            run_diagnostic_crew = _get_crew_runner()
            result = run_diagnostic_crew(
                conversation_history=st.session_state["messages"],
                user_message=sanitized,
                ui_context=ui_context,
            )
            status.update(label="Complete ✓", state="complete")
            st.session_state["_kb_ready"] = True
        final_md = _strip_code_fence(result.get("final_markdown", "(no textual output captured)"))
        st.markdown(final_md)

    st.session_state["messages"].append({"role": "assistant", "content": final_md})


def _friendly_env_hint() -> None:
    csv_path = _BASE / "data" / "logistics_vehicle_maintenance_history.csv"
    if csv_path.exists():
        return
    st.warning(
        "Place your Kaggle-style maintenance CSV at "
        f"`{html.escape(str(csv_path))}` before expecting KB-backed answers."
    )


if __name__ == "__main__":
    main()
