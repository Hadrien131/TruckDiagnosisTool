from crewai import LLM

from utils.config import get_main_llm, get_planner_llm


def get_supervisor_llm() -> LLM:
    return get_main_llm()


def get_planner_safety_llm() -> LLM:
    return get_planner_llm()
