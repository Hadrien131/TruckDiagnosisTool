import os
from dataclasses import dataclass
from pathlib import Path

from crewai import LLM

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

TRUCK_KB_PATH = DATA_DIR / "logistics_vehicle_maintenance_history.csv"

SAFETY_BULLETINS_PATH = DATA_DIR / "safety_bulletins.csv"


@dataclass
class ModelConfig:
    main_model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    planner_model: str = os.getenv("OPENAI_PLANNER_MODEL_NAME", "gpt-4o-mini")


def get_main_llm() -> LLM:
    cfg = ModelConfig()
    return LLM(model=cfg.main_model, temperature=float(os.getenv("OPENAI_MAIN_TEMP", "0.4")))


def get_planner_llm() -> LLM:
    cfg = ModelConfig()
    return LLM(model=cfg.planner_model, temperature=float(os.getenv("OPENAI_PLANNER_TEMP", "0.2")))
