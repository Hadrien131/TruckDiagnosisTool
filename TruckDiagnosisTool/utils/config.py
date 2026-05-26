import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from crewai import LLM

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

TRUCK_KB_PATH = DATA_DIR / "logistics_vehicle_maintenance_history.csv"

SAFETY_BULLETINS_PATH = DATA_DIR / "safety_bulletins.csv"

SECRET_KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_API_KEY2", "openai_api_key2", "openai_api_key")


def _key_from_streamlit_secrets_file() -> str | None:
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    for candidate in SECRET_KEY_NAMES:
        value = data.get(candidate)
        if value:
            return str(value).strip()

    return None


def ensure_openai_api_key() -> str | None:
    """
    Normalize project-specific key names to the environment variable expected by CrewAI.
    """
    if value := os.getenv("OPENAI_API_KEY"):
        return value

    for candidate in SECRET_KEY_NAMES[1:]:
        value = os.getenv(candidate)
        if value:
            os.environ["OPENAI_API_KEY"] = value
            return value

    if value := _key_from_streamlit_secrets_file():
        os.environ["OPENAI_API_KEY"] = value
        return value

    return None


@dataclass
class ModelConfig:
    main_model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    planner_model: str = os.getenv("OPENAI_PLANNER_MODEL_NAME", "gpt-4o-mini")


def get_main_llm() -> LLM:
    api_key = ensure_openai_api_key()
    cfg = ModelConfig()
    return LLM(model=cfg.main_model, temperature=float(os.getenv("OPENAI_MAIN_TEMP", "0.4")), api_key=api_key)


def get_planner_llm() -> LLM:
    api_key = ensure_openai_api_key()
    cfg = ModelConfig()
    return LLM(model=cfg.planner_model, temperature=float(os.getenv("OPENAI_PLANNER_TEMP", "0.2")), api_key=api_key)
