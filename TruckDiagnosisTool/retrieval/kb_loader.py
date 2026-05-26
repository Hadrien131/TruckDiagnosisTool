import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import SAFETY_BULLETINS_PATH, TRUCK_KB_PATH

_ISSUES_CACHE_PATH: Optional[str] = None
_ISSUES_DF = None


def load_truck_issues(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the logistics vehicle maintenance CSV. Cached in-memory by path+mtime so
    Streamlit reruns don't re-parse tens of thousands of rows every interaction.
    """
    global _ISSUES_CACHE_PATH, _ISSUES_DF

    resolved = Path(path or TRUCK_KB_PATH)
    try:

        key = f"{resolved.resolve()}::{os.path.getmtime(resolved)}"

    except OSError:

        key = str(resolved)

    if _ISSUES_DF is not None and _ISSUES_CACHE_PATH == key:

        return _ISSUES_DF

    _ISSUES_CACHE_PATH = key
    _ISSUES_DF = pd.read_csv(resolved)
    return _ISSUES_DF


def truck_kb_mtime_key() -> str:
    resolved = Path(TRUCK_KB_PATH)
    try:

        return f"{resolved.resolve()}:{os.path.getmtime(resolved)}"

    except OSError:

        return str(resolved)


def invalidate_truck_issues_cache() -> None:

    global _ISSUES_CACHE_PATH, _ISSUES_DF

    _ISSUES_CACHE_PATH = None

    _ISSUES_DF = None

    try:

        from retrieval import retriever as _retriever

        _retriever.clear_retriever_cache()

    except ImportError:

        pass


def load_safety_bulletins(path: Optional[Path] = None) -> pd.DataFrame:
    resolved = Path(path or SAFETY_BULLETINS_PATH)

    df = pd.read_csv(resolved)

    df["subsystem"] = df["subsystem"].astype(str).str.strip().str.lower()

    return df
