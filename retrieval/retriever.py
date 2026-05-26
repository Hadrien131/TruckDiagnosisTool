"""TF-IDF retrieval over the truck maintenance KB with caching and similarity gating."""

from __future__ import annotations

import os
import re
from threading import Lock
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.kb_loader import load_truck_issues, load_safety_bulletins, truck_kb_mtime_key

# Lowest max cosine similarity to accept retrieved rows as "related" (tune via env).
_MIN_TOP_SIM = float(os.getenv("TRUCK_KB_MIN_TFIDF_SIM", "0.06"))
# Margin between top-1 and top-2; tiny margin => ambiguous / bogus match.
_MIN_MARGIN = float(os.getenv("TRUCK_KB_MIN_TFIDF_MARGIN", "0.005"))

_CACHE_LOCK = Lock()
_ISSUES_INDEX_CACHE: dict[str, Any] = {}


def clear_retriever_cache() -> None:
    """Drop in-process TF-IDF index (useful after swapping CSV files)."""
    global _ISSUES_INDEX_CACHE
    with _CACHE_LOCK:
        _ISSUES_INDEX_CACHE.clear()


def prime_truck_issues_index() -> None:
    """Warm the KB index once (call from Streamlit @st.cache_resource)."""
    _build_issues_search_index(force=False)


def _electrical_patterns(text_lower: str) -> bool:
    """Avoid substring 'ground' matching LLM jargon like 'grounding in answers'."""
    electrical_tokens = (
        r"\bbattery\b",
        r"\bstarter\b",
        r"\balternator\b",
        r"\bvoltage\b",
        r"\b12v\b",
        r"\belectrical\b",
        r"ground\s*strap",
        r"chassis\s*ground",
        r"negative\s*cable",
        r"positive\s*cable",
        r"\bloom\b.*\bwiring\b",
        r"\becm\b",
    )
    return any(re.search(p, text_lower) for p in electrical_tokens)


def classify_subsystem(symptom_text: str) -> str:
    text = symptom_text.lower()
    if any(k in text for k in ("overheat", "coolant", "radiator", "fan", "temperature")):
        return "cooling"
    if any(k in text for k in ("turbo", "boost", "whistle")):
        return "turbo"
    if any(k in text for k in ("fuel", "injector", "rail", "def fluid", "adblue")):
        return "fuel"
    # "filter" alone is ambiguous; tie to intake/air when obvious
    if "air filter" in text or "fuel filter" in text:
        return "fuel" if "fuel filter" in text else "general"
    if _electrical_patterns(text):
        return "electrical"
    return "general"


def _find_make_model_column(cols: list[str]) -> str | None:
    for c in cols:
        cl = c.lower()
        if "make_and_model" in cl or ("make" in cl and "model" in cl):
            return c
    return None


def _find_subsystem_like_column(cols: list[str]) -> str | None:
    for c in cols:
        cl = c.lower()
        if "subsystem" in cl:
            return c
        if cl in {"system", "component"} or "system" == cl:
            return c
        if "component" in cl:
            return c
    return None


def _build_issues_search_index(force: bool) -> tuple[Any, TfidfVectorizer, Any, list[str]]:
    """
    Cached: full dataframe reference (not copied), sparse TF-IDF matrix, text column list.
    """
    global _ISSUES_INDEX_CACHE

    with _CACHE_LOCK:

        df = load_truck_issues()

        cols = df.columns.tolist()

        keysig = "::".join(cols[: min(40, len(cols))])

        sig = keysig + "::" + str(df.shape)

        key = f"{truck_kb_mtime_key()}:{hash(sig)}"

        if not force and key == _ISSUES_INDEX_CACHE.get("key"):

            b = _ISSUES_INDEX_CACHE

            return b["df"], b["vectorizer"], b["matrix"], b["text_cols"]

        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        text_cols = [c for c in text_cols if c.lower() != "vehicle_id"]

        blob = df[text_cols].fillna("").agg(" ".join, axis=1) if text_cols else None

        if blob is None:

            corpus = []

        else:

            corpus = blob.astype(str).tolist()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=int(os.getenv("TRUCK_KB_TFIDF_MAX_FEATURES", "6000")),
            ngram_range=(1, 1),
            min_df=2,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(corpus)

        bundle = {
            "key": key,
            "df": df,
            "vectorizer": vectorizer,
            "matrix": matrix,
            "text_cols": text_cols or [],
            "blob": blob,
        }

        _ISSUES_INDEX_CACHE = bundle

        return df, vectorizer, matrix, text_cols or []


def retrieve_issues(context: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    """
    Return top_k maintenance rows grounded in cosine similarity against the symptom query.

    - Does NOT widen to the entire 90k KB when structured filters wipe the frame
      (that was the main hallucination trigger).
    - Returns [] when the match is statistically weak vs the query — upstream tools
      should treat this as NO_MATCHES and avoid inventing plausible truck stories from
      unrelated rows.
    """

    df, vectorizer, matrix, text_cols = _build_issues_search_index(force=False)

    make = str(context.get("make", "") or "").strip()
    model = str(context.get("model", "") or "").strip()
    subsystem = str(context.get("subsystem", "") or "").strip().lower()

    symptoms = str(
        context.get("symptoms", "")
        or context.get("primary_symptoms", "")
        or context.get("free_text", "")
        or ""
    ).strip()

    ui_bits = []
    if context.get("mileage"):
        ui_bits.append(f"mileage: {context.get('mileage')}")
    if context.get("year"):
        ui_bits.append(f"year: {context.get('year')}")
    ui_tail = "; ".join(ui_bits)

    if not subsystem and symptoms:
        subsystem = classify_subsystem(symptoms)
        context["subsystem"] = subsystem

    query = symptoms or ""
    if not query.strip():
        if make or model:
            query = " ".join(x for x in (make, model) if x)
        if ui_tail:
            query = (query + " " + ui_tail).strip() if query else ui_tail

    if not query:
        context["retrieval_note"] = "No symptom/query text provided; skipping KB retrieval."
        return []

    cols = df.columns.tolist()
    mm_col = _find_make_model_column(cols)
    sub_col = _find_subsystem_like_column(cols)

    n_docs = matrix.shape[0]
    eligible = np.ones(n_docs, dtype=bool)

    if mm_col and (make or model):
        s = df[mm_col].astype(str).str.strip()
        if make and model:
            mk_l, md_l = make.lower(), model.lower()

            lowered = s.str.lower()

            mm_mask = lowered.str.contains(re.escape(mk_l), regex=True, na=False) & lowered.str.contains(
                re.escape(md_l),
                regex=True,
                na=False,
            )

        elif make:

            mm_mask = s.str.lower().str.contains(re.escape(make.lower()), na=False)

        else:

            mm_mask = s.str.lower().str.contains(re.escape(model.lower()), na=False)

        subset_count = int(mm_mask.sum())

        eligible &= mm_mask.to_numpy(dtype=bool)
        context["kb_make_model_candidates"] = subset_count

        if subset_count == 0:
            context["retrieval_note"] = (
                f"No rows match make/model `{make}` `{model}`; searching full KB "
                "by symptom similarity (no fabricated vehicle linkage)."
            )
            eligible[:] = True

    elif make or model:
        context["retrieval_note"] = "Dataset has no make/model column detected; symptom-only retrieval."

    if sub_col is not None and subsystem and subsystem != "general":
        ssub = df[sub_col].astype(str).str.lower()
        subsystem_mask = ssub.str.contains(re.escape(subsystem), na=False)
        tentative = eligible & subsystem_mask.to_numpy(dtype=bool)

        # Subsystem heuristic can be brittle on synthetic schemas — avoid emptying eligibility.
        if int(tentative.sum()) >= 20:
            eligible = tentative

    q_vec = vectorizer.transform([query])
    cos = q_vec.dot(matrix.transpose())
    dense = cos.toarray().ravel()

    dense_masked = dense.copy()

    masked_out_count = int((~eligible).sum())

    dense_masked[~eligible] = -1.0

    if masked_out_count == n_docs:
        dense_masked = dense.copy()

    ranked = np.argsort(dense_masked)[::-1][: max(top_k * 8, top_k)]

    vals = dense_masked[ranked]
    best_idx: list[int] = []

    for ix, idx in enumerate(ranked):

        sc = vals[ix]

        if sc <= -1e-6:
            continue

        best_idx.append(int(idx))

        if len(best_idx) >= top_k * 3:
            break

    scores = dense_masked[np.array(best_idx, dtype=np.int64)] if best_idx else np.array([])

    max_sim = float(scores.max()) if scores.size else 0.0
    sorted_scores = np.sort(scores)[::-1]
    second = float(sorted_scores[1]) if sorted_scores.size > 1 else 0.0
    margin = max_sim - second

    context["retrieval_best_similarity"] = max_sim

    gate_fail = scores.size == 0 or max_sim < _MIN_TOP_SIM or (
        margin < _MIN_MARGIN and max_sim < max(_MIN_TOP_SIM * 1.5, _MIN_TOP_SIM + 0.02)
    )

    if gate_fail:
        context["retrieval_note"] = (
            (context.get("retrieval_note") or "")
            + f" Retrieval skipped weak matches (top_sim={max_sim:.4f})."
        ).strip()

        context["weak_kb_match"] = True
        return []

    sorted_idx = sorted(zip(scores.tolist(), best_idx), reverse=True)[:top_k]

    out: list[dict[str, Any]] = []
    for sc, ri in sorted_idx:

        rowdict = df.iloc[int(ri)].to_dict()

        rowdict["_retrieval_similarity"] = float(sc)

        out.append(rowdict)

    context["retrieval_similarity_margin"] = margin

    return out


def retrieve_safety_notes(subsystem: str | None) -> list[dict[str, str]]:
    df = load_safety_bulletins()
    subsystem_clean = (subsystem or "").strip().lower()
    if subsystem_clean:

        df_sub = df[df["subsystem"] == subsystem_clean]

        if df_sub.empty:

            df_sub = df[df["subsystem"].isin(["general"])]

    else:

        df_sub = df[df["subsystem"].isin(["general"])]

    return df_sub.to_dict(orient="records")
