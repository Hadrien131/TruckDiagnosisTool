# TruckDiagnosisTool

Final project for *LLMs Theory & Applications* (Yale SOM).

This repo now includes runnable Python modules tuned for grounding, responsiveness, and a clearer Streamlit experience. The authoritative code lives beside the instructional `FinalProjectCode.ipynb`.

---

## Run locally

From the repo root (Python 3.10+):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

On macOS/Linux use `source .venv/bin/activate` instead.

### Data paths

Put the logistics maintenance CSV at:

`data/logistics_vehicle_maintenance_history.csv`

and safety bulletins at:

`data/safety_bulletins.csv`

Configure whichever API keys CrewAI/OpenAI adapters expect in `.env`.

---

## Keep the notebook aligned with Git

After editing files under `utils/`, `retrieval/`, `crewai_layer/`, `app/`, or `data/safety_bulletins.csv`, regenerate the `%%writefile` cells and intro copy with:

```bash
node tools/polish_notebook.mjs
```

That script injects the top markdown overview, clears stale cell outputs, and overwrites each `%%writefile` block with the on-disk source plus a short purpose banner.


## Retrieval tuning knobs

| Variable | Default | Meaning |
|---------|---------|---------|
| `TRUCK_KB_MIN_TFIDF_SIM` | `0.06` | Minimum TF-IDF cosine similarity before emitting KB rows |
| `TRUCK_KB_MIN_TFIDF_MARGIN` | `0.005` | Minimum gap between the best score and runner-up |
| `TRUCK_KB_TFIDF_MAX_FEATURES` | `20000` | Vocabulary ceiling for quicker vector fits |

Raise the similarity floor if bogus historical rows sneak through; lower it gently if sensible matches disappear. Optional `OPENAI_MAIN_TEMP` / `OPENAI_PLANNER_TEMP` tweak LLM stochasticity (`utils/config.py`).
