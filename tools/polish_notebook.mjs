#!/usr/bin/env node
/**
 * Maintains FinalProjectCode.ipynb:
 * - Inserts/upgrades a coherent markdown guide at the top
 * - Clears execution outputs for a readable notebook export
 * - Overwrites %%writefile cells with sibling files under the repo root
 *
 * Usage: node tools/polish_notebook.mjs
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const NB = path.join(ROOT, "FinalProjectCode.ipynb");

const INTRO_MARKDOWN = `## Fleet truck diagnostics notebook

This workbook **materializes the same Python modules** tracked in Git (\`utils/\`, \`retrieval/\`, \`crewai_layer/\`, \`app/\`, \`models/\`). Lecture cells used \`%%writefile\`; the files on disk are the **reference implementation** synced here for clarity.

### What the system does (end-to-end)

1. **Streamlit UI** collects sidebar vehicle facts plus free-text symptoms.
2. **CrewAI crew** (\`crew_setup.py\`) runs five sequential tasks: structure the complaint → TF-IDF retrieval over the CSV KB → planner tool → safety bulletins → final Markdown synthesis.
3. **Retrieval** (\`retrieval/retriever.py\`) caches the vector index, applies optional filters, and **rejects statistically weak cosine matches** so the LLM cannot "shop" random historical trucks.
4. **Tasks** inject \`{user_message}\`, \`{conversation_history}\`, and \`{ui_context}\` into prompts so every agent sees the same operator truth.

### Colab vs local

| Where | What to do |
|-------|------------|
| **Google Colab** | Mount Drive, \`%cd\` into this project folder, run cells top-to-bottom, then use the Streamlit tunnel cell (Colab-specific). |
| **Local clone** | Install deps per \`requirements.txt\`, put the Kaggle CSV at \`data/logistics_vehicle_maintenance_history.csv\`, then \`python -m streamlit run app/streamlit_app.py\` from the repo root. You can skip Drive cells. |

### Cell tour (read the \`#\` banners inside each code cell)

- **Environment & deps** — API keys, pip installs, warning filters.
- **Scaffold / tiny data** — create folders plus \`safety_bulletins.csv\`.
- **Library modules** — every \`%%writefile\` block mirrors one production file (\`README.md\` documents env tuning such as \`TRUCK_KB_MIN_TFIDF_SIM\`).
- **Optional probes** — load the big CSV sanity check when you finish exports.
- **Streamlit launcher** — Colab tunnels; locals should prefer the README command.

_No step should be mysterious: each code section starts with comments describing inputs, artifacts on disk, and downstream consumers._

`;

const SYNC_TARGETS = [
  "utils/config.py",
  "utils/logging_utils.py",
  "utils/security.py",
  "retrieval/kb_loader.py",
  "retrieval/retriever.py",
  "models/llm_client.py",
  "crewai_layer/tools.py",
  "crewai_layer/agents.py",
  "crewai_layer/tasks.py",
  "crewai_layer/crew_setup.py",
  "app/streamlit_app.py",
];

const BANNERS = {
  "utils/config.py":
    "# --- Exported module: utils/config.py ---\n# Purpose: Canonical paths for the KB CSV / safety bulletin file + CrewAI LLM factories.\n# Tuning knobs: OPENAI_MODEL_NAME, OPENAI_PLANNER_MODEL_NAME, OPENAI_MAIN_TEMP, OPENAI_PLANNER_TEMP.\n\n",
  "utils/logging_utils.py":
    "# --- Exported module: utils/logging_utils.py ---\n# Purpose: Thin logging helper used by crew_setup when kicking off sequential tasks.\n\n",
  "utils/security.py":
    "# --- Exported module: utils/security.py ---\n# Purpose: Naive substring guard against obvious prompt-abuse payloads before CrewAI runs.\n\n",
  "retrieval/kb_loader.py":
    "# --- Exported module: retrieval/kb_loader.py ---\n# Purpose: Load and cache pandas DataFrames (issues CSV + bulletin CSV keyed by subsystem).\n\n",
  "retrieval/retriever.py":
    "# --- Exported module: retrieval/retriever.py ---\n# Purpose: TF-IDF cosine retrieval with global index caching + similarity gating.\n\n",
  "models/llm_client.py":
    "# --- Exported module: models/llm_client.py ---\n# Purpose: Bridge between Fleet agents (supervisor vs planner personas) and config.py LLMs.\n\n",
  "crewai_layer/tools.py":
    "# --- Exported module: crewai_layer/tools.py ---\n# Purpose: CrewAI @tool wrappers around retrieval plus diagnostic draft + safety overlays.\n\n",
  "crewai_layer/agents.py":
    "# --- Exported module: crewai_layer/agents.py ---\n# Purpose: Four specialist personas (Supervisor / Retrieval / Planner / Safety Review).\n\n",
  "crewai_layer/tasks.py":
    "# --- Exported module: crewai_layer/tasks.py ---\n# Purpose: Task prompts with explicit template variables for user + UI context wiring.\n\n",
  "crewai_layer/crew_setup.py":
    "# --- Exported module: crewai_layer/crew_setup.py ---\n# Purpose: Sequential Crew wiring + conversation compaction + primes TF-IDF cache.\n\n",
  "app/streamlit_app.py":
    "# --- Exported module: app/streamlit_app.py ---\n# Purpose: Styled chat UX (sidebar dossier -> run_diagnostic_crew) with warmed KB caches.\n\n",
  "data/safety_bulletins.csv":
    "# --- Generated asset: data/safety_bulletins.csv ---\n# Purpose: Hand-authored subsystem hazards merged into the SAFETY agent output.\n\n",
};

function normalizeSource(cell) {
  if (typeof cell.source === "string") return cell.source;
  return cell.source.join("");
}

function setSource(cell, body) {
  const b = body.endsWith("\n") ? body : `${body}\n`;
  cell.source = b.split(/(?<=\n)/);
}

function syncWritefiles(cell) {
  if (cell.cell_type !== "code") return;

  const txt = normalizeSource(cell);

  const wfMatch = txt.match(/%%writefile\s+(\S+)/);

  if (!wfMatch) return;

  const rel = wfMatch[1].replace(/\\/g, "/");

  if (!SYNC_TARGETS.includes(rel) && rel !== "data/safety_bulletins.csv") return;

  const abs = path.join(ROOT, rel);

  if (!fs.existsSync(abs)) {
    console.warn("Skip sync (missing on disk):", rel);
    return;

  }

  const fileBody = fs.readFileSync(abs, "utf8");

  const banner = BANNERS[rel] ?? `# --- Exported artifact: ${rel} ---\n\n`;

  const rebuilt = `${banner}%%writefile ${rel}\n${fileBody.replace(/\s+$/, "")}\n`;

  setSource(cell, rebuilt);

}

function annotateSpecialCells(cell) {
  if (cell.cell_type !== "code") return;

  let txt = normalizeSource(cell);

  if (/from google\.colab import drive/.test(txt)) {
    if (!/Section 1: Drive mount/.test(txt)) {
      txt =
        `# --- Section 1: Drive mount + secrets (Colab lecture path) ---\n` +
        `# Mount Drive, cd into YOUR copy of TruckDiagnosisTool, then hydrate OPENAI_* via Colab Secrets.\n` +
        `# Locally: SKIP this entire cell and load keys from '.env' / your shell instead.\n\n` +
        txt.replace(/^#\s*Cell 1[^\n]*\n\n?/i, "");
      setSource(cell, txt);
    }
    return;
  }

  if (/%%capture\s*\n!pip install/.test(txt)) {
    if (!/^#\s*---\s*Dependencies:/m.test(txt)) {
      setSource(cell, `# --- Dependencies: CrewAI + Streamlit + pandas/scikit-learn ---\n${txt}`);
    }
    return;
  }

  if (txt.includes("warnings.filterwarnings")) {
    if (!/^#\s*---\s*Silence noisy/m.test(txt)) {
      txt = txt.replace(/^#\s*Warning control\s*\n/m, "");
      setSource(cell, `# --- Silence noisy library warnings during iterative demos ---\n${txt}`);
    }
    return;
  }

  if (/\bdirs\b\s*=\s*\[/.test(txt) && txt.includes('"app"')) {


    if (!txt.includes("Section 2")) {

      txt =
        `# --- Section 2: Scaffold folders that %%writefile cells will populate ---\n` +
        `# Lecture path below targets Google Drive; on your laptop, point base_dir at Path.cwd() instead.\n\n` +
        txt.replace(/^#\s*Cell 2[^\n]*\n\n?/i, "");


      setSource(cell, txt);


    }


    return;


  }

  if (txt.includes("subprocess.Popen") && txt.includes("streamlit")) {

    if (!txt.includes("Launcher:")) {

      setSource(

        cell,

        `# --- Launcher: Colab-only Streamlit + tunnel helpers ---\n` +
          `# Local development: see README -> python -m streamlit run app/streamlit_app.py\n\n` +

          txt,


      );


    }


    return;


  }

  if (/Cell 15/i.test(txt) && /load_truck_issues/.test(txt)) {


    setSource(cell, txt.replace(/^#\s*Cell 15[^\n]*\n\n?/i, "# --- Optional QA: inspect KB columns after exports ---\n\n"));


  }

}

function clearOutputs(nb) {

  for (const cell of nb.cells) {

    if (cell.cell_type !== "code") continue;

    cell.outputs = [];

    cell.execution_count = null;

  }

}

function pruneIntro(nb) {

  while (nb.cells[0]?.cell_type === "markdown") {

    const head = Array.isArray(nb.cells[0].source) ? nb.cells[0].source.join("") : nb.cells[0].source;

    if (head.includes("Fleet truck diagnostics notebook")) nb.cells.shift();

    else break;

  }

}

function upsertMarkdownGuide(nb) {

  pruneIntro(nb);

  const src = INTRO_MARKDOWN.endsWith("\n") ? INTRO_MARKDOWN : `${INTRO_MARKDOWN}\n`;

  nb.cells.unshift({

    cell_type: "markdown",

    metadata: {},

    source: src.split(/(?<=\n)/),

  });

}

const nb = JSON.parse(fs.readFileSync(NB, "utf8"));

upsertMarkdownGuide(nb);

clearOutputs(nb);

for (const cell of nb.cells) {

  annotateSpecialCells(cell);

  syncWritefiles(cell);

}

fs.writeFileSync(NB, JSON.stringify(nb, null, 2), "utf8");

console.log("Polished notebook:", NB);
