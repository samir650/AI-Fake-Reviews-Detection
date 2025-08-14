# 🔍 AI‑Powered Fake Review Detector (RAG + LLM + Streamlit)

Detects potentially fake vs. real reviews using a Retrieval‑Augmented Generation (RAG) pipeline, TF‑IDF similarity, and Groq‑hosted LLMs — with a fast Streamlit UI.

---

## ✨ What’s inside
- RAG pipeline TF‑IDF vectorizer + cosine similarity to fetch similar training reviews as context.
- LLM reasoning (Groq API) Structured JSON decision with confidence and key indicators.
- Knowledge Base Extracts stylistic patterns and statistics from your dataset and saves them to `knowledge_base.pkl`.
- Caching & rate‑limiting Smarter Groq API client with retries, backoff, and a small in‑memory cache.
- Fallback parsing If the LLM returns non‑JSON, we still produce a best‑effort decision.
- Streamlit app Single review analysis, batch analysis, analytics dashboard.

---

## 🗂️ Project structure
```
.
├─ fake_review_detection.py   # Core RAG + LLM pipeline (CLI-friendly)
├─ streamlit_app.py           # Streamlit UI
├─ fake reviews dataset.csv   # (optional) Training CSV 'text_', 'label'
└─ knowledge_base.pkl         # Auto-generated after building KB
```

---

## ⚙️ Requirements
- Python 3.9–3.12
- A Groq API key (free tier available)

### Python packages
If you don’t have a `requirements.txt`, install these
```bash
pip install -U 
  streamlit pandas numpy scikit-learn requests plotly
```
 On some systems you may also need `typing-extensions`.

You can also create a `requirements.txt` like
```txt
streamlit=1.33
pandas=2.0
numpy=1.24
scikit-learn=1.2
requests=2.31
plotly=5.18
```

---

## 🔑 Set your Groq API key
The app reads `GROQ_API_KEY` from the environment (or you can paste it in the Streamlit sidebar).

macOS  Linux
```bash
export GROQ_API_KEY=your_key_here
```

Windows (PowerShell)
```powershell
$envGROQ_API_KEY=your_key_here
```

Get a key from httpsconsole.groq.com

---

## 🚀 Quick start

### 1) Create & activate a virtual environment (recommended)
macOS  Linux
```bash
python3 -m venv .venv
source .venvbinactivate
```

Windows (PowerShell)
```powershell
python -m venv .venv
..venvScriptsActivate.ps1
```
 If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2) Install deps
```bash
pip install -r requirements.txt
# or use the inline pip command above
```

### 3) (Optional) Put a dataset in the project folder
CSV with columns
- `text_` the raw review text
- `label` either integers `{0,1}` or strings `{'CG','OR'}` which the code maps to `{0,1}`
  - In the current logic 0 = fake, 1 = real

Minimal example (`fake reviews dataset.csv`)
```csv
text_,label
This product is amazing!!! Best ever!,0
I bought it last week; decent quality for the price.,1
```

### 4) Build the Knowledge Base
You can do this either from the Streamlit UI or the CLI run below. The app will save it to `knowledge_base.pkl`.

### 5) Run the Streamlit app
```bash
streamlit run streamlit_app.py
```
Open the browser link that appears (usually `httplocalhost8501`).

### 6) (Alternative) Run from the command line
```bash
python fake_review_detection.py
```
This will test the API key → loadbuild the KB → analyze a couple of sample reviews → print results.

---

## 🧠 How it works (high level)
1. Preprocessing (`preprocess_data`)
   - Cleans text (lowercase, remove URLsHTML, normalize punctuation).
   - Keeps reviews with length ≥ 10 chars.
   - Maps labels `'CG'→0`, `'OR'→1` (or uses integer labels if already present).

2. Knowledge Base (`ReviewKnowledgeBase`)
   - TF‑IDF vectorizer (1–3 n‑grams) fits on cleaned text.
   - Extracts patternsstatistics for fake vs. real (length, caps ratio, punctuation density, exclamations, sentiment words).
   - Supports similarity search and returns top‑k similar reviews as RAG context.
   - Savedloaded via `knowledge_base.pkl`.

3. LLM Reasoning (`RAGDetector` + `GroqClient`)
   - Builds a system prompt + analysis prompt including
     - Similar reviews (with labels & similarity)
     - Query features
     - Statistical benchmarks (fake vs. real means)
   - Calls Groq (models `fast`, `smart`, `balanced`, `gemma`) and expects JSON with
     ```json
     {
       prediction fake  real,
       confidence 0-100,
       reasoning ...,
       key_indicators [...]
     }
     ```
   - If parsing fails, uses a fallback heuristic to extract a decision.

4. Streamlit UI (`streamlit_app.py`)
   - Single Review paste text or use curated samples, view reasoning, indicators, similarity gauge.
   - Batch Analysis upload CSV, choose the text column, analyze ≤ N rows, download results.
   - Analytics history dashboard (confidence over time, processing time, distribution).

---

## 🧪 Labels & metrics
- Predictions `fake` or `real`
- Confidence 0–100 (LLM‑reported or fallback estimate)
- Similarity score 0–1 (max similarity against KB examples in the current query)

---

## 🛠️ Configuration knobs (Streamlit → Sidebar)
- API Configuration Enter and Test your Groq key.
- Model Settings `fast`, `smart`, `balanced`, `gemma` (💡 smart is a good default).
- Confidence Threshold purely UI‑level cue to highlight lower‑confidence results.
- Dataset Management Upload CSV and click Build Knowledge Base.

---

## 📁 IO files
- Input `fake reviews dataset.csv` (or any CSV you upload via the app)
- Model KB `knowledge_base.pkl` (auto‑created)
- Batch output downloadable CSV via UI (includes review, prediction, confidence, time)

---

## ❗ Troubleshooting

### 1) “Groq API key not found”  connection failed
- Ensure `GROQ_API_KEY` is set (see above), or paste it in the sidebar → Test API Connection.
- Network restrictions or proxies can block the request; try another connection.

### 2) Streamlit “Fatal error in launcher Unable to create process using … streamlit.exe” (Windows)
This usually means a broken Python association or PATH mismatch. Try
```powershell
# Ensure you're in your virtualenv first, then
python -m pip install --upgrade --force-reinstall streamlit
# Or run Streamlit through Python explicitly
python -m streamlit run streamlit_app.py
```
If you have multiple Python installs, confirm you’re using the same interpreter where you installed packages.

### 3) “Failed to import detection modules” in the app
- Make sure both `streamlit_app.py` and `fake_review_detection.py` are in the same folder.
- Run `streamlit run streamlit_app.py` from that folder.

### 4) “Dataset file not found”
- Place `fake reviews dataset.csv` next to the scripts or upload via the sidebar, then click Build Knowledge Base.

---

## 🔒 Notes on use
This tool is for educationalresearch purposes. Automated moderationdecision systems should be reviewed by humans, and evaluated for dataset bias and edge cases.

---

## 📜 License
Add a license of your choice (e.g., MITApache‑2.0). If you plan to publish, include a `LICENSE` file.

---

## 🙌 Acknowledgements
- Groq for fast LLM serving.
- Streamlit, scikit‑learn, pandas, plotly for the tooling.
