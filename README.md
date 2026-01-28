# Pizza Interview Search Tool

Search tool for pizza interview/survey data: full-text (Whoosh), semantic (sentence transformers), and hybrid. Includes a FastAPI + static Web UI.

## Setup

```bash
git clone https://github.com/ltraum/American_Pizza_Project_Search_Tool.git
cd American_Pizza_Project_Search_Tool

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: edit for API keys, model, etc.
```

## Run

**Web UI (recommended):**
```bash
./run_web_ui.sh
```
Then open http://127.0.0.1:8000

**CLI:**
```bash
python search_tool.py --info
python search_tool.py "your query" [--type hybrid|fulltext|semantic] [--limit 20]
python search_tool.py --rebuild   # rebuild indices
```

**Python API:**
```python
from search_tool import PizzaSearchTool
search = PizzaSearchTool()
results = search.search("customer preferences", search_type="hybrid", limit=10)
```

## Config

Edit `.env`. Default model: `all-MiniLM-L6-v2`. For better quality set `SEMANTIC_MODEL_NAME=all-mpnet-base-v2`. Optional: `OPENAI_API_KEY` for AI query expansion.

### "API unavailable" / AI query expansion not running

The Web UI shows this when **OpenAI**-based query expansion fails. The app uses the **OpenAI API** (GPT-3.5-turbo) for that feature.

**Fix:**

1. **Create `.env`** (if missing): `cp .env.example .env`
2. **Set a valid OpenAI key** in `.env`:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
   Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Use your own key; never commit real keys to git.
3. **Restart the server** after changing `.env` (e.g. stop and re-run `./run_web_ui.sh`).

**Check the terminal** where the server runs. You’ll see one of:

- `No OpenAI API key found. Set OPENAI_API_KEY for AI expansion.` → key missing or not loaded
- `OpenAI expansion failed: ...` → key invalid, rate limit, timeout, or network issue

Ensure `openai` and `python-dotenv` are installed (`pip install -r requirements.txt`).

## Project layout

- `search_tool.py` — main CLI/API
- `api_server.py` + `index.html` — Web UI
- `data_loader.py`, `fulltext_search.py`, `semantic_search.py`, `query_expander.py`, `config.py` 
- `pizza_interviews copy.xlsx` — American Pizza Project interview collection


