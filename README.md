# Pizza Interview Search Tool

Search tool for pizza interview/survey data: full-text (Whoosh), semantic (sentence transformers), and hybrid. Includes a FastAPI + static Web UI.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/American_Pizza_Project_Search.git
cd American_Pizza_Project_Search

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

## Project layout

- `search_tool.py` — main CLI/API
- `api_server.py` + `index.html` — Web UI
- `data_loader.py`, `fulltext_search.py`, `semantic_search.py`, `query_expander.py`, `config.py`
- `pizza_interviews copy.xlsx` — put your data here
- `MAC_SETUP.md`, `docs/DEPLOYMENT_GUIDE.md` — extra docs

## License

MIT. See [LICENSE](LICENSE).
