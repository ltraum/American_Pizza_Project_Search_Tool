#!/bin/bash
# Run the web UI (FastAPI + static frontend). No Streamlit; filter changes and
# navigation stay in the browser; only search and metadata hit the server.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found. Run setup_mac.sh or pip install -r requirements.txt first."
fi

echo "Starting web UI at http://127.0.0.1:8000"
echo "Open that URL in your browser. First search may take a minute while the model loads."
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
