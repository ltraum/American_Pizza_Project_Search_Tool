# Pizza Interview Search Tool

A simple tool for exploring American Pizza Project interview data.

You can:
- search interviews by **keywords**
- search by **meaning** (semantic search)
- combine both (hybrid search)
- explore patterns using **Theme Mode** (LLooM-inspired)

Please note that an Open AI API key is needed for all features to function.

## Quick start

```bash
git clone https://github.com/ltraum/American_Pizza_Project_Search_Tool.git
cd American_Pizza_Project_Search_Tool

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_web_ui.sh
