# Labs Chat with MCP & Lang Chains


## Prerequisites
- Python 3.11.x (recommended) and optionally `pyenv`.
- [Ollama](https://ollama.com/download) installed with `llama3.2` (default local model).

## Install and run an Ollama model
```sh
ollama run llama3.2      # downloads if needed and runs once
ollama list              # verify it is available
# optional additional models
ollama pull gemma3:12b
ollama pull qwen3:30b
```

## Setup virtual environment and install dependencies
```sh
pyenv install 3.11.9                # optional: install Python
pyenv local 3.11.9                  # set local version
python --version                    # verify
python -m venv .venv                # create venv
source .venv/Scripts/activate       # Windows
# or: source .venv/bin/activate     # Linux/Mac
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup the environment configuration

By default, the application will run using llama 3.2 in local mode; to set another configuration, edit the .env file and set the required configuration for the query agent and llm agent.

```env
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
QUERY_AGENT_PROVIDER=ollama
QUERY_AGENT_MODEL=llama3.2
QUERY_AGENT_TEMPERATURE=0.1
LLM_SERVER_PROVIDER=ollama
LLM_SERVER_MODEL=llama3.2
LLM_SERVER_TEMPERATURE=0.7
```


## Run the client (web + API)

```sh
uvicorn startup:app --host 0.0.0.0 --port 8000 --reload
```

- Open `http://localhost:8000/` for a minimal chat UI.


