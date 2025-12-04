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

# Help commands
ollama --help
ollama [Command] --help
```

## Setup virtual environment
```sh
pyenv install 3.11.9                # optional: install Python
pyenv local 3.11.9                  # set local version
python --version                    # verify
python -m venv .venv                # create venv
source .venv/Scripts/activate       # Windows
# or: source .venv/bin/activate     # Linux/Mac
```

## Install dependencies using poetry

```sh
pip install poetry          # Install poetry
poetry init                 # Init the pyproject.toml file

# Add dependencies    
poetry add fastapi uvicorn python-dotenv \
sqlalchemy pydantic pydantic-settings \
langchain langgraph langsmith \
langchain-openai langchain-anthropic langchain-google-genai \
langchain-ollama langchain-huggingface \
transformers mcp \
python-multipart requests

```

## Install dependencies using requirements.txt (optional)

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup the environment configuration

By default, the application will run using llama 3.2 in local mode; to set another configuration, edit the .env file and set the required configuration for the query agent and llm agent.

```env
# API keys per provider (set only what you use)
OPENAI_API_KEY=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=

# Router agent (tool picker) config
# Providers: ollama | openai | gemini | anthropic
# Models (examples):
# - ollama: llama3.2, llama3.1, llama3.3, gemma2:9b, qwen2.5:14b, mistral, phi4
# - openai: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
# - gemini: gemini-1.5-flash, gemini-1.5-pro
# - anthropic: claude-3-haiku-20240307, claude-3-5-sonnet-20240620
QUERY_AGENT_PROVIDER=ollama
QUERY_AGENT_MODEL=llama3.2
QUERY_AGENT_TEMPERATURE=0.1

# LLM server (tools) config
# Providers: ollama | openai | gemini | anthropic
# Models follow the same examples as above (for ollama: llama3.2, llama3.1, llama3.3, gemma2:9b, qwen2.5:14b, mistral, phi4)
LLM_SERVER_PROVIDER=ollama
LLM_SERVER_MODEL=llama3.2
LLM_SERVER_TEMPERATURE=0.7

# Ollama config (local models)
OLLAMA_BASE_URL=http://localhost:11434
```

## Run tests and coverage

```sh
# Run tests
pytest

# Run test with coverage
pytest --cov=src --cov=tests --cov-report=term-missing
```

## Run the client (web + API)

```sh
uvicorn startup:app --host 0.0.0.0 --port 8000 --reload
```

- Open `http://localhost:8000/` for a minimal chat UI.


