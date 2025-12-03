# Agent and Model Context Protocol

## Pre-requirements

- Download and install [Ollama](https://ollama.com/download), this will install llama 3.2  LLMs.

## Install and run llama 3.2

```sh
# Install and run
ollama run llama3.2

# List the download ollama llms
ollama list

# other llms models
ollama pull gemma3:12b
ollama pull qwen3:30b
```

## Create virtual environment and install dependencies

```sh
pyenv install 3.11.9                # Install the python
pyenv local 3.11.9                  # Setup this version as local
python --version                    # Check the python version
python -m venv .venv                # Create the virtual environment 
source .venv/Scripts/activate       # Activate the virtual environment
python -m pip install --upgrade pip # Upgrade pip
pip install -r requirements.txt     # Install dependencies
```


## Active the virtual environment (each time you closed session)

```sh
# Windows
.venv/Script/activate

# Linux and Mac
source .venv/bin/activate
```

## Execute Scripts

### Run Model Context Protocol Client and Server Demo

```sh
python client.py
```

### Run Text Summarize using Map Reduce Tokenizen

optional: to run with openain create the environment variables.

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=
LLM_TEMPERATURE=0.1
SUMMARY_SOURCE_FILE=data/to_summarize.txt
CHUNK_SIZE=1000
CHUNK_OVERLAP=0
```

1. Edit the file data/to_summarize.txt with the context that you want to query


```sh
python src/text_summarize.py
```

