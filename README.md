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


## Start MCP Client and MCP Server Demo

```sh

# Run client
python client.py

```
