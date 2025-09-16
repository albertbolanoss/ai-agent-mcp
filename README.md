# Agent and Model Context Protocol

## Pre-requirements

- [Ollama](https://ollama.com/download)
- [python 3.13.7](https://www.python.org/downloads/release/python-3137/)

## Install and run llama model v3.2

```sh
ollama pull llama3.2
```

## Create virtual environment

```sh
python --version                    # 3.13.5
python -m venv .venv                
```

## Start 

```sh
# Run llama LLM
ollama serve

# Active the virtual environment
source .venv/Script/activate

# Run client
python client.py

```
