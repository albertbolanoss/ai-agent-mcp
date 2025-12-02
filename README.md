# Agent and Model Context Protocol

## Pre-requirements

- Download and install [Ollama](https://ollama.com/download), this will install llama 3.2  LLMs.
- Install [python 3.13](https://www.python.org/downloads/release/python-3137/)

## Other Ollama commands

```sh
# Run Ollama LLM
ollama serve

# List the download ollama llms
ollama list

# Pull a image
ollama pull gemma3:12b
ollama pull qwen3:30b
```

## Create virtual environment

```sh
python --version                    # Verify your version of python 3.13.0
python -m venv .venv              
```

## Active the virtual environment

```sh
# Windows
.venv/Script/activate

# Linux and Mac
source .venv/bin/activate
```

## Install the dependencies

```sh
py -m pip install -r requirements.txt
py -m pip install [Dependency_Name]  # Install a new dependency 

```

## Start

```sh


# Active the virtual environment
.venv/Script/activate

# Run client
python client.py

```
