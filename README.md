# Project 3: Local RAG System

A local Retrieval-Augmented Generation (RAG) system with a FastAPI backend, Streamlit UI, and optional Ollama LLM support. Upload `.txt` or `.pdf` documents, ingest them, and ask questions with cited sources.

## Features

- Local embeddings with `sentence-transformers`.
- Vector search using Chroma.
- FastAPI backend with `/ask`, `/sources`, `/ingest` endpoints.
- Streamlit UI with document upload and chat.
- Optional Ollama or OpenAI LLMs (mock LLM available by default).

## Prerequisites

- Windows + PowerShell
- Python 3.10+ (venv recommended)

## Setup

```powershell
# From the project root
python -m venv venv
& .\venv\Scripts\Activate.ps1

pip install -U pip
pip install langchain-text-splitters langchain-community langchain chromadb sentence-transformers pypdf fastapi uvicorn streamlit requests langchain-openai
```

## Build the Vector Store (one-time)

```powershell
& .\venv\Scripts\python.exe .\rag_core.py
```

## Run the API

```powershell
& .\venv\Scripts\python.exe .\rag_api.py
```

Optional dev reload:

```powershell
& .\venv\Scripts\python.exe -m uvicorn rag_api:app --reload --port 8000
```

## Run the UI

```powershell
& .\venv\Scripts\streamlit.exe run .\ui.py
```

## Add Documents

- Use the Streamlit sidebar **Add Document** uploader, then click **Ingest Document**.
- Only `.txt` and `.pdf` are supported.
- Uploaded files are saved in `documents/` and immediately indexed.

## Ask Questions

- Use the chat input in the UI.
- The backend will return an answer plus citations (source + page).

## Optional: Ollama LLM

1. Install Ollama: https://ollama.ai
2. Pull a model:

```powershell
ollama pull llama2
# or: ollama pull mistral
# or: ollama pull phi
```

3. Run `rag_with_llm.py`:

```powershell
& .\venv\Scripts\python.exe .\rag_with_llm.py
```

Choose a model with an environment variable:

```powershell
$env:OLLAMA_MODEL = "mistral"
```

## Troubleshooting

- **API 500 error about citations.page type**: Restart the API after code changes.
- **Vector store not initialized**: Run `rag_core.py` or ingest a document first.
- **Large model download**: First run may take several minutes for the embedding model download.

## Project Structure

```
project3-rag-system/
├─ rag_core.py
├─ rag_with_llm.py
├─ rag_api.py
├─ ui.py
├─ documents/
├─ chroma_db/
└─ README.md
```
