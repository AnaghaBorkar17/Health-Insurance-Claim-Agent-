# Multi-Agent Health Insurance Claim Pre-Authorization and Denial Prevention System

> **Using LangGraph, Ollama, and Google Gemini**

---

## Overview

This project implements an explainable, multi-agent artificial intelligence workflow for health insurance pre-authorization claims. The system evaluates synthetic patient claims against fictional health insurance policies, validates medical codes (ICD-10 / CPT), performs fraud/anomaly checks, provides policy clause citations, and generates structured recommendations (**Approve**, **Query**, or **Reject**).

A licensed human reviewer must review and confirm the final decision. Automated rejections are never finalized without human authorization.

---

## Architecture

The system consists of 9 modular agents managed via a **LangGraph StateGraph**:

1. **Supervisor / Router Agent** (`app/agents/supervisor.py`)
2. **Document Extraction Agent** (`app/agents/extraction.py`)
3. **PII Redaction Agent** (`app/agents/redaction.py`)
4. **Policy RAG Agent** (`app/agents/policy_rag.py`)
5. **Code Validation Agent** (`app/agents/code_validation.py`)
6. **Fraud & Anomaly Subgraph** (`app/agents/fraud/`)
7. **Decision Agent** (`app/agents/decision.py`)
8. **Human Review Node** (`app/graph.py` with `interrupt`)
9. **Communication Agent** (`app/agents/communication.py`)

---

## Directory Structure

```
claim-agents/
|-- app/
|   |-- __init__.py
|   |-- config.py              # Pydantic Settings & environment config
|   |-- state.py               # Typed LangGraph state
|   |-- llm.py                 # Multi-LLM provider abstraction
|   |-- graph.py               # Main StateGraph assembly
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- supervisor.py
|   |   |-- extraction.py
|   |   |-- redaction.py
|   |   |-- policy_rag.py
|   |   |-- code_validation.py
|   |   |-- decision.py
|   |   |-- communication.py
|   |   `-- fraud/
|   |       |-- __init__.py
|   |       |-- graph.py
|   |       `-- nodes.py
|   `-- tools/
|       |-- __init__.py
|       |-- document_parser.py
|       |-- ocr.py
|       |-- policy_ingestion.py
|       |-- database.py
|       `-- validators.py
|-- data/
|   |-- policies/              # Fictional insurance policy documents
|   |-- synthetic_cases/       # Labelled synthetic test cases
|   |-- reference_tables/      # ICD-10 and procedure reference tables
|   `-- chroma/                # Persistent vector database
|-- ui/
|   `-- streamlit_app.py       # Streamlit user interface
|-- tests/                     # Unit and integration test suite
|-- eval/
|   `-- run_eval.py            # Quantitative benchmark evaluation
|-- .env                       # Local secrets (excluded from Git)
|-- .env.example               # Template environment configuration
|-- .gitignore                 # Git ignore rules
|-- requirements.txt           # Project dependencies
|-- verify_phase1.py           # Phase 1 verification script
`-- README.md                  # Project documentation
```

---

## Setup & Verification

### 1. Activate Virtual Environment
```powershell
cd "c:\Users\HP\WPS Cloud\WPSDrive\567096015736\WPSDrive\Developers\claim-agents"
.\venv\Scripts\Activate.ps1
```

### 2. Verify Local Ollama Models
Ensure Ollama is running and has required models:
```powershell
ollama list
```
Required models:
- `qwen2.5:3b` (Local LLM)
- `nomic-embed-text` (Local Embeddings for ChromaDB)

### 3. Verify Phase 1 Setup
Run the verification script:
```powershell
python verify_phase1.py
```
