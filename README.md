
---

### README.md


# Aether Engine

High-performance RAG (Retrieval-Augmented Generation) system with a localized reasoning model and vector database.

## Features
* Reasoning Model: Llama 3.2 3B via Ollama.
* Vector Database: Qdrant for semantic search.
* Multilingual Support: Optimized for Italian and English via paraphrase-multilingual-MiniLM-L12-v2.
* GPU Acceleration: Configured for NVIDIA GPUs (GTX 1080 Ti).
* Automated Management: Python script for builds, logs, and maintenance.

## Prerequisites
* OS: Arch Linux (recommended) or any modern Linux distribution.
* Hardware: NVIDIA GPU with 8GB+ VRAM.
* Software: Docker, Docker Compose, NVIDIA Container Toolkit, Python 3.x.

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/username/aether-engine.git](https://github.com/username/aether-engine.git)
   cd aether-engine


2. Build and start the system:
```bash
python manager.py rebuild

```


3. Pull the required LLM model:
```bash
docker exec -it aether-engine-ollama-1 ollama pull llama3.2:3b

```



## Usage

1. Open http://localhost:8000.
2. Click "New Chat (Reset)" to initialize the vector collection.
3. Upload a PDF or provide a URL for ingestion.
4. Interact with the chat interface.

## Management Commands

* View logs: `python manager.py logs`
* Stop services: `python manager.py stop`
* Clean cache and locks: `python manager.py clean`
* Check status: `python manager.py status`

## Project Structure

* /src: Rust backend source code.
* /static: Frontend assets (HTML/JS/CSS).
* manager.py: System automation script.
* docker-compose.yml: Service orchestration.
* Dockerfile: Multi-stage Rust build.

```

---

### .gitignore

```text
# Rust
/target
Cargo.lock

# Data and Cache
.fastembed_cache/
*.pdf
*.log

# Python
__pycache__/
*.pyc

# Environment and IDE
.env
.vscode/
.idea/

# Docker
.docker

```

---
