# Aether Engine

![Rust](https://img.shields.io/badge/Backend-Rust%20Rocket-orange?style=flat&logo=rust)
![Docker](https://img.shields.io/badge/Container-Docker-blue?style=flat&logo=docker)
![AI](https://img.shields.io/badge/Model-Llama%203.2-purple?style=flat&logo=openai)
![License](https://img.shields.io/badge/License-MIT-green)

**Aether Engine** is a high-performance, local Retrieval-Augmented Generation (RAG) system designed for semantic analysis and reasoning on private documents. It leverages the safety and speed of **Rust** for the API, **Qdrant** for vector search, and **Ollama** for local inference, all optimized for NVIDIA GPUs.

## Key Features

* **Rust Backend**: Built with the **Rocket** framework for ultra-low latency and type safety.
* **Local Reasoning**: Uses **Llama 3.2 3B** (via Ollama) for inference, ensuring data privacy.
* **Semantic Search**: **Qdrant** vector database with `paraphrase-multilingual-MiniLM-L12-v2` embeddings for precise retrieval in Italian and English.
* **📊 Hardware Monitor**: Real-time tracking of **CPU** and **VRAM** usage directly in the UI.
* **LLM-as-a-Judge**: Automated self-evaluation pipeline that scores answers based on *Faithfulness* and *Relevance*.
* **Page-Aware Ingestion**: PDF processing using `lopdf` to track and cite specific page numbers.
* **NVIDIA Native**: Docker container optimized with `nvidia/cuda` base images for direct GPU pass-through.

## Prerequisites

* **OS**: Arch Linux, Ubuntu 22.04+, or any Linux distro with Docker support.
* **Hardware**: NVIDIA GPU with minimum 8GB VRAM (Tested on GTX 1080 Ti).
* **Software**:
    * Docker & Docker Compose
    * NVIDIA Container Toolkit
    * Python 3.x (for the management script)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Amorlot/aether-engine.git]
    cd aether-engine
    ```

2.  **Build the System**
    Use the included Python manager to handle the multi-stage build:
    ```bash
    python manager.py rebuild
    ```

3.  **Download the LLM**
    Once containers are running, pull the specific model version:
    ```bash
    docker exec -it aether-ollama ollama pull llama3.2:3b
    ```

##  Usage

1.  Access the dashboard at **[http://localhost:8000](http://localhost:8000)**.
2.  **Resource Check**: Verify CPU and VRAM stats in the sidebar.
3.  **Ingestion**:
    * Click the 📎 icon to upload a PDF.
    * Wait for the "Stored X chunks" confirmation.
4.  **Query**: Type your question. The system will retrieve context, reason, and provide a cited answer with an evaluation score.

##  Management Script

A `manager.py` utility is provided to simplify DevOps tasks:

| Command | Description |
| :--- | :--- |
| `python manager.py start` | Starts all services in detached mode. |
| `python manager.py stop` | Gracefully stops containers. |
| `python manager.py rebuild` | Cleans cache and forces a full rebuild. |
| `python manager.py logs` | Streams logs from the Rust backend. |
| `python manager.py clean` | Removes temporary cache files. |
| `python manager.py status` | Shows container health status. |

##  Project Structure

```text
aether-engine/
├── api-rust/           # Rust Backend (Rocket framework)
│   ├── src/            # Source code (main.rs)
│   ├── Dockerfile      # Multi-stage NVIDIA/Rust build
│   └── Cargo.toml      # Dependencies
├── static/             # Frontend Assets
│   └── index.html      # Dark Mode UI
├── k8s/                # Kubernetes Manifests
├── docker-compose.yml  # Service Orchestration
├── manager.py          # DevOps Automation Script
└── README.md           # Documentation