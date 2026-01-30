# Aether Engine

![Rust](https://img.shields.io/badge/Backend-Rust%20Rocket-orange?style=flat&logo=rust)
![Docker](https://img.shields.io/badge/Container-Docker-blue?style=flat&logo=docker)
![AI](https://img.shields.io/badge/Model-Llama%203.2-purple?style=flat&logo=openai)
![License](https://img.shields.io/badge/License-MIT-green)

**Aether Engine** is a high-performance, local Retrieval-Augmented Generation (RAG) system designed for semantic analysis and reasoning on private documents. It leverages the safety and speed of **Rust** for the API, **Qdrant** for vector search, and **Ollama** for local inference, all optimized for NVIDIA GPUs.

## Key Features

* **Rust Backend**: Built with the **Rocket** framework for ultra-low latency, type safety, and asynchronous processing.
* **Advanced Retrieval Pipeline**: Implements a two-stage retrieval process using Qdrant for initial vector search and a **Cross-Encoder (BGE-Reranker-Base)** for high-precision context reranking.
* **Real-Time Streaming**: Delivers responses token-by-token to the frontend using asynchronous Rust streams, minimizing perceived latency.
* **Multi-Source Ingestion**:
    * **PDF Processing**: Extracts text using `lopdf` with page-level metadata tracking.
    * **Web Scraping**: Intelligent HTML parsing to ingest and analyze content from URLs.
* **Local Reasoning**: Uses **Llama 3.2 3B** (via Ollama) for inference, ensuring complete data privacy.
* **Hardware Monitor**: Real-time tracking of **CPU** and **VRAM** usage displayed directly in the user interface.
* **LLM-as-a-Judge**: Automated post-processing pipeline that evaluates the generated answer for *Faithfulness* and *Relevance* using a secondary LLM pass.
* **Smart Caching**: Docker volumes are configured to persist AI models (Embeddings and Reranker) on the host disk, ensuring near-instant server startup times.
* **NVIDIA Native**: Docker container optimized with `nvidia/cuda` base images for direct GPU pass-through.

## Prerequisites

* **OS**: Arch Linux, Ubuntu 22.04+, or any Linux distribution with Docker support.
* **Hardware**: NVIDIA GPU with minimum 8GB VRAM (Tested on GTX 1080 Ti).
* **Software**:
    * Docker & Docker Compose
    * NVIDIA Container Toolkit
    * Python 3.x (for the management script)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Amorlot/aether-engine.git](https://github.com/Amorlot/aether-engine.git)
    cd aether-engine
    ```

2.  **Build the System**
    Use the included Python manager to handle the multi-stage build. The first run will download the embedding and reranking models to the local cache.
    ```bash
    python manager.py rebuild
    ```

3.  **Download the LLM**
    Once containers are running, pull the specific model version:
    ```bash
    docker exec -it aether-ollama ollama pull llama3.2:3b
    ```

## Usage

1.  Access the dashboard at **http://localhost:8000**.
2.  **Resource Check**: Verify CPU and VRAM stats in the sidebar to ensure the GPU is recognized.
3.  **Ingestion**:
    * **PDF**: Use the upload button to index a local PDF file.
    * **Web**: Use the URL button to scrape and index a website.
    * Wait for the system confirmation (e.g., "Stored X chunks").
4.  **Query**: Type your question. The system will retrieve context, rerank the results, stream the answer, and provide an evaluation score upon completion.

## Management Script

A `manager.py` utility is provided to simplify DevOps tasks:

| Command | Description |
| :--- | :--- |
| `python manager.py start` | Starts all services in detached mode. |
| `python manager.py stop` | Gracefully stops containers. |
| `python manager.py rebuild` | Cleans cache and forces a full rebuild of the backend. |
| `python manager.py logs` | Streams logs from the Rust backend. |
| `python manager.py clean` | Removes temporary cache files. |
| `python manager.py status` | Shows container health status. |

## Project Structure

```text
aether-engine/
├── api-rust/           # Rust Backend (Rocket framework)
│   ├── src/            # Source code (main.rs)
│   ├── static/         # Frontend Assets (HTML/JS/CSS)
│   ├── Dockerfile      # Multi-stage NVIDIA/Rust build
│   └── Cargo.toml      # Dependencies
├── model_cache/        # Local storage for AI models (Auto-generated)
├── docker-compose.yml  # Service Orchestration
├── manager.py          # DevOps Automation Script
└── README.md           # Documentation