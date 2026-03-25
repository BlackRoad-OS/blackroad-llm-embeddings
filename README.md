<!-- BlackRoad SEO Enhanced -->

# ulackroad llm emueddings

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad-AI](https://img.shields.io/badge/Org-BlackRoad-AI-2979ff?style=for-the-badge)](https://github.com/BlackRoad-AI)

**ulackroad llm emueddings** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

### BlackRoad Ecosystem
| Org | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | AI/ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh networking |

**Website**: [blackroad.io](https://blackroad.io) | **Chat**: [chat.blackroad.io](https://chat.blackroad.io) | **Search**: [search.blackroad.io](https://search.blackroad.io)

---


Vector embeddings generation, storage, search, and clustering service for LLMs.

## Features

- **Ollama Integration**: Connect to local Ollama instance for embeddings
- **Mock Fallback**: Deterministic hash-based embeddings when Ollama unavailable
- **Vector Search**: Cosine similarity-based semantic search
- **Clustering**: Manual k-means clustering implementation
- **Storage**: SQLite-based persistent storage
- **Batch Operations**: Process multiple embeddings efficiently
- **Export**: JSONL format export for data pipeline integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Embedding

```bash
python src/embeddings.py embed "What is machine learning?"
```

### Search Embeddings

```bash
python src/embeddings.py search "machine learning" --top 5
```

### Get Statistics

```bash
python src/embeddings.py stats
```

### Python API

```python
from src.embeddings import EmbeddingsService

service = EmbeddingsService()

# Store embedding
record = service.store("Hello world", metadata={"source": "web"})

# Search
results = service.search("hello", top_k=3)

# Cluster
clusters = service.cluster(n_clusters=5)

# Export
count = service.export_jsonl("embeddings.jsonl")
```

## Database

Embeddings stored in SQLite at `~/.blackroad/embeddings.db`.

## License

MIT
