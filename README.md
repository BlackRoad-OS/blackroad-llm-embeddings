# LLM Embeddings Service

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
