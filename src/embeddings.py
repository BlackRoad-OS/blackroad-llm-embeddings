"""
LLM Embeddings Service and Vector Operations

Provides embeddings generation, storage, search, and clustering capabilities
using Ollama or mock vectors. No external ML dependencies required.
"""

import sqlite3
import json
import math
import hashlib
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import urllib.request
import urllib.error


DB_PATH = Path.home() / ".blackroad" / "embeddings.db"
OLLAMA_URL = "http://localhost:11434"


@dataclass
class EmbeddingRecord:
    """Represents a stored embedding."""
    id: str
    text: str
    model: str
    vector: List[float]
    dim: int
    created_at: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class EmbeddingsService:
    """LLM embeddings service with vector operations."""

    def __init__(self):
        """Initialize the embeddings service."""
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and tables exist."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                vector TEXT NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def embed(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Generate embedding for text using Ollama or mock."""
        try:
            # Try to call Ollama
            return self._embed_ollama(text, model)
        except Exception:
            # Fallback to mock embedding
            return self._embed_mock(text)

    def _embed_ollama(self, text: str, model: str) -> List[float]:
        """Call Ollama embedding API."""
        url = f"{OLLAMA_URL}/api/embeddings"
        data = json.dumps({"model": model, "prompt": text}).encode()
        
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode())
                return result.get("embedding", [])
        except (urllib.error.URLError, urllib.error.HTTPError):
            raise Exception("Ollama not available")

    def _embed_mock(self, text: str) -> List[float]:
        """Generate mock embedding using hash-based approach."""
        # Deterministic mock embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Generate 384-dimensional vector
        vector = []
        for i in range(384):
            seed = hash_int + i
            # Simple pseudo-random number generator
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            vector.append((seed % 1000) / 1000.0 - 0.5)
        
        return vector

    def embed_batch(self, texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
        """Batch embedding generation."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text, model))
        return embeddings

    def store(self, text: str, vector: List[float] = None, metadata: Dict = None) -> EmbeddingRecord:
        """Store an embedding."""
        if vector is None:
            vector = self.embed(text)
        if metadata is None:
            metadata = {}

        # Generate record ID
        record_id = hashlib.md5(f"{text}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        dim = len(vector)

        record = EmbeddingRecord(
            id=record_id,
            text=text,
            model="unknown",
            vector=vector,
            dim=dim,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata,
        )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO embeddings (id, text, model, vector, dim, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.text,
                record.model,
                json.dumps(record.vector),
                record.dim,
                record.created_at,
                json.dumps(record.metadata),
            ),
        )
        conn.commit()
        conn.close()

        return record

    def search(self, query_text: str, top_k: int = 5, model: str = "nomic-embed-text") -> List[Tuple[str, float]]:
        """Search stored embeddings by similarity."""
        query_vector = self.embed(query_text, model)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, vector FROM embeddings")
        rows = cursor.fetchall()
        conn.close()

        # Calculate similarities
        similarities = []
        for row in rows:
            stored_vector = json.loads(row["vector"])
            similarity = self.cosine_similarity(query_vector, stored_vector)
            similarities.append((row["text"], similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def cluster(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """K-means clustering of stored embeddings (manual implementation)."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, vector FROM embeddings")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) == 0:
            return {}

        # Convert to list of vectors
        embeddings = [(row["text"], json.loads(row["vector"])) for row in rows]

        if len(embeddings) <= n_clusters:
            # Each embedding is its own cluster
            return {i: [text] for i, (text, _) in enumerate(embeddings)}

        # Simple k-means initialization
        import random
        random.seed(42)
        centroids = [embeddings[i][1] for i in random.sample(range(len(embeddings)), n_clusters)]

        # Run k-means iterations
        for _ in range(10):
            clusters = [[] for _ in range(n_clusters)]

            # Assign points to nearest centroid
            for text, vector in embeddings:
                distances = [
                    self.cosine_similarity(vector, centroid) for centroid in centroids
                ]
                cluster_idx = distances.index(max(distances))
                clusters[cluster_idx].append(text)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    # Find embeddings for cluster texts
                    cluster_vectors = [
                        json.loads(row["vector"]) 
                        for row in rows 
                        if row["text"] in cluster
                    ]
                    # Compute mean
                    mean_vector = [
                        sum(vec[i] for vec in cluster_vectors) / len(cluster_vectors)
                        for i in range(len(cluster_vectors[0]))
                    ]
                    new_centroids.append(mean_vector)
                else:
                    new_centroids.append(centroids[len(new_centroids)])

            centroids = new_centroids

        # Final assignment
        result = {}
        for text, vector in embeddings:
            distances = [self.cosine_similarity(vector, centroid) for centroid in centroids]
            cluster_idx = distances.index(max(distances))
            if cluster_idx not in result:
                result[cluster_idx] = []
            result[cluster_idx].append(text)

        return result

    def list_records(self, limit: int = 50) -> List[Dict]:
        """List stored embeddings with text preview."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text, model, dim, created_at FROM embeddings LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            text_preview = row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"]
            records.append({
                "id": row["id"],
                "text_preview": text_preview,
                "model": row["model"],
                "dim": row["dim"],
                "created_at": row["created_at"],
            })

        return records

    def delete(self, record_id: str) -> bool:
        """Delete an embedding record."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings WHERE id = ?", (record_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def export_jsonl(self, output_path: str) -> int:
        """Export all embeddings to JSONL format."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        rows = cursor.fetchall()
        conn.close()

        count = 0
        with open(output_path, "w") as f:
            for row in rows:
                record = {
                    "id": row["id"],
                    "text": row["text"],
                    "model": row["model"],
                    "vector": json.loads(row["vector"]),
                    "dim": row["dim"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata"]),
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        return count

    def stats(self) -> Dict:
        """Get embeddings statistics."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        total = cursor.fetchone()["count"]

        # Models used
        cursor.execute("SELECT DISTINCT model FROM embeddings")
        models = [row["model"] for row in cursor.fetchall()]

        # Average dimension
        cursor.execute("SELECT AVG(dim) as avg_dim FROM embeddings")
        avg_dim = cursor.fetchone()["avg_dim"] or 0

        conn.close()

        # Estimate storage size
        storage_size_mb = (Path(DB_PATH).stat().st_size if Path(DB_PATH).exists() else 0) / (1024 * 1024)

        return {
            "total_embeddings": total,
            "models_used": models,
            "avg_dim": avg_dim,
            "storage_size_mb": storage_size_mb,
        }


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="LLM Embeddings Service")
    subparsers = parser.add_subparsers(dest="command")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embedding")
    embed_parser.add_argument("text", help="Text to embed")
    embed_parser.add_argument("--model", default="nomic-embed-text")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search embeddings")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--top", type=int, default=5, help="Number of results")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")

    args = parser.parse_args()

    service = EmbeddingsService()

    if args.command == "embed":
        vector = service.embed(args.text, args.model)
        print(f"Generated {len(vector)}-dimensional embedding")

    elif args.command == "search":
        results = service.search(args.query, args.top)
        for i, (text, similarity) in enumerate(results, 1):
            print(f"{i}. {text[:60]} (similarity: {similarity:.4f})")

    elif args.command == "stats":
        stats = service.stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
