import logging
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float


class QdrantVectorStore:
    def __init__(self, settings: Settings):
        self.collection_name = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url)

    def _get_vector_size(self, collection_name: str) -> int:
        details = self.client.get_collection(collection_name)
        vectors = details.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            if hasattr(first, "size"):
                return int(first.size)
        raise ValueError("Could not determine vector size for existing Qdrant collection.")

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            existing_size = self._get_vector_size(self.collection_name)
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' vector size mismatch: "
                    f"existing={existing_size}, requested={vector_size}"
                )
            return

        logger.info(
            "Creating Qdrant collection '%s' with vector size %s",
            self.collection_name,
            vector_size,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: Sequence[dict], vectors: Sequence[list[float]]) -> None:
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = dict(chunk["metadata"])
            payload["text"] = chunk["text"]
            points.append(PointStruct(id=chunk["id"], vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    with_payload=True,
                    limit=top_k,
                )
                hits = response.points
            else:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    with_payload=True,
                    limit=top_k,
                )
        except UnexpectedResponse as error:
            if error.status_code == 404:
                logger.warning("Qdrant collection '%s' not found during search.", self.collection_name)
                return []
            raise

        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedChunk(
                    id=str(hit.id),
                    text=str(payload.get("text", "")).strip(),
                    metadata={str(key): value for key, value in payload.items()},
                    score=float(hit.score),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def close(self) -> None:
        self.client.close()
