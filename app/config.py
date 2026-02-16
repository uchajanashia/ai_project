import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    qdrant_url: str
    qdrant_collection: str
    csv_path: str
    top_k: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_batch_size: int
    temperature: float
    min_context_score: float
    conversation_history_size: int
    near_duplicate_threshold: float


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    return Settings(
        openai_api_key=_required_env("OPENAI_API_KEY"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1").strip(),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-large",
        ).strip(),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        qdrant_collection="infohub_documents",
        csv_path=os.getenv("CSV_PATH", "data/infohub.csv").strip(),
        top_k=8,
        chunk_size_tokens=900,
        chunk_overlap_tokens=250,
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        temperature=0.2,
        min_context_score=0.2,
        conversation_history_size=5,
        near_duplicate_threshold=float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.92")),
    )
