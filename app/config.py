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
    qdrant_api_key: str | None
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
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


load_dotenv()


DEFAULT_SETTINGS = Settings(
    openai_api_key=_required_env("OPENAI_API_KEY"),
    openai_chat_model="gpt-4.1",
    openai_embedding_model="text-embedding-3-large",
    qdrant_url=_required_env("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    qdrant_collection=os.getenv("QDRANT_COLLECTION", "infohub_documents"),
    csv_path="data/infohub.csv",
    top_k=8,
    chunk_size_tokens=900,
    chunk_overlap_tokens=250,
    embedding_batch_size=64,
    temperature=0.2,
    min_context_score=0.2,
    conversation_history_size=5,
    near_duplicate_threshold=0.92,
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return DEFAULT_SETTINGS
