from dataclasses import dataclass
from functools import lru_cache

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


DEFAULT_SETTINGS = Settings(
    openai_api_key="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",
    openai_chat_model="gpt-4.1",
    openai_embedding_model="text-embedding-3-large",
    qdrant_url="https://dde671cd-9d27-4bd6-815b-9783ac42dd78.sa-east-1-0.aws.cloud.qdrant.io:6333",
    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uiYQWuXXr_Y5JdODvDgUhT5PpqsBn6Wx312vSQRG9hA",
    qdrant_collection="infohub_documents",
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
