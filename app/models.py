from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=5000)
    conversation_id: str | None = None


class SourceItem(BaseModel):
    document: str
    order_number: str
    category: str
    date: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence_score: float = Field(ge=0.0, le=1.0)
    conversation_id: str | None = None


class IngestionResponse(BaseModel):
    indexed_rows: int
    indexed_chunks: int
    skipped_rows: int

