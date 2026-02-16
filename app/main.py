import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.ingestion import ingest_documents
from app.models import AskRequest, AskResponse, IngestionResponse
from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InfoHub Georgian Legal RAG API",
    version="3.0.0",
    description="Production-ready Georgian legal RAG system for tax and customs knowledge.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings=get_settings())


def reset_pipeline() -> None:
    if get_pipeline.cache_info().currsize:
        pipeline = get_pipeline()
        pipeline.close()
    get_pipeline.cache_clear()


@app.on_event("shutdown")
def on_shutdown() -> None:
    reset_pipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestionResponse)
def ingest() -> IngestionResponse:
    try:
        settings = get_settings()
        result = ingest_documents(settings=settings)
        reset_pipeline()
        return IngestionResponse(
            indexed_rows=result.indexed_rows,
            indexed_chunks=result.indexed_chunks,
            skipped_rows=result.skipped_rows,
        )
    except ValueError as error:
        logger.exception("Validation failure in /ingest")
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unhandled error in /ingest")
        raise HTTPException(status_code=500, detail=f"Internal error: {error}") from error


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        pipeline = get_pipeline()
        return pipeline.ask(question=question, conversation_id=payload.conversation_id)
    except ValueError as error:
        logger.exception("Validation failure in /ask")
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=500, detail=f"Internal error: {error}") from error

