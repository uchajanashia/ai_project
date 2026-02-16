import argparse
import logging
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pandas as pd

from app.chunking import (
    REQUIRED_COLUMNS,
    add_canonical_columns,
    build_chunk_records,
    get_tokenizer,
    normalize_column_name,
    validate_required_columns,
)
from app.config import Settings, get_settings
from app.embeddings import OpenAIEmbeddingClient
from app.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    indexed_rows: int
    indexed_chunks: int
    skipped_rows: int


def load_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [normalize_column_name(column) for column in df.columns]
    df = add_canonical_columns(df)
    validate_required_columns(df)
    return df


def build_chunks(df: pd.DataFrame, settings: Settings) -> tuple[list[dict], int]:
    tokenizer = get_tokenizer(settings.openai_embedding_model)
    chunks: list[dict] = []
    skipped = 0

    for row in df.to_dict(orient="records"):
        has_required_content = any(str(row.get(column, "")).strip() for column in REQUIRED_COLUMNS)
        if not has_required_content:
            skipped += 1
            continue

        row_chunks = build_chunk_records(
            row=row,
            tokenizer=tokenizer,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not row_chunks:
            skipped += 1
            continue
        chunks.extend(row_chunks)
    return chunks, skipped


def ingest_documents(settings: Settings) -> IngestionStats:
    df = load_dataframe(settings.csv_path)
    chunks, skipped_rows = build_chunks(df, settings)

    if not chunks:
        raise ValueError("No chunks were generated from CSV. Check source data quality.")

    embedding_client = OpenAIEmbeddingClient(settings=settings)
    vector_store = QdrantVectorStore(settings=settings)
    try:
        texts = [chunk["text"] for chunk in chunks]
        vectors = embedding_client.embed_texts(
            texts=texts,
            batch_size=settings.embedding_batch_size,
        )
        if not vectors:
            raise ValueError("Embedding generation returned no vectors.")

        vector_size = len(vectors[0])
        vector_store.ensure_collection(vector_size=vector_size)

        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vector_batch = vectors[start : start + batch_size]
            vector_store.upsert_chunks(chunk_batch, vector_batch)

        indexed_rows = len(df) - skipped_rows
        logger.info(
            "Ingestion finished. indexed_rows=%s indexed_chunks=%s skipped_rows=%s",
            indexed_rows,
            len(chunks),
            skipped_rows,
        )
        return IngestionStats(
            indexed_rows=indexed_rows,
            indexed_chunks=len(chunks),
            skipped_rows=skipped_rows,
        )
    finally:
        embedding_client.close()
        vector_store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest InfoHub CSV into Qdrant.")
    parser.add_argument("--csv-path", default=None, help="Path to CSV file.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    settings = get_settings()
    if args.csv_path:
        settings = replace(settings, csv_path=args.csv_path)

    stats = ingest_documents(settings=settings)
    print(
        "Indexed rows: "
        f"{stats.indexed_rows}, Indexed chunks: {stats.indexed_chunks}, Skipped rows: {stats.skipped_rows}"
    )


if __name__ == "__main__":
    main()

