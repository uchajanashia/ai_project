from collections.abc import Sequence

from openai import OpenAI

from app.config import Settings


class OpenAIEmbeddingClient:
    def __init__(self, settings: Settings):
        self.model = settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_query(self, question: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=question,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def close(self) -> None:
        self.client.close()

