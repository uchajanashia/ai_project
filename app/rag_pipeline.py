import logging
import uuid
from difflib import SequenceMatcher
from statistics import mean

from openai import OpenAI

from app.chunking import deduplicate_sources
from app.config import Settings
from app.embeddings import OpenAIEmbeddingClient
from app.memory import ConversationMemory
from app.models import AskResponse, SourceItem
from app.prompts import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    build_context_message,
    build_latest_question_message,
    format_source_block,
)
from app.vector_store import QdrantVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


def normalize_similarity(score: float) -> float:
    if -1.0 <= score <= 1.0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))
    return max(0.0, min(1.0, score))


def strip_source_block(answer: str) -> str:
    if "წყარო:" in answer:
        return answer.split("წყარო:", maxsplit=1)[0].strip()
    return answer.strip()


def remove_near_duplicate_chunks(
    chunks: list[RetrievedChunk],
    similarity_threshold: float,
) -> list[RetrievedChunk]:
    unique: list[RetrievedChunk] = []
    for candidate in chunks:
        candidate_text = " ".join(candidate.text.lower().split())
        is_duplicate = False
        for existing in unique:
            existing_text = " ".join(existing.text.lower().split())
            ratio = SequenceMatcher(None, candidate_text, existing_text).ratio()
            if ratio >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(candidate)
    return unique


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_client = OpenAIEmbeddingClient(settings=settings)
        self.vector_store = QdrantVectorStore(settings=settings)
        self.llm_client = OpenAI(api_key=settings.openai_api_key)
        self.memory = ConversationMemory(max_history=settings.conversation_history_size)

    def ask(self, question: str, conversation_id: str | None = None) -> AskResponse:
        conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(
            "RAG question received. conversation_id=%s top_k=%s",
            conversation_id,
            self.settings.top_k,
        )

        query_vector = self.embedding_client.embed_query(question)
        retrieved = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.settings.top_k,
        )
        retrieved = remove_near_duplicate_chunks(
            chunks=retrieved,
            similarity_threshold=self.settings.near_duplicate_threshold,
        )
        retrieved = sorted(retrieved, key=lambda item: item.score, reverse=True)[: self.settings.top_k]

        similarity_scores = [normalize_similarity(item.score) for item in retrieved]
        average_similarity = round(mean(similarity_scores), 4) if similarity_scores else 0.0
        highest_similarity = max(similarity_scores) if similarity_scores else 0.0
        logger.info(
            "Retrieval scores: highest=%.4f average=%.4f all=%s",
            highest_similarity,
            average_similarity,
            [round(score, 4) for score in similarity_scores],
        )

        if (
            not retrieved
            or highest_similarity < self.settings.min_context_score
            or average_similarity < self.settings.min_context_score
        ):
            answer_body = NOT_FOUND_MESSAGE
            sources: list[dict] = []
        else:
            context_chunks = [
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": normalize_similarity(chunk.score),
                }
                for chunk in retrieved
            ]
            model_answer = self._generate_answer(
                question=question,
                conversation_id=conversation_id,
                context_chunks=context_chunks,
            )
            answer_body = strip_source_block(model_answer) or NOT_FOUND_MESSAGE
            sources = deduplicate_sources(chunk.metadata for chunk in retrieved)

        answer = f"{answer_body}\n\n{format_source_block(sources)}"
        self.memory.add_interaction(
            conversation_id=conversation_id,
            question=question,
            answer=answer_body,
        )

        return AskResponse(
            answer=answer,
            sources=[
                SourceItem(
                    document=source["document"],
                    order_number=source["order_number"],
                    category=source["category"],
                    date=source["date"],
                    url=source["url"],
                )
                for source in sources
            ],
            confidence_score=average_similarity,
            conversation_id=conversation_id,
        )

    def _generate_answer(
        self,
        question: str,
        conversation_id: str,
        context_chunks: list[dict],
    ) -> str:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for item in self.memory.get_history(conversation_id):
            messages.append({"role": "user", "content": item.question})
            messages.append({"role": "assistant", "content": item.answer})

        messages.append({"role": "user", "content": build_context_message(context_chunks)})
        messages.append({"role": "user", "content": build_latest_question_message(question)})

        completion = self.llm_client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=self.settings.temperature,
            messages=messages,
        )
        return (completion.choices[0].message.content or "").strip()

    def close(self) -> None:
        self.embedding_client.close()
        self.vector_store.close()
        self.llm_client.close()

