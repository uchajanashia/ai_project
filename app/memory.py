from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass
class Interaction:
    question: str
    answer: str


class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._lock = Lock()
        self._conversations: dict[str, deque[Interaction]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def get_history(self, conversation_id: str) -> list[Interaction]:
        with self._lock:
            history = self._conversations.get(conversation_id)
            if not history:
                return []
            return list(history)

    def add_interaction(self, conversation_id: str, question: str, answer: str) -> None:
        with self._lock:
            bucket = self._conversations[conversation_id]
            bucket.append(Interaction(question=question, answer=answer))

