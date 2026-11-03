"""Process-local visitor conversations, separate from reusable RAG resources."""

import secrets
import threading

from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class Conversation:
    """Own the completed turns of one visitor's conversation."""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []
        self._lock = threading.Lock()

    def snapshot(self) -> list[BaseMessage]:
        with self._lock:
            return self._messages.copy()

    def append_turn(self, question: str, answer: str) -> None:
        with self._lock:
            self._messages.extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )


class ConversationStore:
    """Resolve only server-issued visitor identifiers to conversation state."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def open(self, visitor_id: str | None) -> tuple[str, Conversation]:
        with self._lock:
            if visitor_id is None or visitor_id not in self._conversations:
                visitor_id = secrets.token_urlsafe(32)
                self._conversations[visitor_id] = Conversation()
            return visitor_id, self._conversations[visitor_id]


_initialization_lock = threading.Lock()


def get_conversation_store(request: Request) -> ConversationStore:
    """Share state within one application, never between independent test apps."""
    with _initialization_lock:
        store = getattr(request.app.state, "conversations", None)
        if store is None:
            store = ConversationStore()
            request.app.state.conversations = store
    return store
