"""Process-local visitor conversations, separate from reusable RAG resources."""

import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationReset(Exception):
    """Generation belongs to a conversation version that has been reset."""


class ConversationBusy(Exception):
    """Another generation is already running for this conversation."""


class Conversation:
    """Own the completed turns of one visitor's conversation."""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []
        self._revision = 0
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Reject overlapping inference without blocking reads or resets."""
        if not self._generation_lock.acquire(blocking=False):
            raise ConversationBusy
        try:
            yield
        finally:
            self._generation_lock.release()

    def snapshot(self) -> tuple[int, list[BaseMessage]]:
        with self._lock:
            return self._revision, self._messages.copy()

    def append_turn(self, question: str, answer: str, *, revision: int) -> None:
        with self._lock:
            if revision != self._revision:
                raise ConversationReset
            self._messages.extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )

    def clear(self) -> None:
        with self._lock:
            self._revision += 1
            self._messages.clear()


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

    def read(self, visitor_id: str | None) -> list[BaseMessage]:
        """Read existing context without allocating or refreshing a session."""
        with self._lock:
            conversation = self._conversations.get(visitor_id) if visitor_id else None
            return conversation.snapshot()[1] if conversation is not None else []

    def clear(self, visitor_id: str | None) -> None:
        """Reset an existing visitor's context without creating another session."""
        with self._lock:
            conversation = self._conversations.get(visitor_id) if visitor_id else None
            if conversation is not None:
                conversation.clear()


_initialization_lock = threading.Lock()


def get_conversation_store(request: Request) -> ConversationStore:
    """Share state within one application, never between independent test apps."""
    with _initialization_lock:
        store = getattr(request.app.state, "conversations", None)
        if store is None:
            store = ConversationStore()
            request.app.state.conversations = store
    return store
