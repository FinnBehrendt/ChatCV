"""Process-local visitor conversations, separate from reusable RAG resources."""

import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic

from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import config


class ConversationReset(Exception):
    """Generation belongs to a conversation version that has been reset."""


class ConversationBusy(Exception):
    """Another generation is already running for this conversation."""


class ConversationExpired(ConversationReset):
    """Idle context was discarded; require an explicit fresh submission."""


class ConversationCapacity(Exception):
    """No slot is available for another visitor conversation."""


class Conversation:
    """Own the completed turns of one visitor's conversation."""

    def __init__(self, *, max_turns: int, idle_seconds: int) -> None:
        self._max_turns = max_turns
        self._idle_seconds = idle_seconds
        self._last_activity = monotonic()
        self._expired = False
        self._retired = False
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
            self._require_active()
            return self._revision, self._messages.copy()

    def prepare(self) -> None:
        """Check expiry without treating cookie preparation as activity."""
        with self._lock:
            self._require_active()

    def read(self) -> list[BaseMessage]:
        with self._lock:
            self._expire_if_idle()
            if self._expired:
                raise ConversationExpired
            return self._messages.copy()

    def _expire_if_idle(self) -> None:
        """Caller holds the state lock; expiration invalidates pending snapshots."""
        if (
            not self._expired
            and monotonic() - self._last_activity >= self._idle_seconds
        ):
            self._revision += 1
            self._messages.clear()
            self._expired = True

    def _require_active(self) -> None:
        if self._retired:
            raise ConversationExpired
        self._expire_if_idle()
        if self._expired:
            # Report once on an active request, never silently retry inference.
            self._expired = False
            self._last_activity = monotonic()
            raise ConversationExpired

    def retire_if_idle(self) -> bool:
        """Invalidate abandoned state permanently before releasing its slot."""
        with self._lock:
            self._expire_if_idle()
            if self._expired:
                self._retired = True
            return self._retired

    def append_turn(self, question: str, answer: str, *, revision: int) -> None:
        with self._lock:
            self._require_active()
            if revision != self._revision:
                raise ConversationReset
            self._messages.extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )
            self._messages = self._messages[-2 * self._max_turns :]
            self._last_activity = monotonic()

    def clear(self) -> None:
        with self._lock:
            self._revision += 1
            self._messages.clear()
            self._expired = False
            self._last_activity = monotonic()


class ConversationStore:
    """Resolve only server-issued visitor identifiers to conversation state."""

    def __init__(self, *, max_turns: int, idle_seconds: int, capacity: int) -> None:
        self._capacity = capacity
        self._max_turns = max_turns
        self._idle_seconds = idle_seconds
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def open(self, visitor_id: str | None) -> tuple[str, Conversation]:
        with self._lock:
            if visitor_id is not None and visitor_id not in self._conversations:
                # A stale or unrecognized cookie never silently resumes as new.
                raise ConversationExpired
            if visitor_id is None:
                self._conversations = {
                    key: conversation
                    for key, conversation in self._conversations.items()
                    if not conversation.retire_if_idle()
                }
                if len(self._conversations) >= self._capacity:
                    raise ConversationCapacity
                visitor_id = secrets.token_urlsafe(32)
                self._conversations[visitor_id] = Conversation(
                    max_turns=self._max_turns, idle_seconds=self._idle_seconds
                )
            return visitor_id, self._conversations[visitor_id]

    def read(self, visitor_id: str | None) -> list[BaseMessage]:
        """Read existing context without allocating or refreshing a session."""
        with self._lock:
            conversation = self._conversations.get(visitor_id) if visitor_id else None
            if visitor_id is not None and conversation is None:
                raise ConversationExpired
            return conversation.read() if conversation is not None else []

    def clear(self, visitor_id: str | None) -> bool:
        """Reset known context and report whether the visitor is still tracked."""
        with self._lock:
            conversation = self._conversations.get(visitor_id) if visitor_id else None
            if conversation is not None:
                conversation.clear()
                return True
            return False


_initialization_lock = threading.Lock()


def get_conversation_store(request: Request) -> ConversationStore:
    """Share state within one application, never between independent test apps."""
    with _initialization_lock:
        store = getattr(request.app.state, "conversations", None)
        if store is None:
            store = ConversationStore(
                max_turns=config.conversation_max_turns,
                idle_seconds=config.conversation_idle_minutes * 60,
                capacity=config.conversation_capacity,
            )
            request.app.state.conversations = store
    return store
