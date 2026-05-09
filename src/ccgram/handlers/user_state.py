"""Centralized user-data helpers for context.user_data access.

All string keys used with PTB's context.user_data dict are defined here
to prevent typos and enable IDE navigation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, MutableMapping

PENDING_THREAD_ID = "_pending_thread_id"
PENDING_THREAD_TEXT = "_pending_thread_text"
RECOVERY_WINDOW_ID = "_recovery_window_id"
RECOVERY_SESSIONS = "_recovery_sessions"
RESUME_SESSIONS = "_resume_sessions"
RESUME_THREAD_ID = "_resume_thread_id"
RESUME_SELECTED_CWD = "_resume_selected_cwd"
RESUME_PROVIDER = "_resume_provider"
RESUME_APPROVAL_MODE = "_resume_approval_mode"
VOICE_PENDING = (
    "_voice_pending"  # dict[tuple[int, int], str]: (chat_id, msg_id) → transcribed text
)

SEND_PATH_KEY = "send_path"
SEND_PAGE_KEY = "send_page"
SEND_ITEMS_KEY = "send_items"
SEND_WINDOW_ID_KEY = "send_window_id"
SEND_CWD_KEY = "send_cwd"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PendingChunk:
    """One Telegram text update that belongs to a pending prompt."""

    text: str
    message_id: int | None = None
    date: datetime | None = None


@dataclass(slots=True)
class PendingPrompt:
    """Text waiting for a picker/recovery flow to resolve before delivery."""

    chunks: list[PendingChunk] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    last_chunk_at: datetime = field(default_factory=_utc_now)
    locked_for_picker: bool = False

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        message_id: int | None = None,
        date: datetime | None = None,
        locked_for_picker: bool = False,
    ) -> "PendingPrompt":
        prompt = cls(locked_for_picker=locked_for_picker)
        prompt.add_chunk(text, message_id=message_id, date=date)
        return prompt

    def add_chunk(
        self,
        text: str,
        *,
        message_id: int | None = None,
        date: datetime | None = None,
    ) -> None:
        timestamp = date or _utc_now()
        if not self.chunks:
            self.created_at = timestamp
        self.chunks.append(PendingChunk(text=text, message_id=message_id, date=date))
        self.last_chunk_at = timestamp

    def combined_text(self, *, separator: str = "\n\n") -> str:
        return separator.join(chunk.text for chunk in self.chunks)

    def __bool__(self) -> bool:
        return any(chunk.text for chunk in self.chunks)


def _message_id(message: Any | None) -> int | None:
    value = getattr(message, "message_id", None)
    return value if isinstance(value, int) else None


def _message_date(message: Any | None) -> datetime | None:
    value = getattr(message, "date", None)
    return value if isinstance(value, datetime) else None


def get_pending_prompt(
    user_data: MutableMapping[str, Any] | None,
) -> PendingPrompt | None:
    """Return pending prompt, upgrading legacy string state in place."""

    if user_data is None:
        return None
    value = user_data.get(PENDING_THREAD_TEXT)
    if value is None:
        return None
    if isinstance(value, PendingPrompt):
        return value
    if isinstance(value, str):
        prompt = PendingPrompt.from_text(value)
        user_data[PENDING_THREAD_TEXT] = prompt
        return prompt
    return None


def get_pending_prompt_text(user_data: MutableMapping[str, Any] | None) -> str | None:
    prompt = get_pending_prompt(user_data)
    if not prompt:
        return None
    return prompt.combined_text()


def flush_pending_prompt_text(
    user_data: MutableMapping[str, Any] | None,
) -> str | None:
    """Return combined pending text and clear pending thread state."""

    text = get_pending_prompt_text(user_data)
    clear_pending_thread(user_data)
    return text


def set_pending_prompt(
    user_data: MutableMapping[str, Any],
    text: str,
    *,
    message: Any | None = None,
    locked_for_picker: bool = False,
) -> PendingPrompt:
    prompt = PendingPrompt.from_text(
        text,
        message_id=_message_id(message),
        date=_message_date(message),
        locked_for_picker=locked_for_picker,
    )
    user_data[PENDING_THREAD_TEXT] = prompt
    return prompt


def append_pending_prompt(
    user_data: MutableMapping[str, Any],
    text: str,
    *,
    message: Any | None = None,
) -> PendingPrompt:
    prompt = get_pending_prompt(user_data)
    if prompt is None:
        prompt = PendingPrompt()
        user_data[PENDING_THREAD_TEXT] = prompt
    prompt.add_chunk(text, message_id=_message_id(message), date=_message_date(message))
    return prompt


def set_pending_thread(
    user_data: MutableMapping[str, Any],
    thread_id: int | None,
    text: str,
    *,
    message: Any | None = None,
    locked_for_picker: bool = False,
) -> PendingPrompt:
    user_data[PENDING_THREAD_ID] = thread_id
    return set_pending_prompt(
        user_data,
        text,
        message=message,
        locked_for_picker=locked_for_picker,
    )


def clear_pending_thread(user_data: MutableMapping[str, Any] | None) -> None:
    if user_data is None:
        return
    user_data.pop(PENDING_THREAD_ID, None)
    user_data.pop(PENDING_THREAD_TEXT, None)
