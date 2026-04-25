"""Safe message sending helpers with entity-based formatting.

Provides utility functions for sending Telegram messages with automatic
conversion to entity-based formatting (no parse errors possible) and
fallback to plain text on failure.

Functions:
  - rate_limit_send: Rate limiter to avoid Telegram flood control
  - rate_limit_send_message: Combined rate limiting + send with fallback
  - safe_reply: Reply with entities, fallback to plain text
  - safe_edit: Edit message with entities, fallback to plain text
  - safe_send: Send message with entities, fallback to plain text
"""

import asyncio
import contextlib
import structlog
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from telegram import Bot, CallbackQuery, LinkPreviewOptions, Message, ReactionTypeEmoji
from telegram.error import BadRequest, RetryAfter, TelegramError

from ..entity_formatting import convert_to_entities

logger = structlog.get_logger()


class EditOutcome(Enum):
    """Explicit result for entity-preserving Telegram edits."""

    APPLIED = "applied"
    NOT_MODIFIED = "not_modified"
    MISSING = "missing"
    TRANSIENT_FAILURE = "transient_failure"
    PERMANENT_FAILURE = "permanent_failure"


def is_thread_gone(exc: TelegramError) -> bool:
    """Check if error indicates the Telegram topic/thread no longer exists."""
    if isinstance(exc, BadRequest):
        msg = exc.message.lower()
        return "thread not found" in msg or "topic_id_invalid" in msg
    return False


def is_message_not_modified(exc: TelegramError) -> bool:
    """Check if Telegram rejected an edit because the message is unchanged."""
    if isinstance(exc, BadRequest):
        msg = exc.message.lower()
        return "message is not modified" in msg or "not modified" in msg
    return False


def is_message_missing(exc: TelegramError) -> bool:
    """Check if Telegram says the edited message no longer exists."""
    if isinstance(exc, BadRequest):
        msg = exc.message.lower()
        return (
            "message to edit not found" in msg
            or "message not found" in msg
            or "message_id_invalid" in msg
        )
    return False


def is_permanent_edit_failure(exc: TelegramError) -> bool:
    """Check if retrying the same edit payload is unlikely to succeed."""
    if isinstance(exc, BadRequest):
        msg = exc.message.lower()
        return (
            "message is too long" in msg
            or "message_too_long" in msg
            or "can't parse entities" in msg
            or "entity" in msg
            or "message can't be edited" in msg
            or "message cannot be edited" in msg
        )
    return False


# Disable link previews in all messages to reduce visual noise
NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)


class _MessageGoneError(Exception):
    """Raised when the target message no longer exists (deleted topic)."""


def _retry_after_seconds(exc: RetryAfter) -> int:
    """Extract retry delay from RetryAfter, handling both int and timedelta."""
    ra = exc.retry_after
    return ra if isinstance(ra, int) else int(ra.total_seconds())


# Rate limiting: last send time per chat to avoid Telegram flood control
_last_send_time: dict[int, float] = {}
_rate_limit_locks: dict[int, asyncio.Lock] = {}
MESSAGE_SEND_INTERVAL = 0.5  # seconds between messages to same chat
_LOG_PREVIEW_LIMIT = 220


def _log_preview(text: str) -> str:
    """Return a compact one-line preview for debug logs."""
    preview = " ".join(text.split())
    if len(preview) <= _LOG_PREVIEW_LIMIT:
        return preview
    return f"{preview[:_LOG_PREVIEW_LIMIT]}..."


async def rate_limit_send(chat_id: int) -> None:
    """Wait if necessary to avoid Telegram flood control (max 1 msg/sec per chat).

    Uses a per-chat lock to serialize concurrent senders, preventing two
    coroutines from computing the same wake-up time and sending simultaneously.
    """
    lock = _rate_limit_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        if chat_id in _last_send_time:
            target = _last_send_time[chat_id] + MESSAGE_SEND_INTERVAL
            if target > now:
                await asyncio.sleep(target - now)
                _last_send_time[chat_id] = time.monotonic()
                return
        _last_send_time[chat_id] = time.monotonic()


async def _with_entity_fallback(
    send_fn: Callable[..., Awaitable[Any]],
    text: str,
    context_label: str,
    **kwargs: Any,
) -> Message | None:
    """Convert to entities, send, fall back to plain text on error.

    Entity-based formatting uses character offsets — no syntax to parse,
    no parse errors possible. The only failure mode is Telegram API errors
    (rate limiting, message gone, etc.), which fall back to plain text.

    Args:
        send_fn: Async callable accepting (text, **kwargs).
        text: Raw markdown text (pre-conversion).
        context_label: Label for warning log messages (e.g. "send to 123").
        **kwargs: Extra keyword arguments forwarded to send_fn.

    Returns the result Message on success, None on failure.
    """
    plain_text, entities = convert_to_entities(text)
    logger.debug(
        "telegram %s prepared len=%d entities=%d thread=%s reply_markup=%s preview=%r",
        context_label,
        len(plain_text),
        len(entities),
        kwargs.get("message_thread_id"),
        kwargs.get("reply_markup") is not None,
        _log_preview(plain_text),
    )

    # Phase 1: with entities; Phase 2: plain text fallback.
    # Thread-gone errors (deleted topic) short-circuit both phases.
    last_error: TelegramError | None = None
    for phase_entities in (entities, None):
        send_kwargs = {**kwargs}
        if phase_entities is not None:
            send_kwargs["entities"] = phase_entities
        try:
            sent = await send_fn(plain_text, **send_kwargs)
            msg_id = getattr(sent, "message_id", None)
            logger.debug(
                "telegram %s ok message_id=%s phase=%s",
                context_label,
                msg_id,
                "entities" if phase_entities is not None else "plain",
            )
            return sent
        except RetryAfter as e:
            await asyncio.sleep(_retry_after_seconds(e) + 1)
            try:
                sent = await send_fn(plain_text, **send_kwargs)
                msg_id = getattr(sent, "message_id", None)
                logger.debug(
                    "telegram %s ok after retry message_id=%s phase=%s",
                    context_label,
                    msg_id,
                    "entities" if phase_entities is not None else "plain",
                )
                return sent
            except TelegramError as e2:
                if is_thread_gone(e2):
                    return None
                last_error = e2
        except TelegramError as e:
            if is_thread_gone(e):
                return None
            if context_label.startswith("edit") and is_message_not_modified(e):
                logger.debug("telegram %s not modified", context_label)
                return None
            last_error = e

    if last_error is not None:
        logger.warning("Failed to %s: %s", context_label, last_error)
    return None


async def _send_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Send message with entity formatting, falling back to plain text on failure.

    Returns the sent Message on success, None on failure.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)

    async def _send(text: str, **kw: Any) -> Message:
        return await bot.send_message(chat_id=chat_id, text=text, **kw)

    return await _with_entity_fallback(
        _send, text, f"send message to {chat_id}", **kwargs
    )


async def rate_limit_send_message(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Rate-limited send with entity formatting fallback.

    Combines rate_limit_send() + _send_with_fallback() for convenience.
    Returns the sent Message on success, None on failure.
    """
    await rate_limit_send(chat_id)
    return await _send_with_fallback(bot, chat_id, text, **kwargs)


async def rate_limit_send_message_strict(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Rate-limited send with entities only, no plain-text fallback."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    plain_text, entities = convert_to_entities(text)
    await rate_limit_send(chat_id)
    try:
        sent = await bot.send_message(
            chat_id=chat_id,
            text=plain_text,
            entities=entities,
            **kwargs,
        )
        logger.debug(
            "telegram strict send ok chat=%s message_id=%s entities=%d",
            chat_id,
            getattr(sent, "message_id", None),
            len(entities),
        )
        return sent
    except RetryAfter as exc:
        logger.warning(
            "telegram strict send retry-after chat=%s retry_after=%s",
            chat_id,
            exc.retry_after,
        )
        return None
    except TelegramError as exc:
        if is_thread_gone(exc):
            return None
        logger.debug("telegram strict send failed chat=%s error=%s", chat_id, exc)
        return None


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message | None:
    """Reply with entity formatting, falling back to plain text on failure.

    Returns None if the original message no longer exists (e.g. deleted topic).
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)

    async def _reply(text: str, **kw: Any) -> Message:
        try:
            return await message.reply_text(text, **kw)
        except BadRequest as exc:
            if "not found" in str(exc).lower():
                logger.warning("Cannot reply: original message gone (%s)", exc)
                raise _MessageGoneError from exc
            raise

    try:
        return await _with_entity_fallback(_reply, text, "reply", **kwargs)
    except _MessageGoneError:
        return None


async def safe_edit(target: Message | CallbackQuery, text: str, **kwargs: Any) -> None:
    """Edit message with entity formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    # Message.edit_text vs CallbackQuery.edit_message_text
    raw_edit_fn = (
        target.edit_text if isinstance(target, Message) else target.edit_message_text
    )

    async def _edit(text: str, **kw: Any) -> Any:
        return await raw_edit_fn(text, **kw)

    await _with_entity_fallback(_edit, text, "edit message", **kwargs)


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    message_thread_id: int | None = None,
    **kwargs: Any,
) -> Message | None:
    """Send message with entity formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    if message_thread_id is not None:
        kwargs.setdefault("message_thread_id", message_thread_id)

    async def _send(text: str, **kw: Any) -> Message:
        return await bot.send_message(chat_id=chat_id, text=text, **kw)

    return await _with_entity_fallback(
        _send, text, f"send message to {chat_id}", **kwargs
    )


async def edit_with_fallback(
    bot: Bot,
    chat_id: int,
    message_id: int,
    text: str,
    **kwargs: Any,
) -> bool:
    """Edit a message with entity formatting, falling back to plain text.

    Returns True on success, False on failure.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    plain_text, entities = convert_to_entities(text)
    logger.debug(
        "telegram edit prepared chat=%s message_id=%s len=%d entities=%d "
        "reply_markup=%s preview=%r",
        chat_id,
        message_id,
        len(plain_text),
        len(entities),
        kwargs.get("reply_markup") is not None,
        _log_preview(plain_text),
    )

    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=plain_text,
            entities=entities,
            **kwargs,
        )
        logger.debug("telegram edit ok chat=%s message_id=%s phase=entities", chat_id, message_id)
        return True
    except RetryAfter:
        raise
    except TelegramError as exc:
        if is_message_not_modified(exc):
            logger.debug(
                "telegram edit not modified chat=%s message_id=%s",
                chat_id,
                message_id,
            )
            return True
        try:
            fallback = plain_text
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=fallback,
                **kwargs,
            )
            logger.debug(
                "telegram edit ok chat=%s message_id=%s phase=plain",
                chat_id,
                message_id,
            )
            return True
        except RetryAfter:
            raise
        except TelegramError as fallback_exc:
            if is_message_not_modified(fallback_exc):
                logger.debug(
                    "telegram edit not modified chat=%s message_id=%s phase=plain",
                    chat_id,
                    message_id,
                )
                return True
            logger.debug(
                "telegram edit failed chat=%s message_id=%s",
                chat_id,
                message_id,
            )
            return False


async def edit_with_entities_outcome(
    bot: Bot,
    chat_id: int,
    message_id: int,
    text: str,
    **kwargs: Any,
) -> EditOutcome:
    """Edit with entities only and return a structured outcome."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    plain_text, entities = convert_to_entities(text)
    logger.debug(
        "telegram strict edit prepared chat=%s message_id=%s len=%d entities=%d "
        "reply_markup=%s preview=%r",
        chat_id,
        message_id,
        len(plain_text),
        len(entities),
        kwargs.get("reply_markup") is not None,
        _log_preview(plain_text),
    )
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=plain_text,
            entities=entities,
            **kwargs,
        )
        logger.debug(
            "telegram strict edit ok chat=%s message_id=%s phase=entities",
            chat_id,
            message_id,
        )
        return EditOutcome.APPLIED
    except RetryAfter as exc:
        logger.warning(
            "telegram strict edit retry-after chat=%s message_id=%s retry_after=%s",
            chat_id,
            message_id,
            exc.retry_after,
        )
        return EditOutcome.TRANSIENT_FAILURE
    except TelegramError as exc:
        if is_message_not_modified(exc):
            logger.debug(
                "telegram strict edit not modified chat=%s message_id=%s",
                chat_id,
                message_id,
            )
            return EditOutcome.NOT_MODIFIED
        if is_message_missing(exc):
            logger.debug(
                "telegram strict edit missing chat=%s message_id=%s error=%s",
                chat_id,
                message_id,
                exc,
            )
            return EditOutcome.MISSING
        if is_permanent_edit_failure(exc):
            logger.warning(
                "telegram strict edit permanent failure chat=%s message_id=%s error=%s",
                chat_id,
                message_id,
                exc,
            )
            return EditOutcome.PERMANENT_FAILURE
        logger.debug(
            "telegram strict edit transient failure chat=%s message_id=%s error=%s",
            chat_id,
            message_id,
            exc,
        )
        return EditOutcome.TRANSIENT_FAILURE


async def ack_reaction(bot: Bot, chat_id: int, message_id: int) -> None:
    """React to a message with the configured ack emoji, if enabled."""
    from ..config import config

    if not config.ack_reaction:
        return
    with contextlib.suppress(TelegramError):
        await bot.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[ReactionTypeEmoji(emoji=config.ack_reaction)],
        )


def send_kwargs(thread_id: int | None) -> dict[str, Any]:
    """Build message_thread_id kwargs for bot.send_message()."""
    if thread_id is not None:
        return {"message_thread_id": thread_id}
    return {}
