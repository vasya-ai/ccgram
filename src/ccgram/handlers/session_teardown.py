"""Shared session/topic teardown helpers.

Owns destructive lifecycle cleanup for "this topic/session is done" flows:
user-driven /kill, /sessions kill, Telegram topic close events, and deleted
topic recovery paths.  Non-destructive /unbind intentionally stays separate.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import structlog
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

from ..session_map import session_map_sync
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..window_query import view_window
from ..window_resolver import is_foreign_window
from ..window_state_store import window_store
from .message_sender import is_thread_gone
from .polling_strategies import terminal_poll_state

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class BindingSnapshot:
    user_id: int
    thread_id: int
    chat_id: int


@dataclass(slots=True)
class TeardownResult:
    teardown_id: str
    reason: str
    window_id: str | None
    display: str = ""
    window_status: str = "not_started"
    topic_status: str = "not_requested"
    bindings_removed: int = 0
    session_map_cleared: bool = False
    window_state_removed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.window_status != "failed" and not self.errors


_teardown_locks: dict[str, asyncio.Lock] = {}


def _lock_for(key: str) -> asyncio.Lock:
    return _teardown_locks.setdefault(key, asyncio.Lock())


def _retry_after_seconds(exc: RetryAfter) -> float:
    retry_after = exc.retry_after
    if isinstance(retry_after, timedelta):
        return max(0.0, retry_after.total_seconds())
    return max(0.0, float(retry_after))


def _is_external(window_id: str) -> bool:
    view = view_window(window_id)
    return is_foreign_window(window_id) or bool(view and view.external)


async def _call_topic_method_with_retry(method: Any, **kwargs: Any) -> None:
    try:
        await method(**kwargs)
    except RetryAfter as exc:
        await asyncio.sleep(_retry_after_seconds(exc) + 1.0)
        await method(**kwargs)


async def remove_forum_topic(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    *,
    teardown_id: str,
) -> str:
    """Delete a Telegram forum topic, falling back to close.

    Returns a compact status string for logging/UI:
    deleted, already_gone, closed_fallback, failed, or no_group_chat.
    """
    if chat_id > 0:
        logger.warning(
            "teardown_topic_remove_skipped",
            teardown_id=teardown_id,
            chat_id=chat_id,
            thread_id=thread_id,
            status="no_group_chat",
        )
        return "no_group_chat"

    try:
        await _call_topic_method_with_retry(
            bot.delete_forum_topic,
            chat_id=chat_id,
            message_thread_id=thread_id,
        )
        logger.info(
            "teardown_topic_remove",
            teardown_id=teardown_id,
            chat_id=chat_id,
            thread_id=thread_id,
            status="deleted",
        )
        return "deleted"
    except TelegramError as exc:
        if is_thread_gone(exc):
            logger.info(
                "teardown_topic_remove",
                teardown_id=teardown_id,
                chat_id=chat_id,
                thread_id=thread_id,
                status="already_gone",
            )
            return "already_gone"
        logger.warning(
            "teardown_delete_forum_topic_failed",
            teardown_id=teardown_id,
            chat_id=chat_id,
            thread_id=thread_id,
            error=str(exc),
        )

    try:
        await _call_topic_method_with_retry(
            bot.close_forum_topic,
            chat_id=chat_id,
            message_thread_id=thread_id,
        )
        logger.info(
            "teardown_topic_remove",
            teardown_id=teardown_id,
            chat_id=chat_id,
            thread_id=thread_id,
            status="closed_fallback",
        )
        return "closed_fallback"
    except TelegramError as exc:
        if is_thread_gone(exc):
            logger.info(
                "teardown_topic_remove",
                teardown_id=teardown_id,
                chat_id=chat_id,
                thread_id=thread_id,
                status="already_gone",
            )
            return "already_gone"
        logger.warning(
            "teardown_close_forum_topic_failed",
            teardown_id=teardown_id,
            chat_id=chat_id,
            thread_id=thread_id,
            error=str(exc),
        )
        return "failed"


def _resolve_window_id(
    user_id: int | None, thread_id: int | None, window_id: str | None
) -> str | None:
    if window_id is not None:
        return window_id
    if user_id is None or thread_id is None:
        return None
    return thread_router.get_window_for_thread(user_id, thread_id)


def _new_result(reason: str, window_id: str | None) -> TeardownResult:
    return TeardownResult(
        teardown_id=uuid.uuid4().hex[:12],
        reason=reason,
        window_id=window_id,
        display=thread_router.get_display_name(window_id) if window_id else "",
    )


def _snapshot_bindings(
    window_id: str, user_id: int | None, thread_id: int | None
) -> list[BindingSnapshot]:
    bindings = [
        BindingSnapshot(uid, tid, thread_router.resolve_chat_id(uid, tid))
        for uid, tid, wid in list(thread_router.iter_thread_bindings())
        if wid == window_id
    ]
    if bindings or user_id is None or thread_id is None:
        return bindings

    # Preserve enough context to remove an already-unbound topic if the caller
    # still has it. Cleanup/unbind loops remain no-op.
    return [
        BindingSnapshot(user_id, thread_id, thread_router.resolve_chat_id(user_id, thread_id))
    ]


async def _kill_local_window(window_id: str, teardown_id: str) -> str:
    try:
        window = await tmux_manager.find_window_by_id(window_id)
        if window is None:
            status = "already_gone"
        elif await tmux_manager.kill_window(window.window_id):
            status = "killed"
        else:
            still_live = await tmux_manager.find_window_by_id(window_id)
            status = "failed" if still_live else "already_gone"
        logger.info(
            "teardown_window_kill",
            teardown_id=teardown_id,
            window_id=window_id,
            status=status,
        )
        return status
    except Exception:
        logger.exception(
            "teardown_window_kill_failed",
            teardown_id=teardown_id,
            window_id=window_id,
        )
        return "failed"


async def _cleanup_binding_states(
    result: TeardownResult,
    bot: Bot,
    bindings: list[BindingSnapshot],
    *,
    user_id: int | None,
    user_data: dict[str, Any] | None,
    window_dead: bool,
) -> None:
    from .cleanup import clear_topic_state

    for binding in bindings:
        try:
            await clear_topic_state(
                binding.user_id,
                binding.thread_id,
                bot,
                user_data if binding.user_id == user_id else None,
                window_id=result.window_id,
                window_dead=window_dead,
            )
        except Exception as exc:
            result.errors.append(
                f"cleanup:{binding.user_id}:{binding.thread_id}:{exc}"
            )
            logger.exception(
                "teardown_state_cleanup_failed",
                teardown_id=result.teardown_id,
                user_id=binding.user_id,
                thread_id=binding.thread_id,
                window_id=result.window_id,
            )


def _clear_local_window_state(result: TeardownResult) -> None:
    if result.window_id is None:
        return
    session_map_sync.clear_session_map_entry(result.window_id)
    result.session_map_cleared = True
    result.window_state_removed = window_store.remove_window(result.window_id)


def _unbind_bindings(result: TeardownResult, bindings: list[BindingSnapshot]) -> None:
    if result.window_id is None:
        return
    for binding in bindings:
        removed = thread_router.unbind_thread(binding.user_id, binding.thread_id)
        if removed is not None:
            result.bindings_removed += 1
        terminal_poll_state.reset_probe_failures(result.window_id)


def _summarize_topic_status(statuses: list[str]) -> str:
    if not statuses:
        return "no_binding"
    for status in ("failed", "closed_fallback", "deleted", "already_gone"):
        if status in statuses:
            return status
    return statuses[0]


async def _remove_bound_topics(
    result: TeardownResult, bot: Bot, bindings: list[BindingSnapshot]
) -> None:
    statuses = [
        await remove_forum_topic(
            bot,
            binding.chat_id,
            binding.thread_id,
            teardown_id=result.teardown_id,
        )
        for binding in bindings
    ]
    result.topic_status = _summarize_topic_status(statuses)


async def teardown_topic_session(
    bot: Bot,
    *,
    actor_user_id: int,
    reason: str,
    user_id: int | None = None,
    thread_id: int | None = None,
    window_id: str | None = None,
    user_data: dict[str, Any] | None = None,
    remove_topic: bool = False,
) -> TeardownResult:
    """Kill/detach a topic-bound session and clean all CCGram state.

    At least ``window_id`` or ``(user_id, thread_id)`` must be supplied.
    ``remove_topic`` controls Telegram topic deletion/closure.  The helper is
    intentionally idempotent: already-gone windows/topics/state are normal.
    """
    window_id = _resolve_window_id(user_id, thread_id, window_id)
    result = _new_result(reason, window_id)
    logger.info(
        "teardown_start",
        teardown_id=result.teardown_id,
        reason=reason,
        actor_user_id=actor_user_id,
        user_id=user_id,
        thread_id=thread_id,
        window_id=window_id,
        remove_topic=remove_topic,
    )

    if window_id is None:
        result.window_status = "no_binding"
        logger.info("teardown_done", **_result_log(result))
        return result

    async with _lock_for(window_id):
        external = _is_external(window_id)
        bindings = _snapshot_bindings(window_id, user_id, thread_id)
        if external:
            result.window_status = "external_skipped"
            logger.info(
                "teardown_window_kill",
                teardown_id=result.teardown_id,
                window_id=window_id,
                status=result.window_status,
            )
        else:
            result.window_status = await _kill_local_window(
                window_id, result.teardown_id
            )

        await _cleanup_binding_states(
            result,
            bot,
            bindings,
            user_id=user_id,
            user_data=user_data,
            window_dead=not external,
        )
        if not external:
            _clear_local_window_state(result)
        _unbind_bindings(result, bindings)
        if remove_topic:
            await _remove_bound_topics(result, bot, bindings)

        logger.info("teardown_state_cleanup", **_result_log(result))
        logger.info("teardown_done", **_result_log(result))
        return result


def _result_log(result: TeardownResult) -> dict[str, Any]:
    return {
        "teardown_id": result.teardown_id,
        "reason": result.reason,
        "window_id": result.window_id,
        "display": result.display,
        "window_status": result.window_status,
        "topic_status": result.topic_status,
        "bindings_removed": result.bindings_removed,
        "session_map_cleared": result.session_map_cleared,
        "window_state_removed": result.window_state_removed,
        "errors": result.errors,
    }
