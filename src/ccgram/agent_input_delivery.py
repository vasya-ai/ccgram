"""Verified user-message delivery into agent TUI composers.

This module sits above ``tmux_manager``.  It is intentionally narrow: normal
Telegram user prompts can ask for transcript-backed acceptance, while legacy
toolbar/shell/interactive flows keep using the low-level send helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
import functools
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

import structlog

from . import window_query
from .providers import get_provider_for_window
from .providers.base import AgentProvider
from .tmux_manager import tmux_manager

logger = structlog.get_logger()

_T = TypeVar("_T")

_BASE_SUBMIT_DELAY_SECONDS = 0.5
_MAX_SUBMIT_DELAY_SECONDS = 2.5
_ACK_TIMEOUT_SECONDS = 6.0
_RETRY_ACK_TIMEOUT_SECONDS = 2.0
_ACK_POLL_INTERVAL_SECONDS = 0.25
_MAX_SUBMIT_RETRIES = 2
_FINGERPRINT_CHUNK_SIZE = 40


class UserSubmitStatus(str, Enum):
    """Outcome of a user prompt submit attempt."""

    ACCEPTED = "accepted"
    INJECTED_UNVERIFIED = "injected_unverified"
    WINDOW_MISSING = "window_missing"
    INJECTION_FAILED = "injection_failed"
    ACK_TIMEOUT = "ack_timeout"
    UNSUPPORTED_PROVIDER = "unsupported_provider"


@dataclass(slots=True)
class UserSubmitResult:
    """Result returned by ``submit_user_message``."""

    status: UserSubmitStatus
    message: str
    attempts: int = 0
    transcript_offset: int | None = None
    verified: bool = False

    @property
    def ok(self) -> bool:
        return self.status in (
            UserSubmitStatus.ACCEPTED,
            UserSubmitStatus.INJECTED_UNVERIFIED,
        )


@dataclass(slots=True)
class _AckResult:
    accepted: bool
    offset: int | None


@dataclass(slots=True)
class _RetryOutcome:
    result: UserSubmitResult | None
    attempts: int
    offset: int | None


async def _run_blocking(func: Callable[..., _T], /, *args: Any) -> _T:
    """Run blocking transcript IO without propagating contextvars."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args))


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _line_count(text: str) -> int:
    return text.count("\n") + 1


def _adaptive_submit_delay(text: str) -> float:
    """Return delay between paste and Enter, scaled for long/multiline text."""
    char_extra = min(len(text) / 1000 * 0.5, 1.5)
    line_extra = min(max(_line_count(text) - 1, 0) * 0.03, 0.5)
    return min(
        _MAX_SUBMIT_DELAY_SECONDS,
        _BASE_SUBMIT_DELAY_SECONDS + char_extra + line_extra,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fingerprint_chunks(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= _FINGERPRINT_CHUNK_SIZE * 2:
        return [normalized]

    size = _FINGERPRINT_CHUNK_SIZE
    positions = (
        0,
        max((len(normalized) // 2) - (size // 2), 0),
        max(len(normalized) - size, 0),
    )
    chunks: list[str] = []
    for pos in positions:
        chunk = normalized[pos : pos + size].strip()
        if chunk and chunk not in chunks:
            chunks.append(chunk)
    return chunks


def _draft_likely_present(pane_text: str, text: str) -> bool:
    """Heuristic guard: enough prompt fingerprints are visible near pane bottom."""
    tail = "\n".join(pane_text.splitlines()[-35:])
    normalized_tail = _normalize_text(tail)
    chunks = _fingerprint_chunks(text)
    if not chunks:
        return False
    hits = sum(1 for chunk in chunks if chunk in normalized_tail)
    required = 1 if len(chunks) == 1 else min(2, len(chunks))
    return hits >= required


def _pane_has_agent_status(provider: AgentProvider, pane_text: str) -> bool:
    try:
        status = provider.parse_terminal_status(pane_text)
    except Exception:
        logger.exception("submit_user_message: failed to parse terminal status")
        return False
    return status is not None


def _capture_transcript_offset(provider: AgentProvider, transcript_path: Path) -> int:
    if provider.capabilities.supports_incremental_read:
        return transcript_path.stat().st_size
    _, offset = provider.read_transcript_file(str(transcript_path), 0)
    return offset


def _read_transcript_entries_since(
    provider: AgentProvider,
    transcript_path: Path,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    if not provider.capabilities.supports_incremental_read:
        return provider.read_transcript_file(str(transcript_path), offset)

    entries: list[dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as fh:
        if offset > transcript_path.stat().st_size:
            offset = 0
        fh.seek(offset)
        for line in fh:
            parsed = provider.parse_transcript_line(line)
            if parsed:
                entries.append(parsed)
        new_offset = fh.tell()
    return entries, new_offset


def _has_matching_user_entry(
    provider: AgentProvider,
    entries: list[dict[str, Any]],
    text: str,
) -> bool:
    target = _normalize_text(text)
    for entry in entries:
        if not provider.is_user_transcript_entry(entry):
            continue
        message = provider.parse_history_entry(entry)
        if not message or message.role != "user":
            continue
        if _normalize_text(message.text) == target:
            return True
    return False


async def _wait_for_user_ack(
    provider: AgentProvider,
    transcript_path: Path,
    offset: int,
    text: str,
    *,
    timeout: float,
    poll_interval: float,
) -> _AckResult:
    deadline = time.monotonic() + max(timeout, 0.0)
    current_offset = offset

    while True:
        try:
            entries, current_offset = await _run_blocking(
                _read_transcript_entries_since,
                provider,
                transcript_path,
                current_offset,
            )
        except (OSError, NotImplementedError):
            logger.exception(
                "submit_user_message: transcript ack read failed",
                transcript_path=str(transcript_path),
            )
            return _AckResult(False, current_offset)

        if _has_matching_user_entry(provider, entries, text):
            return _AckResult(True, current_offset)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _AckResult(False, current_offset)
        await asyncio.sleep(min(poll_interval, remaining))


async def _capture_offset_if_available(
    provider: AgentProvider,
    transcript_path: Path | None,
    *,
    can_verify: bool,
    window_id: str,
) -> tuple[bool, int | None]:
    if not can_verify or transcript_path is None:
        return False, None
    try:
        offset = await _run_blocking(
            _capture_transcript_offset,
            provider,
            transcript_path,
        )
    except (OSError, NotImplementedError):
        logger.exception(
            "submit_user_message: cannot capture transcript offset",
            window_id=window_id,
            provider=provider.capabilities.name,
            transcript_path=str(transcript_path),
        )
        return False, None
    return True, offset


async def _retry_visible_draft(
    provider: AgentProvider,
    transcript_path: Path,
    text: str,
    *,
    window_id: str,
    target_window_id: str,
    digest: str,
    attempts: int,
    current_offset: int | None,
    transcript_offset: int,
    retry_ack_timeout: float,
    poll_interval: float,
    max_submit_retries: int,
) -> _RetryOutcome:
    while attempts <= max_submit_retries:
        pane_text = await tmux_manager.capture_pane(target_window_id)
        if not pane_text:
            logger.warning(
                "submit_user_message: no pane text before retry",
                window_id=window_id,
                digest=digest,
                attempts=attempts,
            )
            break
        if _pane_has_agent_status(provider, pane_text):
            logger.warning(
                "submit_user_message: retry blocked by agent status",
                window_id=window_id,
                digest=digest,
                attempts=attempts,
            )
            break
        if not _draft_likely_present(pane_text, text):
            logger.warning(
                "submit_user_message: retry blocked; draft fingerprint absent",
                window_id=window_id,
                digest=digest,
                attempts=attempts,
            )
            break

        attempts += 1
        logger.warning(
            "submit_user_message: retrying Enter for visible draft",
            window_id=window_id,
            digest=digest,
            attempt=attempts,
        )
        submitted = await tmux_manager._submit_enter_locked(target_window_id)
        if not submitted:
            return _RetryOutcome(
                UserSubmitResult(
                    UserSubmitStatus.INJECTION_FAILED,
                    "Failed to retry Enter in terminal",
                    attempts=attempts,
                    transcript_offset=current_offset,
                ),
                attempts,
                current_offset,
            )
        ack = await _wait_for_user_ack(
            provider,
            transcript_path,
            current_offset if current_offset is not None else transcript_offset,
            text,
            timeout=retry_ack_timeout,
            poll_interval=poll_interval,
        )
        current_offset = ack.offset
        if ack.accepted:
            logger.debug(
                "submit_user_message: transcript ack accepted after retry",
                window_id=window_id,
                digest=digest,
                attempts=attempts,
                transcript_offset=ack.offset,
            )
            return _RetryOutcome(
                UserSubmitResult(
                    UserSubmitStatus.ACCEPTED,
                    "Message accepted by agent transcript after retry",
                    attempts=attempts,
                    transcript_offset=ack.offset,
                    verified=True,
                ),
                attempts,
                ack.offset,
            )

    return _RetryOutcome(None, attempts, current_offset)


async def submit_user_message(
    window_id: str,
    text: str,
    *,
    ack_timeout: float = _ACK_TIMEOUT_SECONDS,
    retry_ack_timeout: float = _RETRY_ACK_TIMEOUT_SECONDS,
    poll_interval: float = _ACK_POLL_INTERVAL_SECONDS,
    max_submit_retries: int = _MAX_SUBMIT_RETRIES,
    initial_delay: float | None = None,
) -> UserSubmitResult:
    """Insert and submit a user prompt, verifying transcript acceptance when safe."""
    window = await tmux_manager.find_window_by_id(window_id)
    if not window:
        return UserSubmitResult(
            UserSubmitStatus.WINDOW_MISSING,
            "Window not found (may have been closed)",
        )

    view = window_query.view_window(window_id)
    provider = get_provider_for_window(
        window_id,
        provider_name=view.provider_name if view else None,
    )
    transcript_path = view.transcript_path if view else None
    can_verify = bool(
        transcript_path and provider.capabilities.supports_structured_transcript
    )
    transcript_offset: int | None = None

    delay = initial_delay if initial_delay is not None else _adaptive_submit_delay(text)
    digest = _text_digest(text)
    attempts = 0
    target_window_id = window.window_id
    async with tmux_manager.input_lock(target_window_id):
        can_verify, transcript_offset = await _capture_offset_if_available(
            provider,
            transcript_path,
            can_verify=can_verify,
            window_id=window_id,
        )

        logger.debug(
            "submit_user_message: start",
            window_id=window_id,
            provider=provider.capabilities.name,
            text_len=len(text),
            line_count=_line_count(text),
            digest=digest,
            can_verify=can_verify,
            transcript_path=str(transcript_path) if transcript_path else "",
            transcript_offset=transcript_offset,
            submit_delay=delay,
        )

        inserted = await tmux_manager._insert_literal_text_locked(target_window_id, text)
        logger.debug(
            "submit_user_message: insert result",
            window_id=window_id,
            digest=digest,
            inserted=inserted,
        )
        if not inserted:
            return UserSubmitResult(
                UserSubmitStatus.INJECTION_FAILED,
                "Failed to insert message into terminal",
                attempts=attempts,
                transcript_offset=transcript_offset,
            )

        await asyncio.sleep(delay)

        attempts += 1
        submitted = await tmux_manager._submit_enter_locked(target_window_id)
        logger.debug(
            "submit_user_message: submit attempt",
            window_id=window_id,
            digest=digest,
            attempt=attempts,
            submitted=submitted,
        )
        if not submitted:
            return UserSubmitResult(
                UserSubmitStatus.INJECTION_FAILED,
                "Failed to press Enter in terminal",
                attempts=attempts,
                transcript_offset=transcript_offset,
            )

        if not can_verify or transcript_path is None or transcript_offset is None:
            return UserSubmitResult(
                UserSubmitStatus.INJECTED_UNVERIFIED,
                "Message injected; transcript verification unavailable",
                attempts=attempts,
                transcript_offset=transcript_offset,
            )

        ack = await _wait_for_user_ack(
            provider,
            transcript_path,
            transcript_offset,
            text,
            timeout=ack_timeout,
            poll_interval=poll_interval,
        )
        if ack.accepted:
            logger.debug(
                "submit_user_message: transcript ack accepted",
                window_id=window_id,
                digest=digest,
                attempts=attempts,
                transcript_offset=ack.offset,
            )
            return UserSubmitResult(
                UserSubmitStatus.ACCEPTED,
                "Message accepted by agent transcript",
                attempts=attempts,
                transcript_offset=ack.offset,
                verified=True,
            )

        retry = await _retry_visible_draft(
            provider,
            transcript_path,
            text,
            window_id=window_id,
            target_window_id=target_window_id,
            digest=digest,
            attempts=attempts,
            current_offset=ack.offset,
            transcript_offset=transcript_offset,
            retry_ack_timeout=retry_ack_timeout,
            poll_interval=poll_interval,
            max_submit_retries=max_submit_retries,
        )
        if retry.result:
            return retry.result
        attempts = retry.attempts
        transcript_offset = (
            retry.offset if retry.offset is not None else transcript_offset
        )

    logger.warning(
        "submit_user_message: transcript ack timeout",
        window_id=window_id,
        provider=provider.capabilities.name,
        digest=digest,
        attempts=attempts,
        transcript_offset=transcript_offset,
    )
    return UserSubmitResult(
        UserSubmitStatus.ACK_TIMEOUT,
        "Message was inserted, but the agent did not accept it as a user turn",
        attempts=attempts,
        transcript_offset=transcript_offset,
    )
