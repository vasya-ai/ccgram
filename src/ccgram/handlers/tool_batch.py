"""Agent turn bubbles — ordered Telegram rendering for assistant output.

Accumulates assistant text plus tool_use / tool_result events into one ordered
Telegram bubble per agent turn. Tool calls render as expandable quote sections
between the assistant messages that surround them. When the rendered turn grows
past Telegram's text limit, pages are appended in chronological order instead
of rotating older tool rows out.

Key components:
  - ToolBatchEntry / AgentBubbleSegment / ToolBatch: turn state dataclasses
  - process_tool_event: state-machine entry point (add tool_use or tool_result)
  - process_agent_message: append assistant text to the active turn bubble
  - flush_batch: finalize and send the last edit for a batch
  - format_batch_message: render tool-only entries as ordered paginated content
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import structlog
from telegram import Bot
from telegram.error import TelegramError
from telegramify_markdown import utf16_len as _utf16_len

from ..expandable_quote import EXPANDABLE_QUOTE_END, EXPANDABLE_QUOTE_START
from ..telegram_sender import split_message
from ..utils import atomic_write_json, ccgram_dir, task_done_callback
from ..thread_router import thread_router
from ..topic_state_registry import topic_state
from .message_sender import (
    EditOutcome,
    edit_with_entities_outcome,
    rate_limit_send_message_strict as rate_limit_send_message,
    send_kwargs,
)
from .message_task import ContentTask, thread_key

logger = structlog.get_logger()

TELEGRAM_TEXT_LIMIT = 4096
TOOL_BUBBLE_TITLE = "Tools"
TOOL_SUMMARY_LIMIT = 88
_TOOL_LINE_ELLIPSIS = "…"
_TOOL_BUBBLE_RENDERED_LIMIT = 3800
_PERSISTED_BATCH_STATE_VERSION = 1
_PERSISTED_BATCH_MAX_AGE_SECONDS = 6 * 60 * 60
_AGENT_BUBBLE_RETRY_BASE_SECONDS = 1.0
_AGENT_BUBBLE_RETRY_MAX_SECONDS = 15.0

ToolStatus = Literal["pending", "success", "error"]
AgentBubbleSegmentKind = Literal["text", "tools"]


class _PageSyncResult(Enum):
    SUCCESS = "success"
    TRANSIENT_FAILURE = "transient_failure"
    PERMANENT_FAILURE = "permanent_failure"


@dataclass
class ToolBatchEntry:
    """A single tool call entry within a batch."""

    tool_use_id: str | None
    tool_use_text: str = ""  # Legacy formatted summary from providers.
    tool_result_text: str | None = None  # Legacy result storage.
    tool_name: str | None = None
    summary: str = ""
    status: ToolStatus = "pending"
    result_text: str | None = None

    def __post_init__(self) -> None:
        parsed_name, parsed_summary = _parse_tool_use_text(
            self.tool_use_text,
            self.tool_name,
        )
        if self.tool_name is None:
            self.tool_name = parsed_name
        if not self.summary:
            self.summary = parsed_summary
        self.summary = _normalize_summary(self.summary)
        if self.result_text is None and self.tool_result_text is not None:
            self.result_text = self.tool_result_text
        if self.result_text is not None and self.status == "pending":
            self.status = _status_from_result_text(self.result_text)


@dataclass
class AgentBubbleSegment:
    """One ordered section in an agent turn bubble."""

    kind: AgentBubbleSegmentKind
    text: str = ""
    entries: list[ToolBatchEntry] = field(default_factory=list)


@dataclass
class ToolBatch:
    """Accumulator for one agent turn rendered into Telegram message pages."""

    window_id: str
    thread_id: int  # thread_id_or_0
    entries: list[ToolBatchEntry] = field(default_factory=list)
    telegram_msg_id: int | None = None
    telegram_msg_ids: list[int] = field(default_factory=list)
    segments: list[AgentBubbleSegment] = field(default_factory=list)
    rendered_pages: list[str] = field(default_factory=list)
    total_length: int = 0  # Legacy metric; no longer used as an overflow trigger.


@dataclass
class _DeliveryState:
    dirty_version: int = 0
    flushed_version: int = 0
    permanent_failure_version: int = 0
    scheduled_task: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    retry_count: int = 0


# Active tool batches: (user_id, thread_id_or_0) -> ToolBatch
_active_batches: dict[tuple[int, int], ToolBatch] = {}
_delivery_states: dict[tuple[int, int], _DeliveryState] = {}
_persistent_batches_loaded = False

_MARKDOWN_TOOL_PREFIX_RE = re.compile(r"\*\*([^*]+)\*\*\s*(.*)$")
_PLAIN_TOOL_PREFIX_RE = re.compile(r"^\W*([A-Za-z_][\w.:-]*)\b\s*(.*)$")
_PLAIN_TASK_CREATE_RE = re.compile(r"^TaskCreate\s+(.+)$")
_MIN_BACKTICK_WRAPPED_LENGTH = 2

_BATCH_ERROR_RE = re.compile(
    r"\b(error|FAILED|fail(ed|ure[s]?)?|Exception|Traceback|exit code [1-9]\d*)\b",
    re.IGNORECASE,
)
_BATCH_SUCCESS_RE = re.compile(r"\b(passed|success|exit code 0)\b", re.IGNORECASE)
_BATCH_INTERRUPTED_RE = re.compile(
    r"(\[Request interrupted by user for tool use\]|⏹\s*Interrupted|\binterrupted\b)",
    re.IGNORECASE,
)

_TOOL_NAME_ALIASES = {
    "apply_patch": "Edit",
    "edit_file": "Edit",
    "exec_command": "Bash",
    "read_file": "Read",
    "shell": "Bash",
    "update_plan": "Plan",
    "view_image": "Image",
    "write_stdin": "Input",
}

_TOOL_ICONS = {
    "askuserquestion": "❓",
    "bash": "⚡",
    "edit": "✏️",
    "glob": "🔎",
    "grep": "🔎",
    "image": "🖼️",
    "input": "⌨️",
    "notebookedit": "✏️",
    "plan": "📋",
    "read": "📖",
    "skill": "🧩",
    "task": "🤖",
    "taskcreate": "🤖",
    "tasklist": "📋",
    "taskupdate": "🤖",
    "todoread": "☑️",
    "todowrite": "☑️",
    "webfetch": "🌐",
    "websearch": "🌐",
    "write": "📝",
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_batch_message(
    entries: list[ToolBatchEntry],
    subagent_label: str | None = None,
    provider_label: str | None = None,
) -> str:
    """Render tool entries in chronological order.

    Kept for compatibility with older tests/callers. New delivery uses
    ``format_agent_pages`` so long turns can span multiple Telegram messages.
    """
    del subagent_label  # Legacy signature; task grouping text is intentionally gone.
    pages = format_agent_pages(
        [AgentBubbleSegment("tools", entries=list(entries))],
        provider_label=provider_label,
    )
    return (
        pages[0]
        if pages
        else _render_tool_bubble([], _tool_bubble_title(provider_label))
    )


def format_agent_pages(
    segments: list[AgentBubbleSegment],
    provider_label: str | None = None,
) -> list[str]:
    """Render ordered assistant/tool segments into Telegram-sized pages."""
    builder = _AgentPageBuilder()
    title = _tool_bubble_title(provider_label)
    for segment in segments:
        if segment.kind == "text":
            builder.add_text(segment.text)
        elif segment.kind == "tools":
            builder.add_tool_entries(segment.entries, title)
    return _with_page_footers(builder.finish())


def _batch_result_prefix(result_text: str) -> str:
    """Choose a result indicator prefix based on content."""
    if _BATCH_INTERRUPTED_RE.search(result_text) or _BATCH_ERROR_RE.search(result_text):
        return "\u274c"
    if _BATCH_SUCCESS_RE.search(result_text):
        return "\u2705"
    return "\u23bf"


def _format_batch_entry(entry: ToolBatchEntry) -> str:
    """Render one standard batch row."""
    tool_name = _display_tool_name(entry.tool_name)
    icon = _tool_icon(entry.tool_name)
    glyph = _status_glyph(entry.status)
    return f'{icon} {tool_name}: "{entry.summary}" {glyph}'


def _render_tool_bubble(
    visible_lines: list[str],
    title: str,
    hidden_count: int = 0,
) -> str:
    body = _render_tool_bubble_body(visible_lines, title, hidden_count)
    return f"{EXPANDABLE_QUOTE_START}{body}{EXPANDABLE_QUOTE_END}"


def _render_tool_bubble_body(
    visible_lines: list[str],
    title: str,
    hidden_count: int,
) -> str:
    body_lines: list[str] = [title]
    if hidden_count > 0:
        body_lines.append(
            f"{_TOOL_LINE_ELLIPSIS} {hidden_count} earlier tools {_TOOL_LINE_ELLIPSIS}"
        )
    body_lines.extend(visible_lines)
    return "\n".join(body_lines)


def _telegram_rendered_len(text: str) -> int:
    budget_text = text.replace(EXPANDABLE_QUOTE_START, "").replace(
        EXPANDABLE_QUOTE_END,
        "",
    )
    return _utf16_len(budget_text)


def _fits_rendered_limit(text: str, limit: int = _TOOL_BUBBLE_RENDERED_LIMIT) -> bool:
    return _telegram_rendered_len(text) <= limit


def _truncate_rendered_text(text: str, limit: int) -> str:
    if _fits_rendered_limit(text, limit):
        return text
    if limit <= _telegram_rendered_len(_TOOL_LINE_ELLIPSIS):
        return _TOOL_LINE_ELLIPSIS

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = f"{text[:mid]}{_TOOL_LINE_ELLIPSIS}"
        if _fits_rendered_limit(candidate, limit):
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or _TOOL_LINE_ELLIPSIS


class _AgentPageBuilder:
    """Incrementally build ordered pages without dropping older content."""

    def __init__(self) -> None:
        self._pages: list[str] = []
        self._parts: list[str] = []

    def finish(self) -> list[str]:
        self._flush_page()
        return self._pages

    def add_text(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        for chunk in split_message(clean, max_length=_TOOL_BUBBLE_RENDERED_LIMIT):
            self._append_block(chunk)

    def add_tool_entries(self, entries: list[ToolBatchEntry], title: str) -> None:
        quote_lines: list[str] = []
        for entry in entries:
            line = _format_batch_entry(entry)
            if self._quote_fits_current([*quote_lines, line], title):
                quote_lines.append(line)
                continue

            if quote_lines:
                self._append_known_fitting_block(
                    _render_tool_bubble(quote_lines, title)
                )
                quote_lines = []

            if not self._quote_fits_current([line], title):
                self._flush_page()

            if not self._quote_fits_current([line], title):
                line = _truncate_line_to_fit(line, title, hidden_count=0)
            quote_lines.append(line)

        if quote_lines:
            self._append_known_fitting_block(_render_tool_bubble(quote_lines, title))

    def _current_text(self) -> str:
        return "\n\n".join(self._parts)

    def _block_fits_current(self, block: str) -> bool:
        separator = "\n\n" if self._parts else ""
        candidate = (
            block if not self._parts else f"{self._current_text()}{separator}{block}"
        )
        return _fits_rendered_limit(candidate)

    def _quote_fits_current(self, lines: list[str], title: str) -> bool:
        return self._block_fits_current(_render_tool_bubble(lines, title))

    def _append_block(self, block: str) -> None:
        if self._block_fits_current(block):
            self._parts.append(block)
            return
        self._flush_page()
        if self._block_fits_current(block):
            self._parts.append(block)
            return
        for chunk in split_message(block, max_length=_TOOL_BUBBLE_RENDERED_LIMIT):
            if not self._block_fits_current(chunk):
                self._flush_page()
            self._parts.append(chunk)

    def _append_known_fitting_block(self, block: str) -> None:
        if not self._block_fits_current(block):
            self._flush_page()
        self._parts.append(block)

    def _flush_page(self) -> None:
        if not self._parts:
            return
        self._pages.append(self._current_text())
        self._parts = []


def _with_page_footers(pages: list[str]) -> list[str]:
    total = len(pages)
    if total <= 1:
        return pages
    rendered: list[str] = []
    for index, page in enumerate(pages, 1):
        footer = f"\n\n[{index}/{total}]"
        if _fits_rendered_limit(f"{page}{footer}", TELEGRAM_TEXT_LIMIT):
            rendered.append(f"{page}{footer}")
        else:
            available = TELEGRAM_TEXT_LIMIT - _telegram_rendered_len(footer)
            rendered.append(f"{_truncate_rendered_text(page, available)}{footer}")
    return rendered


def _tool_bubble_fits(
    visible_lines: list[str],
    title: str,
    hidden_count: int,
) -> bool:
    body = _render_tool_bubble_body(visible_lines, title, hidden_count)
    rendered = f"{EXPANDABLE_QUOTE_START}{body}{EXPANDABLE_QUOTE_END}"
    return (
        _telegram_rendered_len(body) <= _TOOL_BUBBLE_RENDERED_LIMIT
        and _telegram_rendered_len(rendered) <= TELEGRAM_TEXT_LIMIT
    )


def _truncate_line_to_fit(line: str, title: str, hidden_count: int) -> str:
    body_overhead = (
        _telegram_rendered_len(_render_tool_bubble_body(["x"], title, hidden_count)) - 1
    )
    rendered_overhead = (
        _telegram_rendered_len(_render_tool_bubble(["x"], title, hidden_count)) - 1
    )
    available = min(
        _TOOL_BUBBLE_RENDERED_LIMIT - body_overhead,
        TELEGRAM_TEXT_LIMIT - rendered_overhead,
    )
    if available <= 0:
        return ""
    if _telegram_rendered_len(line) <= available:
        return line
    if available <= _telegram_rendered_len(_TOOL_LINE_ELLIPSIS):
        return _TOOL_LINE_ELLIPSIS[:available]
    return _truncate_rendered_text(line, available)


def _tool_bubble_title(provider_label: str | None) -> str:
    if not provider_label:
        return TOOL_BUBBLE_TITLE
    label = _one_line(provider_label).strip("`")
    if not label:
        return TOOL_BUBBLE_TITLE
    if label.lower().endswith("tools"):
        return label
    return f"{label} {TOOL_BUBBLE_TITLE}"


def _status_glyph(status: str) -> str:
    if status == "success":
        return "✓"
    if status == "error":
        return "❌"
    return "↻"


def _status_from_result_text(result_text: str) -> ToolStatus:
    if _BATCH_INTERRUPTED_RE.search(result_text) or _BATCH_ERROR_RE.search(result_text):
        return "error"
    return "success"


def _parse_tool_use_text(
    tool_use_text: str,
    tool_name: str | None,
) -> tuple[str, str]:
    text = _one_line(tool_use_text)
    if not text:
        return _display_tool_name(tool_name), ""

    markdown_match = _MARKDOWN_TOOL_PREFIX_RE.search(text)
    if markdown_match:
        parsed_name, suffix = markdown_match.groups()
        return tool_name or parsed_name.strip(), _strip_summary_wrappers(suffix)

    if tool_name:
        summary = _strip_named_prefix(text, tool_name)
        return tool_name, summary

    plain_match = _PLAIN_TOOL_PREFIX_RE.match(text)
    if plain_match:
        parsed_name, suffix = plain_match.groups()
        return parsed_name.strip(), _strip_summary_wrappers(suffix)

    return "Tool", text


def _strip_named_prefix(text: str, tool_name: str) -> str:
    match = re.match(rf"^\W*{re.escape(tool_name)}\b[:\s-]*(.*)$", text)
    if match:
        return _strip_summary_wrappers(match.group(1))
    return _strip_summary_wrappers(text)


def _strip_summary_wrappers(text: str) -> str:
    stripped = text.strip()
    while stripped.startswith((": ", "- ", "— ")):
        stripped = stripped[2:].strip()
    if stripped.startswith(":"):
        stripped = stripped[1:].strip()
    if (
        stripped.startswith("`")
        and stripped.endswith("`")
        and len(stripped) >= _MIN_BACKTICK_WRAPPED_LENGTH
    ):
        stripped = stripped[1:-1].strip()
    return stripped


def _normalize_summary(summary: str) -> str:
    text = _strip_summary_wrappers(_one_line(summary))
    text = _abbreviate_home_paths(text)
    text = text.replace('"', "'").replace("`", "'")
    if len(text) > TOOL_SUMMARY_LIMIT:
        return f"{text[: TOOL_SUMMARY_LIMIT - len(_TOOL_LINE_ELLIPSIS)]}{_TOOL_LINE_ELLIPSIS}"
    return text


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _abbreviate_home_paths(text: str) -> str:
    home = os.path.expanduser("~")
    if not home or home == "~":
        return text
    return text.replace(f"{home}/", "~/").replace(home, "~")


def _display_tool_name(tool_name: str | None) -> str:
    token = _tool_token(tool_name)
    if not token:
        return "Tool"
    alias = _TOOL_NAME_ALIASES.get(token.lower())
    if alias:
        return alias
    if "_" in token or "-" in token:
        return " ".join(part.capitalize() for part in re.split(r"[_-]+", token) if part)
    return token


def _tool_token(tool_name: str | None) -> str:
    if not tool_name:
        return ""
    token = tool_name.strip()
    if token.startswith("mcp__"):
        token = token.split("__")[-1]
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    return token.strip("_")


def _tool_icon(tool_name: str | None) -> str:
    raw_name = tool_name or ""
    if raw_name.startswith("mcp__"):
        return "🔌"
    display = _display_tool_name(tool_name)
    key = re.sub(r"[^a-z0-9]", "", display.lower())
    return _TOOL_ICONS.get(key, "🛠️")


def _extract_task_create_title(entry: ToolBatchEntry) -> str:
    """Extract the visible title from a TaskCreate summary."""
    return _extract_task_tool_suffix(entry)


def _extract_task_tool_suffix(entry: ToolBatchEntry) -> str:
    """Extract the summary text after a markdown/plain task-tool prefix."""
    if entry.summary:
        return entry.summary

    text = entry.tool_use_text.strip()
    if not text:
        return ""

    markdown_match = _MARKDOWN_TOOL_PREFIX_RE.match(text)
    if markdown_match:
        _tool_name, suffix = markdown_match.groups()
        stripped = suffix.strip()
        if (
            stripped.startswith("`")
            and stripped.endswith("`")
            and len(stripped) >= _MIN_BACKTICK_WRAPPED_LENGTH
        ):
            stripped = stripped[1:-1].strip()
        return stripped

    plain_match = _PLAIN_TASK_CREATE_RE.match(text)
    if plain_match:
        return plain_match.group(1).strip()

    return text


# ---------------------------------------------------------------------------
# State machine — process_tool_event / flush_batch
# ---------------------------------------------------------------------------


def _tool_batch_state_path() -> Path:
    return ccgram_dir() / "tool_batches.json"


def _entry_to_data(entry: ToolBatchEntry) -> dict[str, Any]:
    return {
        "tool_use_id": entry.tool_use_id,
        "tool_use_text": entry.tool_use_text,
        "tool_result_text": entry.tool_result_text,
        "tool_name": entry.tool_name,
        "summary": entry.summary,
        "status": entry.status,
        "result_text": entry.result_text,
    }


def _entry_from_data(data: dict[str, Any]) -> ToolBatchEntry | None:
    try:
        status = data.get("status", "pending")
        if status not in ("pending", "success", "error"):
            status = "pending"
        return ToolBatchEntry(
            tool_use_id=data.get("tool_use_id"),
            tool_use_text=str(data.get("tool_use_text") or ""),
            tool_result_text=data.get("tool_result_text"),
            tool_name=data.get("tool_name"),
            summary=str(data.get("summary") or ""),
            status=status,
            result_text=data.get("result_text"),
        )
    except AttributeError, TypeError, ValueError:
        return None


def _segment_to_data(segment: AgentBubbleSegment) -> dict[str, Any]:
    if segment.kind == "text":
        return {"kind": "text", "text": segment.text}
    return {
        "kind": "tools",
        "entries": [_entry_to_data(entry) for entry in segment.entries],
    }


def _segment_from_data(data: dict[str, Any]) -> AgentBubbleSegment | None:
    kind = data.get("kind")
    if kind == "text":
        return AgentBubbleSegment("text", text=str(data.get("text") or ""))
    if kind != "tools":
        return None
    entries_data = data.get("entries") or []
    entries = [
        entry
        for entry_data in entries_data
        if isinstance(entry_data, dict)
        for entry in [_entry_from_data(entry_data)]
        if entry is not None
    ]
    if not entries:
        return None
    return AgentBubbleSegment("tools", entries=entries)


def _batch_to_data(batch: ToolBatch) -> dict[str, Any]:
    return {
        "window_id": batch.window_id,
        "thread_id": batch.thread_id,
        "telegram_msg_id": batch.telegram_msg_id,
        "telegram_msg_ids": batch.telegram_msg_ids,
        "total_length": batch.total_length,
        "entries": [_entry_to_data(entry) for entry in batch.entries],
        "segments": [_segment_to_data(segment) for segment in batch.segments],
    }


def _batch_from_data(data: dict[str, Any]) -> ToolBatch | None:
    try:
        segments_data = data.get("segments") or []
        segments = [
            segment
            for segment_data in segments_data
            if isinstance(segment_data, dict)
            for segment in [_segment_from_data(segment_data)]
            if segment is not None
        ]
        entries = [
            entry
            for segment in segments
            if segment.kind == "tools"
            for entry in segment.entries
        ]

        if not segments:
            entries_data = data.get("entries") or []
            entries = [
                entry
                for entry_data in entries_data
                if isinstance(entry_data, dict)
                for entry in [_entry_from_data(entry_data)]
                if entry is not None
            ]
            if entries:
                segments = [AgentBubbleSegment("tools", entries=entries)]

        if not segments:
            return None
        msg_ids = [
            int(msg_id)
            for msg_id in data.get("telegram_msg_ids") or []
            if msg_id is not None
        ]
        legacy_msg_id = data.get("telegram_msg_id")
        if not msg_ids and legacy_msg_id is not None:
            msg_ids = [int(legacy_msg_id)]
        return ToolBatch(
            window_id=str(data["window_id"]),
            thread_id=int(data["thread_id"]),
            entries=entries,
            telegram_msg_id=msg_ids[0] if msg_ids else None,
            telegram_msg_ids=msg_ids,
            segments=segments,
            total_length=int(data.get("total_length") or 0),
        )
    except KeyError, TypeError, ValueError:
        return None


def _read_persisted_batch_data() -> dict[str, Any] | None:
    path = _tool_batch_state_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to load persisted tool batches: %s", exc)
        return None

    if not isinstance(data, dict):
        return None
    if data.get("version") != _PERSISTED_BATCH_STATE_VERSION:
        return None
    return data


def _restore_persisted_batch_item(
    item: Any,
    now: float,
) -> tuple[tuple[int, int], ToolBatch] | None | Literal["stale"]:
    if not isinstance(item, dict):
        return None
    try:
        user_id = int(item["user_id"])
        thread_id = int(item["thread_id"])
        updated_at = float(item.get("updated_at") or 0)
    except KeyError, TypeError, ValueError:
        return None
    if now - updated_at > _PERSISTED_BATCH_MAX_AGE_SECONDS:
        return "stale"

    batch_data = item.get("batch")
    if not isinstance(batch_data, dict):
        return None
    batch = _batch_from_data(batch_data)
    if batch is None:
        return None
    return (user_id, thread_id), batch


def _load_active_batches_if_needed() -> None:
    global _persistent_batches_loaded
    if _persistent_batches_loaded:
        return
    _persistent_batches_loaded = True

    data = _read_persisted_batch_data()
    if data is None:
        return

    now = time.time()
    loaded_count = 0
    stale_count = 0
    for item in data.get("batches", []):
        restored = _restore_persisted_batch_item(item, now)
        if restored == "stale":
            stale_count += 1
            continue
        if restored is None:
            continue
        key, batch = restored
        _active_batches.setdefault(key, batch)
        loaded_count += 1

    if stale_count > 0:
        _persist_active_batches()
    if loaded_count:
        logger.debug("loaded persisted tool batches count=%d", loaded_count)


def _persist_active_batches() -> None:
    path = _tool_batch_state_path()
    active_items = [
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "updated_at": time.time(),
            "batch": _batch_to_data(batch),
        }
        for (user_id, thread_id), batch in sorted(_active_batches.items())
        if _render_segments(batch)
    ]
    if not active_items:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("failed to remove persisted tool batches: %s", exc)
        return

    try:
        atomic_write_json(
            path,
            {"version": _PERSISTED_BATCH_STATE_VERSION, "batches": active_items},
        )
    except OSError as exc:
        logger.warning("failed to persist tool batches: %s", exc)


def _get_delivery_state(key: tuple[int, int]) -> _DeliveryState:
    state = _delivery_states.get(key)
    if state is None:
        state = _DeliveryState()
        _delivery_states[key] = state
    return state


def _cancel_delivery_state(key: tuple[int, int]) -> None:
    state = _delivery_states.pop(key, None)
    if state is None:
        return
    task = state.scheduled_task
    if task is not None and not task.done():
        task.cancel()


async def shutdown_delivery_tasks() -> None:
    """Cancel pending background agent-bubble delivery tasks."""
    tasks: list[asyncio.Task[None]] = []
    for state in list(_delivery_states.values()):
        task = state.scheduled_task
        if task is not None and not task.done():
            task.cancel()
            tasks.append(task)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _delivery_states.clear()


async def wait_for_pending_deliveries() -> None:
    """Test/support helper: wait for currently scheduled delivery tasks."""
    tasks = [
        task
        for state in _delivery_states.values()
        for task in [state.scheduled_task]
        if task is not None and not task.done()
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _agent_bubble_debounce_seconds() -> float:
    from ..config import config

    return max(0, config.agent_bubble_debounce_ms) / 1000


def _delivery_retry_delay(state: _DeliveryState) -> float:
    exponent = min(max(state.retry_count - 1, 0), 4)
    return min(
        _AGENT_BUBBLE_RETRY_MAX_SECONDS,
        _AGENT_BUBBLE_RETRY_BASE_SECONDS * (2**exponent),
    )


async def _mark_batch_dirty(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    *,
    immediate: bool = False,
) -> bool:
    """Mark the agent bubble dirty and schedule one serialized delivery."""
    key = (user_id, thread_id_or_0)
    state = _get_delivery_state(key)
    state.dirty_version += 1
    if immediate:
        return await _flush_delivery_now(bot, user_id, thread_id_or_0, force=True)
    delay = _agent_bubble_debounce_seconds()
    if delay <= 0:
        return await _flush_delivery_now(bot, user_id, thread_id_or_0)
    _ensure_delivery_task(bot, user_id, thread_id_or_0, delay)
    return True


def _ensure_delivery_task(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    delay: float,
) -> None:
    key = (user_id, thread_id_or_0)
    state = _get_delivery_state(key)
    existing = state.scheduled_task
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _delayed_delivery_flush(bot, user_id, thread_id_or_0, delay)
    )
    task.add_done_callback(task_done_callback)
    state.scheduled_task = task


async def _delayed_delivery_flush(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    delay: float,
) -> None:
    key = (user_id, thread_id_or_0)
    current_task = asyncio.current_task()
    try:
        if delay > 0:
            await asyncio.sleep(delay)
        if key in _active_batches:
            await _flush_delivery_now(bot, user_id, thread_id_or_0)
    finally:
        state = _delivery_states.get(key)
        if state is not None:
            if state.scheduled_task is current_task:
                state.scheduled_task = None
            if _should_reschedule_delivery(key, state):
                _ensure_delivery_task(
                    bot,
                    user_id,
                    thread_id_or_0,
                    _delivery_retry_delay(state),
                )


def _should_reschedule_delivery(
    key: tuple[int, int],
    state: _DeliveryState,
) -> bool:
    if key not in _active_batches:
        return False
    if state.dirty_version <= state.flushed_version:
        return False
    return state.permanent_failure_version != state.dirty_version


async def _flush_delivery_now(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    *,
    force: bool = False,
) -> bool:
    key = (user_id, thread_id_or_0)
    state = _get_delivery_state(key)
    async with state.lock:
        target_version = state.dirty_version
        if not force and state.flushed_version >= target_version:
            return True
        if not force and state.permanent_failure_version == target_version:
            return False

        batch = _active_batches.get(key)
        if not batch or not _render_segments(batch):
            state.flushed_version = target_version
            state.retry_count = 0
            return True

        thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        result = await _sync_batch_pages(
            bot,
            user_id,
            batch,
            chat_id,
            thread_id,
            thread_id_or_0,
            clear_status=not batch.telegram_msg_ids,
        )
        if result == _PageSyncResult.SUCCESS:
            state.flushed_version = target_version
            state.retry_count = 0
            state.permanent_failure_version = 0
            _persist_active_batches()
            return True
        if result == _PageSyncResult.PERMANENT_FAILURE:
            state.permanent_failure_version = target_version
            state.retry_count = 0
            _persist_active_batches()
            return False

        state.retry_count += 1
        _persist_active_batches()
        return False


async def _sync_batch_pages(
    bot: Bot,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    thread_id_or_0: int,
    *,
    clear_status: bool,
) -> _PageSyncResult:
    pages = _prepare_batch_pages(batch)
    if not pages:
        return _PageSyncResult.SUCCESS
    if clear_status:
        await _clear_status_before_agent_bubble(bot, user_id, thread_id_or_0)
    return await _sync_rendered_pages(
        bot, user_id, batch, chat_id, raw_thread_id, pages
    )


def _prepare_batch_pages(batch: ToolBatch) -> list[str]:
    pages = format_agent_pages(_render_segments(batch))
    if batch.telegram_msg_id and not batch.telegram_msg_ids:
        batch.telegram_msg_ids = [batch.telegram_msg_id]
    return pages


async def _clear_status_before_agent_bubble(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
) -> None:
    from .status_bubble import clear_status_message

    await clear_status_message(bot, user_id, thread_id_or_0)


async def _sync_rendered_pages(
    bot: Bot,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    pages: list[str],
) -> _PageSyncResult:
    disable_notification = not _batch_has_text(batch)
    for index, page in enumerate(pages):
        result = await _sync_one_page(
            bot,
            user_id,
            batch,
            chat_id,
            raw_thread_id,
            pages,
            index,
            page,
            disable_notification,
        )
        if result != _PageSyncResult.SUCCESS:
            return result
    await _delete_stale_pages(bot, batch, chat_id, len(pages))
    batch.telegram_msg_id = (
        batch.telegram_msg_ids[0] if batch.telegram_msg_ids else None
    )
    batch.rendered_pages = list(pages)
    return _PageSyncResult.SUCCESS


async def _sync_one_page(
    bot: Bot,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    pages: list[str],
    index: int,
    page: str,
    disable_notification: bool,
) -> _PageSyncResult:
    if index < len(batch.telegram_msg_ids):
        if index < len(batch.rendered_pages) and batch.rendered_pages[index] == page:
            return _PageSyncResult.SUCCESS
        outcome = await _edit_page(bot, user_id, batch, chat_id, pages, index, page)
        if outcome in (EditOutcome.APPLIED, EditOutcome.NOT_MODIFIED):
            return _PageSyncResult.SUCCESS
        if outcome != EditOutcome.MISSING:
            if outcome == EditOutcome.PERMANENT_FAILURE:
                return _PageSyncResult.PERMANENT_FAILURE
            return _PageSyncResult.TRANSIENT_FAILURE

    sent = await _send_fresh_batch_message(
        bot,
        batch,
        chat_id,
        raw_thread_id,
        page,
        disable_notification=disable_notification,
    )
    if sent is not None:
        _store_page_message_id(batch, index, sent)
        return _PageSyncResult.SUCCESS
    return _PageSyncResult.TRANSIENT_FAILURE


async def _edit_page(
    bot: Bot,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    pages: list[str],
    index: int,
    page: str,
) -> EditOutcome:
    msg_id = batch.telegram_msg_ids[index]
    logger.debug(
        "agent bubble edit user=%s thread=%s window=%s message_id=%s page=%d/%d",
        user_id,
        batch.thread_id,
        batch.window_id,
        msg_id,
        index + 1,
        len(pages),
    )
    return await edit_with_entities_outcome(
        bot,
        chat_id,
        msg_id,
        page,
        local_throttle=False,
    )


def _store_page_message_id(batch: ToolBatch, index: int, msg_id: int) -> None:
    if index < len(batch.telegram_msg_ids):
        batch.telegram_msg_ids[index] = msg_id
    else:
        batch.telegram_msg_ids.append(msg_id)


async def _delete_stale_pages(
    bot: Bot,
    batch: ToolBatch,
    chat_id: int,
    keep_count: int,
) -> None:
    for msg_id in batch.telegram_msg_ids[keep_count:]:
        with contextlib.suppress(TelegramError):
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
    batch.telegram_msg_ids = batch.telegram_msg_ids[:keep_count]


async def _send_fresh_batch_message(
    bot: Bot,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    batch_text: str,
    *,
    disable_notification: bool = True,
) -> int | None:
    sent = await rate_limit_send_message(
        bot,
        chat_id,
        batch_text,
        **send_kwargs(raw_thread_id),
        disable_notification=disable_notification,
    )
    if sent:
        logger.debug(
            "tool batch tracked thread=%s window=%s message_id=%s entries=%d",
            batch.thread_id,
            batch.window_id,
            sent.message_id,
            len(batch.entries),
        )
        return sent.message_id
    return None


def _batch_has_text(batch: ToolBatch) -> bool:
    return any(
        segment.kind == "text" and segment.text.strip()
        for segment in _render_segments(batch)
    )


def _render_segments(batch: ToolBatch) -> list[AgentBubbleSegment]:
    if batch.segments:
        return batch.segments
    if batch.entries:
        return [AgentBubbleSegment("tools", entries=batch.entries)]
    return []


async def _handle_tool_result(
    task: ContentTask,
    batch: ToolBatch | None,
    thread_id_or_0: int,
) -> tuple[ToolBatch | None, ContentTask | None]:
    """Process a tool_result event, updating the matching batch entry.

    Returns (updated_batch, followup) — followup is non-None when the result
    could not be absorbed into the batch and should be delivered as content.
    """
    if not batch:
        logger.debug(
            "tool result falls through window=%s thread=%s tool_id=%s has_batch=%s",
            task.window_id,
            thread_id_or_0,
            task.tool_use_id,
            bool(batch),
        )
        return None, task
    if not task.tool_use_id:
        logger.debug(
            "tool result without id suppressed window=%s thread=%s entries=%d",
            task.window_id,
            thread_id_or_0,
            len(batch.entries),
        )
        return batch, None
    for entry in batch.entries:
        if entry.tool_use_id == task.tool_use_id:
            text = "\n".join(task.parts) if task.parts else ""
            entry.tool_result_text = text
            entry.result_text = text
            entry.status = _status_from_result_text(text)
            logger.debug(
                "tool result absorbed window=%s thread=%s tool_id=%s status=%s",
                task.window_id,
                thread_id_or_0,
                task.tool_use_id,
                entry.status,
            )
            return batch, None
    text = "\n".join(task.parts) if task.parts else ""
    for entry in batch.entries:
        if entry.status == "pending":
            entry.tool_result_text = text
            entry.result_text = text
            entry.status = _status_from_result_text(text)
            logger.debug(
                "tool result unmatched mapped window=%s thread=%s tool_id=%s "
                "mapped_to=%s status=%s",
                task.window_id,
                thread_id_or_0,
                task.tool_use_id,
                entry.tool_use_id,
                entry.status,
            )
            return batch, None
    logger.debug(
        "tool result unmatched suppressed window=%s thread=%s tool_id=%s entries=%d",
        task.window_id,
        thread_id_or_0,
        task.tool_use_id,
        len(batch.entries),
    )
    return batch, None


def _add_tool_use_entry(
    task: ContentTask,
    batch: ToolBatch,
) -> None:
    """Append a tool_use entry to the batch."""
    entry_text = "\n".join(task.parts) if task.parts else "tool call"
    entry = ToolBatchEntry(
        tool_use_id=task.tool_use_id,
        tool_use_text=entry_text,
        tool_name=task.tool_name,
    )
    batch.entries.append(entry)
    _current_tool_segment(batch).entries.append(entry)
    batch.total_length += len(entry_text)
    logger.debug(
        "tool use added window=%s thread=%s tool_id=%s tool_name=%s entries=%d summary=%r",
        task.window_id,
        thread_key(task.thread_id),
        task.tool_use_id,
        entry.tool_name,
        len(batch.entries),
        entry.summary,
    )


def _current_tool_segment(batch: ToolBatch) -> AgentBubbleSegment:
    if batch.segments and batch.segments[-1].kind == "tools":
        return batch.segments[-1]
    segment = AgentBubbleSegment("tools")
    batch.segments.append(segment)
    return segment


def _append_text_segment(batch: ToolBatch, text: str) -> None:
    clean = text.strip()
    if not clean:
        return
    if batch.segments and batch.segments[-1].kind == "text":
        existing = batch.segments[-1].text.strip()
        batch.segments[-1].text = f"{existing}\n\n{clean}" if existing else clean
    else:
        batch.segments.append(AgentBubbleSegment("text", text=clean))
    batch.total_length += len(clean)


async def process_tool_event(
    bot: Bot,
    user_id: int,
    task: ContentTask,
) -> ContentTask | None:
    """Add a tool_use or tool_result to the active batch, send/edit the batch message.

    Returns None if absorbed into the batch; returns a ContentTask if the queue
    worker should deliver it as regular content (overflow, unmatched result, etc).
    """
    window_id = task.window_id
    thread_id_or_0 = thread_key(task.thread_id)
    bkey = (user_id, thread_id_or_0)
    _load_active_batches_if_needed()
    batch = _active_batches.get(bkey)

    if task.content_type == "tool_result":
        batch, followup = await _handle_tool_result(task, batch, thread_id_or_0)
        if batch is None:
            return followup
    elif task.content_type == "tool_use":
        result = await _handle_tool_use_event(
            bot, user_id, task, batch, window_id, thread_id_or_0, bkey
        )
        if isinstance(result, ContentTask):
            return result
        if result is None:
            return None
        batch = result
    else:
        return task

    await _mark_batch_dirty(bot, user_id, thread_id_or_0)
    _persist_active_batches()
    return None


async def process_agent_message(bot: Bot, user_id: int, task: ContentTask) -> None:
    """Append assistant text to the active ordered turn bubble."""
    if task.role != "assistant":
        return

    window_id = task.window_id
    thread_id_or_0 = thread_key(task.thread_id)
    bkey = (user_id, thread_id_or_0)
    _load_active_batches_if_needed()
    batch = _active_batches.get(bkey)

    if batch and batch.window_id != window_id:
        await flush_batch(bot, user_id, thread_id_or_0)
        batch = None

    if not batch:
        batch = ToolBatch(window_id=window_id, thread_id=thread_id_or_0)
        _active_batches[bkey] = batch

    for part in task.parts:
        _append_text_segment(batch, part)

    await _mark_batch_dirty(
        bot,
        user_id,
        thread_id_or_0,
        immediate=task.phase == "final_answer",
    )
    if task.phase == "final_answer":
        _active_batches.pop(bkey, None)
        _cancel_delivery_state(bkey)
    _persist_active_batches()


async def _handle_tool_use_event(
    bot: Bot,
    user_id: int,
    task: ContentTask,
    batch: ToolBatch | None,
    window_id: str,
    thread_id_or_0: int,
    bkey: tuple[int, int],
) -> ToolBatch | ContentTask | None:
    """Process a tool_use event, creating/flushing batches as needed.

    Returns a ToolBatch to continue with send/edit, a ContentTask if the caller
    should deliver it as regular content (double-overflow), or None on error.
    """
    if batch and batch.window_id != window_id:
        logger.debug(
            "tool batch window changed thread=%s old_window=%s new_window=%s",
            thread_id_or_0,
            batch.window_id,
            window_id,
        )
        await flush_batch(bot, user_id, thread_id_or_0)
        batch = None

    if not batch:
        batch = ToolBatch(window_id=window_id, thread_id=thread_id_or_0)
        _active_batches[bkey] = batch
        logger.debug(
            "tool batch created user=%s thread=%s window=%s",
            user_id,
            thread_id_or_0,
            window_id,
        )

    _add_tool_use_entry(task, batch)

    return batch


async def flush_if_active(bot: Bot, user_id: int, task: ContentTask) -> None:
    """Flush any active batch for the same topic before delivering non-batchable content."""
    thread_id_or_0 = thread_key(task.thread_id)
    if has_active_batch(user_id, thread_id_or_0):
        logger.debug(
            "tool batch flush before content user=%s thread=%s window=%s "
            "role=%s phase=%s content_type=%s",
            user_id,
            thread_id_or_0,
            task.window_id,
            task.role,
            task.phase,
            task.content_type,
        )
        await flush_batch(bot, user_id, thread_id_or_0)


async def flush_batch(bot: Bot, user_id: int, thread_id_or_0: int) -> None:
    """Finalize the active batch: do a final edit and clear state."""
    _load_active_batches_if_needed()
    bkey = (user_id, thread_id_or_0)
    batch = _active_batches.get(bkey)
    if not batch or not _render_segments(batch):
        _active_batches.pop(bkey, None)
        _delivery_states.pop(bkey, None)
        _persist_active_batches()
        return

    logger.debug(
        "agent bubble flush user=%s thread=%s window=%s message_id=%s entries=%d",
        user_id,
        thread_id_or_0,
        batch.window_id,
        batch.telegram_msg_id,
        len(batch.entries),
    )
    state = _get_delivery_state(bkey)
    state.dirty_version += 1
    await _flush_delivery_now(bot, user_id, thread_id_or_0, force=True)
    _active_batches.pop(bkey, None)
    _cancel_delivery_state(bkey)
    _persist_active_batches()


def has_active_batch(user_id: int, thread_id_or_0: int) -> bool:
    """Check if there is an active batch for a (user, thread) pair."""
    _load_active_batches_if_needed()
    return (user_id, thread_id_or_0) in _active_batches


@topic_state.register("topic")
def clear_batch_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear active batch for a specific topic (called on topic cleanup)."""
    _load_active_batches_if_needed()
    key = (user_id, thread_key(thread_id))
    _active_batches.pop(key, None)
    _cancel_delivery_state(key)
    _persist_active_batches()


def clear_all_batches() -> None:
    """Clear process-local batches without deleting restart recovery state."""
    _active_batches.clear()
    for key in list(_delivery_states):
        _cancel_delivery_state(key)
