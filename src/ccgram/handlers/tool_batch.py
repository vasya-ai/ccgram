"""Tool-use batching — one rotating Telegram tool bubble.

Accumulates consecutive tool_use / tool_result messages into one compact
Telegram message that is edited in place as tools arrive and complete.  The
full in-memory entry list is retained for the active turn; rendering rotates to
the newest entries that fit Telegram's text limit.

Key components:
  - ToolBatchEntry / ToolBatch: batch state dataclasses
  - process_tool_event: state-machine entry point (add tool_use or tool_result)
  - flush_batch: finalize and send the last edit for a batch
  - is_batch_eligible: predicate combining task eligibility and window mode
  - format_batch_message: render entries as a compact fenced tool bubble
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
from telegram import Bot

from ..window_query import get_batch_mode
from ..thread_router import thread_router
from ..topic_state_registry import topic_state
from .message_sender import edit_with_fallback, rate_limit_send_message, send_kwargs
from .message_task import ContentTask, thread_key

logger = structlog.get_logger()

TELEGRAM_TEXT_LIMIT = 4096
TOOL_BUBBLE_TITLE = "Tools"
TOOL_SUMMARY_LIMIT = 160
_TOOL_LINE_ELLIPSIS = "…"

ToolStatus = Literal["pending", "success", "error"]


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
class ToolBatch:
    """Accumulator for consecutive tool calls to batch into one Telegram message."""

    window_id: str
    thread_id: int  # thread_id_or_0
    entries: list[ToolBatchEntry] = field(default_factory=list)
    telegram_msg_id: int | None = None
    total_length: int = 0  # Legacy metric; no longer used as an overflow trigger.


# Active tool batches: (user_id, thread_id_or_0) -> ToolBatch
_active_batches: dict[tuple[int, int], ToolBatch] = {}

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
# Public predicates
# ---------------------------------------------------------------------------


def is_batch_eligible(task: ContentTask) -> bool:
    """Check if a task should go through the batching pipeline."""
    return (
        task.content_type in ("tool_use", "tool_result")
        and get_batch_mode(task.window_id) == "batched"
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_batch_message(
    entries: list[ToolBatchEntry],
    subagent_label: str | None = None,
    provider_label: str | None = None,
) -> str:
    """Render the active tool list as a compact fenced Telegram bubble."""
    del subagent_label  # Legacy signature; task grouping text is intentionally gone.
    title = _tool_bubble_title(provider_label)
    lines = [_format_batch_entry(entry) for entry in entries]
    return _rotate_tool_lines(lines, title)


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


def _rotate_tool_lines(lines: list[str], title: str) -> str:
    """Render the newest suffix of lines that fits Telegram's text limit."""
    if not lines:
        return _render_tool_bubble([], title, hidden_count=0)

    visible: list[str] = []
    for index in range(len(lines) - 1, -1, -1):
        candidate = [lines[index], *visible]
        hidden_count = index
        if len(_render_tool_bubble(candidate, title, hidden_count)) <= TELEGRAM_TEXT_LIMIT:
            visible = candidate
            continue
        break

    if not visible:
        hidden_count = len(lines) - 1
        visible = [_truncate_line_to_fit(lines[-1], title, hidden_count)]

    hidden_count = len(lines) - len(visible)
    return _render_tool_bubble(visible, title, hidden_count)


def _render_tool_bubble(
    visible_lines: list[str],
    title: str,
    hidden_count: int,
) -> str:
    body_lines: list[str] = []
    if hidden_count > 0:
        body_lines.append(f"{_TOOL_LINE_ELLIPSIS} {hidden_count} earlier tools {_TOOL_LINE_ELLIPSIS}")
    body_lines.extend(visible_lines)
    body = "\n".join(body_lines)
    if body:
        return f"```{title}\n{body}\n```"
    return f"```{title}\n```"


def _truncate_line_to_fit(line: str, title: str, hidden_count: int) -> str:
    line_overhead = len(_render_tool_bubble(["x"], title, hidden_count)) - 1
    available = TELEGRAM_TEXT_LIMIT - line_overhead
    if available <= 0:
        return ""
    if len(line) <= available:
        return line
    if available <= len(_TOOL_LINE_ELLIPSIS):
        return _TOOL_LINE_ELLIPSIS[:available]
    return f"{line[: available - len(_TOOL_LINE_ELLIPSIS)]}{_TOOL_LINE_ELLIPSIS}"


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


async def _send_or_edit_batch(
    bot: Bot,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    thread_id_or_0: int,
) -> None:
    """Send a new batch message or edit the existing one in place."""
    from .status_bubble import clear_status_message

    batch_text = format_batch_message(batch.entries)
    await clear_status_message(bot, user_id, thread_id_or_0)

    if batch.telegram_msg_id is None:
        logger.debug(
            "tool batch send user=%s thread=%s window=%s entries=%d",
            user_id,
            thread_id_or_0,
            batch.window_id,
            len(batch.entries),
        )
        await _send_fresh_batch_message(
            bot,
            batch,
            chat_id,
            raw_thread_id,
            batch_text,
        )
    else:
        logger.debug(
            "tool batch edit user=%s thread=%s window=%s message_id=%s entries=%d",
            user_id,
            thread_id_or_0,
            batch.window_id,
            batch.telegram_msg_id,
            len(batch.entries),
        )
        success = await edit_with_fallback(
            bot,
            chat_id,
            batch.telegram_msg_id,
            batch_text,
        )
        if not success:
            await _send_fresh_batch_message(
                bot,
                batch,
                chat_id,
                raw_thread_id,
                batch_text,
            )


async def _send_fresh_batch_message(
    bot: Bot,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    batch_text: str,
) -> None:
    sent = await rate_limit_send_message(
        bot,
        chat_id,
        batch_text,
        **send_kwargs(raw_thread_id),
        disable_notification=True,
    )
    if sent:
        batch.telegram_msg_id = sent.message_id
        logger.debug(
            "tool batch tracked thread=%s window=%s message_id=%s entries=%d",
            batch.thread_id,
            batch.window_id,
            sent.message_id,
            len(batch.entries),
        )


async def _handle_tool_result(
    bot: Bot,
    user_id: int,
    task: ContentTask,
    batch: ToolBatch | None,
    thread_id_or_0: int,
) -> tuple[ToolBatch | None, ContentTask | None]:
    """Process a tool_result event, updating the matching batch entry.

    Returns (updated_batch, followup) — followup is non-None when the result
    could not be absorbed into the batch and should be delivered as content.
    """
    if not task.tool_use_id or not batch:
        logger.debug(
            "tool result falls through window=%s thread=%s tool_id=%s has_batch=%s",
            task.window_id,
            thread_id_or_0,
            task.tool_use_id,
            bool(batch),
        )
        return None, task
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
    logger.debug(
        "tool result unmatched window=%s thread=%s tool_id=%s entries=%d",
        task.window_id,
        thread_id_or_0,
        task.tool_use_id,
        len(batch.entries),
    )
    await flush_batch(bot, user_id, thread_id_or_0)
    return None, task


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
    chat_id = thread_router.resolve_chat_id(user_id, task.thread_id)
    batch = _active_batches.get(bkey)

    if task.content_type == "tool_result":
        batch, followup = await _handle_tool_result(
            bot, user_id, task, batch, thread_id_or_0
        )
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

    await _send_or_edit_batch(
        bot, user_id, batch, chat_id, task.thread_id, thread_id_or_0
    )
    return None


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
    bkey = (user_id, thread_id_or_0)
    batch = _active_batches.pop(bkey, None)
    if not batch or not batch.entries:
        return

    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)

    batch_text = format_batch_message(batch.entries)
    logger.debug(
        "tool batch flush user=%s thread=%s window=%s message_id=%s entries=%d",
        user_id,
        thread_id_or_0,
        batch.window_id,
        batch.telegram_msg_id,
        len(batch.entries),
    )

    if batch.telegram_msg_id is None:
        await _send_fresh_batch_message(
            bot,
            batch,
            chat_id,
            thread_id,
            batch_text,
        )
        return

    success = await edit_with_fallback(
        bot,
        chat_id,
        batch.telegram_msg_id,
        batch_text,
    )
    if not success:
        await _send_fresh_batch_message(
            bot,
            batch,
            chat_id,
            thread_id,
            batch_text,
        )


def has_active_batch(user_id: int, thread_id_or_0: int) -> bool:
    """Check if there is an active batch for a (user, thread) pair."""
    return (user_id, thread_id_or_0) in _active_batches


@topic_state.register("topic")
def clear_batch_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear active batch for a specific topic (called on topic cleanup)."""
    _active_batches.pop((user_id, thread_key(thread_id)), None)


def clear_all_batches() -> None:
    """Clear all active batches (called on shutdown)."""
    _active_batches.clear()
