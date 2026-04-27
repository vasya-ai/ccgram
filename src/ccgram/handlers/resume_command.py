"""Resume command — choose a directory/provider/mode, then resume a session.

Key functions:
  - resume_command: /resume handler
  - handle_resume_command_callback: callback dispatcher for resume UI
  - scan_resumable_sessions: discover resumable sessions for a provider/cwd
"""

import json
import asyncio
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..providers import get_provider_for_window, has_yolo_mode, resolve_launch_command
from ..providers.base import UUID_RE
from .. import window_query
from ..session import session_manager
from ..session_map import session_map_sync
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..utils import read_session_metadata_from_jsonl
from ..window_resolver import is_foreign_window
from .callback_data import (
    CB_RESUME_CANCEL,
    CB_RESUME_DIR_BACK,
    CB_RESUME_MODE_SELECT,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
    CB_RESUME_PROV_SELECT,
)
from .callback_helpers import get_thread_id
from .callback_registry import register
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    build_directory_browser,
    clear_browse_state,
)
from .message_sender import safe_edit, safe_reply
from .session_teardown import teardown_topic_session
from .topic_emoji import format_topic_name_for_mode
from .user_state import (
    PENDING_THREAD_ID,
    RESUME_APPROVAL_MODE,
    RESUME_PROVIDER,
    RESUME_SELECTED_CWD,
    RESUME_SESSIONS,
    RESUME_THREAD_ID,
)

logger = structlog.get_logger()

_SESSIONS_PER_PAGE = 6
_CODEX_METADATA_SCAN_LINES = 50
_SUMMARY_MAX_CHARS = 80
_GEMINI_METADATA_SCAN_MESSAGES = 20
_PI_METADATA_SCAN_LINES = 50
_RESUME_DISCOVERY_PROVIDERS = ("claude", "codex", "gemini", "pi")

_IndexParseError = (json.JSONDecodeError, OSError)


@dataclass
class ResumeEntry:
    """A resumable session discovered from provider session storage."""

    session_id: str
    summary: str
    cwd: str
    provider_name: str = "claude"
    transcript_path: str = ""


@dataclass(frozen=True)
class _ResumeModeSelection:
    provider_name: str
    approval_mode: str
    selected_path: str


def scan_all_sessions(provider_name: str | None = None) -> list[ResumeEntry]:
    """Scan provider-specific storage for resumable sessions."""
    provider = provider_name or "claude"
    if provider == "claude":
        return _scan_claude_sessions()
    if provider == "codex":
        return _scan_codex_sessions()
    return []


def scan_resumable_sessions(provider_name: str, cwd: str) -> list[ResumeEntry]:
    """Return resumable sessions for one provider and one working directory."""
    if not cwd or provider_name not in _RESUME_DISCOVERY_PROVIDERS:
        return []

    if provider_name == "gemini":
        return _scan_gemini_sessions_for_cwd(cwd)
    if provider_name == "pi":
        return _scan_pi_sessions_for_cwd(cwd)

    target = _normalize_path(cwd)
    if not target:
        return []
    entries = []
    for entry in scan_all_sessions(provider_name):
        if _normalize_path(entry.cwd) != target:
            continue
        if not _is_resumable_entry(entry):
            continue
        entries.append(entry)
    return entries


def _normalize_path(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return path


def _is_resumable_entry(entry: ResumeEntry) -> bool:
    if not entry.transcript_path or not Path(entry.transcript_path).exists():
        return False
    if entry.provider_name == "claude":
        return bool(UUID_RE.match(entry.session_id))
    return bool(entry.session_id)


def _scan_claude_sessions() -> list[ResumeEntry]:
    """Scan Claude project directories for resumable sessions.

    Supports both legacy sessions-index.json and bare JSONL files
    (Claude Code >= Feb 2026 no longer writes index files).

    Returns entries sorted by file mtime (most recent first),
    deduplicated by session_id.
    """
    if not config.claude_projects_path.exists():
        return []

    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()

    for project_dir in config.claude_projects_path.iterdir():
        if not project_dir.is_dir():
            continue

        # Try legacy sessions-index.json first
        index_file = project_dir / "sessions-index.json"
        if index_file.exists():
            _scan_index_file(index_file, seen_ids, candidates)

        # Pick up bare JSONL files (no index required)
        _scan_bare_jsonl(project_dir, seen_ids, candidates)

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _scan_index_file(
    index_file: Path,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumeEntry]],
) -> None:
    """Scan a sessions-index.json for resumable sessions."""
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except _IndexParseError:
        return

    original_path = index_data.get("originalPath", "")
    for entry in index_data.get("entries", []):
        session_id = entry.get("sessionId", "")
        full_path = entry.get("fullPath", "")
        if not session_id or not full_path or session_id in seen_ids:
            continue

        file_path = Path(full_path)
        if not file_path.exists():
            continue

        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0

        cwd = entry.get("projectPath", original_path)
        summary = (
            entry.get("summary", "") or entry.get("firstPrompt", "") or session_id[:12]
        )
        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumeEntry(
                    session_id,
                    summary,
                    cwd,
                    transcript_path=str(file_path),
                ),
            )
        )


def _scan_bare_jsonl(
    project_dir: Path,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumeEntry]],
) -> None:
    """Scan bare JSONL files not covered by a sessions-index."""
    try:
        jsonl_iter = project_dir.glob("*.jsonl")
    except OSError:
        return

    for jsonl_file in jsonl_iter:
        session_id = jsonl_file.stem
        if session_id in seen_ids:
            continue

        cwd, summary = read_session_metadata_from_jsonl(jsonl_file)
        if not cwd:
            continue

        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumeEntry(
                    session_id,
                    summary or session_id[:12],
                    cwd,
                    transcript_path=str(jsonl_file),
                ),
            )
        )


def _scan_codex_sessions() -> list[ResumeEntry]:
    """Scan Codex CLI session JSONL files for resumable sessions."""
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.is_dir():
        return []

    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()
    try:
        jsonl_files = list(sessions_dir.rglob("*.jsonl"))
    except OSError:
        return []

    for jsonl_file in jsonl_files:
        metadata = _read_codex_resume_metadata(jsonl_file)
        if metadata is None:
            continue
        session_id, cwd, summary = metadata
        if session_id in seen_ids:
            continue
        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0
        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumeEntry(
                    session_id,
                    summary or session_id[:12],
                    cwd,
                    "codex",
                    str(jsonl_file),
                ),
            )
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _scan_gemini_sessions_for_cwd(cwd: str) -> list[ResumeEntry]:
    """Scan Gemini chat JSON files for sessions matching one cwd."""
    try:
        from ccgram.providers.gemini import (
            _collect_gemini_sessions,
            _read_gemini_session_meta,
            _read_project_alias,
        )
    except ImportError:
        return []

    config_dir = Path.home() / ".gemini"
    sessions_root = config_dir / "tmp"
    if not sessions_root.is_dir():
        return []

    resolved_cwd = _normalize_path(cwd)
    expected_hash = hashlib.sha256(resolved_cwd.encode()).hexdigest()
    candidate_dirs = [sessions_root / expected_hash / "chats"]
    alias = _read_project_alias(config_dir, resolved_cwd)
    if alias:
        candidate_dirs.append(sessions_root / alias / "chats")

    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()
    seen_dirs: set[Path] = set()
    for chats_dir in candidate_dirs:
        if chats_dir in seen_dirs:
            continue
        seen_dirs.add(chats_dir)
        for mtime, fpath in _collect_gemini_sessions(chats_dir):
            meta = _read_gemini_session_meta(fpath)
            if not meta:
                continue
            session_id, project_hash = meta
            if project_hash != expected_hash or session_id in seen_ids:
                continue
            seen_ids.add(session_id)
            candidates.append(
                (
                    mtime,
                    ResumeEntry(
                        session_id=session_id,
                        summary=_read_gemini_summary(fpath) or session_id[:12],
                        cwd=resolved_cwd,
                        provider_name="gemini",
                        transcript_path=str(fpath),
                    ),
                )
            )

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _read_gemini_summary(fpath: Path) -> str:
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return ""
    for entry in messages[:_GEMINI_METADATA_SCAN_MESSAGES]:
        if not isinstance(entry, dict) or entry.get("type") != "user":
            continue
        text = _extract_gemini_content_text(entry.get("content"))
        if text:
            return text[:_SUMMARY_MAX_CHARS]
    return ""


def _extract_gemini_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _scan_pi_sessions_for_cwd(cwd: str) -> list[ResumeEntry]:
    """Scan Pi JSONL v3 sessions for sessions matching one cwd."""
    try:
        from ccgram.providers.pi import _candidate_transcripts
        from ccgram.providers.pi_format import read_session_header
    except ImportError:
        return []

    resolved_cwd = _normalize_path(cwd)
    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()
    for mtime, fpath in _candidate_transcripts(cwd):
        header = read_session_header(str(fpath))
        if not header:
            continue
        session_id = header.get("id", "")
        header_cwd = header.get("cwd", "")
        if not session_id or session_id in seen_ids:
            continue
        if _normalize_path(header_cwd) != resolved_cwd:
            continue
        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumeEntry(
                    session_id=session_id,
                    summary=_read_pi_summary(fpath) or session_id[:12],
                    cwd=header_cwd,
                    provider_name="pi",
                    transcript_path=str(fpath),
                ),
            )
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _read_pi_summary(fpath: Path) -> str:
    try:
        from ccgram.providers.pi_format import extract_text
    except ImportError:
        return ""

    for data in _iter_jsonl_dicts(fpath, _PI_METADATA_SCAN_LINES):
        msg = data.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = extract_text(msg.get("content", "")).strip()
        if text:
            return text[:_SUMMARY_MAX_CHARS]
    return ""


def _read_codex_resume_metadata(jsonl_file: Path) -> tuple[str, str, str] | None:
    """Read Codex session id, cwd, and first user prompt from a JSONL file."""
    session_id = ""
    cwd = ""
    summary = ""
    for data in _iter_jsonl_dicts(jsonl_file, _CODEX_METADATA_SCAN_LINES):
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            continue
        entry_type = data.get("type", "")
        if entry_type == "session_meta":
            if _is_codex_subagent_source(payload.get("source")):
                return None
            session_id = _first_text(payload.get("id"), session_id)
            cwd = _first_text(payload.get("cwd"), cwd)
        elif entry_type == "turn_context":
            cwd = _first_text(payload.get("cwd"), cwd)
        if not summary:
            summary = _extract_codex_user_summary(entry_type, payload)
        if session_id and cwd and summary:
            break

    if not session_id or not cwd:
        return None
    return session_id, cwd, summary


def _iter_jsonl_dicts(jsonl_file: Path, max_lines: int) -> Iterator[dict[str, Any]]:
    try:
        with open(jsonl_file, encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                if line_index >= max_lines:
                    break
                parsed = _parse_jsonl_dict(line)
                if parsed is not None:
                    yield parsed
    except OSError:
        return


def _parse_jsonl_dict(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _first_text(value: Any, current: str) -> str:
    if current:
        return current
    return value if isinstance(value, str) and value else ""


def _is_codex_subagent_source(source: Any) -> bool:
    return isinstance(source, dict) and "subagent" in source


def _extract_codex_user_summary(entry_type: str, payload: dict[str, Any]) -> str:
    if entry_type not in ("response_item", "input_item"):
        return ""
    if payload.get("role") != "user":
        return ""
    text = _extract_codex_content_text(payload.get("content"))
    if _is_codex_injected_context(text):
        return ""
    return text[:_SUMMARY_MAX_CHARS]


def _is_codex_injected_context(text: str) -> bool:
    return text.startswith(
        (
            "<permissions",
            "<environment_context",
            "# AGENTS.md instructions",
        )
    )


def _extract_codex_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("input_text", "output_text", "text"):
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _build_resume_keyboard(
    sessions: list[dict[str, str]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for resume session picker with pagination."""
    total = len(sessions)
    start = page * _SESSIONS_PER_PAGE
    end = min(start + _SESSIONS_PER_PAGE, total)
    page_sessions = sessions[start:end]

    rows: list[list[InlineKeyboardButton]] = []
    current_cwd = ""
    for idx_offset, entry in enumerate(page_sessions):
        global_idx = start + idx_offset
        cwd = entry.get("cwd", "")
        # Show project header when cwd changes
        if cwd != current_cwd:
            current_cwd = cwd
            short_path = Path(cwd).name if cwd else "unknown"
            rows.append(
                [
                    InlineKeyboardButton(
                        f"\U0001f4c1 {short_path}",
                        callback_data=CB_RESUME_DIR_BACK,
                    )
                ]
            )
        label = entry.get("summary", "")[:40] or entry["session_id"][:12]
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"{CB_RESUME_PICK}{global_idx}"[:64],
                )
            ]
        )

    # Pagination row
    nav_buttons: list[InlineKeyboardButton] = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(
                "\u2b05 Prev",
                callback_data=f"{CB_RESUME_PAGE}{page - 1}"[:64],
            )
        )
    total_pages = (total + _SESSIONS_PER_PAGE - 1) // _SESSIONS_PER_PAGE
    if page < total_pages - 1:
        nav_buttons.append(
            InlineKeyboardButton(
                "Next \u27a1",
                callback_data=f"{CB_RESUME_PAGE}{page + 1}"[:64],
            )
        )
    nav_buttons.append(
        InlineKeyboardButton("\u2716 Cancel", callback_data=CB_RESUME_CANCEL)
    )
    rows.append(nav_buttons)

    return InlineKeyboardMarkup(rows)


_PROVIDER_META: dict[str, tuple[str, str]] = {
    "claude": ("Claude", "\U0001f7e0"),
    "codex": ("Codex", "\U0001f9e9"),
    "gemini": ("Gemini", "\u264a"),
    "pi": ("Pi", "\U0001f967"),
}


def build_resume_provider_picker(selected_path: str) -> tuple[str, InlineKeyboardMarkup]:
    """Build provider picker for cwd-scoped resume."""
    display_path = selected_path.replace(str(Path.home()), "~")
    text = (
        "*Select Provider To Resume*\n\n"
        f"Directory: `{display_path}`\n\n"
        "Which agent history should be searched?"
    )
    rows: list[list[InlineKeyboardButton]] = []
    for provider_name in _RESUME_DISCOVERY_PROVIDERS:
        provider = _resolve_resume_provider(provider_name)
        if provider is None:
            continue
        if not provider.capabilities.supports_resume:
            continue
        label, icon = _provider_label(provider_name)
        rows.append(
            [
                InlineKeyboardButton(
                    f"{icon} {label}",
                    callback_data=f"{CB_RESUME_PROV_SELECT}{provider_name}",
                )
            ]
        )
    rows.append([InlineKeyboardButton("Cancel", callback_data=CB_RESUME_CANCEL)])
    return text, InlineKeyboardMarkup(rows)


def _resolve_resume_provider(provider_name: str) -> Any | None:
    if provider_name not in _RESUME_DISCOVERY_PROVIDERS:
        return None
    provider = get_provider_for_window("", provider_name=provider_name)
    if provider.capabilities.name != provider_name:
        return None
    return provider


def _provider_label(provider_name: str) -> tuple[str, str]:
    return _PROVIDER_META.get(provider_name, (provider_name.title(), "\U0001f916"))


def build_resume_mode_picker(
    selected_path: str, provider_name: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Build approval-mode picker for resume launch."""
    display_path = selected_path.replace(str(Path.home()), "~")
    label, icon = _provider_label(provider_name)
    text = (
        "*Select Resume Mode*\n\n"
        f"Directory: `{display_path}`\n"
        f"Provider: {icon} {label}\n\n"
        "Choose how many approvals you want for the resumed session."
    )
    rows = [
        [
            InlineKeyboardButton(
                "\u2705 Standard",
                callback_data=f"{CB_RESUME_MODE_SELECT}{provider_name}:normal",
            )
        ]
    ]
    if has_yolo_mode(provider_name):
        rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f3b2 YOLO",
                    callback_data=f"{CB_RESUME_MODE_SELECT}{provider_name}:yolo",
                )
            ]
        )
    rows.append([InlineKeyboardButton("Cancel", callback_data=CB_RESUME_CANCEL)])
    return text, InlineKeyboardMarkup(rows)


async def show_resume_provider_picker(
    query: CallbackQuery,
    selected_path: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Transition from shared directory browser to resume provider picker."""
    thread_id = get_thread_id(update)
    if not _resume_thread_matches(context.user_data, thread_id):
        clear_resume_state(context.user_data)
        await query.answer("Stale resume browser", show_alert=True)
        return
    if not selected_path or not Path(selected_path).is_dir():
        clear_resume_state(context.user_data)
        await safe_edit(query, "\u274c Directory no longer exists.")
        return
    if context.user_data is not None:
        context.user_data[RESUME_SELECTED_CWD] = selected_path
    logger.info(
        "resume_directory_selected",
        thread_id=thread_id,
        cwd=selected_path,
    )
    text, keyboard = build_resume_provider_picker(selected_path)
    await safe_edit(query, text, reply_markup=keyboard)


def is_resume_flow(user_data: dict | None) -> bool:
    return bool(user_data and user_data.get(RESUME_THREAD_ID) is not None)


def clear_resume_state(user_data: dict | None) -> None:
    """Remove resume-related keys from user_data."""
    if user_data is None:
        return
    for key in (
        RESUME_SESSIONS,
        RESUME_THREAD_ID,
        RESUME_SELECTED_CWD,
        RESUME_PROVIDER,
        RESUME_APPROVAL_MODE,
    ):
        user_data.pop(key, None)


def clear_resume_flow_state(user_data: dict | None) -> None:
    """Remove resume wizard state, including shared directory browser state."""
    clear_resume_state(user_data)
    clear_browse_state(user_data)
    if user_data is not None:
        user_data.pop(PENDING_THREAD_ID, None)


def _resume_thread_matches(user_data: dict | None, thread_id: int | None) -> bool:
    if user_data is None or thread_id is None:
        return False
    pending = user_data.get(RESUME_THREAD_ID)
    return pending == thread_id


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /resume — start directory/provider/mode resume wizard."""
    if not update.message:
        return

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(
            update.message,
            "\u274c Please use /resume in a named topic.",
        )
        return

    clear_browse_state(context.user_data)
    clear_resume_state(context.user_data)
    start_path = str(Path.cwd())
    msg_text, keyboard, subdirs = build_directory_browser(start_path, user_id=user.id)
    if context.user_data is not None:
        context.user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        context.user_data[BROWSE_PATH_KEY] = start_path
        context.user_data[BROWSE_PAGE_KEY] = 0
        context.user_data[BROWSE_DIRS_KEY] = subdirs
        context.user_data[PENDING_THREAD_ID] = thread_id
        context.user_data[RESUME_THREAD_ID] = thread_id
    logger.info(
        "resume_flow_start",
        user_id=user.id,
        thread_id=thread_id,
        start_path=start_path,
    )
    await safe_reply(
        update.message,
        msg_text,
        reply_markup=keyboard,
    )


async def handle_resume_command_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Dispatch resume command callbacks."""
    if data.startswith(CB_RESUME_PROV_SELECT):
        await _handle_resume_provider_select(query, user_id, data, update, context)
    elif data.startswith(CB_RESUME_MODE_SELECT):
        await _handle_resume_mode_select(query, user_id, data, update, context)
    elif data.startswith(CB_RESUME_PICK):
        await _handle_pick(query, user_id, data, update, context)
    elif data.startswith(CB_RESUME_PAGE):
        await _handle_page(query, user_id, data, update, context)
    elif data == CB_RESUME_DIR_BACK:
        await _handle_directory_back(query, user_id, update, context)
    elif data == CB_RESUME_CANCEL:
        await _handle_cancel(query, context)


async def _handle_resume_provider_select(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle provider selection for cwd-scoped resume."""
    thread_id = get_thread_id(update)
    if not _resume_thread_matches(context.user_data, thread_id):
        await query.answer("Stale resume browser", show_alert=True)
        return

    provider_name = data[len(CB_RESUME_PROV_SELECT) :]
    provider_error = _resume_provider_error(provider_name)
    if provider_error:
        await query.answer(provider_error, show_alert=True)
        return

    selected_path = _resume_selected_cwd(context.user_data)
    if not selected_path or not Path(selected_path).is_dir():
        clear_resume_state(context.user_data)
        await safe_edit(query, "\u274c Directory no longer exists.")
        await query.answer("Failed")
        return

    if context.user_data is not None:
        context.user_data[RESUME_PROVIDER] = provider_name
    logger.info(
        "resume_provider_selected",
        thread_id=thread_id,
        provider=provider_name,
        cwd=selected_path,
    )
    text, keyboard = build_resume_mode_picker(selected_path, provider_name)
    await safe_edit(query, text, reply_markup=keyboard)
    await query.answer()


async def _handle_resume_mode_select(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle approval mode selection, then show filtered sessions."""
    selection = await _validate_resume_mode_selection(query, data, update, context)
    if selection is None:
        return

    provider_name = selection.provider_name
    approval_mode = selection.approval_mode
    selected_path = selection.selected_path

    sessions = await asyncio.to_thread(
        scan_resumable_sessions, provider_name, selected_path
    )
    thread_id = get_thread_id(update)
    logger.info(
        "resume_sessions_scanned",
        thread_id=thread_id,
        provider=provider_name,
        cwd=selected_path,
        count=len(sessions),
    )
    if not sessions:
        await _show_no_resumable_sessions(query, context)
        return

    session_dicts = [_entry_to_dict(s) for s in sessions]
    if context.user_data is not None:
        context.user_data[RESUME_PROVIDER] = provider_name
        context.user_data[RESUME_APPROVAL_MODE] = approval_mode
        context.user_data[RESUME_SESSIONS] = session_dicts

    keyboard = _build_resume_keyboard(session_dicts, page=0)
    await safe_edit(
        query,
        _resume_picker_text(provider_name, selected_path),
        reply_markup=keyboard,
    )
    await query.answer()


async def _validate_resume_mode_selection(
    query: CallbackQuery,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> _ResumeModeSelection | None:
    thread_id = get_thread_id(update)
    if not _resume_thread_matches(context.user_data, thread_id):
        await query.answer("Stale resume browser", show_alert=True)
        return None
    parsed = _parse_resume_mode_select(data)
    if parsed is None:
        await query.answer("Invalid mode", show_alert=True)
        return None
    provider_name, approval_mode = parsed
    error = _resume_provider_error(provider_name) or _resume_approval_mode_error(
        provider_name, approval_mode
    )
    if error:
        await query.answer(error, show_alert=True)
        return None
    selected_path = _resume_selected_cwd(context.user_data)
    if not selected_path or not Path(selected_path).is_dir():
        clear_resume_state(context.user_data)
        await safe_edit(query, "\u274c Directory no longer exists.")
        await query.answer("Failed")
        return None
    return _ResumeModeSelection(provider_name, approval_mode, selected_path)


def _resume_provider_error(provider_name: str) -> str:
    if provider_name not in _RESUME_DISCOVERY_PROVIDERS:
        return "Unsupported provider"
    provider = _resolve_resume_provider(provider_name)
    if provider is None:
        return "Unknown provider"
    if not provider.capabilities.supports_resume:
        return "Resume not supported"
    return ""


def _resume_approval_mode_error(provider_name: str, approval_mode: str) -> str:
    if approval_mode not in ("normal", "yolo"):
        return "Unknown mode"
    if approval_mode == "yolo" and not has_yolo_mode(provider_name):
        return "YOLO not supported for this provider"
    return ""


async def _show_no_resumable_sessions(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    selected_path = _resume_selected_cwd(context.user_data)
    display_path = selected_path.replace(str(Path.home()), "~") if selected_path else ""
    rows = [
        [
            InlineKeyboardButton(
                f"\U0001f4c1 {display_path or 'Change directory'}",
                callback_data=CB_RESUME_DIR_BACK,
            )
        ],
        [InlineKeyboardButton("\u2716 Cancel", callback_data=CB_RESUME_CANCEL)],
    ]
    await safe_edit(
        query,
        "\u274c No resumable sessions found for this provider and directory.",
        reply_markup=InlineKeyboardMarkup(rows),
    )
    await query.answer("No sessions")


def _parse_resume_mode_select(data: str) -> tuple[str, str] | None:
    raw = data[len(CB_RESUME_MODE_SELECT) :]
    provider_name, sep, approval_mode = raw.partition(":")
    if not sep:
        return None
    return provider_name, approval_mode.lower()


def _entry_to_dict(entry: ResumeEntry) -> dict[str, str]:
    return {
        "session_id": entry.session_id,
        "summary": entry.summary,
        "cwd": entry.cwd,
        "provider_name": entry.provider_name,
        "transcript_path": entry.transcript_path,
    }


def _resume_selected_cwd(user_data: dict | None) -> str:
    if user_data is None:
        return ""
    selected = user_data.get(RESUME_SELECTED_CWD) or user_data.get(BROWSE_PATH_KEY, "")
    return selected if isinstance(selected, str) else ""


def _resume_picker_text(provider_name: str, cwd: str) -> str:
    display_path = cwd.replace(str(Path.home()), "~")
    label, icon = _provider_label(provider_name)
    return (
        "\u23ea Select a session to resume:\n"
        f"Provider: {icon} {label}\n"
        f"Directory: `{display_path}`"
    )


async def _create_resume_window(
    user_id: int,
    thread_id: int,
    session_id: str,
    cwd: str,
    provider_name: str,
    approval_mode: str,
    transcript_path: str = "",
) -> tuple[bool, str, str, str]:
    """Create a new window with provider-specific resume args.

    Returns (success, message, window_name, window_id).
    """
    provider = get_provider_for_window("", provider_name=provider_name)
    launch_args = provider.make_launch_args(resume_id=session_id)
    launch_command = resolve_launch_command(
        provider.capabilities.name, approval_mode=approval_mode
    )
    success, message, created_wname, created_wid = await tmux_manager.create_window(
        cwd, agent_args=launch_args, launch_command=launch_command
    )
    if success:
        thread_router.bind_thread(
            user_id, thread_id, created_wid, window_name=created_wname
        )
        session_manager.set_window_cwd(created_wid, cwd)
        session_manager.set_window_provider(created_wid, provider.capabilities.name)
        session_manager.set_window_approval_mode(created_wid, approval_mode)
        await tmux_manager.stamp_pane_title(created_wid, provider.capabilities.name)
        if approval_mode == "yolo" and provider.capabilities.has_yolo_confirmation:
            from .directory_callbacks import _accept_yolo_confirmation

            await _accept_yolo_confirmation(created_wid)
        if provider.capabilities.supports_hook:
            await session_map_sync.wait_for_session_map_entry(created_wid)
        elif transcript_path:
            session_map_sync.claim_hookless_session(
                window_id=created_wid,
                session_id=session_id,
                cwd=cwd,
                transcript_path=transcript_path,
                provider_name=provider.capabilities.name,
            )
            await asyncio.to_thread(
                session_map_sync.write_hookless_session_map,
                window_id=created_wid,
                session_id=session_id,
                cwd=cwd,
                transcript_path=transcript_path,
                provider_name=provider.capabilities.name,
            )

    return success, message, created_wname, created_wid


async def _replace_bound_session_if_needed(
    user_id: int,
    thread_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> tuple[bool, str]:
    """Kill and unbind the current local session before reusing the topic."""
    old_window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if not old_window_id:
        return True, ""

    view = window_query.view_window(old_window_id)
    if is_foreign_window(old_window_id) or bool(view and view.external):
        return (
            False,
            "External session cannot be replaced automatically. Run /unbind first.",
        )

    logger.info(
        "resume_replace_start",
        user_id=user_id,
        thread_id=thread_id,
        old_window_id=old_window_id,
    )
    result = await teardown_topic_session(
        context.bot,
        actor_user_id=user_id,
        user_id=user_id,
        thread_id=thread_id,
        window_id=old_window_id,
        user_data=context.user_data,
        reason="resume_replace",
        remove_topic=False,
    )
    logger.info(
        "resume_replace_done",
        user_id=user_id,
        thread_id=thread_id,
        old_window_id=old_window_id,
        window_status=result.window_status,
        bindings_removed=result.bindings_removed,
        errors=result.errors,
    )
    if result.window_status == "failed" or result.errors:
        return False, "Could not kill the currently bound session. Check logs and try again."
    return True, ""


async def _handle_pick(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle session selection from the resume picker."""
    idx_str = data[len(CB_RESUME_PICK) :]
    try:
        idx = int(idx_str)
    except ValueError:
        await query.answer("Invalid selection", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    if (
        context.user_data
        and context.user_data.get(RESUME_THREAD_ID) is not None
        and not _resume_thread_matches(context.user_data, thread_id)
    ):
        await query.answer("Stale resume browser", show_alert=True)
        return

    stored = context.user_data.get(RESUME_SESSIONS) if context.user_data else None
    if not stored or idx < 0 or idx >= len(stored):
        await query.answer("Invalid session index", show_alert=True)
        return

    picked = stored[idx]
    session_id = picked["session_id"]
    cwd = picked.get("cwd", "")
    stored_provider = context.user_data.get(RESUME_PROVIDER, "claude") if context.user_data else "claude"
    provider_name = picked.get("provider_name") or stored_provider
    approval_mode = (
        context.user_data.get(RESUME_APPROVAL_MODE, "normal")
        if context.user_data
        else "normal"
    )
    transcript_path = picked.get("transcript_path", "")

    if not cwd or not Path(cwd).is_dir():
        await safe_edit(query, "\u274c Project directory no longer exists.")
        clear_resume_flow_state(context.user_data)
        await query.answer("Failed")
        return

    replaced, replace_message = await _replace_bound_session_if_needed(
        user_id, thread_id, context
    )
    if not replaced:
        await safe_edit(query, f"\u274c {replace_message}")
        clear_resume_flow_state(context.user_data)
        await query.answer("Failed")
        return

    success, message, created_wname, created_wid = await _create_resume_window(
        user_id,
        thread_id,
        session_id,
        cwd,
        provider_name,
        approval_mode,
        transcript_path,
    )
    if not success:
        await safe_edit(query, f"\u274c {message}")
        clear_resume_flow_state(context.user_data)
        await query.answer("Failed")
        return

    # Store group chat_id for routing
    chat = query.message.chat if query.message else None
    if chat and chat.type in ("group", "supergroup"):
        thread_router.set_group_chat_id(user_id, thread_id, chat.id)

    # Rename topic to match the window
    try:
        await context.bot.edit_forum_topic(
            chat_id=thread_router.resolve_chat_id(user_id, thread_id),
            message_thread_id=thread_id,
            name=format_topic_name_for_mode(
                created_wname, session_manager.get_approval_mode(created_wid)
            ),
        )
    except TelegramError as e:
        logger.debug("Failed to rename topic: %s", e)

    summary_short = picked.get("summary", "")[:40]
    await safe_edit(
        query,
        f"\u2705 Resuming session: {summary_short}\n\U0001f4c2 `{cwd}`",
    )
    logger.info(
        "resume_launch_done",
        user_id=user_id,
        thread_id=thread_id,
        provider=provider_name,
        approval_mode=approval_mode,
        session_id=session_id,
        new_window_id=created_wid,
        transcript_path=transcript_path,
    )
    clear_resume_flow_state(context.user_data)
    await query.answer("Resumed")


async def _handle_page(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    _update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle pagination in resume picker."""
    page_str = data[len(CB_RESUME_PAGE) :]
    try:
        page = int(page_str)
    except ValueError:
        await query.answer("Invalid page", show_alert=True)
        return

    stored = context.user_data.get(RESUME_SESSIONS) if context.user_data else None
    if not stored:
        await query.answer("No sessions available", show_alert=True)
        return

    keyboard = _build_resume_keyboard(stored, page=page)
    await safe_edit(
        query,
        _resume_picker_text(
            context.user_data.get(RESUME_PROVIDER, "claude") if context.user_data else "claude",
            _resume_selected_cwd(context.user_data),
        ),
        reply_markup=keyboard,
    )
    await query.answer()


async def _handle_directory_back(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Return from the filtered session picker to directory browsing."""
    thread_id = get_thread_id(update)
    if not _resume_thread_matches(context.user_data, thread_id):
        await query.answer("Stale resume browser", show_alert=True)
        return

    selected_path = _resume_selected_cwd(context.user_data)
    if not selected_path or not Path(selected_path).is_dir():
        clear_resume_flow_state(context.user_data)
        await safe_edit(query, "\u274c Directory no longer exists.")
        await query.answer("Failed")
        return

    msg_text, keyboard, subdirs = build_directory_browser(selected_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        context.user_data[BROWSE_PATH_KEY] = selected_path
        context.user_data[BROWSE_PAGE_KEY] = 0
        context.user_data[BROWSE_DIRS_KEY] = subdirs
        context.user_data[PENDING_THREAD_ID] = thread_id
        context.user_data[RESUME_THREAD_ID] = thread_id
        context.user_data[RESUME_SELECTED_CWD] = selected_path
        context.user_data.pop(RESUME_SESSIONS, None)
        context.user_data.pop(RESUME_PROVIDER, None)
        context.user_data.pop(RESUME_APPROVAL_MODE, None)

    logger.info(
        "resume_back_to_directory",
        user_id=user_id,
        thread_id=thread_id,
        cwd=selected_path,
    )
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer("Choose directory")


async def _handle_cancel(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle cancel in resume picker."""
    clear_resume_flow_state(context.user_data)
    await safe_edit(query, "Resume cancelled.")
    await query.answer("Cancelled")


def _clear_resume_state(user_data: dict | None) -> None:
    """Backward-compatible alias for tests/imports."""
    clear_resume_state(user_data)


# --- Registry dispatch entry point ---


@register(
    CB_RESUME_PROV_SELECT,
    CB_RESUME_MODE_SELECT,
    CB_RESUME_PICK,
    CB_RESUME_PAGE,
    CB_RESUME_DIR_BACK,
    CB_RESUME_CANCEL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_resume_command_callback(query, user.id, query.data, update, context)
