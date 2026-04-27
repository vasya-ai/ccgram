import json
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import (
    CB_RESUME_CANCEL,
    CB_RESUME_DIR_BACK,
    CB_RESUME_MODE_SELECT,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
    CB_RESUME_PROV_SELECT,
)
from ccgram.handlers.directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
)
from ccgram.handlers.resume_command import (
    ResumeEntry,
    _build_resume_keyboard,
    handle_resume_command_callback,
    resume_command,
    scan_all_sessions,
    scan_resumable_sessions,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    RESUME_APPROVAL_MODE,
    RESUME_PROVIDER,
    RESUME_SELECTED_CWD,
    RESUME_SESSIONS,
    RESUME_THREAD_ID,
)
from ccgram.providers.pi import encode_cwd_dirname

_RC = "ccgram.handlers.resume_command"


def _make_provider(name: str = "claude", *, supports_hook: bool = False) -> MagicMock:
    provider = MagicMock()
    provider.capabilities.name = name
    provider.capabilities.supports_resume = True
    provider.capabilities.supports_hook = supports_hook
    provider.capabilities.has_yolo_confirmation = False

    def _launch_args(*, resume_id: str | None = None, **_kwargs) -> str:
        if not resume_id:
            return ""
        if name == "codex":
            return f"resume {resume_id}"
        if name == "pi":
            return f"--session {resume_id}"
        return f"--resume {resume_id}"

    provider.make_launch_args.side_effect = _launch_args
    return provider


@pytest.fixture(autouse=True)
def _inline_resume_to_thread():
    """Avoid default-executor teardown hangs in resume command unit tests."""

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        f"{_RC}.asyncio.to_thread",
        new=AsyncMock(side_effect=_run_inline),
    ):
        yield


@pytest.fixture(autouse=True)
def _default_resume_provider():
    """Keep resume launch tests off real provider/session-map side effects."""

    def _resolve_provider(_window_id: str = "", *, provider_name: str | None = None):
        return _make_provider(provider_name or "claude")

    with patch(
        f"{_RC}.get_provider_for_window",
        side_effect=_resolve_provider,
    ):
        yield


def _make_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
    thread_id: int = 42,
    text: str = "/resume",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=chat_id)
    msg = MagicMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True
    msg.is_topic_message = True
    update.message = msg
    update.callback_query = None
    return update


def _make_callback_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
    thread_id: int = 42,
    data: str = "",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=chat_id)
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = chat_id
    query.message.message_thread_id = thread_id
    query.message.chat.is_forum = True
    query.message.is_topic_message = True
    update.callback_query = query
    update.message = None
    return update


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    return ctx


def _write_codex_session(
    path,
    *,
    session_id: str = "codex-session-1",
    cwd: str = "/tmp/codex-proj",
    prompt: str = "Fix Codex resume",
    source: object = "cli",
) -> None:
    payload: dict[str, object] = {"id": session_id, "cwd": cwd}
    if source is not None:
        payload["source"] = source
    lines = [
        {"type": "session_meta", "payload": payload},
        {
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


class TestScanAllSessions:
    def test_returns_sessions_from_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Fix the bug",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "sess-1"
        assert result[0].summary == "Fix the bug"
        assert result[0].cwd == "/tmp/myproj"

    def test_returns_empty_when_projects_path_missing(self, tmp_path) -> None:
        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = tmp_path / "nonexistent"
            result = scan_all_sessions()

        assert result == []

    def test_deduplicates_by_session_id(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"

        for name in ("proj-a", "proj-b"):
            proj_dir = projects_path / name
            proj_dir.mkdir(parents=True)
            sf = proj_dir / "sess-dup.jsonl"
            sf.write_text('{"type":"summary"}\n')
            index = {
                "originalPath": f"/tmp/{name}",
                "entries": [
                    {
                        "sessionId": "sess-dup",
                        "fullPath": str(sf),
                        "projectPath": f"/tmp/{name}",
                        "summary": f"From {name}",
                    }
                ],
            }
            (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1

    def test_skips_missing_session_files(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-gone",
                    "fullPath": str(proj_dir / "nonexistent.jsonl"),
                    "projectPath": "/tmp/myproj",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert result == []

    def test_uses_session_id_as_summary_fallback(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-abc123.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "a1b2c3d4-0000-0000-0000-abc123000000",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "a1b2c3d4-000"

    def test_sorted_by_mtime_descending(self, tmp_path) -> None:
        import time

        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        old_file = proj_dir / "sess-old.jsonl"
        old_file.write_text('{"type":"summary"}\n')
        time.sleep(0.05)

        new_file = proj_dir / "sess-new.jsonl"
        new_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-old",
                    "fullPath": str(old_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Old session",
                },
                {
                    "sessionId": "sess-new",
                    "fullPath": str(new_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "New session",
                },
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 2
        assert result[0].session_id == "sess-new"
        assert result[1].session_id == "sess-old"

    def test_scans_multiple_projects(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"

        for i, name in enumerate(("proj-a", "proj-b")):
            proj_dir = projects_path / name
            proj_dir.mkdir(parents=True)
            sf = proj_dir / f"sess-{i}.jsonl"
            sf.write_text('{"type":"summary"}\n')
            index = {
                "originalPath": f"/tmp/{name}",
                "entries": [
                    {
                        "sessionId": f"sess-{i}",
                        "fullPath": str(sf),
                        "projectPath": f"/tmp/{name}",
                        "summary": f"Session {i}",
                    }
                ],
            }
            (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 2
        ids = {r.session_id for r in result}
        assert ids == {"sess-0", "sess-1"}

    def test_skips_invalid_json(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "sessions-index.json").write_text("not valid json{{{")

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert result == []

    def test_bare_jsonl_without_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "abc-123.jsonl"
        jsonl.write_text(
            '{"type":"user","cwd":"/tmp/myproj","message":{"content":[{"type":"text","text":"Fix the bug"}]}}\n'
        )

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].session_id == "abc-123"
        assert result[0].cwd == "/tmp/myproj"
        assert result[0].summary == "Fix the bug"

    def test_bare_jsonl_skips_no_cwd(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "no-cwd.jsonl"
        jsonl.write_text('{"type":"file-history-snapshot"}\n')

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert result == []

    def test_bare_jsonl_deduplicates_with_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text(
            '{"type":"user","cwd":"/tmp/myproj","message":{"content":"hi"}}\n'
        )

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "From index",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "From index"

    def test_uses_first_prompt_as_summary_fallback(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-fp.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "sess-fp",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "firstPrompt": "Implement auth",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions()

        assert len(result) == 1
        assert result[0].summary == "Implement auth"

    def test_scan_codex_sessions_reads_codex_jsonl(self, tmp_path) -> None:
        jsonl = (
            tmp_path / ".codex" / "sessions" / "2026" / "04" / "25" / "rollout.jsonl"
        )
        _write_codex_session(
            jsonl,
            session_id="codex-session-1",
            cwd="/tmp/codex-proj",
            prompt="Fix Codex resume",
        )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_all_sessions("codex")

        assert len(result) == 1
        assert result[0].session_id == "codex-session-1"
        assert result[0].cwd == "/tmp/codex-proj"
        assert result[0].summary == "Fix Codex resume"
        assert result[0].provider_name == "codex"
        assert result[0].transcript_path == str(jsonl)

    def test_scan_codex_sessions_uses_payload_id_not_filename(self, tmp_path) -> None:
        jsonl = tmp_path / ".codex" / "sessions" / "rollout-outer-name.jsonl"
        _write_codex_session(jsonl, session_id="payload-id-123")

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_all_sessions("codex")

        assert [entry.session_id for entry in result] == ["payload-id-123"]

    def test_scan_codex_sessions_skips_injected_agents_prompt(self, tmp_path) -> None:
        jsonl = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
        _write_codex_session(
            jsonl,
            session_id="codex-session-1",
            prompt="# AGENTS.md instructions for /tmp/proj\n<INSTRUCTIONS>noise",
        )
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Real user task"}
                            ],
                        },
                    }
                )
                + "\n"
            )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_all_sessions("codex")

        assert result[0].summary == "Real user task"

    def test_scan_codex_sessions_skips_subagents(self, tmp_path) -> None:
        sessions = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions / "primary.jsonl",
            session_id="primary-session",
            source="cli",
        )
        _write_codex_session(
            sessions / "subagent.jsonl",
            session_id="subagent-session",
            source={"subagent": "worker"},
        )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_all_sessions("codex")

        assert [entry.session_id for entry in result] == ["primary-session"]

    def test_scan_unknown_provider_returns_empty(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        projects_path.mkdir()
        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_all_sessions("shell")

        assert result == []


class TestScanResumableSessions:
    def test_codex_filters_by_selected_cwd(self, tmp_path) -> None:
        selected = tmp_path / "selected"
        other = tmp_path / "other"
        selected.mkdir()
        other.mkdir()
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir / "selected.jsonl",
            session_id="selected-session",
            cwd=str(selected),
            prompt="Selected prompt",
        )
        _write_codex_session(
            sessions_dir / "other.jsonl",
            session_id="other-session",
            cwd=str(other),
            prompt="Other prompt",
        )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_resumable_sessions("codex", str(selected))

        assert [entry.session_id for entry in result] == ["selected-session"]
        assert result[0].provider_name == "codex"

    def test_claude_rejects_invalid_session_id_for_resume(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)
        session_file = proj_dir / "not-a-uuid.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        index = {
            "originalPath": "/tmp/myproj",
            "entries": [
                {
                    "sessionId": "not-a-uuid",
                    "fullPath": str(session_file),
                    "projectPath": "/tmp/myproj",
                    "summary": "Invalid",
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        with patch(f"{_RC}.config") as mock_config:
            mock_config.claude_projects_path = projects_path
            result = scan_resumable_sessions("claude", "/tmp/myproj")

        assert result == []

    def test_gemini_filters_by_project_hash(self, tmp_path) -> None:
        cwd = tmp_path / "gemini-project"
        cwd.mkdir()
        resolved_cwd = str(cwd.resolve())
        project_hash = hashlib.sha256(resolved_cwd.encode()).hexdigest()
        chat_path = (
            tmp_path
            / ".gemini"
            / "tmp"
            / project_hash
            / "chats"
            / "session-gemini-1.json"
        )
        chat_path.parent.mkdir(parents=True)
        chat_path.write_text(
            json.dumps(
                {
                    "sessionId": "gemini-1",
                    "projectHash": project_hash,
                    "messages": [{"type": "user", "content": "Ask Gemini"}],
                }
            )
        )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_resumable_sessions("gemini", str(cwd))

        assert [entry.session_id for entry in result] == ["gemini-1"]
        assert result[0].summary == "Ask Gemini"
        assert result[0].provider_name == "gemini"

    def test_pi_filters_by_session_header_cwd(self, tmp_path) -> None:
        cwd = tmp_path / "pi-project"
        cwd.mkdir()
        session_dir = (
            tmp_path / ".pi" / "agent" / "sessions" / encode_cwd_dirname(str(cwd))
        )
        session_dir.mkdir(parents=True)
        transcript = session_dir / "pi-1.jsonl"
        transcript.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session",
                            "id": "pi-1",
                            "cwd": str(cwd),
                            "version": 3,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": "Ask Pi"}],
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        with patch(f"{_RC}.Path.home", return_value=tmp_path):
            result = scan_resumable_sessions("pi", str(cwd))

        assert [entry.session_id for entry in result] == ["pi-1"]
        assert result[0].summary == "Ask Pi"
        assert result[0].provider_name == "pi"


class TestBuildResumeKeyboard:
    def _sessions(self, count: int = 3) -> list[dict[str, str]]:
        return [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(count)
        ]

    def test_session_buttons(self) -> None:
        sessions = self._sessions(2)
        kb = _build_resume_keyboard(sessions)
        assert len(kb.inline_keyboard) == 4
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_RESUME_PICK}0"
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_RESUME_PICK}1"

    def test_project_header(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions)
        header = kb.inline_keyboard[0][0]
        assert "proj" in header.text
        assert header.callback_data == CB_RESUME_DIR_BACK

    def test_cancel_button_present(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions)
        nav_row = kb.inline_keyboard[-1]
        cancel = [b for b in nav_row if b.callback_data == CB_RESUME_CANCEL]
        assert len(cancel) == 1

    def test_no_prev_on_first_page(self) -> None:
        sessions = self._sessions(1)
        kb = _build_resume_keyboard(sessions, page=0)
        nav_row = kb.inline_keyboard[-1]
        prev_btns = [
            b
            for b in nav_row
            if isinstance(b.callback_data, str) and CB_RESUME_PAGE in b.callback_data
        ]
        assert len(prev_btns) == 0

    def test_next_button_on_first_page(self) -> None:
        sessions = self._sessions(10)
        kb = _build_resume_keyboard(sessions, page=0)
        nav_row = kb.inline_keyboard[-1]
        next_btns = [
            b
            for b in nav_row
            if isinstance(b.callback_data, str)
            and b.callback_data.startswith(CB_RESUME_PAGE)
        ]
        assert len(next_btns) == 1
        assert "Next" in next_btns[0].text

    def test_prev_button_on_second_page(self) -> None:
        sessions = self._sessions(10)
        kb = _build_resume_keyboard(sessions, page=1)
        nav_row = kb.inline_keyboard[-1]
        prev_btns = [b for b in nav_row if "Prev" in b.text]
        assert len(prev_btns) == 1

    def test_callback_data_truncated_to_64(self) -> None:
        sessions = [
            {"session_id": f"sess-{'x' * 60}", "summary": "Long", "cwd": "/tmp/proj"}
        ]
        kb = _build_resume_keyboard(sessions)
        for row in kb.inline_keyboard:
            for btn in row:
                if isinstance(btn.callback_data, str):
                    assert len(btn.callback_data) <= 64

    def test_grouped_by_cwd(self) -> None:
        sessions = [
            {"session_id": "s1", "summary": "A", "cwd": "/proj/a"},
            {"session_id": "s2", "summary": "B", "cwd": "/proj/b"},
        ]
        kb = _build_resume_keyboard(sessions)
        headers = [
            row[0]
            for row in kb.inline_keyboard
            if row[0].callback_data == CB_RESUME_DIR_BACK
        ]
        assert len(headers) == 2


class TestResumeCommand:
    @patch(f"{_RC}.build_directory_browser")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_starts_directory_browser(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_build: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        keyboard = MagicMock()
        mock_build.return_value = ("Directory browser", keyboard, ["src"])

        update = _make_update()
        user_data: dict = {}
        ctx = _make_context(user_data)

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert mock_safe_reply.call_args.args[1] == "Directory browser"
        assert mock_safe_reply.call_args.kwargs["reply_markup"] is keyboard
        mock_build.assert_called_once_with(str(Path.cwd()), user_id=100)
        assert user_data[STATE_KEY] == STATE_BROWSING_DIRECTORY
        assert user_data[BROWSE_PATH_KEY] == str(Path.cwd())
        assert user_data[BROWSE_PAGE_KEY] == 0
        assert user_data[BROWSE_DIRS_KEY] == ["src"]
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[RESUME_THREAD_ID] == 42

    @patch(f"{_RC}.scan_all_sessions")
    @patch(f"{_RC}.build_directory_browser")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_does_not_scan_sessions_before_provider_and_mode(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        _mock_safe_reply: AsyncMock,
        mock_build: MagicMock,
        mock_scan: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_build.return_value = ("Directory browser", MagicMock(), [])

        update = _make_update()
        ctx = _make_context({})

        await resume_command(update, ctx)

        mock_scan.assert_not_called()

    @patch(f"{_RC}.build_directory_browser")
    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    @patch(f"{_RC}.config")
    async def test_replaces_existing_resume_state_on_new_resume_command(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
        mock_build: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_build.return_value = ("Directory browser", MagicMock(), [])

        update = _make_update()
        user_data = {
            RESUME_SESSIONS: [{"session_id": "old"}],
            RESUME_PROVIDER: "codex",
        }
        ctx = _make_context(user_data)

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert RESUME_SESSIONS not in user_data
        assert RESUME_PROVIDER not in user_data
        assert user_data[RESUME_THREAD_ID] == 42

    @patch(f"{_RC}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=None)
    @patch(f"{_RC}.config")
    async def test_no_topic_rejected(
        self,
        mock_config: MagicMock,
        _mock_thread_id: MagicMock,
        mock_safe_reply: AsyncMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        update = _make_update()
        ctx = _make_context()

        await resume_command(update, ctx)

        mock_safe_reply.assert_called_once()
        assert "named topic" in mock_safe_reply.call_args.args[1]

    async def test_no_message_returns_early(self) -> None:
        update = MagicMock()
        update.message = None
        ctx = _make_context()

        await resume_command(update, ctx)


class TestResumeWizardCallbacks:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_provider_select_stores_provider_and_shows_mode_picker(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        tmp_path: Path,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PROV_SELECT}codex")
        user_data = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert user_data[RESUME_PROVIDER] == "codex"
        mock_safe_edit.assert_called_once()
        assert "Select Resume Mode" in mock_safe_edit.call_args.args[1]
        keyboard = mock_safe_edit.call_args.kwargs["reply_markup"]
        callbacks = [
            btn.callback_data for row in keyboard.inline_keyboard for btn in row
        ]
        assert f"{CB_RESUME_MODE_SELECT}codex:normal" in callbacks

    @patch(f"{_RC}.scan_resumable_sessions")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_mode_select_scans_provider_and_directory(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        mock_scan: MagicMock,
        tmp_path: Path,
    ) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text("{}\n")
        mock_scan.return_value = [
            ResumeEntry(
                "codex-session-1",
                "Fix Codex resume",
                str(tmp_path),
                "codex",
                str(transcript),
            )
        ]
        update = _make_callback_update(data=f"{CB_RESUME_MODE_SELECT}codex:normal")
        user_data = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
            RESUME_PROVIDER: "codex",
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_scan.assert_called_once_with("codex", str(tmp_path))
        assert user_data[RESUME_APPROVAL_MODE] == "normal"
        assert user_data[RESUME_SESSIONS][0]["provider_name"] == "codex"
        mock_safe_edit.assert_called_once()
        assert "Select a session" in mock_safe_edit.call_args.args[1]

    @patch(f"{_RC}.scan_resumable_sessions", return_value=[])
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_mode_select_no_sessions_keeps_resume_state_and_shows_back(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        _mock_scan: MagicMock,
        tmp_path: Path,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_MODE_SELECT}codex:normal")
        user_data = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
            RESUME_PROVIDER: "codex",
            PENDING_THREAD_ID: 42,
            STATE_KEY: STATE_BROWSING_DIRECTORY,
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert user_data[RESUME_THREAD_ID] == 42
        assert user_data[RESUME_SELECTED_CWD] == str(tmp_path)
        assert user_data[RESUME_PROVIDER] == "codex"
        assert user_data[PENDING_THREAD_ID] == 42
        mock_safe_edit.assert_called_once()
        assert "No resumable sessions" in mock_safe_edit.call_args.args[1]
        keyboard = mock_safe_edit.call_args.kwargs["reply_markup"]
        callbacks = [
            btn.callback_data for row in keyboard.inline_keyboard for btn in row
        ]
        assert CB_RESUME_DIR_BACK in callbacks
        assert CB_RESUME_CANCEL in callbacks
        query.answer.assert_called_once_with("No sessions")

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=99)
    async def test_mode_select_stale_topic_rejected(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        tmp_path: Path,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_MODE_SELECT}codex:normal")
        user_data = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once_with(
            "Stale resume browser",
            show_alert=True,
        )
        mock_safe_edit.assert_not_called()


class TestResumePickCallback:
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_creates_window_with_resume(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            PENDING_THREAD_ID: 42,
            STATE_KEY: STATE_BROWSING_DIRECTORY,
            BROWSE_PATH_KEY: "/tmp/proj",
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tm.create_window.assert_called_once_with(
            "/tmp/proj",
            agent_args="--resume a1b2c3d4-0000-0000-0000-000000000001",
            launch_command="claude",
        )
        mock_tr.bind_thread.assert_called_once_with(
            100, 42, "@5", window_name="project"
        )

    @patch(f"{_RC}.resolve_launch_command", side_effect=lambda name, **_kwargs: name)
    @patch(f"{_RC}.get_provider_for_window")
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_uses_stored_codex_provider(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_get_provider_for_window: MagicMock,
        _mock_resolve_launch: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        codex_provider = MagicMock()
        codex_provider.capabilities.name = "codex"
        codex_provider.capabilities.supports_hook = False
        codex_provider.capabilities.has_yolo_confirmation = False
        codex_provider.make_launch_args.return_value = "resume codex-session-1"
        mock_get_provider_for_window.return_value = codex_provider

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "codex-session-1",
                    "summary": "Fix Codex resume",
                    "cwd": "/tmp/codex",
                    "provider_name": "codex",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tm.create_window.assert_called_once_with(
            "/tmp/codex",
            agent_args="resume codex-session-1",
            launch_command="codex",
        )

    @patch(f"{_RC}.resolve_launch_command", side_effect=lambda name, **_kwargs: name)
    @patch(f"{_RC}.get_provider_for_window")
    @patch(f"{_RC}.session_map_sync")
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_codex_pick_claims_selected_transcript_for_new_window(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        _mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_session_map: MagicMock,
        mock_get_provider_for_window: MagicMock,
        _mock_resolve_launch: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = -100999
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()

        codex_provider = MagicMock()
        codex_provider.capabilities.name = "codex"
        codex_provider.capabilities.supports_hook = False
        codex_provider.capabilities.has_yolo_confirmation = False
        codex_provider.make_launch_args.return_value = "resume codex-session-1"
        mock_get_provider_for_window.return_value = codex_provider

        transcript_path = "/tmp/codex-session.jsonl"
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "codex-session-1",
                    "summary": "Fix Codex resume",
                    "cwd": "/tmp/codex",
                    "provider_name": "codex",
                    "transcript_path": transcript_path,
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_session_map.claim_hookless_session.assert_called_once_with(
            window_id="@5",
            session_id="codex-session-1",
            cwd="/tmp/codex",
            transcript_path=transcript_path,
            provider_name="codex",
        )
        mock_session_map.write_hookless_session_map.assert_called_once_with(
            window_id="@5",
            session_id="codex-session-1",
            cwd="/tmp/codex",
            transcript_path=transcript_path,
            provider_name="codex",
        )

    @patch(f"{_RC}.teardown_topic_session", new_callable=AsyncMock)
    @patch(f"{_RC}.is_foreign_window", return_value=False)
    @patch(f"{_RC}.window_query.view_window", return_value=None)
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_replaces_old_window_without_deleting_topic(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        _mock_view: MagicMock,
        _mock_foreign: MagicMock,
        mock_teardown: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999
        teardown_result = MagicMock()
        teardown_result.window_status = "killed"
        teardown_result.bindings_removed = 1
        teardown_result.errors = []
        mock_teardown.return_value = teardown_result

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            PENDING_THREAD_ID: 42,
            STATE_KEY: STATE_BROWSING_DIRECTORY,
            BROWSE_PATH_KEY: "/tmp/proj",
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_teardown.assert_awaited_once_with(
            ctx.bot,
            actor_user_id=100,
            user_id=100,
            thread_id=42,
            window_id="@0",
            user_data=user_data,
            reason="resume_replace",
            remove_topic=False,
        )
        mock_tm.create_window.assert_called_once()

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_invalid_cwd_fails(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/gone",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = False
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "no longer exists" in mock_safe_edit.call_args.args[1].lower()
        assert RESUME_SESSIONS not in user_data

    async def test_pick_invalid_index_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}99")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()
        assert (
            "invalid"
            in query.answer.call_args.kwargs.get(
                "text",
                query.answer.call_args.args[0] if query.answer.call_args.args else "",
            ).lower()
        )

    async def test_pick_no_sessions_stored_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        ctx = _make_context({})
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=42):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    async def test_pick_no_topic_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.get_thread_id", return_value=None):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_second_session(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}1")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000002",
                    "summary": "Add tests",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tm.create_window.assert_called_once_with(
            "/tmp/proj",
            agent_args="--resume a1b2c3d4-0000-0000-0000-000000000002",
            launch_command="claude",
        )

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_sets_group_chat_id(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -100999)

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_clears_resume_state(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            PENDING_THREAD_ID: 42,
            STATE_KEY: STATE_BROWSING_DIRECTORY,
            BROWSE_PATH_KEY: "/tmp/proj",
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert RESUME_SESSIONS not in user_data
        assert PENDING_THREAD_ID not in user_data
        assert STATE_KEY not in user_data

    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_create_window_failure(
        self,
        _mock_thread_id: MagicMock,
        mock_safe_edit: AsyncMock,
        mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(False, "Tmux error", None, None)
        )

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "Tmux error" in mock_safe_edit.call_args.args[1]
        assert RESUME_SESSIONS not in user_data

    async def test_pick_invalid_value_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PICK}notanumber")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()


class TestResumePageCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_page_shows_sessions(
        self,
        mock_safe_edit: AsyncMock,
    ) -> None:
        sessions = [
            {"session_id": f"sess-{i}", "summary": f"Session {i}", "cwd": "/tmp/proj"}
            for i in range(10)
        ]
        user_data: dict = {RESUME_SESSIONS: sessions}
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}1")
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_safe_edit.assert_called_once()
        assert "Select a session" in mock_safe_edit.call_args.args[1]

    async def test_page_invalid_number_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}abc")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()

    async def test_page_no_sessions_rejected(self) -> None:
        update = _make_callback_update(data=f"{CB_RESUME_PAGE}0")
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once()


class TestResumeDirectoryBackCallback:
    async def test_directory_header_returns_to_directory_browser(
        self,
        tmp_path: Path,
    ) -> None:
        keyboard = MagicMock()
        user_data: dict = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
            RESUME_PROVIDER: "claude",
            RESUME_APPROVAL_MODE: "normal",
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": str(tmp_path),
                },
            ],
        }
        update = _make_callback_update(data=CB_RESUME_DIR_BACK)
        ctx = _make_context(user_data)
        query = update.callback_query

        with (
            patch(f"{_RC}.get_thread_id", return_value=42),
            patch(f"{_RC}.safe_edit", new_callable=AsyncMock) as mock_safe_edit,
            patch(f"{_RC}.build_directory_browser") as mock_browser,
        ):
            mock_browser.return_value = (
                "Select Working Directory",
                keyboard,
                ["child"],
            )

            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_browser.assert_called_once_with(str(tmp_path), user_id=100)
        mock_safe_edit.assert_called_once_with(
            query,
            "Select Working Directory",
            reply_markup=keyboard,
        )
        assert user_data[STATE_KEY] == STATE_BROWSING_DIRECTORY
        assert user_data[BROWSE_PATH_KEY] == str(tmp_path)
        assert user_data[BROWSE_PAGE_KEY] == 0
        assert user_data[BROWSE_DIRS_KEY] == ["child"]
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[RESUME_THREAD_ID] == 42
        assert user_data[RESUME_SELECTED_CWD] == str(tmp_path)
        assert RESUME_PROVIDER not in user_data
        assert RESUME_APPROVAL_MODE not in user_data
        assert RESUME_SESSIONS not in user_data
        query.answer.assert_called_once_with("Choose directory")

    async def test_directory_header_stale_topic_rejected(self, tmp_path: Path) -> None:
        user_data = {
            RESUME_THREAD_ID: 42,
            RESUME_SELECTED_CWD: str(tmp_path),
        }
        update = _make_callback_update(data=CB_RESUME_DIR_BACK)
        ctx = _make_context(user_data)
        query = update.callback_query

        with (
            patch(f"{_RC}.get_thread_id", return_value=99),
            patch(f"{_RC}.safe_edit", new_callable=AsyncMock) as mock_safe_edit,
        ):
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once_with(
            "Stale resume browser",
            show_alert=True,
        )
        mock_safe_edit.assert_not_called()


class TestResumeCancelCallback:
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_cancel_clears_state(
        self,
        mock_safe_edit: AsyncMock,
    ) -> None:
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "a1b2c3d4-0000-0000-0000-000000000001",
                    "summary": "test",
                    "cwd": "/tmp",
                },
            ],
        }
        update = _make_callback_update(data=CB_RESUME_CANCEL)
        ctx = _make_context(user_data)
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        assert RESUME_SESSIONS not in user_data
        mock_safe_edit.assert_called_once()
        assert "cancelled" in mock_safe_edit.call_args.args[1].lower()

    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    async def test_cancel_answers_query(
        self,
        _mock_safe_edit: AsyncMock,
    ) -> None:
        update = _make_callback_update(data=CB_RESUME_CANCEL)
        ctx = _make_context({})
        query = update.callback_query

        await handle_resume_command_callback(query, 100, query.data, update, ctx)

        query.answer.assert_called_once_with("Cancelled")


class TestResumeProviderLaunch:
    @patch(f"{_RC}.get_provider_for_window")
    @patch(f"{_RC}.teardown_topic_session", new_callable=AsyncMock)
    @patch(f"{_RC}.is_foreign_window", return_value=False)
    @patch(f"{_RC}.window_query.view_window", return_value=None)
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_uses_selected_provider_after_replacing_bound_session(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        _mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        _mock_view: MagicMock,
        _mock_foreign: MagicMock,
        mock_teardown: AsyncMock,
        mock_gpw: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@3"
        mock_tr.resolve_chat_id.return_value = -100999
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        teardown_result = MagicMock()
        teardown_result.window_status = "killed"
        teardown_result.bindings_removed = 1
        teardown_result.errors = []
        mock_teardown.return_value = teardown_result
        mock_gpw.return_value = _make_provider("codex")

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "sess-1",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                    "provider_name": "codex",
                },
            ],
            RESUME_PROVIDER: "codex",
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_teardown.assert_awaited_once()
        mock_gpw.assert_called_once_with("", provider_name="codex")

    @patch(f"{_RC}.get_provider_for_window")
    @patch(f"{_RC}.tmux_manager")
    @patch(f"{_RC}.thread_router")
    @patch(f"{_RC}.session_manager")
    @patch(f"{_RC}.safe_edit", new_callable=AsyncMock)
    @patch(f"{_RC}.get_thread_id", return_value=42)
    async def test_pick_defaults_to_claude_provider_when_entry_has_no_provider(
        self,
        _mock_thread_id: MagicMock,
        _mock_safe_edit: AsyncMock,
        _mock_sm: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_gpw: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Window created", "project", "@5")
        )
        mock_tm.stamp_pane_title = AsyncMock()
        mock_tr.resolve_chat_id.return_value = -100999
        mock_gpw.return_value = _make_provider("claude")

        update = _make_callback_update(data=f"{CB_RESUME_PICK}0")
        user_data: dict = {
            RESUME_SESSIONS: [
                {
                    "session_id": "sess-1",
                    "summary": "Fix bug",
                    "cwd": "/tmp/proj",
                },
            ],
        }
        ctx = _make_context(user_data)
        query = update.callback_query

        with patch(f"{_RC}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            await handle_resume_command_callback(query, 100, query.data, update, ctx)

        mock_gpw.assert_called_once_with("", provider_name="claude")
        mock_gpw.return_value.make_launch_args.assert_called_once_with(
            resume_id="sess-1"
        )
