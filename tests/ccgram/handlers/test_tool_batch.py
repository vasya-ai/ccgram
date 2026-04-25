import ast
import inspect
import os

import pytest

from ccgram.handlers.message_task import ContentTask
from ccgram.handlers.tool_batch import (
    TELEGRAM_TEXT_LIMIT,
    TOOL_BUBBLE_TITLE,
    ToolBatch,
    ToolBatchEntry,
    _batch_result_prefix,
    _extract_task_create_title,
    _status_from_result_text,
    flush_if_active,
    format_batch_message,
    is_batch_eligible,
    process_tool_event,
)


class TestFormatBatchMessage:
    def test_pending_success_error_glyphs(self) -> None:
        entries = [
            ToolBatchEntry("t1", "Read src/foo.py", tool_name="Read"),
            ToolBatchEntry("t2", "Bash make test", "23 passed", tool_name="Bash"),
            ToolBatchEntry("t3", "Bash bad", "FAILED test_foo", tool_name="Bash"),
        ]

        result = format_batch_message(entries)

        assert result.startswith("```\nTools\n")
        assert '📖 Read: "src/foo.py" ↻' in result
        assert '⚡ Bash: "make test" ✓' in result
        assert '⚡ Bash: "bad" ❌' in result
        assert "23 passed" not in result
        assert "FAILED test_foo" not in result

    def test_ccbot_style_markdown_summary(self) -> None:
        entries = [
            ToolBatchEntry("t1", "📖 **Read** `/tmp/a.py`"),
            ToolBatchEntry("t2", "**Edit** `src/app.py`"),
            ToolBatchEntry("t3", "**Bash** `npm test`"),
        ]

        result = format_batch_message(entries)

        assert '📖 Read: "/tmp/a.py" ↻' in result
        assert '✏️ Edit: "src/app.py" ↻' in result
        assert '⚡ Bash: "npm test" ↻' in result

    def test_home_path_abbreviation_and_one_line_summary(self) -> None:
        home = os.path.expanduser("~")
        entry = ToolBatchEntry("t1", f"Read {home}/project/file.py\nnext line")

        result = format_batch_message([entry])

        assert '~/project/file.py next line' in result
        assert home not in result

    def test_mcp_style_tool_uses_mcp_icon_and_short_name(self) -> None:
        entry = ToolBatchEntry(
            "t1",
            "**mcp__codex_apps__github._fetch_pr** `openai/repo#12`",
            tool_name="mcp__codex_apps__github._fetch_pr",
        )

        result = format_batch_message([entry])

        assert '🔌 Fetch Pr: "openai/repo#12" ↻' in result

    def test_provider_aware_title(self) -> None:
        result = format_batch_message(
            [ToolBatchEntry("t1", "Read x", tool_name="Read")],
            provider_label="Codex",
        )

        assert result.startswith("```\nCodex Tools\n")

    def test_below_limit_renders_all_tools(self) -> None:
        entries = [
            ToolBatchEntry(f"t{i}", f"Read file{i}.py", tool_name="Read")
            for i in range(4)
        ]

        result = format_batch_message(entries)

        assert "earlier tools" not in result
        for i in range(4):
            assert f'file{i}.py' in result
        assert len(result) <= TELEGRAM_TEXT_LIMIT

    def test_above_limit_hides_oldest_tools(self) -> None:
        entries = [
            ToolBatchEntry(
                f"t{i}",
                f"Bash command-{i}-" + ("x" * 120),
                tool_name="Bash",
            )
            for i in range(80)
        ]

        result = format_batch_message(entries)

        assert len(result) <= TELEGRAM_TEXT_LIMIT
        assert "earlier tools" in result
        assert "command-79" in result
        assert "command-0" not in result

    def test_pathological_single_entry_is_truncated_not_split(self) -> None:
        entry = ToolBatchEntry("t1", "Bash short", tool_name="Bash")
        entry.summary = "x" * (TELEGRAM_TEXT_LIMIT * 2)

        result = format_batch_message([entry])

        assert len(result) <= TELEGRAM_TEXT_LIMIT
        assert result.count("```") == 2
        assert result.endswith("```")
        assert "…" in result


class TestExtractTaskCreateTitle:
    def test_markdown_format(self) -> None:
        entry = ToolBatchEntry(
            tool_use_id="t1",
            tool_use_text="**TaskCreate** `Build the widget`",
        )
        assert _extract_task_create_title(entry) == "Build the widget"

    def test_plain_format(self) -> None:
        entry = ToolBatchEntry(
            tool_use_id="t1",
            tool_use_text="TaskCreate Build the widget",
        )
        assert _extract_task_create_title(entry) == "Build the widget"

    def test_empty_text(self) -> None:
        entry = ToolBatchEntry(tool_use_id="t1", tool_use_text="")
        assert _extract_task_create_title(entry) == ""


class TestIsBatchEligible:
    def _make_task(
        self, content_type: str = "text", window_id: str = "@0"
    ) -> ContentTask:
        return ContentTask(
            window_id=window_id,
            parts=("hello",),
            content_type=content_type,  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize("content_type", ["tool_use", "tool_result"])
    def test_tool_types_eligible_with_batched_window(
        self, content_type: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.handlers import tool_batch

        monkeypatch.setattr(tool_batch, "get_batch_mode", lambda _wid: "batched")
        task = self._make_task(content_type=content_type)
        assert is_batch_eligible(task) is True

    @pytest.mark.parametrize("content_type", ["text", "thinking", "status"])
    def test_non_tool_types_not_eligible(
        self, content_type: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.handlers import tool_batch

        monkeypatch.setattr(tool_batch, "get_batch_mode", lambda _wid: "batched")
        task = self._make_task(content_type=content_type)
        assert is_batch_eligible(task) is False

    def test_not_eligible_when_batch_mode_verbose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.handlers import tool_batch

        monkeypatch.setattr(tool_batch, "get_batch_mode", lambda _wid: "verbose")
        task = self._make_task(content_type="tool_use")
        assert is_batch_eligible(task) is False

    def test_window_id_derived_from_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ccgram.handlers import tool_batch

        captured: list[str] = []

        def capture_get_batch_mode(wid: str) -> str:
            captured.append(wid)
            return "batched"

        monkeypatch.setattr(tool_batch, "get_batch_mode", capture_get_batch_mode)
        task = self._make_task(content_type="tool_use", window_id="@7")
        is_batch_eligible(task)
        assert captured == ["@7"]


class TestBatchResultStatus:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("All tests passed", "success"),
            ("success", "success"),
            ("exit code 0", "success"),
            ("error: file not found", "error"),
            ("FAILED test_foo", "error"),
            ("exit code 1", "error"),
            ("⏹ Interrupted", "error"),
        ],
    )
    def test_status_selection(self, text: str, expected: str) -> None:
        assert _status_from_result_text(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("All tests passed", "\u2705"),
            ("error: file not found", "\u274c"),
            ("⏹ Interrupted", "\u274c"),
            ("42 lines", "\u23bf"),
        ],
    )
    def test_legacy_prefix_selection(self, text: str, expected: str) -> None:
        assert _batch_result_prefix(text) == expected


class TestBatchDataStructures:
    def test_tool_batch_entry_defaults(self) -> None:
        entry = ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        assert entry.tool_result_text is None
        assert entry.tool_name == "Read"
        assert entry.summary == "foo.py"
        assert entry.status == "pending"

    def test_tool_batch_entry_result_sets_status(self) -> None:
        entry = ToolBatchEntry("t1", "Bash make test", "exit code 1")
        assert entry.status == "error"
        assert entry.result_text == "exit code 1"

    def test_tool_batch_defaults(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=42)
        assert batch.entries == []
        assert batch.telegram_msg_id is None
        assert batch.total_length == 0

    def test_constants(self) -> None:
        assert TOOL_BUBBLE_TITLE == "Tools"
        assert TELEGRAM_TEXT_LIMIT == 4096


class TestProcessToolEventSignature:
    def test_accepts_content_task_and_returns_optional(self) -> None:
        sig = inspect.signature(process_tool_event)
        params = list(sig.parameters.values())
        assert params[2].name == "task"
        assert params[2].annotation == "ContentTask"
        assert sig.return_annotation == "ContentTask | None"

    def test_flush_if_active_exists_and_accepts_content_task(self) -> None:
        sig = inspect.signature(flush_if_active)
        params = list(sig.parameters.values())
        assert params[2].name == "task"
        assert params[2].annotation == "ContentTask"


class TestNoImportFromMessageQueue:
    def test_no_import_from_message_queue(self) -> None:
        import ccgram.handlers.tool_batch as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "message_queue" in node.module
            ):
                violations.append(f"line {node.lineno}: from {node.module} import ...")
        assert violations == [], f"tool_batch imports from message_queue: {violations}"
