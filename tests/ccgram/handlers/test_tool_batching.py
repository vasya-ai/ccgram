import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import MessageEntity
from telegram.error import RetryAfter

from ccgram.entity_formatting import convert_to_entities
from ccgram.expandable_quote import EXPANDABLE_QUOTE_END as EXP_END
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START as EXP_START
from ccgram.handlers.message_queue import (
    _handle_content_task,
    get_or_create_queue,
    shutdown_workers,
)
from ccgram.handlers.message_task import ContentTask, MessageTask
from ccgram.handlers.tool_batch import (
    AgentBubbleSegment,
    TELEGRAM_TEXT_LIMIT,
    ToolBatch,
    ToolBatchEntry,
    _active_batches,
    clear_all_batches,
    clear_batch_for_topic,
    flush_batch,
    format_agent_pages,
    format_batch_message,
    process_tool_event,
)
from ccgram.session import (
    WindowState,
)


@pytest.fixture
def batch_env():
    with (
        patch(
            "ccgram.handlers.status_bubble.clear_status_message",
            new_callable=AsyncMock,
        ) as mock_clear,
        patch("ccgram.handlers.tool_batch.rate_limit_send_message") as mock_send,
        patch("ccgram.handlers.tool_batch.thread_router") as mock_tr,
    ):
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg
        yield AsyncMock(), mock_send, mock_clear


@pytest.fixture(autouse=True)
def _clear_batches(tmp_path, monkeypatch):
    from ccgram.handlers import tool_batch

    state_path = tmp_path / "tool_batches.json"
    monkeypatch.setattr(tool_batch, "_tool_batch_state_path", lambda: state_path)
    tool_batch._persistent_batches_loaded = False
    _active_batches.clear()
    yield
    _active_batches.clear()
    tool_batch._persistent_batches_loaded = False


def _make_tool_use(
    window_id: str = "@0",
    tool_use_id: str = "tu1",
    text: str = "Read src/foo.py",
    tool_name: str | None = "Read",
    thread_id: int | None = 10,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_use",
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        thread_id=thread_id,
    )


def _make_tool_result(
    tool_use_id: str | None = "tu1",
    text: str = "42 lines",
    thread_id: int | None = 10,
    window_id: str = "@0",
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_result",
        tool_use_id=tool_use_id,
        thread_id=thread_id,
    )


class TestFormatBatchMessage:
    def test_single_bubble_shape(self) -> None:
        result = format_batch_message(
            [
                ToolBatchEntry("t1", "Read src/a.py", "10 lines", tool_name="Read"),
                ToolBatchEntry("t2", "Edit src/a.py", tool_name="Edit"),
                ToolBatchEntry("t3", "Bash make test", tool_name="Bash"),
            ]
        )

        assert result.startswith(f"{EXP_START}Tools\n")
        assert result.endswith(EXP_END)
        plain, entities = convert_to_entities(result)
        assert plain.startswith("Tools\n")
        assert any(e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE for e in entities)
        assert '📖 Read: "src/a.py" ✓' in result
        assert '✏️ Edit: "src/a.py" ↻' in result
        assert '⚡ Bash: "make test" ↻' in result
        assert "10 lines" not in result

    def test_paginates_entries_oldest_first_under_telegram_limit(self) -> None:
        entries = [
            ToolBatchEntry(
                f"t{i}",
                f"Bash run-{i}-" + ("x" * 120),
                tool_name="Bash",
            )
            for i in range(100)
        ]

        pages = format_agent_pages([AgentBubbleSegment("tools", entries=entries)])

        assert len(pages) > 1
        assert all(len(page) <= TELEGRAM_TEXT_LIMIT for page in pages)
        assert "earlier tools" not in "\n".join(pages)
        assert "run-0" in pages[0]
        assert "run-99" in pages[-1]


class TestBatchDataStructures:
    def test_tool_batch_entry_defaults(self) -> None:
        entry = ToolBatchEntry(tool_use_id="t1", tool_use_text="Read x")
        assert entry.result_text is None
        assert entry.status == "pending"
        assert entry.summary == "x"

    def test_tool_batch_defaults(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        assert batch.entries == []
        assert batch.telegram_msg_id is None
        assert batch.total_length == 0

    def test_batch_entry_accumulation_keeps_all_entries(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        for i in range(20):
            entry = ToolBatchEntry(f"t{i}", f"Read file{i}.py")
            batch.entries.append(entry)
            batch.total_length += len(entry.tool_use_text)
        assert len(batch.entries) == 20
        assert batch.total_length == sum(len(f"Read file{i}.py") for i in range(20))


class TestWindowStateLegacyBatchMode:
    def test_legacy_batch_mode_is_ignored(self) -> None:
        ws = WindowState.from_dict(
            {"session_id": "s1", "cwd": "/tmp", "batch_mode": "verbose"}
        )
        assert "batch_mode" not in ws.to_dict()


class TestProcessBatchTask:
    async def test_tool_use_creates_silent_bubble(self, batch_env) -> None:
        bot, mock_send, mock_clear = batch_env

        await process_tool_event(bot, 1, _make_tool_use())

        bkey = (1, 10)
        assert bkey in _active_batches
        batch = _active_batches[bkey]
        assert len(batch.entries) == 1
        assert batch.entries[0].tool_use_id == "tu1"
        assert batch.telegram_msg_id == 100
        mock_clear.assert_awaited_once()
        mock_send.assert_awaited_once()
        assert mock_send.await_args.kwargs["disable_notification"] is True

    async def test_many_tool_calls_keep_one_telegram_message(self, batch_env) -> None:
        bot, mock_send, _ = batch_env

        for i in range(20):
            await process_tool_event(
                bot,
                1,
                _make_tool_use(tool_use_id=f"tu{i}", text=f"Bash command {i}", tool_name="Bash"),
            )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 20
        assert batch.telegram_msg_id == 100
        mock_send.assert_awaited_once()
        assert bot.edit_message_text.await_count == 19

    async def test_overflow_creates_ordered_paginated_bubbles(
        self, batch_env
    ) -> None:
        bot, mock_send, _ = batch_env

        for i in range(90):
            await process_tool_event(
                bot,
                1,
                _make_tool_use(
                    tool_use_id=f"tu{i}",
                    text=f"Bash command-{i}-" + ("x" * 120),
                    tool_name="Bash",
                ),
            )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 90
        assert batch.telegram_msg_id == 100
        assert mock_send.await_count > 1
        rendered_pages = batch.rendered_pages
        assert all(len(page) <= TELEGRAM_TEXT_LIMIT for page in rendered_pages)
        assert "command-0" in rendered_pages[0]
        assert "command-89" in rendered_pages[-1]
        assert "earlier tools" not in "\n".join(rendered_pages)

    async def test_tool_result_updates_status_without_result_snippet(
        self, batch_env
    ) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_name="Read"))

        await process_tool_event(bot, 1, _make_tool_result(text="42 lines"))

        batch = _active_batches[(1, 10)]
        assert batch.entries[0].status == "success"
        edited_text = bot.edit_message_text.await_args.kwargs["text"]
        assert 'Read: "src/foo.py" ✓' in edited_text
        assert "42 lines" not in edited_text

    async def test_tool_result_marks_error(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot,
            1,
            _make_tool_use(text="Bash make test", tool_name="Bash"),
        )

        await process_tool_event(bot, 1, _make_tool_result(text="FAILED test_foo"))

        batch = _active_batches[(1, 10)]
        assert batch.entries[0].status == "error"
        edited_text = bot.edit_message_text.await_args.kwargs["text"]
        assert 'Bash: "make test" ❌' in edited_text
        assert "FAILED test_foo" not in edited_text

    async def test_assistant_text_between_tool_groups_keeps_one_batch(
        self, batch_env
    ) -> None:
        bot, mock_send, _ = batch_env
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()

        await process_tool_event(
            bot,
            1,
            _make_tool_use(tool_use_id="tu1", text="Bash first", tool_name="Bash"),
        )
        await _handle_content_task(
            bot,
            1,
            ContentTask(window_id="@0", parts=("assistant update",), thread_id=10),
            queue,
            lock,
        )
        await process_tool_event(
            bot,
            1,
            _make_tool_use(tool_use_id="tu2", text="Bash second", tool_name="Bash"),
        )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 2
        assert [segment.kind for segment in batch.segments] == ["tools", "text", "tools"]
        assert batch.telegram_msg_id == 100
        mock_send.assert_awaited_once()
        edited_text = bot.edit_message_text.await_args.kwargs["text"]
        assert 'Bash: "first" ↻' in edited_text
        assert "assistant update" in edited_text
        assert 'Bash: "second" ↻' in edited_text

    async def test_tool_result_no_matching_entry_updates_pending_without_flushing(
        self, batch_env
    ) -> None:
        bot, _, _ = batch_env
        with patch(
            "ccgram.handlers.tool_batch.flush_batch", new_callable=AsyncMock
        ) as mock_flush:
            await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
            task = _make_tool_result(tool_use_id="tu_unknown")
            followup = await process_tool_event(bot, 1, task)

        mock_flush.assert_not_awaited()
        assert followup is None
        assert (1, 10) in _active_batches
        assert _active_batches[(1, 10)].entries[0].status == "success"

    async def test_multiple_unmatched_results_update_pending_entries_in_order(
        self, batch_env
    ) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
        await process_tool_event(
            bot, 1, _make_tool_use(tool_use_id="tu2", text="Bash second", tool_name="Bash")
        )

        await process_tool_event(bot, 1, _make_tool_result(tool_use_id="other1"))
        await process_tool_event(
            bot,
            1,
            _make_tool_result(tool_use_id="other2", text="exit code 1"),
        )

        batch = _active_batches[(1, 10)]
        assert [entry.status for entry in batch.entries] == ["success", "error"]

    async def test_tool_result_no_active_batch_falls_through(self, batch_env) -> None:
        bot, _, _ = batch_env
        task = _make_tool_result(tool_use_id="tu1", text="result text")
        result = await process_tool_event(bot, 1, task)
        assert result == task

    async def test_tool_result_none_tool_use_id_is_suppressed_with_active_batch(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
        task = _make_tool_result(tool_use_id=None, text="result text")
        result = await process_tool_event(bot, 1, task)
        assert result is None
        assert len(_active_batches[(1, 10)].entries) == 1

    async def test_different_window_flushes_old_batch(self, batch_env) -> None:
        bot, _, _ = batch_env
        with patch(
            "ccgram.handlers.tool_batch.flush_batch", new_callable=AsyncMock
        ) as mock_flush:
            await process_tool_event(bot, 1, _make_tool_use(window_id="@0"))
            await process_tool_event(
                bot, 1, _make_tool_use(window_id="@1", tool_use_id="tu2")
            )
        mock_flush.assert_awaited_once()
        assert _active_batches[(1, 10)].window_id == "@1"

    async def test_lost_edit_target_reclaims_silent_bubble(self, batch_env) -> None:
        bot, mock_send, _ = batch_env
        first_msg = MagicMock()
        first_msg.message_id = 100
        second_msg = MagicMock()
        second_msg.message_id = 101
        mock_send.side_effect = [first_msg, second_msg]

        with patch(
            "ccgram.handlers.tool_batch.edit_with_fallback",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
            await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu2"))

        batch = _active_batches[(1, 10)]
        assert batch.telegram_msg_id == 101
        assert mock_send.await_count == 2
        assert mock_send.await_args.kwargs["disable_notification"] is True

    async def test_persisted_batch_after_restart_edits_existing_message(
        self, batch_env
    ) -> None:
        from ccgram.handlers import tool_batch

        bot, mock_send, _ = batch_env
        await process_tool_event(
            bot,
            1,
            _make_tool_use(tool_use_id="tu1", text="Bash first", tool_name="Bash"),
        )

        _active_batches.clear()
        tool_batch._persistent_batches_loaded = False
        await process_tool_event(
            bot,
            1,
            _make_tool_use(tool_use_id="tu2", text="Bash second", tool_name="Bash"),
        )

        batch = _active_batches[(1, 10)]
        assert batch.telegram_msg_id == 100
        assert [entry.tool_use_id for entry in batch.entries] == ["tu1", "tu2"]
        mock_send.assert_awaited_once()
        assert bot.edit_message_text.await_args.kwargs["message_id"] == 100
        edited_text = bot.edit_message_text.await_args.kwargs["text"]
        assert 'Bash: "first" ↻' in edited_text
        assert 'Bash: "second" ↻' in edited_text

    async def test_flush_removes_persisted_batch(self, batch_env) -> None:
        from ccgram.handlers import tool_batch

        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use())
        state_path = tool_batch._tool_batch_state_path()
        assert state_path.exists()

        await flush_batch(bot, 1, 10)

        assert not state_path.exists()

    async def test_shutdown_clear_keeps_persisted_batch_for_restart(
        self, batch_env
    ) -> None:
        from ccgram.handlers import tool_batch

        bot, mock_send, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
        state_path = tool_batch._tool_batch_state_path()

        clear_all_batches()
        assert (1, 10) not in _active_batches
        assert state_path.exists()

        tool_batch._persistent_batches_loaded = False
        await process_tool_event(
            bot,
            1,
            _make_tool_use(tool_use_id="tu2", text="Read src/bar.py"),
        )

        assert [entry.tool_use_id for entry in _active_batches[(1, 10)].entries] == [
            "tu1",
            "tu2",
        ]
        mock_send.assert_awaited_once()


class TestHandleContentTask:
    @patch("ccgram.handlers.message_queue.process_tool_event", new_callable=AsyncMock)
    async def test_batch_eligible_routes_to_batch(self, mock_batch) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="tool_use",
            window_id="@0",
            parts=("Read x",),
        )
        mock_batch.return_value = None
        extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_batch.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.process_tool_event", new_callable=AsyncMock)
    async def test_tool_event_routes_to_ordered_bubble(self, mock_tool_event) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="tool_use",
            window_id="@0",
            parts=("Read x",),
        )
        extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_tool_event.assert_awaited_once_with(bot, 1, task)

    @patch("ccgram.handlers.message_queue.flush_if_active", new_callable=AsyncMock)
    @patch("ccgram.handlers.message_queue.process_agent_message", new_callable=AsyncMock)
    async def test_assistant_text_keeps_active_batch(self, mock_agent, mock_flush) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="text",
            window_id="@0",
            parts=("Hello",),
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_not_awaited()
        mock_agent.assert_awaited_once_with(bot, 1, task)

    @patch("ccgram.handlers.message_queue.flush_if_active", new_callable=AsyncMock)
    @patch("ccgram.handlers.message_queue.process_agent_message", new_callable=AsyncMock)
    async def test_final_text_finishes_agent_bubble(self, mock_agent, mock_flush) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="text",
            window_id="@0",
            parts=("Done",),
            phase="final_answer",
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_not_awaited()
        mock_agent.assert_awaited_once_with(bot, 1, task)


class TestFlushBatch:
    @patch("ccgram.handlers.tool_batch.thread_router")
    async def test_flush_removes_batch(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 10)] = ToolBatch(
            window_id="@0",
            thread_id=10,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 10)
        assert (1, 10) not in _active_batches

    async def test_flush_noop_when_no_batch(self) -> None:
        bot = AsyncMock()
        await flush_batch(bot, 1, 10)

    @patch("ccgram.handlers.tool_batch.thread_router")
    async def test_flush_edits_final_message(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[
                ToolBatchEntry("t1", "Read a.py", "10 lines"),
                ToolBatchEntry("t2", "Edit a.py"),
            ],
            telegram_msg_id=200,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        bot.edit_message_text.assert_awaited()

    @patch("ccgram.handlers.tool_batch.thread_router")
    @patch("ccgram.handlers.tool_batch.rate_limit_send_message")
    async def test_flush_sends_when_no_telegram_msg_id(
        self, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=None,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        mock_send.assert_awaited_once()
        assert mock_send.await_args.kwargs["disable_notification"] is True
        assert (1, 0) not in _active_batches

    @patch("ccgram.handlers.tool_batch.thread_router")
    @patch("ccgram.handlers.tool_batch.rate_limit_send_message")
    async def test_flush_reclaims_when_edit_target_lost(
        self, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=200,
        )

        bot = AsyncMock()
        with patch(
            "ccgram.handlers.tool_batch.edit_with_fallback",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await flush_batch(bot, 1, 0)

        mock_send.assert_awaited_once()
        assert mock_send.await_args.kwargs["disable_notification"] is True
        assert (1, 0) not in _active_batches

    async def test_flush_empty_entries_noop(self) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0, entries=[])
        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        assert (1, 0) not in _active_batches
        bot.edit_message_text.assert_not_awaited()


class TestBatchIsolation:
    async def test_different_threads_separate_batches(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=20, tool_use_id="tu2")
        )

        assert (1, 10) in _active_batches
        assert (1, 20) in _active_batches
        assert len(_active_batches[(1, 10)].entries) == 1
        assert len(_active_batches[(1, 20)].entries) == 1

    async def test_different_users_same_thread_separate_batches(
        self, batch_env
    ) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await process_tool_event(
            bot, 2, _make_tool_use(thread_id=10, tool_use_id="tu2")
        )

        assert (1, 10) in _active_batches
        assert (2, 10) in _active_batches
        assert _active_batches[(1, 10)].entries[0].tool_use_id == "tu1"
        assert _active_batches[(2, 10)].entries[0].tool_use_id == "tu2"


class TestShutdownClearsBatches:
    async def test_shutdown_clears_active_batches(self) -> None:
        await shutdown_workers()
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0)
        _active_batches[(2, 5)] = ToolBatch(window_id="@1", thread_id=5)
        await shutdown_workers()
        assert len(_active_batches) == 0


class TestQueueWorkerRetryAfter:
    @patch("ccgram.handlers.message_queue.asyncio.sleep", new_callable=AsyncMock)
    @patch("ccgram.handlers.message_queue._handle_content_task", new_callable=AsyncMock)
    async def test_retry_after_retries_same_task(self, mock_handle, mock_sleep) -> None:
        await shutdown_workers()
        mock_handle.side_effect = [RetryAfter(1), 0]

        bot = AsyncMock()
        queue = get_or_create_queue(bot, 1)
        queue.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("hello",),
                content_type="text",
                thread_id=10,
            )
        )

        try:
            await asyncio.wait_for(queue.join(), timeout=1)
            assert mock_handle.await_count == 2
            mock_sleep.assert_awaited_once()
        finally:
            await shutdown_workers()


class TestTopicCleanupClearsBatch:
    def test_clear_batch_for_topic(self) -> None:
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)
        clear_batch_for_topic(1, 10)
        assert (1, 10) not in _active_batches

    def test_clear_batch_for_topic_noop(self) -> None:
        clear_batch_for_topic(1, 999)

    def test_clear_batch_none_thread(self) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0)
        clear_batch_for_topic(1, None)
        assert (1, 0) not in _active_batches


class TestDefensiveElseBranch:
    @patch("ccgram.handlers.tool_batch.thread_router")
    async def test_unexpected_content_type_routes_to_normal(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        bot = AsyncMock()
        task = ContentTask(
            window_id="@0",
            parts=("hello",),
            content_type="text",
            thread_id=10,
        )
        result = await process_tool_event(bot, 1, task)
        assert result == task
