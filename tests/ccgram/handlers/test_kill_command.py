"""Tests for session kill via sessions dashboard (two-step confirmation)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.kill_command import kill_command
from ccgram.handlers.sessions_dashboard import (
    handle_sessions_kill,
    handle_sessions_kill_confirm,
)
from ccgram.session import WindowState


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.sessions_dashboard.view_window") as mock_view,
        patch("ccgram.handlers.sessions_dashboard.thread_router") as mock_tr,
        patch("ccgram.handlers.sessions_dashboard.tmux_manager") as mock_tm,
        patch(
            "ccgram.handlers.sessions_dashboard.teardown_topic_session",
            new_callable=AsyncMock,
        ) as mock_teardown,
    ):
        mock_tr.get_display_name.side_effect = lambda wid: wid
        mock_view.side_effect = lambda wid: WindowState()
        mock_tr.get_all_thread_windows.return_value = {}
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_teardown.return_value = SimpleNamespace(
            window_status="killed",
            topic_status="deleted",
        )
        yield mock_view, mock_tr, mock_tm, mock_teardown


class TestHandleSessionsKill:
    async def test_shows_confirmation(self, _patch_deps) -> None:
        _mock_sm, mock_tr, _, _ = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"

        query = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill(query, 100, "@5")
            mock_edit.assert_called_once()
            text = mock_edit.call_args[0][1]
            assert "Kill session" in text
            assert "myproj" in text
            keyboard = mock_edit.call_args.kwargs["reply_markup"]
            data = [
                btn.callback_data for row in keyboard.inline_keyboard for btn in row
            ]
            assert any("sess:killok:" in d for d in data)


class TestHandleSessionsKillConfirm:
    async def test_delegates_to_teardown(self, _patch_deps) -> None:
        _mock_sm, mock_tr, _mock_tm, mock_teardown = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit"):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_teardown.assert_awaited_once_with(
            bot,
            actor_user_id=100,
            window_id="@5",
            reason="sessions_kill",
            remove_topic=True,
        )

    async def test_teardown_failure_reported(self, _patch_deps) -> None:
        _mock_sm, mock_tr, _mock_tm, mock_teardown = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"
        mock_teardown.return_value = SimpleNamespace(
            window_status="failed",
            topic_status="not_requested",
        )

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        text = mock_edit.call_args[0][1]
        assert "Could not kill" in text

    async def test_refreshes_dashboard_after_kill(self, _patch_deps) -> None:
        _mock_sm, mock_tr, _mock_tm, _ = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "proj"

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill_confirm(query, 100, "@5", bot)
            mock_edit.assert_called_once()
            text = mock_edit.call_args[0][1]
            assert "Killed" in text


class TestKillCommand:
    @patch("ccgram.config.Config.is_user_allowed", return_value=True)
    @patch("ccgram.handlers.kill_command.thread_router")
    async def test_kill_command_shows_confirmation(
        self, mock_tr: MagicMock, _allowed: MagicMock
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@5"
        mock_tr.get_display_name.return_value = "proj"
        update = MagicMock()
        update.effective_user.id = 100
        update.message.message_thread_id = 42

        with (
            patch("ccgram.handlers.kill_command.view_window", return_value=WindowState()),
            patch("ccgram.handlers.kill_command.safe_reply", new_callable=AsyncMock) as reply,
        ):
            await kill_command(update, MagicMock())

        text = reply.call_args[0][1]
        assert "Kill session" in text
        keyboard = reply.call_args.kwargs["reply_markup"]
        data = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
        assert any(d.startswith("kill:ok:") for d in data)

    @patch("ccgram.config.Config.is_user_allowed", return_value=True)
    @patch("ccgram.handlers.kill_command.thread_router")
    async def test_kill_command_unbound_topic(
        self, mock_tr: MagicMock, _allowed: MagicMock
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        update = MagicMock()
        update.effective_user.id = 100
        update.message.message_thread_id = 42

        with patch(
            "ccgram.handlers.kill_command.safe_reply", new_callable=AsyncMock
        ) as reply:
            await kill_command(update, MagicMock())

        assert "No session bound" in reply.call_args[0][1]
