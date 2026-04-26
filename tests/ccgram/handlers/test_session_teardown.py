from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.session_teardown import teardown_topic_session
from ccgram.session import WindowState


class TestTeardownTopicSession:
    async def test_local_window_killed_and_topics_removed(self) -> None:
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.session_teardown.thread_router") as mock_tr,
            patch("ccgram.handlers.session_teardown.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.cleanup.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_clear,
            patch("ccgram.handlers.session_teardown.session_map_sync") as mock_sms,
            patch("ccgram.handlers.session_teardown.window_store") as mock_store,
            patch(
                "ccgram.handlers.session_teardown.remove_forum_topic",
                new_callable=AsyncMock,
            ) as mock_remove_topic,
            patch(
                "ccgram.handlers.session_teardown.view_window",
                return_value=WindowState(),
            ),
        ):
            mock_tr.get_display_name.return_value = "proj"
            mock_tr.iter_thread_bindings.return_value = [
                (100, 42, "@5"),
                (200, 99, "@5"),
                (300, 10, "@9"),
            ]
            mock_tr.resolve_chat_id.side_effect = lambda uid, _tid: -1000 - uid
            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@5"))
            mock_tm.kill_window = AsyncMock(return_value=True)
            mock_store.remove_window.return_value = True
            mock_remove_topic.return_value = "deleted"

            result = await teardown_topic_session(
                bot,
                actor_user_id=100,
                window_id="@5",
                reason="test",
                remove_topic=True,
            )

        assert result.window_status == "killed"
        assert result.topic_status == "deleted"
        assert result.bindings_removed == 2
        mock_tm.kill_window.assert_awaited_once_with("@5")
        assert mock_clear.await_count == 2
        mock_clear.assert_any_await(
            100, 42, bot, None, window_id="@5", window_dead=True
        )
        mock_clear.assert_any_await(
            200, 99, bot, None, window_id="@5", window_dead=True
        )
        mock_sms.clear_session_map_entry.assert_called_once_with("@5")
        mock_store.remove_window.assert_called_once_with("@5")
        mock_tr.unbind_thread.assert_any_call(100, 42)
        mock_tr.unbind_thread.assert_any_call(200, 99)
        assert mock_remove_topic.await_count == 2

    async def test_external_window_unbinds_without_kill_or_state_removal(self) -> None:
        bot = AsyncMock()
        state = WindowState()
        state.external = True
        with (
            patch("ccgram.handlers.session_teardown.thread_router") as mock_tr,
            patch("ccgram.handlers.session_teardown.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.cleanup.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_clear,
            patch("ccgram.handlers.session_teardown.session_map_sync") as mock_sms,
            patch("ccgram.handlers.session_teardown.window_store") as mock_store,
            patch(
                "ccgram.handlers.session_teardown.view_window",
                return_value=state,
            ),
        ):
            mock_tr.get_display_name.return_value = "external"
            mock_tr.iter_thread_bindings.return_value = [
                (100, 42, "emdash-main:@1")
            ]
            mock_tr.resolve_chat_id.return_value = -100

            result = await teardown_topic_session(
                bot,
                actor_user_id=100,
                window_id="emdash-main:@1",
                reason="test",
                remove_topic=False,
            )

        assert result.window_status == "external_skipped"
        mock_tm.find_window_by_id.assert_not_called()
        mock_tm.kill_window.assert_not_called()
        mock_clear.assert_awaited_once_with(
            100,
            42,
            bot,
            None,
            window_id="emdash-main:@1",
            window_dead=False,
        )
        mock_sms.clear_session_map_entry.assert_not_called()
        mock_store.remove_window.assert_not_called()
        mock_tr.unbind_thread.assert_called_once_with(100, 42)
