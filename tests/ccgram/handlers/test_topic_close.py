"""Tests for FORUM_TOPIC_CLOSED handler (terminate bound session)."""

from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(thread_id: int = 42, user_id: int = 1) -> MagicMock:
    """Create a mock Update for FORUM_TOPIC_CLOSED."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.message_thread_id = thread_id
    return update


_PATCH_ALLOWED = patch("ccgram.config.Config.is_user_allowed", return_value=True)


class TestTopicClosedHandler:
    @_PATCH_ALLOWED
    @patch(
        "ccgram.handlers.topic_lifecycle.teardown_topic_session",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topic_lifecycle.thread_router")
    async def test_unbinds_bound_topic(
        self, mock_tr: MagicMock, mock_teardown: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.handlers.topic_lifecycle import topic_closed_handler

        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tr.get_display_name.return_value = "my-project"

        update = _make_update()
        ctx = MagicMock()
        await topic_closed_handler(update, ctx)

        mock_tr.get_window_for_thread.assert_called_once_with(1, 42)
        mock_teardown.assert_awaited_once_with(
            ctx.bot,
            actor_user_id=1,
            user_id=1,
            thread_id=42,
            window_id="@0",
            user_data=ctx.user_data,
            reason="telegram_topic_closed",
            remove_topic=False,
        )

    @_PATCH_ALLOWED
    @patch(
        "ccgram.handlers.topic_lifecycle.teardown_topic_session",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topic_lifecycle.thread_router")
    async def test_skips_unbound_topic(
        self, mock_tr: MagicMock, mock_teardown: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.handlers.topic_lifecycle import topic_closed_handler

        mock_tr.get_window_for_thread.return_value = None

        update = _make_update()
        await topic_closed_handler(update, MagicMock())

        mock_teardown.assert_not_called()

    @patch(
        "ccgram.handlers.topic_lifecycle.teardown_topic_session",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topic_lifecycle.thread_router")
    async def test_skips_disallowed_user(
        self, mock_tr: MagicMock, mock_teardown: AsyncMock
    ) -> None:
        from ccgram.handlers.topic_lifecycle import topic_closed_handler

        update = _make_update()
        with patch("ccgram.config.Config.is_user_allowed", return_value=False):
            await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_teardown.assert_not_called()

    @_PATCH_ALLOWED
    @patch(
        "ccgram.handlers.topic_lifecycle.teardown_topic_session",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topic_lifecycle.thread_router")
    async def test_skips_general_topic(
        self, mock_tr: MagicMock, mock_teardown: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.handlers.topic_lifecycle import topic_closed_handler

        update = MagicMock()
        update.effective_user.id = 1
        update.message.message_thread_id = 1

        await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_teardown.assert_not_called()

    @_PATCH_ALLOWED
    @patch(
        "ccgram.handlers.topic_lifecycle.teardown_topic_session",
        new_callable=AsyncMock,
    )
    @patch("ccgram.handlers.topic_lifecycle.thread_router")
    async def test_skips_no_thread_id(
        self, mock_tr: MagicMock, mock_teardown: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.handlers.topic_lifecycle import topic_closed_handler

        update = MagicMock()
        update.effective_user.id = 1
        update.message.message_thread_id = None

        await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_teardown.assert_not_called()
