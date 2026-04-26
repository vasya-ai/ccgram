"""User-facing /kill command for terminating a session and removing its topic."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..config import config
from ..thread_router import thread_router
from ..window_query import view_window
from .callback_data import CB_KILL_CANCEL, CB_KILL_CONFIRM
from .callback_helpers import get_thread_id
from .callback_registry import register
from .message_sender import safe_edit, safe_reply
from .session_teardown import teardown_topic_session


async def kill_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /kill — terminate the bound session and delete/close this topic."""
    user = update.effective_user
    if not user or not update.message:
        return
    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "Use /kill inside a session topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "No session bound to this topic.")
        return

    view = view_window(window_id)
    if view and view.external:
        await safe_reply(
            update.message,
            "External session cannot be killed from CCGram; use /unbind.",
        )
        return

    display = thread_router.get_display_name(window_id)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"Confirm kill {display}",
                    callback_data=f"{CB_KILL_CONFIRM}{window_id}"[:64],
                )
            ],
            [InlineKeyboardButton("Cancel", callback_data=CB_KILL_CANCEL)],
        ]
    )
    await safe_reply(
        update.message,
        f"Kill session `{display}` and delete this topic?",
        reply_markup=keyboard,
    )


@register(CB_KILL_CONFIRM, CB_KILL_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if not query or not query.data or not user:
        return

    if query.data == CB_KILL_CANCEL:
        await safe_edit(query, "Kill cancelled.")
        await query.answer("Cancelled")
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a session topic", show_alert=True)
        return

    window_id = query.data[len(CB_KILL_CONFIRM) :]
    bound_window = thread_router.get_window_for_thread(user.id, thread_id)
    if bound_window != window_id:
        await query.answer("Stale kill confirmation", show_alert=True)
        return

    view = view_window(window_id)
    if view and view.external:
        await query.answer("External session cannot be killed", show_alert=True)
        return

    await query.answer("Killing session...")
    result = await teardown_topic_session(
        context.bot,
        actor_user_id=user.id,
        user_id=user.id,
        thread_id=thread_id,
        window_id=window_id,
        user_data=context.user_data,
        reason="kill_command",
        remove_topic=True,
    )

    if result.window_status == "failed":
        await safe_edit(query, "Could not kill the session. Check logs and try again.")
    elif result.topic_status in {"failed", "no_group_chat"}:
        await safe_edit(
            query,
            "Session killed, but topic could not be deleted/closed. Run /sync later.",
        )
