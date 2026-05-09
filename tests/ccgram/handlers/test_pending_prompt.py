from datetime import datetime, timezone

from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PendingPrompt,
    append_pending_prompt,
    clear_pending_thread,
    flush_pending_prompt_text,
    get_pending_prompt,
    get_pending_prompt_text,
    set_pending_thread,
)


def test_pending_prompt_combines_chunks() -> None:
    prompt = PendingPrompt.from_text("first")

    prompt.add_chunk("second")

    assert prompt.combined_text() == "first\n\nsecond"


def test_legacy_string_state_upgrades_in_place() -> None:
    user_data = {PENDING_THREAD_TEXT: "hello"}

    prompt = get_pending_prompt(user_data)

    assert prompt is user_data[PENDING_THREAD_TEXT]
    assert get_pending_prompt_text(user_data) == "hello"


def test_set_pending_thread_records_message_metadata() -> None:
    ts = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    message = type("Message", (), {"message_id": 123, "date": ts})()
    user_data: dict = {}

    prompt = set_pending_thread(user_data, 42, "hello", message=message)

    assert user_data[PENDING_THREAD_ID] == 42
    assert prompt.chunks[0].message_id == 123
    assert prompt.chunks[0].date == ts


def test_append_pending_prompt_preserves_existing_chunks() -> None:
    user_data: dict = {}

    append_pending_prompt(user_data, "one")
    append_pending_prompt(user_data, "two")

    assert get_pending_prompt_text(user_data) == "one\n\ntwo"


def test_clear_pending_thread_removes_thread_and_prompt() -> None:
    user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hello"}

    clear_pending_thread(user_data)

    assert PENDING_THREAD_ID not in user_data
    assert PENDING_THREAD_TEXT not in user_data


def test_flush_pending_prompt_text_returns_text_and_clears() -> None:
    user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hello"}

    text = flush_pending_prompt_text(user_data)

    assert text == "hello"
    assert PENDING_THREAD_ID not in user_data
    assert PENDING_THREAD_TEXT not in user_data
