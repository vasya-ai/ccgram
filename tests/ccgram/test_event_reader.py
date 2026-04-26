"""Tests for event_reader — incremental events.jsonl reading."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ccgram.event_reader import read_new_events
from ccgram.providers.base import HookEvent


class _InlineAsyncFile:
    def __init__(self, path: Path, *args, **kwargs) -> None:
        self._path = Path(path)
        self._args = args
        self._kwargs = kwargs
        self._file: Any = None

    async def __aenter__(self):
        self._file = self._path.open(*self._args, **self._kwargs)
        return self

    async def __aexit__(self, *_exc_info) -> None:
        if self._file is not None:
            self._file.close()

    async def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    async def tell(self):
        return self._file.tell()

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = self._file.readline()
        if line == "":
            raise StopAsyncIteration
        return line


@pytest.fixture(autouse=True)
def _inline_aiofiles_open():
    """Avoid aiofiles threadpool startup in these unit tests."""

    with patch(
        "ccgram.event_reader.aiofiles.open",
        side_effect=lambda path, *args, **kwargs: _InlineAsyncFile(
            path, *args, **kwargs
        ),
    ):
        yield


def _write_event(path: Path, event_type: str, window_key: str, session_id: str) -> None:
    with path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "event": event_type,
                    "window_key": window_key,
                    "session_id": session_id,
                    "data": {},
                    "ts": 1234567890.0,
                }
            )
            + "\n"
        )


async def test_returns_empty_when_file_missing(tmp_path: Path) -> None:
    events, offset = await read_new_events(tmp_path / "missing.jsonl", 0)
    assert events == []
    assert offset == 0


async def test_reads_new_events_from_zero(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    _write_event(path, "SessionStart", "ccgram:@1", "sess-2")

    events, offset = await read_new_events(path, 0)
    assert len(events) == 2
    assert events[0].event_type == "Stop"
    assert events[0].window_key == "ccgram:@0"
    assert events[1].event_type == "SessionStart"
    assert offset == path.stat().st_size


async def test_reads_only_new_events_after_offset(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    _, offset_after_first = await read_new_events(path, 0)

    _write_event(path, "SessionStart", "ccgram:@1", "sess-2")
    events, offset = await read_new_events(path, offset_after_first)
    assert len(events) == 1
    assert events[0].event_type == "SessionStart"
    assert offset > offset_after_first


async def test_skips_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("\n\n")
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    path.open("a").write("\n")

    events, offset = await read_new_events(path, 0)
    assert len(events) == 1
    assert events[0].event_type == "Stop"


async def test_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n")
    _write_event(path, "Stop", "ccgram:@0", "sess-1")

    events, offset = await read_new_events(path, 0)
    assert len(events) == 1
    assert events[0].event_type == "Stop"


async def test_resets_offset_on_truncation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    file_size = path.stat().st_size

    stale_offset = file_size + 9999
    events, offset = await read_new_events(path, stale_offset)
    assert offset <= file_size


async def test_returns_hook_event_dataclass(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Notification", "ccgram:@5", "abc-123")

    events, _ = await read_new_events(path, 0)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, HookEvent)
    assert ev.event_type == "Notification"
    assert ev.window_key == "ccgram:@5"
    assert ev.session_id == "abc-123"
    assert ev.timestamp == pytest.approx(1234567890.0)
