import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from ccgram.agent_input_delivery import (
    UserSubmitStatus,
    _adaptive_submit_delay,
    submit_user_message,
)
from ccgram.providers.base import AgentMessage, ProviderCapabilities, StatusUpdate


class FakeProvider:
    def __init__(self, *, structured: bool = True, active_status: bool = False) -> None:
        self.active_status = active_status
        self.capabilities = ProviderCapabilities(
            name="codex",
            launch_command="codex",
            supports_structured_transcript=structured,
            supports_incremental_read=True,
        )

    def parse_transcript_line(self, line: str) -> dict | None:
        return json.loads(line)

    def read_transcript_file(
        self, file_path: str, last_offset: int
    ) -> tuple[list[dict], int]:
        entries: list[dict] = []
        with Path(file_path).open("r", encoding="utf-8") as fh:
            fh.seek(last_offset)
            for line in fh:
                entries.append(json.loads(line))
            return entries, fh.tell()

    def is_user_transcript_entry(self, entry: dict) -> bool:
        payload = entry.get("payload", {})
        return entry.get("type") == "response_item" and payload.get("role") == "user"

    def parse_history_entry(self, entry: dict) -> AgentMessage | None:
        payload = entry.get("payload", {})
        if payload.get("role") != "user":
            return None
        texts = [
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "input_text"
        ]
        return AgentMessage(
            text="\n".join(texts),
            role="user",
            content_type="text",
        )

    def parse_terminal_status(self, _pane_text: str, *, _pane_title: str = ""):
        if self.active_status:
            return StatusUpdate(raw_text="working", display_label="...working")
        return None


class FakeTmux:
    def __init__(
        self,
        transcript_path: Path,
        text: str,
        *,
        append_on_attempt: int | str | None = 1,
        pane_text: str = "",
    ) -> None:
        self.transcript_path = transcript_path
        self.text = text
        self.append_on_attempt = append_on_attempt
        self.pane_text = pane_text
        self.lock = asyncio.Lock()
        self.inserts: list[str] = []
        self.latest_text = text
        self.submit_attempts = 0
        self.max_active = 0
        self._active = 0

    async def find_window_by_id(self, window_id: str):
        return SimpleNamespace(window_id=window_id)

    def input_lock(self, window_id: str) -> asyncio.Lock:
        return self.lock

    async def _insert_literal_text_locked(self, window_id: str, text: str) -> bool:
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        self.inserts.append(text)
        self.latest_text = text
        await asyncio.sleep(0)
        self._active -= 1
        return True

    async def _submit_enter_locked(self, window_id: str) -> bool:
        self.submit_attempts += 1
        if (
            self.append_on_attempt == "all"
            or self.submit_attempts == self.append_on_attempt
        ):
            _append_user_entry(self.transcript_path, self.latest_text)
        return True

    async def capture_pane(self, window_id: str) -> str:
        return self.pane_text


def _append_user_entry(path: Path, text: str) -> None:
    entry = {
        "type": "response_item",
        "payload": {
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _patch_context(monkeypatch, tmux: FakeTmux, provider: FakeProvider):
    import ccgram.agent_input_delivery as mod

    async def run_blocking_inline(func, /, *args):
        return func(*args)

    transcript = tmux.transcript_path
    monkeypatch.setattr(mod, "tmux_manager", tmux)
    monkeypatch.setattr(mod, "_run_blocking", run_blocking_inline)
    monkeypatch.setattr(
        mod.window_query,
        "view_window",
        lambda _wid: SimpleNamespace(
            provider_name="codex",
            transcript_path=transcript,
        ),
    )
    monkeypatch.setattr(mod, "get_provider_for_window", lambda *_a, **_kw: provider)


async def test_short_message_accepted_after_first_enter(tmp_path, monkeypatch) -> None:
    text = "hello"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, text)
    _patch_context(monkeypatch, tmux, FakeProvider())

    result = await submit_user_message("@1", text, initial_delay=0, poll_interval=0)

    assert result.status == UserSubmitStatus.ACCEPTED
    assert result.verified is True
    assert tmux.submit_attempts == 1


def test_long_multiline_message_uses_adaptive_delay() -> None:
    short = _adaptive_submit_delay("hello")
    long = _adaptive_submit_delay(("hello\n" * 40) + ("x" * 2000))

    assert short == 0.5025
    assert long > short
    assert long <= 2.5


async def test_retry_enter_when_draft_fingerprint_visible(tmp_path, monkeypatch) -> None:
    text = "first part " + ("middle " * 80) + "last part"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, text, append_on_attempt=2, pane_text=text)
    _patch_context(monkeypatch, tmux, FakeProvider())

    result = await submit_user_message(
        "@1",
        text,
        initial_delay=0,
        ack_timeout=0,
        retry_ack_timeout=0,
        poll_interval=0,
        max_submit_retries=1,
    )

    assert result.status == UserSubmitStatus.ACCEPTED
    assert tmux.submit_attempts == 2
    assert tmux.inserts == [text]


async def test_no_retry_when_draft_fingerprint_absent(tmp_path, monkeypatch) -> None:
    text = "long message " * 100
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, text, append_on_attempt=None, pane_text="idle prompt")
    _patch_context(monkeypatch, tmux, FakeProvider())

    result = await submit_user_message(
        "@1",
        text,
        initial_delay=0,
        ack_timeout=0,
        poll_interval=0,
    )

    assert result.status == UserSubmitStatus.ACK_TIMEOUT
    assert tmux.submit_attempts == 1


async def test_no_retry_when_agent_status_present(tmp_path, monkeypatch) -> None:
    text = "visible draft " * 100
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, text, append_on_attempt=None, pane_text=text)
    _patch_context(monkeypatch, tmux, FakeProvider(active_status=True))

    result = await submit_user_message(
        "@1",
        text,
        initial_delay=0,
        ack_timeout=0,
        poll_interval=0,
    )

    assert result.status == UserSubmitStatus.ACK_TIMEOUT
    assert tmux.submit_attempts == 1


async def test_repeated_text_before_offset_does_not_ack(tmp_path, monkeypatch) -> None:
    text = "repeat me"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    _append_user_entry(transcript, text)
    tmux = FakeTmux(transcript, text, append_on_attempt=None, pane_text="")
    _patch_context(monkeypatch, tmux, FakeProvider())

    result = await submit_user_message(
        "@1",
        text,
        initial_delay=0,
        ack_timeout=0,
        poll_interval=0,
    )

    assert result.status == UserSubmitStatus.ACK_TIMEOUT


async def test_unstructured_provider_returns_unverified(tmp_path, monkeypatch) -> None:
    text = "hello"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, text, append_on_attempt=None)
    _patch_context(monkeypatch, tmux, FakeProvider(structured=False))

    result = await submit_user_message("@1", text, initial_delay=0)

    assert result.status == UserSubmitStatus.INJECTED_UNVERIFIED
    assert tmux.submit_attempts == 1


async def test_concurrent_submits_are_serialized(tmp_path, monkeypatch) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    tmux = FakeTmux(transcript, "one", append_on_attempt="all")
    _patch_context(monkeypatch, tmux, FakeProvider())

    await asyncio.gather(
        submit_user_message("@1", "one", initial_delay=0, poll_interval=0),
        submit_user_message("@1", "one", initial_delay=0, poll_interval=0),
    )

    assert tmux.max_active == 1
