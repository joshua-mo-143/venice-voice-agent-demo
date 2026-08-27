from __future__ import annotations

from io import BytesIO
from typing import Any
from wave import open as open_wav

import pytest

import audio


class FakeInputStream:
    def __init__(self, callback: Any, pcm: bytes = b"") -> None:
        self.callback = callback
        self.pcm = pcm
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        if self.pcm:
            self.callback(self.pcm, len(self.pcm) // 2, None, None)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeOutputStream:
    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.aborted = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        return None

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))

    def abort(self) -> None:
        self.aborted = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_pcm_to_wav_wraps_s16le() -> None:
    pcm = b"\x00\x01" * 16
    wav_bytes = audio.pcm_to_wav(pcm, 16_000)
    assert wav_bytes.startswith(b"RIFF")
    with open_wav(BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.readframes(16) == pcm


def test_cancel_during_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeInputStream(lambda *args: None)
    monkeypatch.setattr(audio, "require_audio", lambda: None)
    monkeypatch.setattr(audio, "_open_input_stream", lambda callback: stream)
    monkeypatch.setattr(
        audio,
        "_wait_for_enter_or_limit",
        lambda stopped, seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(audio.AudioError, match="cancelled"):
        audio.record_until_enter()

    assert stream.closed


def test_record_returns_in_memory_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm = b"\x01\x00" * 20

    def open_stream(callback: Any) -> FakeInputStream:
        return FakeInputStream(callback, pcm)

    monkeypatch.setattr(audio, "require_audio", lambda: None)
    monkeypatch.setattr(audio, "_open_input_stream", open_stream)
    monkeypatch.setattr(audio, "_wait_for_enter_or_limit", lambda stopped, seconds: None)

    recorded = audio.record_until_enter()
    with open_wav(BytesIO(recorded), "rb") as wav:
        assert wav.readframes(20) == pcm


def test_pcm_player_close_does_not_mask_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio, "require_audio", lambda: None)
    player = audio.PcmPlayer()
    stream = FakeOutputStream()

    def boom(_data: bytes) -> None:
        raise RuntimeError("device gone")

    stream.write = boom  # type: ignore[method-assign]
    player._stream = stream
    player._pending = b"\x00"
    with pytest.raises(RuntimeError, match="original"):
        try:
            raise RuntimeError("original")
        finally:
            player.close(raise_on_error=False)


def test_pcm_player_abort_stops_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "require_audio", lambda: None)
    player = audio.PcmPlayer()
    stream = FakeOutputStream()
    player._stream = stream
    player.abort()
    assert player._stream is None
    assert stream.aborted
    assert stream.closed
