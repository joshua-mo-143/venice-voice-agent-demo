from __future__ import annotations

import io
import signal
from pathlib import Path
from typing import Any

import pytest

import audio


class FakeProcess:
    def __init__(self, *, fail_immediately: bool = False) -> None:
        self.returncode: int | None = 1 if fail_immediately else None
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"boom" if fail_immediately else b"")
        self.signals: list[int] = []
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = -sig

    def terminate(self) -> None:
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.killed = True
        self.returncode = -signal.SIGKILL

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_cancel_during_listen_deletes_temp_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40 + b"speech")
    process = FakeProcess()

    class FakeTemp:
        def __init__(self, **kwargs: Any) -> None:
            self.name = str(wav)

        def close(self) -> None:
            return None

    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio.tempfile, "NamedTemporaryFile", lambda **kwargs: FakeTemp())
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(audio, "_wait_until_recording", lambda proc, path: None)
    monkeypatch.setattr(
        audio,
        "_wait_for_enter_or_limit",
        lambda proc, seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(audio.AudioError, match="cancelled"):
        audio.record_until_enter()

    assert not wav.exists()
    assert signal.SIGINT in process.signals


def test_successful_record_deletes_temp_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "clip.wav"
    payload = b"RIFF" + b"\x00" * 40 + b"speech"
    wav.write_bytes(payload)
    process = FakeProcess()

    class FakeTemp:
        def __init__(self, **kwargs: Any) -> None:
            self.name = str(wav)

        def close(self) -> None:
            return None

    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio.tempfile, "NamedTemporaryFile", lambda **kwargs: FakeTemp())
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(audio, "_wait_until_recording", lambda proc, path: None)
    monkeypatch.setattr(audio, "_wait_for_enter_or_limit", lambda proc, seconds: None)

    recorded = audio.record_until_enter()
    assert recorded == payload
    assert not wav.exists()


def test_pcm_player_close_does_not_mask_existing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    player = audio.PcmPlayer()
    process = FakeProcess()
    process.returncode = 1
    process.stderr = io.BytesIO(b"device gone")
    player._process = process
    with pytest.raises(RuntimeError, match="original"):
        try:
            raise RuntimeError("original")
        finally:
            player.close(raise_on_error=False)


def test_pcm_player_abort_stops_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    player = audio.PcmPlayer()
    process = FakeProcess()
    player._process = process
    player.abort()
    assert player._process is None
    assert process.returncode is not None
