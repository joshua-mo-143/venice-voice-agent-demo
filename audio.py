"""Local mic and speaker I/O via PipeWire (`pw-record` / `pw-play`).

This is not part of the Venice API. Linux-only. The API only sees a WAV
blob on the way in and raw PCM on the way out.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import TracebackType

WAV_HEADER_BYTES = 44
RECORD_RATE = 16_000
RECORD_BYTES_PER_SECOND = RECORD_RATE * 2  # mono s16
DEFAULT_PCM_RATE = 24_000
MAX_RECORD_SECONDS = 30
MAX_RECORD_BYTES = WAV_HEADER_BYTES + RECORD_BYTES_PER_SECOND * MAX_RECORD_SECONDS
RECORDER_READY_TIMEOUT = 1.5


class AudioError(RuntimeError):
    """Microphone or speaker failure."""


def require_pipewire() -> None:
    missing = [name for name in ("pw-record", "pw-play") if shutil.which(name) is None]
    if missing:
        tools = " and ".join(missing)
        raise AudioError(
            f"{tools} not found. On Arch, install PipeWire with `paru -S --needed pipewire`."
        )


def _stop_recorder(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _wait_until_recording(process: subprocess.Popen[bytes], path: Path) -> None:
    deadline = time.monotonic() + RECORDER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        try:
            if path.stat().st_size > 0:
                return
        except FileNotFoundError:
            pass
        time.sleep(0.05)


def _wait_for_enter_or_limit(
    process: subprocess.Popen[bytes], seconds: float
) -> None:
    """Block on Enter. Stop the recorder if the time cap is hit first."""
    release = threading.Event()

    def auto_stop() -> None:
        if not release.wait(seconds):
            print(
                f"\nRecording reached {int(seconds)}s. Press Enter to send.",
                flush=True,
            )
            _stop_recorder(process)

    timer = threading.Thread(target=auto_stop, daemon=True)
    timer.start()
    try:
        input()
    finally:
        release.set()


def record_until_enter() -> bytes:
    """Record 16 kHz mono WAV until Enter, a 30s cap, or cancel. Always deletes the temp file."""
    require_pipewire()
    handle = tempfile.NamedTemporaryFile(prefix="venice-voice-", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()
    process: subprocess.Popen[bytes] | None = None
    try:
        command = ["pw-record", "--channels=1", f"--rate={RECORD_RATE}", "--format=s16"]
        source = os.environ.get("AUDIO_SOURCE", "").strip()
        if source:
            command.extend(("--target", source))
        command.append(str(path))
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _wait_until_recording(process, path)
        if process.poll() is not None:
            detail = (process.stderr.read() if process.stderr else b"").decode(
                "utf-8", errors="replace"
            ).strip()
            suffix = f": {detail}" if detail else ""
            raise AudioError(f"pw-record failed{suffix}")
        try:
            _wait_for_enter_or_limit(process, MAX_RECORD_SECONDS)
        except (EOFError, KeyboardInterrupt) as exc:
            raise AudioError("Recording cancelled.") from exc
        finally:
            _stop_recorder(process)
        if process.returncode not in {0, -signal.SIGINT, -signal.SIGTERM, 130}:
            detail = (process.stderr.read() if process.stderr else b"").decode(
                "utf-8", errors="replace"
            ).strip()
            suffix = f": {detail}" if detail else ""
            raise AudioError(f"pw-record failed{suffix}")
        recorded = path.read_bytes()
        if len(recorded) > MAX_RECORD_BYTES:
            recorded = recorded[:MAX_RECORD_BYTES]
        if len(recorded) <= WAV_HEADER_BYTES:
            raise AudioError(
                "That recording was empty. Press Enter, speak, then press Enter again."
            )
        return recorded
    finally:
        path.unlink(missing_ok=True)


def _play_command(*arguments: str) -> list[str]:
    command = ["pw-play"]
    sink = os.environ.get("AUDIO_SINK", "").strip()
    if sink:
        command.extend(("--target", sink))
    command.extend(arguments)
    return command


def _process_error(process: subprocess.Popen[bytes], fallback: str) -> AudioError:
    detail = (process.stderr.read() if process.stderr else b"").decode(
        "utf-8", errors="replace"
    ).strip()
    suffix = f": {detail}" if detail else ""
    return AudioError(f"{fallback}{suffix}")


class PcmPlayer:
    """One pw-play process that accepts concatenated s16le mono PCM on stdin."""

    def __init__(self, sample_rate: int = DEFAULT_PCM_RATE) -> None:
        require_pipewire()
        if sample_rate <= 0:
            raise AudioError("PCM sample rate must be positive")
        self.sample_rate = sample_rate
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        command = _play_command(
            "--raw",
            "--channels=1",
            f"--rate={self.sample_rate}",
            "--format=s16",
            "-",
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            self.abort()
            raise AudioError("pw-play stdin is unavailable")

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise AudioError("pw-play stdin is unavailable")
        if process.poll() is not None:
            raise _process_error(process, "pw-play failed")
        try:
            process.stdin.write(pcm)
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise AudioError("pw-play closed before playback finished") from exc

    def abort(self) -> None:
        """Kill playback immediately. Does not raise."""
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

    def close(self, *, raise_on_error: bool = True) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if raise_on_error:
                raise AudioError("pw-play did not exit")
            return
        if returncode != 0 and raise_on_error:
            raise _process_error(process, "pw-play failed")

    def __enter__(self) -> PcmPlayer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(raise_on_error=exc_type is None)
