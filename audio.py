"""Local mic and speaker I/O via PortAudio (`sounddevice`).

This is not part of the Venice API. It works on macOS, Windows, and Linux.
The API only sees a WAV blob on the way in and raw PCM on the way out.
"""

from __future__ import annotations

import os
import threading
import wave
from collections.abc import Callable
from io import BytesIO
from types import TracebackType
from typing import Any

RECORD_RATE = 16_000
RECORD_BYTES_PER_SECOND = RECORD_RATE * 2  # mono s16
DEFAULT_PCM_RATE = 24_000
MAX_RECORD_SECONDS = 30
MAX_RECORD_PCM_BYTES = RECORD_BYTES_PER_SECOND * MAX_RECORD_SECONDS


class AudioError(RuntimeError):
    """Microphone or speaker failure."""


def _sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioError("sounddevice is not installed. Run `uv sync`.") from exc
    except OSError as exc:
        raise AudioError(
            "PortAudio is missing. On macOS: `brew install portaudio`. "
            "On Arch: `paru -S --needed portaudio`. "
            "On Windows, re-run `uv sync`."
        ) from exc
    return sd


def _device(env_name: str) -> int | str | None:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return value


def require_audio() -> None:
    """Fail early if PortAudio cannot be loaded."""
    _sounddevice()


def pcm_to_wav(pcm: bytes, sample_rate: int, *, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a WAV header so STT can consume it from memory."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _wait_for_enter_or_limit(stopped: threading.Event, seconds: float) -> None:
    """Block on Enter. Signal `stopped` if the time cap is hit first."""

    def auto_stop() -> None:
        if not stopped.wait(seconds):
            print(
                f"\nRecording reached {int(seconds)}s. Press Enter to send.",
                flush=True,
            )
            stopped.set()

    timer = threading.Thread(target=auto_stop, daemon=True)
    timer.start()
    try:
        input()
    finally:
        stopped.set()


def _open_input_stream(callback: Callable[..., None]) -> Any:
    sd = _sounddevice()
    try:
        return sd.RawInputStream(
            samplerate=RECORD_RATE,
            channels=1,
            dtype="int16",
            device=_device("AUDIO_SOURCE"),
            callback=callback,
        )
    except Exception as exc:
        raise AudioError(f"Could not open the microphone: {exc}") from exc


def record_until_enter() -> bytes:
    """Record 16 kHz mono WAV in memory until Enter, a 30s cap, or cancel."""
    require_audio()
    chunks: list[bytes] = []
    stopped = threading.Event()

    def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        if stopped.is_set():
            return
        chunks.append(bytes(indata))

    stream = _open_input_stream(callback)
    stream.start()
    try:
        try:
            _wait_for_enter_or_limit(stopped, MAX_RECORD_SECONDS)
        except (EOFError, KeyboardInterrupt) as exc:
            raise AudioError("Recording cancelled.") from exc
    finally:
        stopped.set()
        try:
            stream.stop()
        finally:
            stream.close()

    pcm = b"".join(chunks)
    if len(pcm) > MAX_RECORD_PCM_BYTES:
        pcm = pcm[:MAX_RECORD_PCM_BYTES]
        pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    if not pcm:
        raise AudioError(
            "That recording was empty. Press Enter, speak, then press Enter again."
        )
    return pcm_to_wav(pcm, RECORD_RATE)


class PcmPlayer:
    """One PortAudio output stream that accepts concatenated s16le mono PCM."""

    def __init__(self, sample_rate: int = DEFAULT_PCM_RATE) -> None:
        require_audio()
        if sample_rate <= 0:
            raise AudioError("PCM sample rate must be positive")
        self.sample_rate = sample_rate
        self._stream: Any | None = None
        self._pending = b""

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _sounddevice()
        try:
            stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=_device("AUDIO_SINK"),
            )
            stream.start()
        except Exception as exc:
            raise AudioError(f"Could not open the speakers: {exc}") from exc
        self._stream = stream

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._stream is None:
            self.start()
        stream = self._stream
        if stream is None:
            raise AudioError("Speaker stream is unavailable")
        data = self._pending + pcm
        aligned = len(data) - (len(data) % 2)
        try:
            if aligned:
                stream.write(data[:aligned])
        except Exception as exc:
            raise AudioError(f"Playback failed: {exc}") from exc
        self._pending = data[aligned:]

    def abort(self) -> None:
        """Stop playback immediately. Does not raise."""
        stream = self._stream
        self._stream = None
        self._pending = b""
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def close(self, *, raise_on_error: bool = True) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            leftover = self._pending
            self._pending = b""
            if leftover:
                stream.write(leftover + b"\x00")
            stream.stop()
            stream.close()
        except Exception as exc:
            if raise_on_error:
                raise AudioError(f"Playback failed: {exc}") from exc

    def __enter__(self) -> PcmPlayer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(raise_on_error=exc_type is None)
