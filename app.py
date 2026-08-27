"""Terminal loop: press Enter to talk, hear a spoken reply.

Each turn is STT → streamed chat → streamed TTS. venice.py owns the API
calls; audio.py owns PipeWire record/play. This file is just the prompt.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections.abc import Iterator

from dotenv import load_dotenv
from openai import OpenAI

import audio
import venice

load_dotenv()

MAX_HISTORY_TURNS = 8
QUIT_WORDS = {"q", "quit", "exit"}
RESET_WORDS = {"reset", "new", "clear"}


def _dim(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[2m{text}\033[0m"


def _print_help() -> None:
    print(
        _dim(
            "Enter to talk, type a message to send text, "
            "reset for a new conversation, q to quit."
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk to a Venice voice agent from the terminal."
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("VENICE_TTS_VOICE", venice.DEFAULT_TTS_VOICE),
        help="Kokoro voice id (default: af_sky). Examples: af_sky, af_heart, am_adam",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip the microphone and type every turn",
    )
    return parser.parse_args()


def _listen(client: OpenAI) -> str:
    print(_dim("Listening… press Enter to send"))
    clip = audio.record_until_enter()
    print(_dim("Transcribing…"))
    return venice.transcribe(client, clip, "speech.wav")


def _queued_sentences(
    client: OpenAI,
    history: list[dict[str, str]],
    user_text: str,
) -> Iterator[str]:
    """Drain the LLM stream on a side thread so TTS can overlap later sentences."""
    pending: queue.Queue[str | BaseException | None] = queue.Queue()
    cancel = threading.Event()

    def produce() -> None:
        try:
            for sentence in venice.iter_sentences(
                client, history, user_text, cancel=cancel
            ):
                pending.put(sentence)
            pending.put(None)
        except BaseException as exc:
            pending.put(exc)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = pending.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()


def _speak_turn(
    client: OpenAI,
    history: list[dict[str, str]],
    user_text: str,
    voice: str,
    sample_rate: int,
    *,
    play: bool,
) -> str:
    print(_dim("Thinking…"), flush=True)
    player: audio.PcmPlayer | None = None
    parts: list[str] = []
    started = time.perf_counter()
    first_audio: float | None = None
    failed = False
    try:
        for sentence in _queued_sentences(client, history, user_text):
            parts.append(sentence)
            if len(parts) == 1:
                print(f"Venice: {sentence}", flush=True)
            else:
                print(sentence, flush=True)
            if not play:
                continue
            for chunk in venice.iter_pcm(client, sentence, voice):
                if player is None:
                    player = audio.PcmPlayer(sample_rate)
                    print(_dim("Speaking…"), flush=True)
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                player.write(chunk)
    except KeyboardInterrupt:
        failed = True
        if player is not None:
            player.abort()
            player = None
        raise audio.AudioError("Playback cancelled.") from None
    except BaseException:
        failed = True
        raise
    finally:
        if player is not None:
            player.close(raise_on_error=not failed)
    if not parts:
        raise venice.VeniceError("Venice returned an empty reply. Please try again.")
    if play and first_audio is not None:
        print(_dim(f"First audio in {first_audio:.2f}s"), flush=True)
    return " ".join(parts)


def main() -> None:
    args = _parse_args()
    try:
        client = venice.load_client()
        voice = venice.resolve_voice(args.voice)
        if not args.text_only:
            audio.require_pipewire()
        print(_dim("Warming up…"), flush=True)
        sample_rate = venice.warmup(client, voice, tts=not args.text_only)
    except (venice.VeniceError, audio.AudioError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    config = venice.public_config()
    print("Venice voice demo")
    print(
        _dim(
            f"{config['stt_model']} → {config['llm_model']} → {config['tts_model']}"
        )
    )
    _print_help()
    print()

    history: list[dict[str, str]] = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        if stripped.lower() in QUIT_WORDS:
            break
        if stripped.lower() in RESET_WORDS:
            history.clear()
            print(_dim("New conversation."))
            continue
        if stripped.lower() in {"help", "?"}:
            _print_help()
            continue

        try:
            if stripped:
                user_text = stripped
            elif args.text_only:
                print(_dim("Type a message, or q to quit."))
                continue
            else:
                user_text = _listen(client)
            print(f"You: {user_text}")
            assistant_text = _speak_turn(
                client,
                history,
                user_text,
                voice,
                sample_rate,
                play=not args.text_only,
            )
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
            history = history[-(MAX_HISTORY_TURNS * 2) :]
        except KeyboardInterrupt:
            print()
            print(_dim("Cancelled."))
        except audio.AudioError as exc:
            print(f"{exc}")
        except venice.VeniceError as exc:
            print(f"{exc}")


if __name__ == "__main__":
    main()
