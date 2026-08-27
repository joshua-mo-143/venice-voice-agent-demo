# Building a Venice voice agent

Venice can hear you and talk back. In this demo we stitch three ordinary HTTP calls together:

1. Speech to text (`POST /audio/transcriptions`)
2. A short chat reply (`POST /chat/completions`)
3. Text to speech (`POST /audio/speech`)

In this demo, we'll build that loop as a terminal app. Press Enter, speak, press Enter again, and the reply plays through your speakers. You can type a line instead of using the mic.

This is the same pipeline as the [LiveKit Agents guide](https://docs.venice.ai/guides/integrations/livekit-agents), without LiveKit, wake words, or tools. The interesting part is the Venice API. Recording and playback use [sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio), so the same code runs on macOS, Windows, and Linux.

By the end you should be able to lift `venice.py` into your own app and know why we stream both the chat tokens and the PCM.

## What we're building

| Stage | Venice endpoint | Default model |
| --- | --- | --- |
| Speech to text | `POST /audio/transcriptions` | `nvidia/parakeet-tdt-0.6b-v3` |
| Reply | `POST /chat/completions` | `zai-org-glm-5-2` |
| Text to speech | `POST /audio/speech` | `tts-kokoro` (`af_sky`) |

The source tree stays small on purpose:

```text
.
├── app.py          # prompt loop: listen, print, play
├── venice.py       # the three API calls
├── audio.py        # local record / play via PortAudio (not the API)
├── tests/          # sentence splitting, WAV wrapping, PCM checks
├── .env.example
└── pyproject.toml
```

## Pre-requisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A Venice API key from [venice.ai](https://venice.ai)
- A microphone and speakers if you want the full voice loop

Recording and playback go through [sounddevice](https://python-sounddevice.readthedocs.io/), which uses PortAudio. `uv sync` installs the Python package. On Windows that is enough. On macOS and Linux you also need the PortAudio library:

```bash
# macOS
brew install portaudio

# Debian / Ubuntu
sudo apt install libportaudio2

# Arch
paru -S --needed portaudio
```

`--text-only` skips the mic and still hits chat + TTS. Use that if you just want to try the API calls.

## Getting started

```bash
cp .env.example .env
```

Put your key in `.env` as `VENICE_API_KEY`, then:

```bash
uv sync
uv run python app.py
```

Press Enter, speak, press Enter again. Type a line if you do not want to use the mic. `reset` starts a new conversation. `q` quits. Ctrl+C during a reply stops playback and returns to the prompt.

```bash
uv run python app.py --voice am_adam
uv run python app.py --voice af_heart
uv run python app.py --text-only
```

`--voice` is passed through to Kokoro. Unknown ids fail at the API instead of silently falling back.

If the mic or speakers are wrong, list devices and set `AUDIO_SOURCE` / `AUDIO_SINK` in `.env` to a name or index:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

The API key stays in your environment. Microphone audio stays in memory, gets a WAV header, and is sent to Venice. Nothing is written to disk. Recordings stop after 30 seconds.

## How a turn works

Venice's API is OpenAI-compatible, so we use the official `openai` Python client and point it at `https://api.venice.ai/api/v1`. All you need to do is swap the base URL and model ids.

```python
client = OpenAI(
    api_key=os.environ["VENICE_API_KEY"],
    base_url="https://api.venice.ai/api/v1",
)
```

### 1. Transcribe

`POST /audio/transcriptions` takes a WAV (16 kHz mono in this demo) and returns text.

```python
result = client.audio.transcriptions.create(
    model="nvidia/parakeet-tdt-0.6b-v3",
    file=("speech.wav", wav_bytes, "audio/wav"),
    response_format="json",
)
user_text = result.text
```

### 2. Stream the reply

`POST /chat/completions` with `stream=True`. We keep answers short (20 words) so they sound like speech, not a blog post.

Two Venice-specific extras matter here:

- `include_venice_system_prompt: False` — otherwise Venice adds its own system prompt on top of ours
- `disable_thinking: True` (and `reasoning.enabled: False`) — GLM will otherwise spend tokens on a hidden chain of thought before it talks

As tokens arrive, we split on sentence boundaries (`Hello.` / `Wait!`) and start TTS on the first complete sentence. The chat stream keeps draining on a side thread while earlier sentences play. Abbreviations like `Dr.` and `U.S.` are not treated as the end of a sentence.

```python
stream = client.chat.completions.create(
    model="zai-org-glm-5-2",
    messages=messages,
    max_tokens=48,
    stream=True,
    extra_body={
        "venice_parameters": {
            "include_venice_system_prompt": False,
            "disable_thinking": True,
        },
        "reasoning": {"enabled": False},
    },
)
```

### 3. Speak

`POST /audio/speech` with `response_format="pcm"` and `streaming: True`. We write raw s16le (24 kHz mono) straight into a PortAudio output stream, so playback does not wait for an MP3 to finish downloading.

```python
with client.audio.speech.with_streaming_response.create(
    model="tts-kokoro",
    voice="af_sky",
    input=sentence,
    response_format="pcm",
    extra_body={"streaming": True},
) as response:
    for chunk in response.iter_bytes():
        player.write(chunk)
```

Check the HTTP status and content type before you treat the body as PCM. A JSON error written into a raw speaker stream is a loud burst of noise.

## Notes

The LiveKit guide uses this same STT → LLM → TTS shape.

The system prompt lives in `venice.py`. It is sent as `role=system` on every turn. If someone asks what Venice is, we mean the product, not the Italian city.

`audio.py` is the only OS-specific file, and PortAudio is the portability layer. The API calls in `venice.py` do not care which machine you are on.

```bash
uv run pytest
```

## Finishing up

The thing to take away: a voice agent on Venice is three OpenAI-compatible endpoints, streamed.

Swap `VENICE_LLM_MODEL` if you want a different chat model. Full GLM 5.3 is `z-ai-glm-5-3`; Flash is `z-ai-glm-5-3-flash`.

Docs for the endpoints:

- [Chat completions](https://docs.venice.ai/api-reference/endpoint/chat/completions)
- [Audio transcriptions](https://docs.venice.ai/api-reference/endpoint/audio/transcriptions)
- [Audio speech](https://docs.venice.ai/api-reference/endpoint/audio/speech)
- [Text models](https://docs.venice.ai/models/text)
- [Text to speech](https://docs.venice.ai/guides/media/text-to-speech)
