from types import SimpleNamespace

import pytest

import venice


@pytest.mark.parametrize(
    ("buffer", "expected", "rest"),
    [
        ("Hello. ", ["Hello."], ""),
        ("Hello.", [], "Hello."),
        ("Hello. World is big. ", ["Hello.", "World is big."], ""),
        ('He said "Go." Next. ', ['He said "Go."', "Next."], ""),
        ("Wait! Now. ", ["Wait!", "Now."], ""),
    ],
)
def test_pop_sentences(buffer: str, expected: list[str], rest: str) -> None:
    sentences, leftover = venice.pop_sentences(buffer)
    assert sentences == expected
    assert leftover == rest


def test_flush_yields_final_sentence_without_trailing_space() -> None:
    assert venice.flush_sentences("Venice is a privacy-first AI platform.") == [
        "Venice is a privacy-first AI platform."
    ]


def test_abbreviations_do_not_split_early() -> None:
    sentences, rest = venice.pop_sentences("Dr. Smith arrived. Next. ")
    assert sentences == ["Dr. Smith arrived.", "Next."]
    assert rest == ""


def test_initials_do_not_split_early() -> None:
    sentences, leftover = venice.pop_sentences("The U.S. Navy arrived. ")
    assert sentences == ["The U.S. Navy arrived."]
    assert leftover == ""


def test_resolve_voice_passes_kokoro_ids_through() -> None:
    assert venice.resolve_voice("af_heart") == "af_heart"
    assert venice.resolve_voice("  am_adam ") == "am_adam"
    assert venice.resolve_voice("") == "af_sky"
    assert venice.resolve_voice(None) == "af_sky"


def test_looks_like_non_pcm() -> None:
    assert venice.looks_like_non_pcm(b'{"error":"nope"}')
    assert venice.looks_like_non_pcm(b"RIFF....WAVE")
    assert venice.looks_like_non_pcm(b"ID3....")
    assert not venice.looks_like_non_pcm(b"\x00\x01\x02\x03")


def test_ensure_pcm_response_rejects_http_errors() -> None:
    response = SimpleNamespace(status_code=400, headers={}, http_response=SimpleNamespace())
    response.http_response.json = lambda: {"error": {"message": "bad voice"}}
    with pytest.raises(venice.VeniceError, match="400"):
        venice.ensure_pcm_response(response)


def test_ensure_pcm_response_rejects_json_content_type() -> None:
    response = SimpleNamespace(
        status_code=200,
        headers={"content-type": "application/json; charset=utf-8"},
    )
    with pytest.raises(venice.VeniceError, match="application/json"):
        venice.ensure_pcm_response(response)


def test_translate_includes_non_http_detail() -> None:
    from openai import OpenAIError

    err = venice._translate(OpenAIError("connection timed out"))
    assert "connection timed out" in str(err)
