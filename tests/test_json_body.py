"""``JsonBody``: parse like ``json.loads``, but rewrite by splicing.

The passthrough forwards the caller's bytes with only ``model`` (and, on a
stream, ``stream``) replaced. These pin the two halves of that: the parse
accepts and rejects exactly what ``json.loads`` does, and a replacement leaves
every other byte where the caller put it.
"""

from __future__ import annotations

import json

import pytest

from openai_api_bridge.util.json_body import JsonBody, parse_object

_OBJECTS: list[bytes] = [
    b"{}",
    b"  { }  ",
    b'{"model":"p/m","messages":[{"role":"user","content":"hi"}]}',
    b'{\n  "model" : "p/m" ,\n  "n": 1.5e3, "x": null, "t": true, "f": false\n}\n',
    '{"model": "p/m", "text": "caf\u00e9 \U0001f600"}'.encode(),
    b'{"model": "p/m", "esc": "\\"model\\": \\"other\\"", "u": "\\u00e9\\ud83d\\ude00"}',
    b'{"nested": {"model": "p/m"}, "model": "p/m"}',
    b'{"model": "a", "model": "b"}',
    b'{"n": NaN, "i": -Infinity}',
]


@pytest.mark.parametrize("raw", _OBJECTS)
def test_fields_match_json_loads(raw: bytes) -> None:
    body = parse_object(raw)
    # Compared as re-encoded text, since NaN != NaN.
    assert json.dumps(dict(body.fields)) == json.dumps(json.loads(raw))
    assert body.raw == raw  # plain UTF-8 is forwarded as the caller's own bytes


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"   ",
        b"[]",
        b'"x"',
        b"1",
        b"{",
        b'{"a"}',
        b'{"a":}',
        b'{"a":1,}',
        b'{"a":1 "b":2}',
        b"{a:1}",
        b'{"a":1} x',
        b'{"a":1}{}',
        b'{"a":"\x01"}',
        b"\xff\xfe\xfd",
    ],
)
def test_rejects_what_json_loads_rejects_or_is_not_an_object(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_object(raw)
    try:
        loaded = json.loads(raw)
    except ValueError:
        return
    assert not isinstance(loaded, dict)


@pytest.mark.parametrize(
    "encoding", ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-be"]
)
def test_other_encodings_are_accepted_and_normalised_to_utf8(encoding: str) -> None:
    doc = {"model": "p/m", "text": "caf\u00e9"}
    raw = json.dumps(doc, ensure_ascii=False).encode(encoding)
    body = parse_object(raw)
    assert dict(body.fields) == doc
    assert json.loads(body.raw.decode("utf-8")) == doc


def test_replace_splices_and_leaves_every_other_byte() -> None:
    raw = b'{ "messages" : [ {"role":"user", "content":"\\"model\\": \\"p/m\\""} ] ,\n "model":"p/m"  }'
    out = parse_object(raw).replace_top_level({"model": "m"})
    assert out.raw == raw.replace(b'"model":"p/m"  }', b'"model":"m"  }')
    assert out.fields["model"] == "m"
    assert json.loads(out.raw) == {**json.loads(raw), "model": "m"}


def test_replace_reaches_every_duplicate_and_not_a_nested_key() -> None:
    raw = b'{"model": "p/a", "meta": {"model": "p/a"}, "model": "p/a"}'
    out = parse_object(raw).replace_top_level({"model": "a"})
    assert out.raw == b'{"model": "a", "meta": {"model": "p/a"}, "model": "a"}'


def test_successive_replacements_keep_their_offsets() -> None:
    raw = b'{"stream": 1, "model": "provider/slug", "x": [1, 2]}'
    out = parse_object(raw).replace_top_level({"model": "s"}).replace_top_level({"stream": True})
    assert out.raw == b'{"stream": true, "model": "s", "x": [1, 2]}'
    again = out.replace_top_level({"x": None, "model": "longer-than-before"})
    assert again.raw == b'{"stream": true, "model": "longer-than-before", "x": null}'


def test_replacing_an_absent_key_falls_back_to_reencoding() -> None:
    out = parse_object(b'{"model": "m"}').replace_top_level({"stream": True})
    assert json.loads(out.raw) == {"model": "m", "stream": True}


def test_from_fields_round_trips() -> None:
    body = JsonBody.from_fields({"model": "m", "messages": [{"content": "\u00e9"}]})
    assert json.loads(body.raw) == {"model": "m", "messages": [{"content": "\u00e9"}]}


def test_many_duplicate_keys_replace_in_bounded_time() -> None:
    """A body repeating a top-level key thousands of times, alongside many
    other keys, must not make the splice quadratic on the event loop."""
    import time

    count = 20_000
    members = [f'"model": "p/m", "k{i}": {i}' for i in range(count)]
    raw = ("{" + ", ".join(members) + "}").encode()
    body = parse_object(raw)

    started = time.perf_counter()
    out = body.replace_top_level({"model": "m"})
    elapsed = time.perf_counter() - started

    assert out.raw.count(b'"model": "m"') == count
    assert out.fields["k19999"] == 19999
    # Offsets stayed right for the untouched keys after thousands of edits.
    start, end = out.spans["k19999"][0]
    assert out.text[start:end] == "19999"
    assert elapsed < 1.0


def test_the_decoded_text_is_not_held() -> None:
    """A body lives as long as its request (a whole stream, for chat); a str
    with one non-BMP character is four bytes per character. Only the bytes
    and the parsed fields are kept."""
    assert "text" not in JsonBody.__slots__
    body = parse_object('{"model": "p/m", "x": "\U0001f600"}'.encode())
    out = body.replace_top_level({"model": "m"})
    assert json.loads(out.raw) == {"model": "m", "x": "\U0001f600"}


def test_from_fields_does_not_parse_its_own_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bridge-built body (OpenRouter's image-over-chat, base64 references and
    all) is encoded once, never decoded back just to record spans."""
    from openai_api_bridge.util import json_body

    def no_parse(raw: bytes) -> JsonBody:
        raise AssertionError("from_fields parsed its own output")

    monkeypatch.setattr(json_body, "parse_object", no_parse)
    body = JsonBody.from_fields({"model": "m", "messages": [{"content": "x" * 1000}]})
    assert json.loads(body.raw)["model"] == "m"
    assert json.loads(body.replace_top_level({"model": "n"}).raw)["model"] == "n"


def test_padding_between_members_is_skipped_quickly() -> None:
    import time

    pad = " \n\t\r" * 1_000_000
    raw = ("{" + pad + '"model"' + pad + ":" + pad + '"p/m"' + pad + "}" + pad).encode()
    started = time.perf_counter()
    body = parse_object(raw)
    assert time.perf_counter() - started < 0.5
    assert body.fields == {"model": "p/m"}
    assert json.loads(body.replace_top_level({"model": "m"}).raw) == {"model": "m"}
