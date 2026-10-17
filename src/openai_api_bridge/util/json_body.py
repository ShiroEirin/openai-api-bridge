"""A JSON object request body that keeps its original bytes.

The passthrough endpoints read ``model`` out of a request, swap in the
upstream's native slug, and forward everything else untouched. Doing that as
``json.loads`` then ``json.dumps`` re-encodes the whole body on the single
event loop to change one short string — and encoding is the expensive half,
about three times the cost of the parse (a 3MB embeddings batch: ~2ms to
parse, ~5ms more to re-encode; a 5MB base64 vision request: ~3ms and ~8ms).

:func:`parse_object` still parses everything, so the body is validated exactly
as before and every field is available to read. It parses one top-level member
at a time with the C decoder, though, which yields where each top-level value
sits in the text. :meth:`JsonBody.replace_top_level` splices new values in at
those offsets and leaves the rest of the bytes alone.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_DECODER = json.JSONDecoder()
_WS = " \t\n\r"


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return i


@dataclass(frozen=True, slots=True)
class JsonBody:
    """A parsed JSON object plus the bytes to send for it.

    ``fields`` is the whole parsed object; ``raw`` is UTF-8 JSON equivalent to
    it. ``spans`` holds the ``(start, end)`` offsets of each top-level value,
    in characters, in ``raw`` decoded as UTF-8.

    The decoded text is deliberately not kept. It lives as long as the body
    does, which on a streaming chat completion is the whole stream, and a str
    holding even one character outside the BMP (an emoji) is stored at four
    bytes per character. Only :meth:`replace_top_level` needs it, so it
    re-decodes ``raw`` there.
    """

    fields: Mapping[str, Any]
    raw: bytes
    spans: Mapping[str, tuple[tuple[int, int], ...]] = field(repr=False)

    @property
    def text(self) -> str:
        """``raw`` decoded; the string ``spans`` index into."""
        return self.raw.decode("utf-8", "surrogatepass")

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> JsonBody:
        """Encode a body the bridge built itself.

        Encoded once and not parsed back: the fields are already the parsed
        form, and parsing the encoding only to record spans cost a full extra
        decode of every base64 image in an OpenRouter edit, plus a second
        copy of each data URL held for the whole upstream call. With no spans
        recorded, :meth:`replace_top_level` re-encodes instead, which for a
        body the bridge built is what it would have cost anyway.
        """
        return cls(fields=fields, raw=json.dumps(fields).encode("utf-8"), spans={})

    def replace_top_level(self, updates: Mapping[str, Any]) -> JsonBody:
        """A copy with existing top-level ``updates`` keys set to new values.

        Spliced into the original text, so only the replaced values are
        encoded. A key with no recorded span — absent from the body, which
        the passthrough never asks for, or any key of a body built with
        :meth:`from_fields` — falls back to re-encoding the whole object.
        """
        if any(key not in self.spans for key in updates):
            return JsonBody.from_fields({**self.fields, **updates})
        edits = sorted(
            (start, end, json.dumps(value))
            for key, value in updates.items()
            for start, end in self.spans[key]
        )
        source = self.text
        parts: list[str] = []
        pos = 0
        # shift_before[i]: how far the text has moved by the start of edit i.
        shift_before: list[int] = []
        shift = 0
        for start, end, encoded in edits:
            parts.append(source[pos:start])
            parts.append(encoded)
            pos = end
            shift_before.append(shift)
            shift += len(encoded) - (end - start)
        shift_before.append(shift)
        parts.append(source[pos:])
        text = "".join(parts)
        starts = [start for start, _, _ in edits]
        replaced = {start: len(encoded) for start, _, encoded in edits}

        def _shifted(span: tuple[int, int]) -> tuple[int, int]:
            # Top-level values never overlap, so the edits wholly before a
            # span are exactly those that start before it. One bisect per
            # span keeps this linear-logarithmic however many duplicate keys
            # a body repeats; summing over every edit per span was quadratic.
            start, end = span
            idx = bisect_left(starts, start)
            new_start = start + shift_before[idx]
            if start in replaced:
                return new_start, new_start + replaced[start]
            return new_start, new_start + (end - start)

        return JsonBody(
            fields={**self.fields, **updates},
            raw=text.encode("utf-8", "surrogatepass"),
            spans={k: tuple(_shifted(sp) for sp in v) for k, v in self.spans.items()},
        )


def parse_object(raw: bytes) -> JsonBody:
    """Parse ``raw`` as a JSON object, recording each top-level value's span.

    Accepts what ``json.loads`` accepts on bytes (UTF-8/16/32, with or without
    a BOM) and raises ``ValueError`` (``json.JSONDecodeError``) where it would,
    plus a ``ValueError`` when the document is valid JSON but not an object.
    Duplicate keys keep the last value, as ``json.loads`` does, and every
    occurrence's span is recorded so a replacement reaches whichever one the
    upstream's parser honours.
    """
    encoding = json.detect_encoding(raw)
    text = raw.decode(encoding, "surrogatepass")
    i = _skip_ws(text, 0)
    if i >= len(text) or text[i] != "{":
        # Let the real decoder name the error, or report the non-object.
        _DECODER.decode(text)
        raise ValueError("JSON body is not an object")
    fields: dict[str, Any] = {}
    spans: dict[str, list[tuple[int, int]]] = {}
    i = _skip_ws(text, i + 1)
    if i < len(text) and text[i] == "}":
        i += 1
    else:
        while True:
            if i >= len(text) or text[i] != '"':
                raise json.JSONDecodeError(
                    "Expecting property name enclosed in double quotes", text, i
                )
            key, i = _DECODER.raw_decode(text, i)
            i = _skip_ws(text, i)
            if i >= len(text) or text[i] != ":":
                raise json.JSONDecodeError("Expecting ':' delimiter", text, i)
            start = _skip_ws(text, i + 1)
            value, end = _DECODER.raw_decode(text, start)
            fields[key] = value
            spans.setdefault(key, []).append((start, end))
            i = _skip_ws(text, end)
            if i < len(text) and text[i] == ",":
                i = _skip_ws(text, i + 1)
                continue
            if i < len(text) and text[i] == "}":
                i += 1
                break
            raise json.JSONDecodeError("Expecting ',' delimiter", text, i)
    if _skip_ws(text, i) != len(text):
        raise json.JSONDecodeError("Extra data", text, i)
    # Forward the caller's own bytes when they are plain UTF-8, the common
    # case; a UTF-16/32 or BOM-prefixed body is normalised to UTF-8 once.
    out = raw if encoding == "utf-8" else text.encode("utf-8", "surrogatepass")
    return JsonBody(
        fields=fields,
        raw=out,
        spans={k: tuple(v) for k, v in spans.items()},
    )
