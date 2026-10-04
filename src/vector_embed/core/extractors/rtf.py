"""RTF: text by paragraph (tables as rows), embedded PNG/JPEG pictures through OCR.

A small tokenizer, not a full RTF reader: it keeps the visible text and the pictures, and skips
the destinations that only hold formatting (font, colour and style tables, metadata, field
instructions, embedded objects).
"""

import codecs
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from vector_embed.core.extractors.base import (
    KIND_DOC,
    Chunk,
    ExtractContext,
    Extractor,
    register_extractor,
)
from vector_embed.core.extractors.chunking import split_by_lines
from vector_embed.core.extractors.image import image_chunk, image_key

_TOKEN = re.compile(
    rb"\\([a-zA-Z]+)(-?\d+)? ?"  # control word with an optional numeric parameter
    rb"|\\'([0-9a-fA-F]{2})"  # a byte in the document's code page
    rb"|\\(.)"  # control symbol
    rb"|([{}])"  # group start / end
    rb"|[\r\n]+"  # line breaks in the source mean nothing
    rb"|([^\\{}\r\n]+)",  # plain text
    re.DOTALL,
)
_SKIPPED_DESTINATIONS = frozenset(
    {
        "fonttbl", "colortbl", "stylesheet", "info", "listtable", "listoverridetable",
        "rsidtbl", "generator", "xmlnstbl", "themedata", "colorschememapping", "latentstyles",
        "datastore", "fldinst", "objdata", "object", "nonshppict", "header", "footer",
        "headerl", "headerr", "headerf", "footerl", "footerr", "footerf", "filetbl",
        "revtbl", "pgdsctbl", "mmathPr", "wgrffmtfilter",
    }
)  # fmt: skip
_KEPT_STARRED = frozenset({"shppict"})  # \*\shppict holds the real picture; its twin is skipped
_PICTURE_FORMATS = frozenset({"pngblip", "jpegblip"})  # what Pillow can decode; WMF/EMF are not
_BREAKS = {"par": "\n", "line": "\n", "row": "\n", "sect": "\n", "page": "\n"}
_INLINE = {"tab": "\t", "cell": " | ", "emdash": "\u2014", "endash": "\u2013", "bullet": "\u2022"}
_SYMBOLS = {"~": "\u00a0", "_": "-", "-": "", "\\": "\\", "{": "{", "}": "}"}
_DEFAULT_CODEPAGE = 1252
_UNICODE_OFFSET = 65536  # \u takes a signed 16-bit value


@dataclass
class _Group:
    skip: bool = False
    picture: bool = False
    picture_format: str = ""
    unicode_skip: int = 1  # \ucN: fallback characters after each \uN


@dataclass
class _Reader:
    """Walks the tokens once, collecting text and the bytes of each picture."""

    codepage: str = f"cp{_DEFAULT_CODEPAGE}"
    text: list[str] = field(default_factory=list)
    pictures: list[bytes] = field(default_factory=list)
    _stack: list[_Group] = field(default_factory=lambda: [_Group()])
    _hex: list[bytes] = field(default_factory=list)
    _pending: bytearray = field(default_factory=bytearray)  # \'hh bytes not decoded yet
    _starred: bool = False
    _to_skip: int = 0  # fallback characters still to drop after a \uN

    @property
    def group(self) -> _Group:
        return self._stack[-1]

    def feed(self, raw: bytes) -> None:
        for match in _TOKEN.finditer(raw):
            word, param, hex_byte, symbol, brace, plain = match.groups()
            if hex_byte is not None:
                self._byte(int(hex_byte, 16))
                continue
            self._flush_bytes()
            if word is not None:
                self._control(word.decode("ascii"), int(param) if param is not None else None)
            elif symbol is not None:
                self._symbol(symbol.decode("latin-1"))
            elif brace is not None:
                self._brace(brace)
            elif plain is not None:
                self._plain(plain)
        self._flush_bytes()

    # ------------------------------------------------------------------ tokens
    def _brace(self, brace: bytes) -> None:
        if brace == b"{":
            self._stack.append(replace(self.group))
            return
        closed = self._stack.pop() if len(self._stack) > 1 else self.group
        if closed.picture and not self.group.picture:
            self._finish_picture(closed)
        self._starred = False

    def _control(self, word: str, param: int | None) -> None:
        starred, self._starred = self._starred, False
        if word == "ansicpg" and param:
            self.codepage = _codepage(param)
        if (starred and word not in _KEPT_STARRED) or word in _SKIPPED_DESTINATIONS:
            self.group.skip = True
        elif word == "pict":
            self.group.picture = True
            self._hex = []
        elif word in _PICTURE_FORMATS and self.group.picture:
            self.group.picture_format = word
        elif word == "uc" and param is not None:
            self.group.unicode_skip = param
        elif word == "u" and param is not None:
            self._emit(chr(param + _UNICODE_OFFSET if param < 0 else param))
            self._to_skip = self.group.unicode_skip
        elif word in _BREAKS:
            self._emit(_BREAKS[word])
        elif word in _INLINE:
            self._emit(_INLINE[word])

    def _symbol(self, symbol: str) -> None:
        if symbol == "*":
            self._starred = True
        elif symbol in _SYMBOLS:
            self._emit(_SYMBOLS[symbol])

    def _plain(self, chunk: bytes) -> None:
        if self.group.picture and not self.group.skip:
            self._hex.append(chunk)
            return
        text = chunk.decode(self.codepage, errors="replace")
        if self._to_skip:
            dropped = min(self._to_skip, len(text))
            text, self._to_skip = text[dropped:], self._to_skip - dropped
        self._emit(text)

    def _byte(self, value: int) -> None:
        if self._to_skip:  # a \'hh is the ANSI fallback of the \uN before it
            self._to_skip -= 1
            return
        if not self.group.skip:
            self._pending.append(value)

    def _flush_bytes(self) -> None:
        if self._pending:
            self._emit(bytes(self._pending).decode(self.codepage, errors="replace"))
            self._pending.clear()

    def _emit(self, text: str) -> None:
        if text and not self.group.skip and not self.group.picture:
            self.text.append(text)

    def _finish_picture(self, group: _Group) -> None:
        hex_text = b"".join(self._hex).translate(None, b" \t")
        self._hex = []
        if group.skip or group.picture_format not in _PICTURE_FORMATS:
            return
        try:
            self.pictures.append(bytes.fromhex(hex_text.decode("ascii")))
        except ValueError:  # truncated or not hex: just this picture is lost
            return


def _codepage(number: int) -> str:
    name = f"cp{number}"
    try:
        codecs.lookup(name)
    except LookupError:
        return f"cp{_DEFAULT_CODEPAGE}"
    return name


def read_rtf(raw: bytes) -> tuple[str, list[bytes]]:
    """The visible text of an RTF document and the bytes of its PNG/JPEG pictures."""
    reader = _Reader()
    reader.feed(raw)
    return "".join(reader.text), reader.pictures


def _text_chunks(ctx: ExtractContext, text: str) -> list[Chunk]:
    cfg = ctx.chunking
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        Chunk(part, KIND_DOC, "")
        for part, _, _ in split_by_lines(lines, 1, cfg.target_chunk_chars * 2, cfg.line_overlap)
    ]


def _picture_chunks(ctx: ExtractContext, pictures: list[bytes]) -> list[Chunk]:
    chunks: list[Chunk] = []
    budget = ctx.images.max_per_doc
    seen: set[str] = set()
    for data in pictures:
        if budget <= 0:
            break
        if image_key(data) in seen:
            continue  # the same picture used twice: no second OCR, no allowance spent
        budget -= 1
        chunk = image_chunk(ctx, data, 0, seen, "embedded image")
        if chunk:
            chunks.append(chunk)
    return chunks


@register_extractor("rtf")
class RtfExtractor(Extractor):
    name = "rtf"
    priority = 30
    is_document = True

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".rtf"

    def extract(self, path: Path) -> Iterable[Chunk]:
        text, pictures = read_rtf(path.read_bytes())
        return [*_text_chunks(self.ctx, text), *_picture_chunks(self.ctx, pictures)]
