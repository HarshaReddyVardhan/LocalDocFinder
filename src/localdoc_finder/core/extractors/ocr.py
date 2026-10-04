"""Windows built-in OCR (Windows.Media.Ocr via winrt): no model download, reads screenshots."""

import asyncio
import io
import logging
import re

from PIL import Image

from localdoc_finder.core.extractors.base import Untyped

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30  # guards against a winrt callback failure leaving the await pending forever
_DEFAULT_MAX_DIM = 4000  # the engine's own limit is ~10000; far below that is faster

# OCR of a page that holds no real text (a blank or dusty scan, a photo) still returns tokens:
# specks read as "l", "i", "|", "~". Below these limits a page is low-content: kept for keyword
# and name search, but never treated as meaning anything to the embedder.
MIN_OCR_QUALITY = 0.5  # share of real words among the word-like tokens
MIN_OCR_CONTENT_CHARS = 40  # letters and digits; about six words

_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
_NUMBER = re.compile(r"^[\d.,:/$€£%+\-]+$")
_LATIN_VOWELS = frozenset("aeiouyAEIOUY")
_REPEAT_SHARE = 0.6  # "lllll", "iiii": one character making most of a token is speckle


def _is_word(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return False
    # Accented and non-Latin letters count as vowels: the vowel test is for Latin script only.
    if not any(c in _LATIN_VOWELS or not c.isascii() for c in letters):
        return False
    most = max(letters.count(c) for c in set(letters))
    return len(letters) < 3 or most / len(letters) < _REPEAT_SHARE


def ocr_quality(text: str) -> float:
    """Share of word-like tokens that are real words, from 0 (noise) to 1 (prose).

    Numbers (``$1,450``, ``12/03``) are neither: a statement full of figures is not noise. Text
    with no tokens at all scores 0; text of numbers only scores 1.
    """
    words = junk = numbers = 0
    for raw in text.split():
        token = _EDGE_PUNCTUATION.sub("", raw)
        if not token:
            junk += 1  # a lone "|" or "~"
        elif _NUMBER.match(token) and sum(c.isdigit() for c in token) >= 2:
            numbers += 1
        elif _is_word(token):
            words += 1
        else:
            junk += 1
    if words + junk == 0:
        return 1.0 if numbers else 0.0
    return words / (words + junk)


def is_low_content(text: str) -> bool:
    """Too little readable text to say what the page or picture is about."""
    chars = sum(c.isalnum() for c in text)
    return chars < MIN_OCR_CONTENT_CHARS or ocr_quality(text) < MIN_OCR_QUALITY


def to_png(image: Image.Image, max_dim: int) -> bytes:
    """RGB PNG bytes, downscaled so the longest side is at most ``max_dim``."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        rgb = rgb.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return buffer.getvalue()


class WindowsOcr:
    """OCR through the user-profile languages of Windows; unavailable elsewhere."""

    def __init__(self, max_dim: int = _DEFAULT_MAX_DIM) -> None:
        self._max_dim = max_dim
        self._engine: Untyped | None = None
        self._failed = False

    def _get_engine(self) -> Untyped | None:
        if self._engine is None and not self._failed:
            try:
                from winrt.windows.media.ocr import OcrEngine

                self._engine = OcrEngine.try_create_from_user_profile_languages()
            except Exception:  # winrt missing or unsupported OS: OCR is simply unavailable
                logger.debug("ocr: winrt engine unavailable", exc_info=True)
            self._failed = self._engine is None
        return self._engine

    def available(self) -> bool:
        return self._get_engine() is not None

    async def _recognize(self, png: bytes) -> list[str]:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        engine = self._get_engine()
        assert engine is not None
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(png)
        await writer.store_async()
        await writer.flush_async()
        writer.detach_stream()
        stream.seek(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
        return [str(line.text or "") for line in result.lines]

    def ocr_lines(self, image: Image.Image) -> list[str]:
        """Recognised lines, top to bottom (each whitespace-normalised, blanks dropped); ``[]`` if
        OCR is unavailable or finds nothing. Lines let a caller drop a page's header or footer."""
        if self._get_engine() is None:
            return []
        try:
            png = to_png(image, self._max_dim)
            lines = asyncio.run(asyncio.wait_for(self._recognize(png), timeout=_TIMEOUT_SECONDS))
        except Exception:  # one unreadable image must not stop an indexing run
            logger.debug("ocr: recognition failed", exc_info=True)
            return []
        return [" ".join(line.split()) for line in lines if line.strip()]

    def ocr_image(self, image: Image.Image) -> str:
        """Recognised text on one line; ``""`` if OCR is unavailable or finds none."""
        return " ".join(self.ocr_lines(image))
