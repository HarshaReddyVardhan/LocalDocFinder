"""Windows built-in OCR (Windows.Media.Ocr via winrt): no model download, reads screenshots."""

import asyncio
import io
import logging

from PIL import Image

from localdoc_finder.core.extractors.base import Untyped

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30  # guards against a winrt callback failure leaving the await pending forever
_DEFAULT_MAX_DIM = 4000  # the engine's own limit is ~10000; far below that is faster


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

    async def _recognize(self, png: bytes) -> str:
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
        return str(result.text or "")

    def ocr_image(self, image: Image.Image) -> str:
        """Recognised text (whitespace-normalised); ``""`` if OCR is unavailable or finds none."""
        if self._get_engine() is None:
            return ""
        try:
            png = to_png(image, self._max_dim)
            text = asyncio.run(asyncio.wait_for(self._recognize(png), timeout=_TIMEOUT_SECONDS))
        except Exception:  # one unreadable image must not stop an indexing run
            logger.debug("ocr: recognition failed", exc_info=True)
            return ""
        return " ".join(text.split())
