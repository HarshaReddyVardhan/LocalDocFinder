"""Windows built-in OCR (Windows.Media.Ocr via winrt). No model download, reads screenshots/diagrams."""
import asyncio
import io

from PIL import Image

_engine = None
_engine_failed = False
_MAX_DIM = 4000  # OcrEngine.MaxImageDimension is ~10000; stay far below for speed


def _get_engine():
    global _engine, _engine_failed
    if _engine is None and not _engine_failed:
        try:
            from winrt.windows.media.ocr import OcrEngine
            _engine = OcrEngine.try_create_from_user_profile_languages()
            if _engine is None:
                _engine_failed = True
        except Exception:
            _engine_failed = True
    return _engine


def available() -> bool:
    return _get_engine() is not None


def _to_png(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_DIM:
        s = _MAX_DIM / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _recognize(png: bytes) -> str:
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    engine = _get_engine()
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
    return result.text or ""


def ocr_image(img: Image.Image) -> str:
    """OCR a PIL image. Returns '' if OCR is unavailable or finds nothing."""
    if _get_engine() is None:
        return ""
    try:
        # The timeout guards against a winrt callback failure leaving the await pending forever.
        text = asyncio.run(asyncio.wait_for(_recognize(_to_png(img)), timeout=30))
    except Exception:
        return ""
    return " ".join(text.split())


def ocr_bytes(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return ocr_image(img)
    except Exception:
        return ""
