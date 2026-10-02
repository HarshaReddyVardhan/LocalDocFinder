import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw
from tests.core.extractors.conftest import FakeOcr, Writer, png_bytes

from vector_embed.core.extractors import image as image_mod
from vector_embed.core.extractors import ocr as ocr_mod
from vector_embed.core.extractors.base import ExtractContext, ExtractError, ExtractorSet, NullOcr
from vector_embed.core.settings import ImageSettings

Build = Callable[..., ExtractorSet]


def test_image_file_gets_name_and_ocr_text(
    build_with_ocr: Build, write: Writer, ocr: FakeOcr
) -> None:
    chunks = build_with_ocr().extract(write("flow.png", png_bytes()))
    assert chunks[0].kind == "image"
    assert "Image file: flow.png (300x300)" in chunks[0].text
    assert "Text in image: diagram of the OAuth login flow" in chunks[0].text
    assert ocr.calls == 1


def test_image_without_text_is_still_findable_by_name(
    extractors: ExtractorSet, write: Writer
) -> None:
    chunks = extractors.extract(write("photo.png", png_bytes()))
    assert chunks[0].text == "Image file: photo.png (300x300)"


def test_tiny_and_corrupt_images_are_rejected(extractors: ExtractorSet, write: Writer) -> None:
    with pytest.raises(ExtractError, match="too small"):
        extractors.extract(write("icon.png", png_bytes((32, 32))))
    with pytest.raises(ExtractError, match="unreadable"):
        extractors.extract(write("bad.png", b"not an image"))


def test_one_large_side_is_enough(extractors: ExtractorSet, write: Writer) -> None:
    assert extractors.extract(write("banner.png", png_bytes((600, 40))))


def blank_caption(_image: Image.Image) -> str:
    return ""


def test_captions_only_when_enabled_and_provided(build_with_ocr: Build, write: Writer) -> None:
    def captioner(_image: Image.Image) -> str:
        return "a login screen"

    path = write("a.png", png_bytes())
    off = build_with_ocr(captioner=captioner).extract(path)[0].text
    assert "Caption" not in off
    on = build_with_ocr(ImageSettings(enable_captions=True), captioner).extract(path)[0].text
    assert "Caption: a login screen" in on
    no_caption = build_with_ocr(ImageSettings(enable_captions=True), blank_caption).extract(path)
    assert "Caption" not in no_caption[0].text


def test_thumbnail_is_written_once(build_with_ocr: Build, write: Writer, tmp_path: Path) -> None:
    extractors = build_with_ocr(thumbs=True)
    path = write("t.png", png_bytes((900, 600)))
    extractors.extract(path)
    (thumb,) = list((tmp_path / "thumbs").glob("*.jpg"))
    with Image.open(thumb) as img:
        assert max(img.size) <= 480
    stamp = thumb.stat().st_mtime_ns
    extractors.extract(path)
    assert thumb.stat().st_mtime_ns == stamp


def test_thumbnail_failure_is_ignored(
    build_with_ocr: Build, write: Writer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "thumbs").write_text("a file where the directory should be")
    assert build_with_ocr(thumbs=True).extract(write("t.png", png_bytes()))
    assert image_mod.make_thumbnail(build_with_ocr().ctx, Image.new("RGB", (9, 9)), "h") is None


def test_phash_is_stable_for_identical_images() -> None:
    first = Image.new("RGB", (64, 64), "red")
    second = Image.new("RGB", (64, 64), "red")
    assert image_mod.phash(first) == image_mod.phash(second)


def test_embedded_image_chunk_dedupes_and_filters(ctx: ExtractContext, ocr: FakeOcr) -> None:
    ctx_ocr = ExtractContext(ctx.scope, ctx.scope_settings, ocr=ocr)
    seen: set[str] = set()
    data = png_bytes()
    first = image_mod.image_chunk(ctx_ocr, data, 2, seen, "figure")
    assert first is not None
    assert (first.kind, first.page, first.symbol) == ("image", 2, "figure")
    assert image_mod.image_chunk(ctx_ocr, data, 2, seen, "figure") is None  # duplicate
    assert image_mod.image_chunk(ctx_ocr, png_bytes((20, 20), "blue"), 1, seen, "x") is None
    assert image_mod.image_chunk(ctx_ocr, b"garbage", 1, seen, "x") is None
    silent = ExtractContext(ctx.scope, ctx.scope_settings, ocr=NullOcr())
    assert image_mod.image_chunk(silent, png_bytes(color="green"), 1, set(), "x") is None


class FakeVisionClient:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.kwargs: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"message": {"content": self.reply}}


def test_ollama_captioner_sends_image_and_unloads() -> None:
    client = FakeVisionClient("  A cat\n on a sofa ")
    caption = image_mod.OllamaCaptioner(client, "vision")(Image.new("RGB", (2000, 1000)))
    assert caption == "A cat on a sofa"
    assert client.kwargs["keep_alive"] == 0
    assert client.kwargs["model"] == "vision"
    assert len(client.kwargs["messages"][0]["images"]) == 1


def test_ollama_captioner_swallows_errors() -> None:
    caption = image_mod.OllamaCaptioner(FakeVisionClient(ConnectionError("x")), "v")
    assert caption(Image.new("RGB", (10, 10))) == ""


class TestOcrModule:
    def test_to_png_downscales_long_side(self) -> None:
        data = ocr_mod.to_png(Image.new("RGBA", (400, 200)), max_dim=100)
        with Image.open(io.BytesIO(data)) as img:
            assert img.size == (100, 50)
            assert img.mode == "RGB"

    def test_to_png_keeps_small_images(self) -> None:
        with Image.open(io.BytesIO(ocr_mod.to_png(Image.new("RGB", (50, 20)), 100))) as img:
            assert img.size == (50, 20)

    def test_unavailable_engine_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = ocr_mod.WindowsOcr()
        monkeypatch.setattr(engine, "_get_engine", lambda: None)
        assert not engine.available()
        assert engine.ocr_image(Image.new("RGB", (10, 10))) == ""

    def test_recognition_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = ocr_mod.WindowsOcr()
        monkeypatch.setattr(engine, "_get_engine", object)

        async def boom(_png: bytes) -> str:
            raise RuntimeError("winrt failure")

        monkeypatch.setattr(engine, "_recognize", boom)
        assert engine.ocr_image(Image.new("RGB", (10, 10))) == ""

    def test_text_is_whitespace_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = ocr_mod.WindowsOcr()
        monkeypatch.setattr(engine, "_get_engine", object)

        async def fake(_png: bytes) -> str:
            return "a\n  b\t c"

        monkeypatch.setattr(engine, "_recognize", fake)
        assert engine.ocr_image(Image.new("RGB", (10, 10))) == "a b c"

    def test_real_windows_ocr_reads_rendered_text(self) -> None:
        engine = ocr_mod.WindowsOcr()
        if not engine.available():
            pytest.skip("Windows OCR is not available")
        image = Image.new("RGB", (900, 160), "white")
        ImageDraw.Draw(image).text((20, 50), "INVOICE 2026 PAYMENT", fill="black", font_size=48)
        assert "INVOICE" in engine.ocr_image(image).upper()
