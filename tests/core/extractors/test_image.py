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


def tiff_pages(count: int) -> bytes:
    colours = ["white", "gray", "silver", "beige", "ivory"]
    pages = [Image.new("RGB", (600, 800), colours[i % len(colours)]) for i in range(count)]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def test_every_page_of_a_multi_page_tiff_scan_is_read(
    build_with_ocr: Build, write: Writer, ocr: FakeOcr
) -> None:
    chunks = build_with_ocr().extract(write("contract.tif", tiff_pages(3)))
    assert chunks[0].text == "Scanned document: contract.tif (3 pages)"
    assert [(c.kind, c.symbol, c.page) for c in chunks[1:]] == [
        ("doc", "page 1", 1),
        ("doc", "page 2", 2),
        ("doc", "page 3", 3),
    ]
    assert ocr.calls == 3


def test_a_tiff_scan_stops_at_the_page_allowance(
    build_with_ocr: Build, write: Writer, ocr: FakeOcr
) -> None:
    images = ImageSettings(max_scanned_pages=2)
    chunks = build_with_ocr(images).extract(write("long.tiff", tiff_pages(4)))
    assert "(2 pages)" in chunks[0].text
    assert ocr.calls == 2


def test_a_single_page_tiff_is_an_ordinary_image(
    build_with_ocr: Build, write: Writer, ocr: FakeOcr
) -> None:
    chunks = build_with_ocr().extract(write("photo.tif", tiff_pages(1)))
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Image file: photo.tif")


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
        self.generated: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> None:
        self.generated.append(kwargs)

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if isinstance(self.reply, Exception):
            raise self.reply
        return {"message": {"content": self.reply}}


def test_ollama_captioner_sends_image_and_stays_loaded_until_released() -> None:
    client = FakeVisionClient("  A cat   on a sofa ")
    freed: list[int] = []
    captioner = image_mod.OllamaCaptioner(client, "vision", free_gpu=lambda: freed.append(1))
    assert captioner(Image.new("RGB", (2000, 1000))) == "A cat on a sofa"
    assert captioner(Image.new("RGB", (10, 10))) == "A cat on a sofa"
    assert client.kwargs["keep_alive"] == "5m"  # loaded once for the whole pass
    assert client.kwargs["model"] == "vision"
    assert len(client.kwargs["messages"][0]["images"]) == 1
    assert freed == [1]  # the embedder was cleared once, before the vision model loaded
    assert client.generated == []
    captioner.release()
    assert client.generated == [{"model": "vision", "prompt": "", "keep_alive": 0}]
    captioner.release()
    assert len(client.generated) == 1  # nothing resident: no request that would load it


def test_releasing_an_unused_captioner_sends_nothing() -> None:
    client = FakeVisionClient("x")
    image_mod.OllamaCaptioner(client, "vision").release()
    assert client.generated == []


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


class TestImageCost:
    def test_the_decode_limit_is_checked_without_touching_a_global(
        self, build_with_ocr: Build, write: Writer
    ) -> None:
        before = Image.MAX_IMAGE_PIXELS
        small_limit = ImageSettings(max_decode_pixels=10_000)
        with pytest.raises(ExtractError, match="too large"):
            build_with_ocr(small_limit).extract(write("big.png", png_bytes((300, 300))))
        assert before == Image.MAX_IMAGE_PIXELS  # PIL's own setting is left alone

    def test_jpegs_are_decoded_at_a_reduced_size_and_the_real_size_is_reported(
        self, build_with_ocr: Build, write: Writer
    ) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (8000, 6000), "white").save(buffer, format="JPEG", quality=50)
        seen: list[tuple[int, int]] = []
        engine = FakeOcr()
        original = engine.ocr_image

        def spy(image: Image.Image) -> str:
            seen.append(image.size)
            return original(image)

        engine.ocr_image = spy  # type: ignore[method-assign]
        chunks = build_with_ocr(engine=engine).extract(write("huge.jpg", buffer.getvalue()))
        assert "(8000x6000)" in chunks[0].text  # what the user sees is the real size
        assert max(seen[0]) <= 4000  # but the OCR was given a much smaller decode

    def test_the_thumbnail_is_named_from_the_path_and_version_not_the_file_bytes(
        self, build_with_ocr: Build, write: Writer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractors = build_with_ocr(thumbs=True)
        path = write("t.png", png_bytes((900, 600)))
        reads: list[str] = []
        original = Path.read_bytes
        monkeypatch.setattr(
            Path, "read_bytes", lambda self: reads.append(self.name) or original(self)
        )
        extractors.extract(path)
        assert reads == []  # the image is not read a second time just to name its thumbnail
        info = path.stat()
        expected = image_mod.thumbnail_key(path, info.st_mtime_ns, info.st_size)
        assert (tmp_path / "thumbs" / f"{expected}.jpg").is_file()
        assert image_mod.thumbnail_path(tmp_path / "thumbs", path).is_file()

    def test_thumbnail_removal_is_safe_when_nothing_exists(self, tmp_path: Path) -> None:
        image_mod.remove_thumbnail(tmp_path / "thumbs", "x.png", 1, 1)
        image_mod.remove_thumbnail(None, "x.png", 1, 1)

    def test_an_identical_looking_image_is_not_ocrd_again(
        self, build_with_ocr: Build, write: Writer, ocr: FakeOcr
    ) -> None:
        extractors = build_with_ocr()
        first = extractors.extract(write("one.png", png_bytes((300, 300), "white")))
        copy = extractors.extract(write("copy.png", png_bytes((300, 300), "white")))
        assert ocr.calls == 1
        assert first[0].text.split("\n", 1)[1] == copy[0].text.split("\n", 1)[1]
        other = Image.new("RGB", (300, 300), "white")
        for x in range(0, 300, 20):  # a visibly different picture
            for y in range(300):
                other.putpixel((x, y), (0, 0, 0))
        buffer = io.BytesIO()
        other.save(buffer, format="PNG")
        extractors.extract(write("different.png", buffer.getvalue()))
        assert ocr.calls == 2
