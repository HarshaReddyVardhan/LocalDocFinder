import io
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from localdoc_finder.core.extractors.base import Captioner, ExtractContext, ExtractorSet, OcrEngine
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.settings import ChunkingSettings, ImageSettings, ScopeSettings

Writer = Callable[[str, str | bytes], Path]


class FakeOcr:
    """Returns canned text and counts calls.

    ``pages`` gives each successive call its own text (then falls back to ``text``), for scans
    whose pages must differ: a line on every page is a header and is stripped.
    """

    def __init__(self, text: str = "diagram of the OAuth login flow") -> None:
        self.text = text
        self.pages: list[str] = []
        self.calls = 0

    def available(self) -> bool:
        return True

    def ocr_image(self, image: Image.Image) -> str:
        return " ".join(self.ocr_lines(image))

    def ocr_lines(self, image: Image.Image) -> list[str]:
        self.calls += 1
        text = self.pages[self.calls - 1] if self.calls <= len(self.pages) else self.text
        return text.splitlines()


def png_bytes(size: tuple[int, int] = (300, 300), color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def ctx(scope_settings: ScopeSettings) -> ExtractContext:
    return ExtractContext(ScopePolicy(scope_settings), scope_settings)


@pytest.fixture
def extractors(ctx: ExtractContext) -> ExtractorSet:
    return ExtractorSet(ctx)


@pytest.fixture
def with_chunking(ctx: ExtractContext) -> Callable[[ChunkingSettings], ExtractorSet]:
    """Build an ExtractorSet using custom chunk-size settings."""

    def build(chunking: ChunkingSettings) -> ExtractorSet:
        return ExtractorSet(
            ExtractContext(ctx.scope, ctx.scope_settings, chunking=chunking, images=ctx.images)
        )

    return build


@pytest.fixture
def no_merge(with_chunking: Callable[[ChunkingSettings], ExtractorSet]) -> ExtractorSet:
    """Extractors that never merge small neighbouring chunks, so each unit stays visible."""
    return with_chunking(ChunkingSettings(min_chunk_chars=0))


@pytest.fixture
def ocr() -> FakeOcr:
    return FakeOcr()


@pytest.fixture
def build_with_ocr(
    ctx: ExtractContext, ocr: FakeOcr, tmp_path: Path
) -> Callable[..., ExtractorSet]:
    """ExtractorSet wired with the fake OCR (and optionally captions / image limits)."""

    def build(
        images: ImageSettings | None = None,
        captioner: Captioner | None = None,
        engine: OcrEngine | None = None,
        thumbs: bool = False,
    ) -> ExtractorSet:
        return ExtractorSet(
            ExtractContext(
                ctx.scope,
                ctx.scope_settings,
                images=images or ImageSettings(),
                ocr=engine or ocr,
                captioner=captioner,
                thumbs_dir=tmp_path / "thumbs" if thumbs else None,
            )
        )

    return build


@pytest.fixture
def write(tmp_path: Path) -> Writer:
    def _write(name: str, content: str | bytes) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    return _write
