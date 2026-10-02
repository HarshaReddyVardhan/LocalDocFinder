"""Extractor contract, shared context and the registry-driven dispatcher."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from vector_embed.core.registry import Registry, discover_modules
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ChunkingSettings, ImageSettings, ScopeSettings

if TYPE_CHECKING:
    from PIL import Image

KIND_CODE = "code"
KIND_OUTLINE = "outline"
KIND_DOC = "doc"
KIND_IMAGE = "image"


@dataclass
class Chunk:
    """A searchable unit of text from a file."""

    text: str
    kind: str = KIND_DOC  # code | outline | doc | image
    symbol: str = ""  # "Class.method", a heading path, "slide 3"
    start_line: int = 0  # 1-based, 0 = not applicable
    end_line: int = 0
    page: int = 0  # PDF page / slide number, 0 = not applicable


class ExtractError(Exception):
    """A file could not be turned into text (corrupt, encrypted, ...). Never retried."""


class OcrEngine(Protocol):
    def available(self) -> bool: ...

    def ocr_image(self, image: "Image.Image") -> str:
        """Recognised text, or ``""`` when OCR is unavailable or finds nothing."""
        ...


class Captioner(Protocol):
    def __call__(self, image: "Image.Image") -> str: ...


class NullOcr:
    """OCR stand-in for machines without Windows OCR (and for tests)."""

    def available(self) -> bool:
        return False

    def ocr_image(self, image: "Image.Image") -> str:
        return ""


@dataclass(frozen=True)
class ExtractContext:
    """Everything an extractor may need; built once per run by the caller."""

    scope: ScopePolicy
    scope_settings: ScopeSettings
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    images: ImageSettings = field(default_factory=ImageSettings)
    ocr: OcrEngine = field(default_factory=NullOcr)
    captioner: Captioner | None = None
    thumbs_dir: Path | None = None


class Extractor(ABC):
    """One class per file type; subclass, set ``name`` and decorate with ``@register_extractor``."""

    name: ClassVar[str]
    priority: ClassVar[int] = 100  # lower runs first; the plain-text fallback is last
    is_document: ClassVar[bool] = False  # documents get the larger per-file chunk budget

    def __init__(self, ctx: ExtractContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def extract(self, path: Path) -> Iterable[Chunk]:
        """Chunks for ``path``; raise ``ExtractError`` when the file cannot be read."""
        ...


EXTRACTORS: Registry[type[Extractor]] = Registry("extractor")
register_extractor = EXTRACTORS.register

_BUILTIN_PACKAGE = "vector_embed.core.extractors"


class ExtractorSet:
    """Instantiated extractors for one context, ordered by priority."""

    def __init__(self, ctx: ExtractContext) -> None:
        discover_modules(_BUILTIN_PACKAGE)
        classes = sorted(EXTRACTORS, key=lambda cls: cls.priority)
        self.ctx = ctx
        self._extractors = [cls(ctx) for cls in classes]

    def select(self, path: Path) -> Extractor | None:
        return next((e for e in self._extractors if e.supports(path)), None)

    def extract(self, path: str | Path) -> list[Chunk]:
        """Chunks for ``path``, blank chunks dropped and size caps applied."""
        target = Path(path)
        extractor = self.select(target)
        if extractor is None:
            raise ExtractError(f"no extractor for {target.name}")
        cfg = self.ctx.chunking
        limit = cfg.max_chunks_per_doc if extractor.is_document else cfg.max_chunks_per_file
        return finalize(list(extractor.extract(target)), cfg, limit)


def finalize(chunks: list[Chunk], cfg: ChunkingSettings, limit: int) -> list[Chunk]:
    """Drop blank chunks, hard-cap text size and chunk count."""
    out: list[Chunk] = []
    for chunk in chunks:
        chunk.text = chunk.text.strip()
        if not chunk.text:
            continue
        if len(chunk.text) > cfg.max_chunk_chars:
            chunk.text = chunk.text[: cfg.max_chunk_chars]
        out.append(chunk)
    return out[:limit]
