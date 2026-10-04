"""Kinds of files a user can choose to index, and the presets that bundle them.

The extensions of each kind live in ``ScopeSettings.kind_exts`` (overridable); this module only
names the kinds, describes them for people, and says which kinds each preset turns on.
"""

from dataclasses import dataclass
from enum import StrEnum


class FileKind(StrEnum):
    """A family of files, chosen or skipped as a whole."""

    DOCUMENTS = "documents"
    NOTES = "notes"
    IMAGES = "images"
    CODE = "code"
    DATA = "data"


@dataclass(frozen=True)
class KindInfo:
    label: str
    detail: str


KIND_INFO: dict[FileKind, KindInfo] = {
    FileKind.DOCUMENTS: KindInfo(
        "Documents",
        "PDF, Word, PowerPoint and RTF, with the text in their pictures and scanned pages",
    ),
    FileKind.NOTES: KindInfo("Text and notes", "Plain text, Markdown and LaTeX"),
    FileKind.IMAGES: KindInfo(
        "Images and scans",
        "Photos, screenshots and scanned pages (JPG, PNG, TIFF, ...); their text is read too",
    ),
    FileKind.CODE: KindInfo(
        "Source code", "Python, JavaScript, TypeScript, C#, Java, Go, Rust, web pages and more"
    ),
    FileKind.DATA: KindInfo("Data and config", "JSON, YAML, XML, CSV, SQL, INI and TOML"),
}

PRESET_DOCUMENTS = "documents"
PRESET_EVERYTHING = "everything"
PRESET_CUSTOM = "custom"

PRESET_KINDS: dict[str, frozenset[FileKind]] = {
    PRESET_DOCUMENTS: frozenset({FileKind.DOCUMENTS, FileKind.NOTES}),
    PRESET_EVERYTHING: frozenset(FileKind),
}
