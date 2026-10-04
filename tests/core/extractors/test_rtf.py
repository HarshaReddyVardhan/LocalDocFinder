from collections.abc import Callable
from pathlib import Path

from tests.core.extractors.conftest import FakeOcr, Writer, png_bytes

from localdoc_finder.core.extractors.base import ExtractorSet
from localdoc_finder.core.extractors.rtf import read_rtf

Build = Callable[..., ExtractorSet]

HEADER = (
    r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0 Times New Roman;}{\f1 Arial;}}"
    r"{\colortbl;\red0\green0\blue0;}{\stylesheet{\s0 Normal;}}"
    r"{\*\generator Riched20;}{\info{\author Someone}}"
)


def picture(data: bytes, blip: str = "pngblip") -> str:
    return "{\\pict\\" + blip + r"\picw300\pich300 " + data.hex() + "}"


def test_formatting_tables_and_metadata_are_not_text() -> None:
    text, pictures = read_rtf((HEADER + r"\f0 Quarterly \b report\b0\par}").encode())
    assert text.strip() == "Quarterly report"
    assert pictures == []


def test_accents_and_unicode_are_decoded() -> None:
    body = HEADER + r"Caf\'e9 na\'efve \uc1\u8364? total\par Gr\u252\'fc\'dfe\par}"
    text, _ = read_rtf(body.encode())
    assert "Café naïve € total" in text
    assert "Grüße" in text


def test_a_png_picture_is_kept_and_its_wmf_twin_dropped() -> None:
    png = png_bytes((300, 300))
    body = (
        HEADER
        + r"Figure:\par{\*\shppict"
        + picture(png)
        + r"}{\nonshppict"
        + picture(b"wmf-bytes", "wmetafile8")
        + r"}\par}"
    )
    text, pictures = read_rtf(body.encode())
    assert pictures == [png]
    assert png.hex()[:16] not in text


def test_table_cells_become_a_row() -> None:
    body = HEADER + r"\trowd\intbl Name\cell Amount\cell\row\intbl Rent\cell 1200\cell\row}"
    text, _ = read_rtf(body.encode())
    assert "Name | Amount |" in text
    assert "Rent | 1200 |" in text


def test_rtf_with_pictures_is_chunked_and_ocrd(
    build_with_ocr: Build, write: Writer, ocr: FakeOcr
) -> None:
    png = png_bytes((300, 300))
    body = HEADER + r"Invoice from Acme\par" + picture(png) + picture(png) + r"\par}"
    chunks = build_with_ocr().extract(write("scan.rtf", body))
    assert chunks[0].kind == "doc"
    assert chunks[0].text == "Invoice from Acme"
    images = [c for c in chunks if c.kind == "image"]
    assert len(images) == 1  # the same picture twice is read once
    assert "OAuth login flow" in images[0].text
    assert ocr.calls == 1


def test_an_unknown_codepage_falls_back(write: Writer, extractors: ExtractorSet) -> None:
    body = r"{\rtf1\ansi\ansicpg99999 Plain words\par}"
    chunks = extractors.extract(write("odd.rtf", body))
    assert chunks[0].text == "Plain words"


def test_a_truncated_picture_loses_only_itself(tmp_path: Path) -> None:
    body = HEADER + r"Text stays\par{\pict\pngblip 89504e4}}"
    text, pictures = read_rtf(body.encode())
    assert "Text stays" in text
    assert pictures == []
