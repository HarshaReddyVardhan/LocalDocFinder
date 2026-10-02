import io
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from tests.core.extractors.conftest import FakeOcr, Writer, png_bytes

from vector_embed.core.extractors.base import ExtractError, ExtractorSet
from vector_embed.core.settings import ChunkingSettings, ImageSettings

Build = Callable[..., ExtractorSet]


def make_pdf(path: Path, pages: list[str], images: dict[int, bytes] | None = None) -> Path:
    doc = pymupdf.open()
    for number, text in enumerate(pages, 1):
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
        if images and number in images:
            page.insert_image(pymupdf.Rect(72, 200, 372, 500), stream=images[number])
    doc.save(path)
    doc.close()
    return path


class TestPdf:
    def test_text_per_page_with_page_numbers(
        self, extractors: ExtractorSet, tmp_path: Path
    ) -> None:
        pdf = make_pdf(tmp_path / "a.pdf", ["first page text " * 5, "second page text " * 5])
        chunks = extractors.extract(pdf)
        assert [(c.page, c.symbol) for c in chunks] == [(1, "p.1"), (2, "p.2")]
        assert "second page" in chunks[1].text

    def test_long_page_is_split(
        self, with_chunking: Callable[..., ExtractorSet], tmp_path: Path
    ) -> None:
        doc = pymupdf.open()
        page = doc.new_page()
        for i in range(70):
            page.insert_text((20, 20 + 11 * i), f"line {i} of a very long page", fontsize=8)
        doc.save(tmp_path / "long.pdf")
        doc.close()
        cfg = ChunkingSettings(max_chunk_chars=200, target_chunk_chars=100)
        chunks = with_chunking(cfg).extract(tmp_path / "long.pdf")
        assert len(chunks) > 1
        assert {c.page for c in chunks} == {1}

    def test_figure_text_is_ocrd_with_page(
        self, build_with_ocr: Build, tmp_path: Path, ocr: FakeOcr
    ) -> None:
        pdf = make_pdf(tmp_path / "d.pdf", ["intro text " * 5], images={1: png_bytes((300, 300))})
        chunks = build_with_ocr().extract(pdf)
        figure = next(c for c in chunks if c.kind == "image")
        assert figure.page == 1
        assert "OAuth login flow" in figure.text
        assert figure.symbol == "figure on p.1"

    def test_scanned_page_is_ocrd(
        self, build_with_ocr: Build, tmp_path: Path, ocr: FakeOcr
    ) -> None:
        ocr.text = "scanned contract clause seven"
        pdf = make_pdf(tmp_path / "scan.pdf", [""], images={1: png_bytes((600, 800))})
        chunks = build_with_ocr().extract(pdf)
        assert [(c.kind, c.page) for c in chunks] == [("doc", 1)]
        assert "clause seven" in chunks[0].text

    def test_ocr_budget_limits_work(
        self, build_with_ocr: Build, tmp_path: Path, ocr: FakeOcr
    ) -> None:
        pdf = make_pdf(
            tmp_path / "many.pdf",
            [""] * 5,
            images={
                n: png_bytes((600, 800), c)
                for n, c in enumerate(["red", "green", "blue", "gray", "black"], 1)
            },
        )
        build_with_ocr(ImageSettings(max_per_doc=2)).extract(pdf)
        assert ocr.calls == 2

    def test_duplicate_figures_are_ocrd_once(
        self, build_with_ocr: Build, tmp_path: Path, ocr: FakeOcr
    ) -> None:
        same = png_bytes()
        pdf = make_pdf(
            tmp_path / "dup.pdf", ["text " * 10, "more " * 10], images={1: same, 2: same}
        )
        build_with_ocr().extract(pdf)
        assert ocr.calls == 1

    def test_small_figures_are_skipped(
        self, build_with_ocr: Build, tmp_path: Path, ocr: FakeOcr
    ) -> None:
        pdf = make_pdf(tmp_path / "logo.pdf", ["text " * 10], images={1: png_bytes((40, 40))})
        build_with_ocr().extract(pdf)
        assert ocr.calls == 0

    def test_encrypted_pdf_is_rejected(self, extractors: ExtractorSet, tmp_path: Path) -> None:
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "secret")
        target = tmp_path / "enc.pdf"
        doc.save(target, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
        doc.close()
        with pytest.raises(ExtractError, match="encrypted"):
            extractors.extract(target)

    def test_corrupt_pdf_is_rejected(self, extractors: ExtractorSet, write: Writer) -> None:
        with pytest.raises(ExtractError, match="cannot open pdf"):
            extractors.extract(write("bad.pdf", b"%PDF-1.4 garbage"))


class TestDocx:
    def build(self, path: Path, with_image: bool = False) -> Path:
        import docx

        document = docx.Document()
        document.add_heading("Experience", level=1)
        document.add_paragraph("Built payment systems in Python.")
        document.add_heading("Skills", level=2)
        document.add_paragraph("FastAPI, PostgreSQL")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Language"
        table.cell(0, 1).text = "Years"
        table.cell(1, 0).text = "Python"
        table.cell(1, 1).text = "8"
        if with_image:
            document.add_picture(io.BytesIO(png_bytes()))
        document.save(path)
        return path

    def test_headings_become_symbols_and_tables_become_rows(
        self, extractors: ExtractorSet, tmp_path: Path
    ) -> None:
        chunks = extractors.extract(self.build(tmp_path / "r.docx"))
        by_symbol = {c.symbol: c.text for c in chunks}
        assert "Built payment systems" in by_symbol["Experience"]
        assert "FastAPI" in by_symbol["Experience > Skills"]
        assert "Language | Years" in by_symbol["table 1"]
        assert "Python | 8" in by_symbol["table 1"]

    def test_embedded_images_are_ocrd(self, build_with_ocr: Build, tmp_path: Path) -> None:
        chunks = build_with_ocr().extract(self.build(tmp_path / "img.docx", with_image=True))
        picture = next(c for c in chunks if c.kind == "image")
        assert "OAuth login flow" in picture.text

    def test_long_section_is_split(
        self, with_chunking: Callable[..., ExtractorSet], tmp_path: Path
    ) -> None:
        import docx

        document = docx.Document()
        document.add_heading("Big", 1)
        for i in range(40):
            document.add_paragraph(f"paragraph number {i} " * 6)
        document.save(tmp_path / "big.docx")
        cfg = ChunkingSettings(max_chunk_chars=300, target_chunk_chars=150)
        chunks = with_chunking(cfg).extract(tmp_path / "big.docx")
        assert len([c for c in chunks if c.symbol == "Big"]) > 1

    def test_corrupt_docx_is_rejected(self, extractors: ExtractorSet, write: Writer) -> None:
        with pytest.raises(ExtractError, match="cannot open docx"):
            extractors.extract(write("bad.docx", b"not a zip"))

    def test_blank_docx_without_zip_media(self, extractors: ExtractorSet, tmp_path: Path) -> None:
        import docx

        docx.Document().save(tmp_path / "empty.docx")
        assert extractors.extract(tmp_path / "empty.docx") == []


class TestPptx:
    def build(self, path: Path, with_image: bool = False) -> Path:
        from pptx import Presentation
        from pptx.util import Inches

        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[5])
        slide.shapes.title.text = "Quarterly roadmap"
        box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
        box.text_frame.text = "Ship the indexer"
        table = slide.shapes.add_table(1, 2, Inches(1), Inches(3), Inches(4), Inches(1)).table
        table.cell(0, 0).text = "Q1"
        table.cell(0, 1).text = "Search"
        slide.notes_slide.notes_text_frame.text = "mention the GPU budget"
        if with_image:
            slide.shapes.add_picture(io.BytesIO(png_bytes()), Inches(1), Inches(4))
        deck.save(path)
        return path

    def test_slide_text_table_and_notes(self, extractors: ExtractorSet, tmp_path: Path) -> None:
        (chunk,) = extractors.extract(self.build(tmp_path / "d.pptx"))
        assert (chunk.symbol, chunk.page) == ("slide 1", 1)
        for expected in ("Quarterly roadmap", "Ship the indexer", "Q1 | Search", "Notes: mention"):
            assert expected in chunk.text

    def test_pictures_are_ocrd(self, build_with_ocr: Build, tmp_path: Path) -> None:
        chunks = build_with_ocr().extract(self.build(tmp_path / "p.pptx", with_image=True))
        picture = next(c for c in chunks if c.kind == "image")
        assert picture.symbol == "image on slide 1"
        assert picture.page == 1

    def test_corrupt_pptx_is_rejected(self, extractors: ExtractorSet, write: Writer) -> None:
        with pytest.raises(ExtractError, match="cannot open pptx"):
            extractors.extract(write("bad.pptx", b"not a zip"))
