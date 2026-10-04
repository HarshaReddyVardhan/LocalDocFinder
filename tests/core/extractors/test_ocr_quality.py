import pytest

from localdoc_finder.core.extractors.ocr import (
    MIN_OCR_QUALITY,
    is_low_content,
    ocr_quality,
)

# The kind of tokens OCR makes of dust, specks and texture on a page that holds no text.
DUSTY_BLANK_SCAN = "' . , i l | ~ - ii ;: .. ' ' Il 1 l. ,"
SPECKLED_PHOTO = "lll iiii | ~~ -- I1l '' rn ,, . ."
SCANNER_NOISE = "~ '\\ ._ ., l1 |i ;; .: l, II"

# Real text, including OCR slips seen on the eval corpus scans ("$1 ,450", "Scann:ed").
LEASE = (
    "RESIDENTIAL LEASE AGREEMENT This lease is made between the landlord, Maple Court "
    "Properties, and the tenant, Jordan Avery. 2. Monthly rent. The tenant pays $1 ,450."
)
WATERMARK_SLIP = "Scann:ed with CamScanner"
STATEMENT = "01/03 4,210.55 02/03 3,977.10 03/03 5,002.00 Total 13,189.65"
FRENCH = "Le locataire paie un loyer mensuel de mille euros, charges comprises."
CYRILLIC = "Договор аренды квартиры на двенадцать месяцев"


@pytest.mark.parametrize("noise", [DUSTY_BLANK_SCAN, SPECKLED_PHOTO, SCANNER_NOISE])
def test_noise_scores_low(noise: str) -> None:
    assert ocr_quality(noise) < MIN_OCR_QUALITY
    assert is_low_content(noise)


@pytest.mark.parametrize("text", [LEASE, STATEMENT, FRENCH, CYRILLIC])
def test_real_text_scores_high(text: str) -> None:
    assert ocr_quality(text) >= 0.8
    assert not is_low_content(text)


def test_an_ocr_slip_inside_a_word_still_counts_as_a_word() -> None:
    assert ocr_quality(WATERMARK_SLIP) == 1.0


def test_a_few_real_words_are_still_too_little() -> None:
    assert is_low_content(WATERMARK_SLIP)  # 21 letters: says nothing about the page
    assert is_low_content("")


def test_numbers_are_neither_words_nor_noise() -> None:
    assert ocr_quality("4,210.55 3,977.10") == 1.0
    assert ocr_quality("rent 4,210.55 ~ |") == pytest.approx(1 / 3)


def test_empty_text_scores_zero() -> None:
    assert ocr_quality("") == 0.0
    assert ocr_quality("  \n ") == 0.0


@pytest.mark.parametrize("token", ["lllll", "iiii", "xx1x", "bcd", "q"])
def test_repeated_or_vowelless_tokens_are_not_words(token: str) -> None:
    assert ocr_quality(token) == 0.0
