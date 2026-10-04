"""Generate the scanned-document and photo fixtures used by ``eval/queries.yaml``.

Run once from the repo root (``.venv\\Scripts\\python scripts\\make_eval_fixtures.py``); the
output is committed so the evaluation is reproducible without this script. The files mimic what
breaks ranking in practice: image-only PDFs, a scanner watermark on every page, a near-blank
scan whose OCR is noise, and photos with and without text.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eval_corpus"
PAGE = (1700, 2200)  # US letter at 200 dpi
FONT = "C:/Windows/Fonts/arial.ttf"
WATERMARK = "Scanned with CamScanner"
SEED = 7

LEASE_PAGES = [
    [
        "RESIDENTIAL LEASE AGREEMENT",
        "",
        "This lease is made between the landlord, Maple Court Properties,",
        "and the tenant, Jordan Avery, for the apartment at 41 Birch Lane, Unit 3B.",
        "",
        "1. Term. The lease begins on March 1 and runs for twelve months.",
        "2. Monthly rent. The tenant pays a monthly rent of $1,450, due on the",
        "   first day of each month. A late fee of $50 applies after the fifth.",
        "3. Security deposit. The tenant pays a security deposit of $1,450,",
        "   returned within 30 days after move-out, less any damage.",
    ],
    [
        "4. Utilities. Water and trash are included; electricity and internet",
        "   are paid by the tenant.",
        "5. Pets. One cat or small dog is allowed with a $300 pet deposit.",
        "6. Maintenance. The landlord repairs the heating, plumbing and",
        "   appliances within seven days of written notice.",
        "",
        "Signed by the landlord and the tenant on February 12.",
    ],
]
WHITEBOARD = [
    "SPRINT PLANNING",
    "- migrate billing database to postgres",
    "- fix invoice PDF export",
    "- deadline: Friday",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def _speckle(image: Image.Image, rng: random.Random, dots: int) -> None:
    """Dust and toner specks, as a cheap scanner leaves them."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(dots):
        x, y = rng.randrange(width), rng.randrange(height)
        r = rng.choice((1, 1, 2, 3))
        draw.ellipse((x, y, x + r, y + r), fill=rng.randrange(0, 120))


def _scan_page(lines: list[str], rng: random.Random, *, watermark: bool) -> Image.Image:
    page = Image.new("L", PAGE, 245)
    draw = ImageDraw.Draw(page)
    body = _font(34)
    y = 200
    for line in lines:
        draw.text((160, y), line, font=body, fill=25)
        y += 60
    if watermark:
        draw.text((PAGE[0] - 520, PAGE[1] - 110), WATERMARK, font=_font(26), fill=110)
    _speckle(page, rng, 400)
    angle = rng.uniform(-0.8, 0.8)  # a page never lies perfectly straight on the glass
    return page.rotate(angle, fillcolor=245).filter(ImageFilter.GaussianBlur(0.6))


def _save_pdf(path: Path, pages: list[Image.Image]) -> None:
    pages[0].save(path, "PDF", resolution=200.0, save_all=True, append_images=pages[1:])


def _photo(rng: random.Random, size: tuple[int, int] = (1200, 900)) -> Image.Image:
    """A blurry, textless "photo": a sky gradient with a few soft shapes."""
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = int(120 + 100 * y / height)
        draw.line((0, y, width, y), fill=(shade // 2, shade - 30, shade))
    for _ in range(12):
        x, y, r = rng.randrange(width), rng.randrange(height), rng.randrange(40, 180)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)
    return image.filter(ImageFilter.GaussianBlur(6))


def _whiteboard(rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (1600, 1100), (232, 234, 228))
    draw = ImageDraw.Draw(image)
    y = 120
    for line in WHITEBOARD:
        draw.text((110, y), line, font=_font(64), fill=(20, 40, 120))
        y += 130
    _speckle(image, rng, 200)
    return image.rotate(rng.uniform(-2, 2), fillcolor=(200, 200, 195))


def main() -> None:
    rng = random.Random(SEED)  # noqa: S311 - reproducible noise, not crypto
    OUT.mkdir(parents=True, exist_ok=True)
    _save_pdf(
        OUT / "lease_agreement_scan.pdf",
        [_scan_page(lines, rng, watermark=True) for lines in LEASE_PAGES],
    )
    # The junk scan: nothing on its pages but the scanner's watermark and dust.
    _save_pdf(OUT / "scan_0042.pdf", [_scan_page([], rng, watermark=True) for _ in range(3)])
    blank = Image.new("L", PAGE, 240)
    _speckle(blank, rng, 3000)
    _save_pdf(OUT / "blank_scan.pdf", [blank.filter(ImageFilter.GaussianBlur(1.2))])
    _photo(rng).save(OUT / "IMG_2041.jpg", quality=80)
    _photo(rng).save(OUT / "passport.jpg", quality=80)
    _whiteboard(rng).save(OUT / "whiteboard_photo.jpg", quality=85)


if __name__ == "__main__":
    main()
