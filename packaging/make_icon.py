"""Draw the app icon (the tray icon's blue disc with an "S") to ``vector_embed.ico``."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
BLUE = "#4c7dff"
CANVAS = 256
MARGIN = 14
FONT_FILES = ("segoeuib.ttf", "arialbd.ttf")  # bold system fonts


def draw_icon() -> Image.Image:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((MARGIN, MARGIN, CANVAS - MARGIN, CANVAS - MARGIN), fill=BLUE)
    font = _bold_font(CANVAS * 5 // 8)
    draw.text((CANVAS // 2, CANVAS // 2), "S", fill="white", font=font, anchor="mm")
    return image


def _bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(target: Path) -> None:
    draw_icon().save(target, format="ICO", sizes=SIZES)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("vector_embed.ico"))
