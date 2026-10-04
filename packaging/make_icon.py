"""Draw the app icon (a magnifier with a sparkle on a blue tile) to PNG and ICO files.

The PNG is what the tray loads at runtime; the ICO is for the exe and installer. Both come from
one drawing, rendered large and scaled down so the edges are smooth.
"""

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
CANVAS = 1024  # drawn at this size, then scaled down
OUTPUT = 256
TILE_MARGIN = 40
TILE_RADIUS = 230
GRADIENT_TOP = (112, 92, 255)  # violet
GRADIENT_BOTTOM = (36, 148, 255)  # sky blue
WHITE = (255, 255, 255, 255)
LENS_CENTRE = (440, 430)
LENS_RADIUS = 215
RING_WIDTH = 74
HANDLE_WIDTH = 96
HANDLE_LENGTH = 250
HANDLE_ANGLE_DEG = 45  # down and to the right
SPARKLE_RADIUS = 120
SPARKLE_WAIST = 0.22  # how thin the sparkle's arms pinch in (0 = needle, 1 = diamond)
LENS_TINT = (255, 255, 255, 46)
ASSET = Path(__file__).parents[1] / "src" / "localdoc_finder" / "app" / "assets" / "icon.png"


def draw_icon() -> Image.Image:
    image = _gradient_tile()
    overlay = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_magnifier(draw)
    _draw_sparkle(draw, LENS_CENTRE, SPARKLE_RADIUS)
    image.alpha_composite(overlay)
    return image.resize((OUTPUT, OUTPUT), Image.Resampling.LANCZOS)


def _gradient_tile() -> Image.Image:
    gradient = Image.new("RGBA", (CANVAS, CANVAS))
    pixels = gradient.load()
    assert pixels is not None
    for y in range(CANVAS):
        blend = y / (CANVAS - 1)
        row = tuple(
            round(top + (bottom - top) * blend)
            for top, bottom in zip(GRADIENT_TOP, GRADIENT_BOTTOM, strict=True)
        )
        for x in range(CANVAS):
            pixels[x, y] = (*row, 255)
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (TILE_MARGIN, TILE_MARGIN, CANVAS - TILE_MARGIN, CANVAS - TILE_MARGIN),
        radius=TILE_RADIUS,
        fill=255,
    )
    tile = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    tile.paste(gradient, mask=mask)
    return tile


def _draw_magnifier(draw: ImageDraw.ImageDraw) -> None:
    cx, cy = LENS_CENTRE
    draw.ellipse(
        (cx - LENS_RADIUS, cy - LENS_RADIUS, cx + LENS_RADIUS, cy + LENS_RADIUS),
        fill=LENS_TINT,
        outline=WHITE,
        width=RING_WIDTH,
    )
    angle = math.radians(HANDLE_ANGLE_DEG)
    start = (cx + math.cos(angle) * LENS_RADIUS, cy + math.sin(angle) * LENS_RADIUS)
    end = (
        start[0] + math.cos(angle) * HANDLE_LENGTH,
        start[1] + math.sin(angle) * HANDLE_LENGTH,
    )
    draw.line((start, end), fill=WHITE, width=HANDLE_WIDTH)
    cap = HANDLE_WIDTH / 2  # round the handle's far end
    draw.ellipse((end[0] - cap, end[1] - cap, end[0] + cap, end[1] + cap), fill=WHITE)


def _draw_sparkle(draw: ImageDraw.ImageDraw, centre: tuple[int, int], radius: int) -> None:
    """A four-point star: the mark for 'smart' search."""
    cx, cy = centre
    waist = radius * SPARKLE_WAIST
    points: list[tuple[float, float]] = []
    for step in range(8):
        angle = math.radians(step * 45 - 90)
        reach = radius if step % 2 == 0 else waist
        points.append((cx + math.cos(angle) * reach, cy + math.sin(angle) * reach))
    draw.polygon(points, fill=WHITE)


def main(target: Path) -> None:
    icon = draw_icon()
    icon.save(target, format="ICO", sizes=SIZES)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    icon.save(ASSET, format="PNG")


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("localdoc_finder.ico")
    )
