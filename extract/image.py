"""Images -> text (OCR + optional caption) so everything shares one text embedding space."""
import base64
import io
import os
from pathlib import Path
from typing import List

import imagehash
import xxhash
from PIL import Image

import indexer_config as cfg
from . import Chunk, ExtractError, ocr

Image.MAX_IMAGE_PIXELS = 200_000_000  # large photos are fine; decompression bombs are not


def caption(img: Image.Image) -> str:
    """One-sentence caption from a small vision model via Ollama (opt-in, idle-time only)."""
    if not cfg.ENABLE_IMAGE_CAPTIONS:
        return ""
    try:
        import ollama
        small = img.convert("RGB")
        small.thumbnail((896, 896))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=85)
        r = ollama.chat(
            model=cfg.CAPTION_MODEL, keep_alive=0,
            messages=[{"role": "user",
                       "content": "Describe this image in one or two sentences, mentioning any visible text, "
                                  "diagrams, UI, objects and scene.",
                       "images": [base64.b64encode(buf.getvalue()).decode()]}])
        return " ".join(r["message"]["content"].split())
    except Exception:
        return ""


def phash(img: Image.Image) -> str:
    return str(imagehash.phash(img))


def thumbnail_path(file_hash: str) -> Path:
    return cfg.DATA_DIR / "thumbs" / f"{file_hash}.jpg"


def make_thumbnail(img: Image.Image, file_hash: str) -> None:
    try:
        p = thumbnail_path(file_hash)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        t = img.convert("RGB")
        t.thumbnail((480, 480))
        t.save(p, format="JPEG", quality=80)
    except Exception:
        pass


def describe(img: Image.Image, label: str, with_caption: bool = True) -> str:
    """OCR text plus optional caption, labelled. '' when the image carries no text/caption."""
    parts = []
    text = ocr.ocr_image(img)
    if len(text) >= 4:
        parts.append(f"Text in image: {text}")
    if with_caption:
        cap = caption(img)
        if cap:
            parts.append(f"Caption: {cap}")
    return "\n".join(parts)


def extract_image(path: str) -> List[Chunk]:
    name = os.path.basename(path)
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        raise ExtractError(f"unreadable image: {e}")
    w, h = img.size
    if w < cfg.MIN_IMAGE_PIXELS and h < cfg.MIN_IMAGE_PIXELS:
        raise ExtractError("image too small")
    with open(path, "rb") as f:
        make_thumbnail(img, xxhash.xxh3_64_hexdigest(f.read()))
    body = describe(img, name)
    # Even with no text, the file name / folder makes the image findable by name.
    text = f"Image file: {name} ({w}x{h})\n{body}".strip()
    return [Chunk(text, "image", name)]
