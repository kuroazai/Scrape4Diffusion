"""Finding, resizing and copying source images.

Nothing here deletes anything. The original pipeline called `os.remove(image)` on
every source file after processing, so a bad run cost you the originals. Copying
is the default and moving is opt-in.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image

#: Formats Pillow reads reliably and Stable Diffusion tooling expects.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def find_images(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """Every image in `folder`, sorted so runs are reproducible."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    walker: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def resize_to_square(image: Image.Image, resolution: int) -> Image.Image:
    """Fit the image inside a square canvas without distorting it.

    The original called `img.resize((res, res))`, which stretches a portrait
    photo into a square and teaches the model that faces are wide. Letterboxing
    preserves the aspect ratio; the padding is black, which bucketing ignores.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    source = image.convert("RGB")
    source.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    offset = ((resolution - source.width) // 2, (resolution - source.height) // 2)
    canvas.paste(source, offset)
    return canvas


def iter_prepared(
    paths: Iterable[Path], resolution: int
) -> Iterator[tuple[Path, Image.Image]]:
    """Yield (source path, resized image) pairs, skipping files Pillow rejects.

    A single corrupt download should not abort a 200-image run.
    """
    for path in paths:
        try:
            with Image.open(path) as handle:
                yield path, resize_to_square(handle, resolution)
        except OSError:
            continue
