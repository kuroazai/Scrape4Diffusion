"""Fixtures. Images are generated here, so the suite needs no assets and no model."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

#: Deliberately not square. The original pipeline called resize((res, res)),
#: which stretches a portrait into a square and teaches the model that faces are
#: wide; these shapes make that regression visible.
SHAPES = [(400, 900), (1200, 300), (600, 600), (1000, 750)]


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    for index, (width, height) in enumerate(SHAPES):
        Image.new("RGB", (width, height), (index * 50, 120, 200)).save(
            folder / f"photo_{index}.jpg"
        )
    return folder


@pytest.fixture
def mixed_dir(tmp_path: Path) -> Path:
    """Images plus files that are not images, plus one that only claims to be."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    Image.new("RGB", (512, 512), "red").save(folder / "real.png")
    Image.new("RGB", (512, 512), "blue").save(folder / "real.webp")
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    (folder / "archive.zip").write_bytes(b"PK\x03\x04nope")
    (folder / "broken.jpg").write_bytes(b"this is not a jpeg")
    return folder


@pytest.fixture
def nested_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "nested"
    (folder / "a" / "b").mkdir(parents=True)
    Image.new("RGB", (300, 300), "green").save(folder / "top.png")
    Image.new("RGB", (300, 300), "green").save(folder / "a" / "mid.png")
    Image.new("RGB", (300, 300), "green").save(folder / "a" / "b" / "deep.png")
    return folder
