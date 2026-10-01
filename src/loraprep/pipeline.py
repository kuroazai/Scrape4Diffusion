"""Folder of photos in, trainable dataset out."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .captions import Captioner, StubCaptioner, with_trigger
from .config import TrainingConfig
from .dataset import DatasetLayout, build_layout
from .images import find_images, iter_prepared


@dataclass
class PrepareResult:
    layout: DatasetLayout
    prepared: int
    skipped: int
    sources_removed: int

    def __str__(self) -> str:
        parts = [f"{self.prepared} image(s) -> {self.layout.image_dir}"]
        if self.skipped:
            parts.append(f"{self.skipped} unreadable, skipped")
        if self.sources_removed:
            parts.append(f"{self.sources_removed} source(s) removed")
        return "; ".join(parts)


def prepare_dataset(
    source: str | Path,
    out_root: str | Path,
    name: str,
    *,
    captioner: Captioner | None = None,
    config: TrainingConfig | None = None,
    trigger: str | None = None,
    recursive: bool = False,
    move_sources: bool = False,
) -> PrepareResult:
    """Resize, caption and lay out a dataset ready for kohya.

    Sources are copied, not moved. `move_sources=True` opts into deleting the
    originals; the original tooling did that unconditionally, so a mistake cost
    you the photographs.
    """
    cfg = config or TrainingConfig()
    cap = captioner or StubCaptioner()
    layout = build_layout(out_root, name, cfg.repeats).create()

    found = find_images(source, recursive=recursive)
    prepared = removed = 0

    for index, (src_path, image) in enumerate(iter_prepared(found, cfg.resolution), start=1):
        stem = f"{name}_{index:04d}"
        image.save(layout.image_dir / f"{stem}.png")
        caption = with_trigger(cap.caption(image), trigger)
        (layout.image_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        prepared += 1
        if move_sources:
            src_path.unlink()
            removed += 1

    return PrepareResult(
        layout=layout,
        prepared=prepared,
        skipped=len(found) - prepared,
        sources_removed=removed,
    )


def copy_into(source: str | Path, destination: str | Path) -> int:
    """Copy images between folders without touching them. Returns the count."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in find_images(source):
        shutil.copy2(path, dest / path.name)
        count += 1
    return count
