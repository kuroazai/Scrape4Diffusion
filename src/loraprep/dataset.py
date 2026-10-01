"""The kohya_ss folder layout.

kohya infers repeat count from the image folder's name: `image/15_mysubject`
means "show each of these 15 times per epoch". Get that wrong and training
silently runs for the wrong number of steps, so the layout is built here rather
than left to the user.

    <root>/
      image/<repeats>_<name>/   images + matching .txt captions
      model/                    output safetensors
      log/                      tensorboard logs
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    """Where everything lives for one LoRA."""

    root: Path
    name: str
    repeats: int

    @property
    def image_root(self) -> Path:
        return self.root / "image"

    @property
    def image_dir(self) -> Path:
        """kohya reads the repeat count out of this folder's name."""
        return self.image_root / f"{self.repeats}_{self.name}"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def log_dir(self) -> Path:
        return self.root / "log"

    @property
    def output_file(self) -> Path:
        return self.model_dir / f"{self.name}.safetensors"

    def create(self) -> DatasetLayout:
        for directory in (self.image_dir, self.model_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def images(self) -> list[Path]:
        if not self.image_dir.is_dir():
            return []
        return sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".txt"
        )

    def is_trained(self) -> bool:
        """True once kohya has written the adapter - used to skip repeat work."""
        return self.output_file.exists()


def build_layout(root: str | Path, name: str, repeats: int) -> DatasetLayout:
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise ValueError(f"not usable as a folder name: {name!r}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return DatasetLayout(Path(root) / name, name, repeats)
