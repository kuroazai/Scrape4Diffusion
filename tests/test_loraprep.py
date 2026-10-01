"""loraprep. Several of these pin defects the original pipeline had."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from loraprep import (
    IMAGE_SUFFIXES,
    DatasetLayout,
    StubCaptioner,
    TrainingConfig,
    as_shell,
    build_layout,
    build_training_command,
    find_images,
    prepare_dataset,
    resize_to_square,
    with_trigger,
)
from loraprep.captions import FilenameCaptioner
from loraprep.images import iter_prepared
from loraprep.pipeline import copy_into


# -- discovery -------------------------------------------------------------
def test_find_images_returns_only_images(mixed_dir):
    found = find_images(mixed_dir)
    names = {p.name for p in found}
    assert "real.png" in names
    assert "real.webp" in names
    assert "notes.txt" not in names
    assert "archive.zip" not in names


def test_find_images_is_sorted_for_reproducibility(photo_dir):
    found = find_images(photo_dir)
    assert found == sorted(found)


def test_find_images_is_not_recursive_by_default(nested_dir):
    assert len(find_images(nested_dir)) == 1


def test_find_images_recursive(nested_dir):
    assert len(find_images(nested_dir, recursive=True)) == 3


def test_find_images_rejects_a_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        find_images(target)


def test_supported_suffixes_are_lowercase():
    assert all(s == s.lower() and s.startswith(".") for s in IMAGE_SUFFIXES)


# -- resizing --------------------------------------------------------------
@pytest.mark.parametrize(("width", "height"), [(400, 900), (1200, 300), (900, 400)])
def test_resize_never_distorts(width, height):
    """The original stretched to a square. A 1200x300 photo became 768x768 and
    every face in the dataset got squashed."""
    source = Image.new("RGB", (width, height), "red")
    out = resize_to_square(source, 768)

    assert out.size == (768, 768)
    scale = min(768 / width, 768 / height)
    expected = (round(width * scale), round(height * scale))
    # The content occupies its aspect-correct box; the rest is padding.
    bbox = out.getbbox()
    content = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    assert abs(content[0] - expected[0]) <= 2
    assert abs(content[1] - expected[1]) <= 2


def test_resize_square_input_fills_the_canvas():
    out = resize_to_square(Image.new("RGB", (600, 600), "red"), 512)
    assert out.size == (512, 512)
    assert out.getbbox() == (0, 0, 512, 512)


def test_resize_converts_to_rgb():
    out = resize_to_square(Image.new("RGBA", (300, 300), (255, 0, 0, 128)), 256)
    assert out.mode == "RGB"


def test_resize_rejects_bad_resolution():
    with pytest.raises(ValueError):
        resize_to_square(Image.new("RGB", (10, 10)), 0)


def test_unreadable_files_are_skipped_not_fatal(mixed_dir):
    """One corrupt download should not abort a 200-image run."""
    paths = find_images(mixed_dir)
    assert any(p.name == "broken.jpg" for p in paths), "fixture should include it"
    prepared = list(iter_prepared(paths, 256))
    assert len(prepared) == len(paths) - 1


# -- dataset layout --------------------------------------------------------
def test_layout_encodes_repeats_in_the_folder_name(tmp_path):
    """kohya reads the repeat count out of the folder name. Get it wrong and
    training silently runs for the wrong number of steps."""
    layout = build_layout(tmp_path, "mysubject", 15)
    assert layout.image_dir.name == "15_mysubject"
    assert layout.image_dir.parent == layout.image_root


def test_layout_create_makes_every_directory(tmp_path):
    layout = build_layout(tmp_path, "sub", 10).create()
    assert layout.image_dir.is_dir()
    assert layout.model_dir.is_dir()
    assert layout.log_dir.is_dir()


def test_layout_is_trained_tracks_the_output_file(tmp_path):
    layout = build_layout(tmp_path, "sub", 10).create()
    assert layout.is_trained() is False
    layout.output_file.write_bytes(b"x")
    assert layout.is_trained() is True


def test_layout_images_excludes_caption_files(tmp_path):
    layout = build_layout(tmp_path, "sub", 10).create()
    Image.new("RGB", (10, 10)).save(layout.image_dir / "a.png")
    (layout.image_dir / "a.txt").write_text("caption", encoding="utf-8")
    assert [p.name for p in layout.images()] == ["a.png"]


@pytest.mark.parametrize("bad", ["", "with/slash", "with\\\\back", "q?mark", "a:b"])
def test_build_layout_rejects_unusable_names(tmp_path, bad):
    with pytest.raises(ValueError):
        build_layout(tmp_path, bad, 10)


def test_build_layout_rejects_zero_repeats(tmp_path):
    with pytest.raises(ValueError):
        build_layout(tmp_path, "sub", 0)


# -- captions --------------------------------------------------------------
def test_with_trigger_prefixes():
    assert with_trigger("a portrait", "mysubj") == "mysubj, a portrait"


def test_with_trigger_handles_empties():
    assert with_trigger("a portrait", None) == "a portrait"
    assert with_trigger("", "mysubj") == "mysubj"


def test_stub_captioner_is_deterministic():
    cap = StubCaptioner("fixed")
    image = Image.new("RGB", (8, 8))
    assert cap.caption(image) == cap.caption(image) == "fixed"


def test_filename_captioner_cleans_the_stem():
    image = Image.new("RGB", (8, 8))
    cap = FilenameCaptioner({id(image): Path("a/my_subject-portrait.jpg")})
    assert cap.caption(image) == "my subject portrait"


def test_importing_loraprep_does_not_load_a_model():
    """The original instantiated BLIP at module scope, so importing it
    downloaded a gigabyte."""
    import sys
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# -- pipeline --------------------------------------------------------------
def test_prepare_writes_an_image_and_a_caption_per_photo(photo_dir, tmp_path):
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner("a photo"))
    assert result.prepared == 4
    images = list(result.layout.image_dir.glob("*.png"))
    captions = list(result.layout.image_dir.glob("*.txt"))
    assert len(images) == len(captions) == 4


def test_every_image_has_a_matching_caption_file(photo_dir, tmp_path):
    """kohya pairs them by stem. A mismatch trains images against the wrong text."""
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner("a photo"))
    stems = {p.stem for p in result.layout.image_dir.glob("*.png")}
    caption_stems = {p.stem for p in result.layout.image_dir.glob("*.txt")}
    assert stems == caption_stems


def test_sources_are_preserved_by_default(photo_dir, tmp_path):
    """The original ran os.remove() on every source. A bad run cost the originals."""
    before = len(find_images(photo_dir))
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner())
    assert len(find_images(photo_dir)) == before
    assert result.sources_removed == 0


def test_move_sources_is_opt_in(photo_dir, tmp_path):
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner(), move_sources=True)
    assert result.sources_removed == 4
    assert find_images(photo_dir) == []


def test_trigger_word_reaches_every_caption(photo_dir, tmp_path):
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner("a photo"), trigger="mysubj")
    for caption in result.layout.image_dir.glob("*.txt"):
        assert caption.read_text(encoding="utf-8").startswith("mysubj,")


def test_prepare_skips_unreadable_and_reports_it(mixed_dir, tmp_path):
    result = prepare_dataset(mixed_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner())
    assert result.prepared == 2
    assert result.skipped == 1


def test_prepare_respects_resolution(photo_dir, tmp_path):
    result = prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                             captioner=StubCaptioner(),
                             config=TrainingConfig(resolution=512))
    with Image.open(next(result.layout.image_dir.glob("*.png"))) as img:
        assert img.size == (512, 512)


def test_copy_into_is_non_destructive(photo_dir, tmp_path):
    count = copy_into(photo_dir, tmp_path / "copy")
    assert count == 4
    assert len(find_images(photo_dir)) == 4


# -- kohya command ---------------------------------------------------------
def _prepared(photo_dir, tmp_path, **kwargs) -> DatasetLayout:
    return prepare_dataset(photo_dir, tmp_path / "ds", "sub",
                           captioner=StubCaptioner(), **kwargs).layout


def test_command_is_built_not_executed(photo_dir, tmp_path, monkeypatch):
    """The original's build_command() called os.system() on a multi-hour GPU run
    and returned the string afterwards, so you could not inspect it first."""
    import os
    called = []
    monkeypatch.setattr(os, "system", lambda cmd: called.append(cmd))
    build_training_command(_prepared(photo_dir, tmp_path), "base-model")
    assert called == []


def test_command_points_at_the_parent_of_the_repeats_folder(photo_dir, tmp_path):
    """kohya wants image/, not image/15_sub. Pointing it at the inner folder
    makes it find nothing and train on an empty set."""
    layout = _prepared(photo_dir, tmp_path)
    command = build_training_command(layout, "base-model")
    arg = next(a for a in command if a.startswith("--train_data_dir="))
    assert arg.endswith(str(layout.image_root))
    assert "15_sub" not in arg


def test_step_count_is_images_times_repeats(photo_dir, tmp_path):
    layout = _prepared(photo_dir, tmp_path)
    command = build_training_command(layout, "base-model", TrainingConfig(repeats=15))
    assert "--max_train_steps=60" in command  # 4 images * 15


def test_command_refuses_an_empty_dataset(tmp_path):
    layout = build_layout(tmp_path, "sub", 15).create()
    with pytest.raises(ValueError, match="no images"):
        build_training_command(layout, "base-model")


def test_command_carries_the_lora_hyperparameters(photo_dir, tmp_path):
    command = build_training_command(_prepared(photo_dir, tmp_path), "base-model")
    assert "--network_module=networks.lora" in command
    assert "--network_dim=8" in command
    assert "--network_alpha=1" in command
    assert "--save_model_as=safetensors" in command


def test_command_is_a_list_so_names_cannot_break_out(photo_dir, tmp_path):
    """A string-interpolated command would let a quote in a name reach the shell."""
    command = build_training_command(_prepared(photo_dir, tmp_path), 'base "model"')
    assert isinstance(command, list)
    rendered = as_shell(command)
    assert rendered.count('base "model"') == 0 or "\\\\" in rendered or "'" in rendered


def test_extra_args_are_appended(photo_dir, tmp_path):
    cfg = TrainingConfig(extra_args=["--xformers"])
    command = build_training_command(_prepared(photo_dir, tmp_path), "m", cfg)
    assert command[-1] == "--xformers"


def test_max_train_steps_helper():
    assert TrainingConfig(repeats=15).max_train_steps(20) == 300
