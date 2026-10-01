# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

`loraprep` turns a folder of images into a dataset kohya_ss can train a LoRA on.
Resize, caption, lay out, build the command.

**It does not train, and it does not acquire images.** Those are both out of
scope on purpose. Training belongs to kohya; acquisition belongs to you.

## The two invariants

> **1. Nothing is deleted unless the caller asked for it.**

Sources are copied. `move_sources=True` is the only path that removes anything.
This repo's predecessor called `os.remove()` on every source after processing, so
one bad run cost you the photographs.

> **2. `build_training_command()` builds. It never runs.**

It returns a list of arguments. The caller decides. The predecessor called
`os.system()` inside a function named `build_command` and returned the string
afterwards, which meant inspecting the command required already having launched a
multi-hour GPU job.

If you add anything that executes training, you have broken the contract the
whole package rests on.

## Layout

```
src/loraprep/
├── config.py      TrainingConfig - every knob, no module-level globals
├── images.py      discovery, aspect-preserving resize, tolerant iteration
├── captions.py    Captioner protocol; Blip / Stub / Filename implementations
├── dataset.py     DatasetLayout - the image/<repeats>_<name> structure
├── kohya.py       command construction only
├── pipeline.py    orchestration
└── cli.py         prepare / command / inspect
```

## Rules

### Resizing preserves aspect ratio.

`resize_to_square` thumbnails then letterboxes. Never `resize((res, res))` — that
distorts, and a dataset of stretched faces trains a model that produces stretched
faces. There is a parametrised test over three aspect ratios; do not weaken it.

### Captions pair by filename stem.

`mysubject_0001.png` needs `mysubject_0001.txt`. kohya matches on the stem, and a
mismatch trains an image against someone else's caption with no error. Covered by
`test_every_image_has_a_matching_caption_file`.

### The repeats folder name is load-bearing.

`image/15_mysubject` tells kohya to show each image 15 times per epoch. It is not
decoration. `build_layout()` constructs it; do not hand-build these paths.

And `--train_data_dir` points at `image/`, **not** `image/15_mysubject`. Point it
at the inner folder and kohya finds nothing and trains on an empty set.

### Heavy imports stay inside functions.

`BlipCaptioner` imports transformers in `_load()`, called on first caption. The
predecessor instantiated BLIP at module scope, so importing the module downloaded
a gigabyte. There is a test asserting `torch` is not in `sys.modules` after
importing the package — keep it passing.

### Commands are lists.

Never build a shell string by interpolation. `as_shell()` exists for display and
uses `shlex.join`. A dataset name with a quote in it must not reach a shell.

### Tolerate bad input, report it.

A corrupt image is skipped and counted, not raised. `PrepareResult.skipped` is how
the user finds out. One bad download should not cost a 200-image run.

## Adding a captioner

```python
class MyCaptioner:
    def caption(self, image: Image.Image) -> str:
        ...
```

That is the whole protocol. No base class, no registration. Import heavy
dependencies lazily inside the method, and raise an `ImportError` naming the
extra if they are missing.

## Testing

```bash
pytest          # 44 tests, ~3 seconds, no torch, no GPU, no network
```

Fixtures generate images with Pillow at non-square aspect ratios, because that is
what exposes the resize regression. There are no committed image assets and there
should not be.

Use `StubCaptioner` in tests. Anything requiring a model download does not belong
in the suite.

## Things not to do

- Don't add image acquisition. The scraper was removed deliberately: it breached
  Instagram's terms, had broken against Selenium 4, and hardcoded local paths.
- Don't make `transformers` or `torch` a hard dependency. They are the `blip`
  extra so the package works on a machine with no GPU stack.
- Don't execute training from library code.
- Don't commit datasets, images or `.safetensors`. `.gitignore` covers them.
