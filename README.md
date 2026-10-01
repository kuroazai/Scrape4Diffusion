# loraprep

**Turn a folder of images into a trainable LoRA dataset for [kohya_ss](https://github.com/bmaltais/kohya_ss).**

Resize without distorting, caption with BLIP, lay out the folder structure kohya
expects, and print the training command. Point it at images you already have.

```bash
pip install -e .

loraprep prepare photos/ --name mysubject --trigger mysubj
loraprep command --name mysubject --base-model runwayml/stable-diffusion-v1-5
```

---

## What it does

Preparing a LoRA dataset is four fiddly steps, and three of them fail silently if
you get them slightly wrong:

1. **Resize to a square** without stretching. Naively calling `resize((768, 768))`
   turns a portrait into a square and teaches the model that faces are wide. This
   letterboxes instead, preserving the aspect ratio.
2. **Caption every image**, one `.txt` per image, matched by filename stem. A
   mismatch trains images against the wrong text and you will not get an error.
3. **Lay out `image/<repeats>_<name>/`**. kohya reads the repeat count *out of the
   folder name*. Name it wrong and training quietly runs for the wrong number of
   steps.
4. **Build the training command** with the right argument set, pointed at the
   right directory — kohya wants `image/`, not `image/15_name`.

This does all four and then stops. **It does not run training**, and it does not
download images from anywhere.

## Usage

```bash
# What am I working with?
loraprep inspect photos/

# Resize, caption, lay out. Sources are COPIED, never moved.
loraprep prepare photos/ --name mysubject --trigger mysubj

# Skip the 1 GB model download while you are experimenting
loraprep prepare photos/ --name mysubject --no-model --caption "a portrait"

# Print the kohya command (does not run it)
loraprep command --name mysubject --base-model runwayml/stable-diffusion-v1-5
```

Result:

```
datasets/mysubject/
├── image/15_mysubject/
│   ├── mysubject_0001.png
│   ├── mysubject_0001.txt      "mysubj, a portrait of a person..."
│   └── ...
├── model/                       kohya writes mysubject.safetensors here
└── log/
```

### As a library

```python
from loraprep import prepare_dataset, build_training_command, as_shell, BlipCaptioner

result = prepare_dataset(
    "photos/", "datasets/", "mysubject",
    captioner=BlipCaptioner(),
    trigger="mysubj",
)
print(result)                      # 42 image(s) -> datasets/mysubject/image/15_mysubject

command = build_training_command(result.layout, "runwayml/stable-diffusion-v1-5")
print(as_shell(command))
```

## Design decisions worth knowing

**Nothing is deleted.** Sources are copied. `--move` opts into removing them, and
that is the only way it happens.

**The command builder only builds.** You get an argument list back and decide
whether to run it. A function that silently launches a multi-hour GPU job is a
trap, and you cannot inspect what it would do without it already having happened.

**The command is a list, not a string.** A dataset name containing a quote cannot
break out into your shell.

**Captioning is behind a protocol.** `BlipCaptioner` loads the model on first use,
not at import, so `import loraprep` stays fast and offline. `StubCaptioner` and
`FilenameCaptioner` need no model at all — which is why the test suite runs in
3 seconds with no torch installed.

**One corrupt file does not abort the run.** Unreadable images are skipped and
counted in the result.

## Tuning

`TrainingConfig` holds every knob:

```python
from loraprep import TrainingConfig

cfg = TrainingConfig(
    resolution=1024,
    repeats=10,
    network_dim=16,        # larger = more capacity, more overfitting risk
    unet_lr=1e-4,
    text_encoder_lr=5e-5,  # deliberately half the unet rate
    extra_args=["--xformers"],
)
```

The defaults produce a usable character or style LoRA from 20–60 images. `dim 8`
with `alpha 1` keeps the adapter small and resists memorisation on small sets;
the cosine schedule with warmup is what stops early steps wrecking the text
encoder, which is why its learning rate is lower.

Total steps are `images × repeats`, reported by `loraprep command`.

## Install

```bash
pip install -e .              # core: resize, layout, command building
pip install -e ".[blip]"      # + BLIP captioning (pulls torch)
pip install -e ".[dev]"       # + pytest, ruff, mypy
```

Python 3.10+. Core needs only Pillow — so the layout, resizing and command
building all work on a machine with no GPU stack at all.

## Development

```bash
pytest          # 44 tests, no model download, no GPU
ruff check src tests
mypy
```

## History

This repo began as an Instagram scraper that fed a LoRA pipeline. The scraper has
been removed: it breached Instagram's terms, it had stopped working against
Selenium 4, and it hardcoded local filesystem paths. What remains is the part
that was actually useful, rebuilt — and it works on any folder of images you
already own.

## Licence

MIT. See [LICENSE](LICENSE).
