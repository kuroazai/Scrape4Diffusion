"""Caption generation.

`Captioner` is a protocol so the pipeline can be tested without downloading a
1 GB model, and so a better captioner can be swapped in without touching the
pipeline. The original imported transformers and instantiated BLIP at module
scope, which meant importing the module downloaded a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


class Captioner(Protocol):
    """Anything that can describe an image."""

    def caption(self, image: Image.Image) -> str: ...


class StubCaptioner:
    """Returns a fixed caption. For tests, and for dry runs."""

    def __init__(self, text: str = "a photograph") -> None:
        self.text = text

    def caption(self, image: Image.Image) -> str:
        return self.text


class FilenameCaptioner:
    """Derives a caption from the file stem - useful when images are already
    named descriptively and you want no model at all."""

    def __init__(self, path_lookup: dict[int, Path] | None = None) -> None:
        self.path_lookup = path_lookup or {}

    def caption(self, image: Image.Image) -> str:
        path = self.path_lookup.get(id(image))
        if path is None:
            return ""
        return path.stem.replace("_", " ").replace("-", " ").strip()


class BlipCaptioner:
    """Salesforce BLIP image captioning.

    The model is loaded on first use, not at import, so `import loraprep` stays
    fast and offline. Requires the `blip` extra.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        *,
        prompt: str = "a photograph of",
        max_new_tokens: int = 40,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "BlipCaptioner needs transformers: pip install 'loraprep[blip]'"
            ) from exc
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)

    def caption(self, image: Image.Image) -> str:
        self._load()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(image, self.prompt, return_tensors="pt")
        output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._processor.decode(output[0], skip_special_tokens=True)
        # BLIP echoes the prompt back; strip it so captions do not all start
        # with the same five words, which weakens the trained association.
        if self.prompt and text.lower().startswith(self.prompt.lower()):
            text = text[len(self.prompt):]
        return text.strip()


def with_trigger(caption: str, trigger: str | None) -> str:
    """Prefix a trigger word, which is how a LoRA gets something to activate on."""
    if not trigger:
        return caption
    if not caption:
        return trigger
    return f"{trigger}, {caption}"
