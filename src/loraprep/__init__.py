"""loraprep - turn a folder of images into a trainable LoRA dataset.

    from loraprep import prepare_dataset, build_training_command, as_shell

    result = prepare_dataset("photos/", "datasets/", "mysubject", trigger="mysubj")
    print(as_shell(build_training_command(result.layout, "runwayml/stable-diffusion-v1-5")))

Resize, caption, lay out the kohya folder structure, and print the training
command. It does not run training, and it does not download your images from
anywhere - point it at files you already have.
"""
from .captions import BlipCaptioner, Captioner, FilenameCaptioner, StubCaptioner, with_trigger
from .config import TrainingConfig
from .dataset import DatasetLayout, build_layout
from .images import IMAGE_SUFFIXES, find_images, resize_to_square
from .kohya import as_shell, build_training_command
from .pipeline import PrepareResult, copy_into, prepare_dataset

__version__ = "0.2.0"

__all__ = [
    "TrainingConfig", "DatasetLayout", "build_layout",
    "find_images", "resize_to_square", "IMAGE_SUFFIXES",
    "Captioner", "BlipCaptioner", "StubCaptioner", "FilenameCaptioner", "with_trigger",
    "build_training_command", "as_shell",
    "prepare_dataset", "copy_into", "PrepareResult",
    "__version__",
]
