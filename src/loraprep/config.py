"""Training and dataset configuration.

Every knob in one dataclass rather than module-level globals, so a caller can run
two configurations in one process - which the original could not, because
`resolution` and `repeats` were import-time constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Defaults that produce a usable character/style LoRA.

    These are not arbitrary. network_dim 8 with alpha 1 keeps the adapter small
    and resists overfitting on the 20-60 image datasets this tooling is aimed at;
    a larger dim mostly memorises. The cosine schedule with warmup is what keeps
    early steps from wrecking the text encoder, which is why its learning rate is
    half the unet's.
    """

    resolution: int = 768
    repeats: int = 15

    network_dim: int = 8
    network_alpha: int = 1
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 324
    lr_scheduler_num_cycles: int = 3
    train_batch_size: int = 1
    save_every_n_epochs: int = 1
    mixed_precision: str = "fp16"
    save_precision: str = "fp16"
    optimizer_type: str = "AdamW"
    seed: int = 1234
    bucket_reso_steps: int = 64
    extra_args: list[str] = field(default_factory=list)

    def max_train_steps(self, image_count: int) -> int:
        """Total steps for a dataset of this size."""
        return image_count * self.repeats
