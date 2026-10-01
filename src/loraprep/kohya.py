"""Build the kohya_ss training command.

**This module builds a command. It never runs one.**

That is the whole design. The original `build_command()` called `os.system()` on
a multi-hour GPU training run and then returned the string, so inspecting what it
would do required already having done it. Here you get the argument list back and
decide.

It also builds a list, not a string, so a dataset name containing a quote cannot
break out into the shell.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from .config import TrainingConfig
from .dataset import DatasetLayout

TRAIN_SCRIPT = "train_network.py"


def build_training_command(
    layout: DatasetLayout,
    base_model: str | Path,
    config: TrainingConfig | None = None,
    *,
    accelerate: str = "accelerate",
    num_cpu_threads: int = 2,
) -> list[str]:
    """Arguments for a kohya LoRA run over `layout`.

    Raises if the dataset has no images: kohya would otherwise start, compute
    zero steps and exit after the model load, which looks like a crash.
    """
    cfg = config or TrainingConfig()
    images = layout.images()
    if not images:
        raise ValueError(
            f"no images in {layout.image_dir} - run `loraprep prepare` first"
        )

    return [
        accelerate, "launch",
        f"--num_cpu_threads_per_process={num_cpu_threads}",
        TRAIN_SCRIPT,
        "--enable_bucket",
        f"--pretrained_model_name_or_path={base_model}",
        # kohya wants the PARENT of the `<repeats>_<name>` folder; pointing it at
        # the image folder itself makes it find nothing and train on an empty set.
        f"--train_data_dir={layout.image_root}",
        f"--resolution={cfg.resolution},{cfg.resolution}",
        f"--output_dir={layout.model_dir}",
        f"--logging_dir={layout.log_dir}",
        f"--output_name={layout.name}",
        "--save_model_as=safetensors",
        "--network_module=networks.lora",
        f"--network_dim={cfg.network_dim}",
        f"--network_alpha={cfg.network_alpha}",
        f"--unet_lr={cfg.unet_lr}",
        f"--text_encoder_lr={cfg.text_encoder_lr}",
        f"--learning_rate={cfg.learning_rate}",
        f"--lr_scheduler={cfg.lr_scheduler}",
        f"--lr_warmup_steps={cfg.lr_warmup_steps}",
        f"--lr_scheduler_num_cycles={cfg.lr_scheduler_num_cycles}",
        f"--train_batch_size={cfg.train_batch_size}",
        f"--max_train_steps={cfg.max_train_steps(len(images))}",
        f"--save_every_n_epochs={cfg.save_every_n_epochs}",
        f"--mixed_precision={cfg.mixed_precision}",
        f"--save_precision={cfg.save_precision}",
        f"--optimizer_type={cfg.optimizer_type}",
        f"--seed={cfg.seed}",
        f"--bucket_reso_steps={cfg.bucket_reso_steps}",
        "--mem_eff_attn",
        "--gradient_checkpointing",
        "--bucket_no_upscale",
        *cfg.extra_args,
    ]


def as_shell(command: list[str]) -> str:
    """Render a command for copy-paste, quoted so paths with spaces survive."""
    return shlex.join(command)
