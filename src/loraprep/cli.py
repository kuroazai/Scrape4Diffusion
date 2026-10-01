"""Command line interface: prepare, command, inspect."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .captions import BlipCaptioner, Captioner, StubCaptioner
from .config import TrainingConfig
from .dataset import build_layout
from .images import find_images
from .kohya import as_shell, build_training_command
from .pipeline import prepare_dataset


def _config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(resolution=args.resolution, repeats=args.repeats)


def _cmd_prepare(args: argparse.Namespace) -> int:
    # Declared then branched rather than a ternary: mypy joins the two concrete
    # classes to `object` in a ternary, because neither inherits the Protocol.
    captioner: Captioner
    if args.no_model:
        captioner = StubCaptioner(args.caption)
    else:
        captioner = BlipCaptioner()
    if args.no_model:
        print("captioning disabled (--no-model); using a fixed caption")
    else:
        print("loading BLIP (first run downloads the model)...")
    try:
        result = prepare_dataset(
            args.source, args.out, args.name,
            captioner=captioner, config=_config(args),
            trigger=args.trigger, recursive=args.recursive,
            move_sources=args.move,
        )
    except (NotADirectoryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result)
    if result.prepared == 0:
        print("no usable images found", file=sys.stderr)
        return 1
    print(f"\nnext: loraprep command --out {args.out} --name {args.name} --base-model <model>")
    return 0


def _cmd_command(args: argparse.Namespace) -> int:
    layout = build_layout(args.out, args.name, args.repeats)
    try:
        command = build_training_command(layout, args.base_model, _config(args))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Run this from your kohya_ss checkout:\n")
    print(as_shell(command))
    print(f"\n{len(layout.images())} image(s), "
          f"{_config(args).max_train_steps(len(layout.images()))} steps")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    source = Path(args.source)
    try:
        images = find_images(source, recursive=args.recursive)
    except NotADirectoryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{source}: {len(images)} image(s)")
    by_suffix: dict[str, int] = {}
    for path in images:
        by_suffix[path.suffix.lower()] = by_suffix.get(path.suffix.lower(), 0) + 1
    for suffix, count in sorted(by_suffix.items()):
        print(f"  {suffix:<6} {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loraprep",
        description="Turn a folder of images into a trainable LoRA dataset.",
    )
    parser.add_argument("--version", action="version", version=f"loraprep {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--resolution", type=int, default=768)
    common.add_argument("--repeats", type=int, default=15)

    p = sub.add_parser("prepare", parents=[common], help="resize, caption and lay out a dataset")
    p.add_argument("source", help="folder of images")
    p.add_argument("-n", "--name", required=True, help="dataset / output LoRA name")
    p.add_argument("-o", "--out", default="datasets", help="where datasets live")
    p.add_argument("--trigger", help="trigger word prefixed to every caption")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--no-model", action="store_true", help="skip BLIP, use a fixed caption")
    p.add_argument("--caption", default="a photograph", help="caption used with --no-model")
    p.add_argument("--move", action="store_true",
                   help="delete source images after preparing (default: copy)")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("command", parents=[common], help="print the kohya training command")
    p.add_argument("-n", "--name", required=True)
    p.add_argument("-o", "--out", default="datasets")
    p.add_argument("-m", "--base-model", required=True, help="base SD model path or hub id")
    p.set_defaults(func=_cmd_command)

    p = sub.add_parser("inspect", help="count images in a folder")
    p.add_argument("source")
    p.add_argument("--recursive", action="store_true")
    p.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
