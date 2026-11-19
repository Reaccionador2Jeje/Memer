"""Unified command-line interface for Memer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memer import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memer",
        description="Memer AI tools by AtDev",
        epilog="Run 'memer <command> --help' for command-specific help.",
    )
    parser.add_argument("--version", action="version", version=f"Memer {__version__}")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("info", help="show installation and runtime information")

    serve = commands.add_parser("serve", help="start the Memer Core HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    train = commands.add_parser("train", help="train the native Memer Transformer")
    train.add_argument("dataset", type=Path, help="UTF-8 text corpus")
    train.add_argument("--output", type=Path, default=Path("memer-transformer.json"))
    train.add_argument("--validation", type=Path)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--dimension", type=int, default=16)
    train.add_argument("--heads", type=int, default=2)
    train.add_argument("--layers", type=int, default=1)
    train.add_argument("--feed-forward-dim", type=int, default=32)
    train.add_argument("--block-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=17)

    checkpoint = commands.add_parser("checkpoint", help="manage local model checkpoints")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command")

    checkpoint_list = checkpoint_commands.add_parser("list", help="list checkpoints in a directory")
    checkpoint_list.add_argument("directory", nargs="?", type=Path, default=Path("checkpoints"))

    checkpoint_inspect = checkpoint_commands.add_parser("inspect", help="show checkpoint architecture")
    checkpoint_inspect.add_argument("path", type=Path)

    checkpoint_validate = checkpoint_commands.add_parser("validate", help="validate a checkpoint")
    checkpoint_validate.add_argument("path", type=Path)

    generate = commands.add_parser("generate", help="generate text from a saved native checkpoint")
    generate.add_argument("checkpoint", type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=80)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=17)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        print(f"Memer {__version__}")
        print("Project: AtDev / Memer")
        print(f"Python: {sys.version.split()[0]}")
        print("Native model: experimental byte-level decoder Transformer")
        print("Run 'memer --help' to see available commands.")
        return 0

    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("API dependencies are missing. Reinstall Memer with pip.") from exc
        uvicorn.run("memer.api:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "train":
        from memer.models.train_cli import main as train_main
        forwarded = [
            str(args.dataset), "--output", str(args.output),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--dimension", str(args.dimension), "--heads", str(args.heads),
            "--layers", str(args.layers), "--feed-forward-dim", str(args.feed_forward_dim),
            "--block-size", str(args.block_size), "--seed", str(args.seed),
        ]
        if args.validation is not None:
            forwarded.extend(["--validation", str(args.validation)])
        return train_main(forwarded)

    if args.command == "checkpoint":
        from memer.models.checkpoints import (
            CheckpointManagerError,
            inspect_checkpoint,
            list_checkpoints,
            validate_checkpoint,
        )

        try:
            if args.checkpoint_command == "list":
                items = list_checkpoints(args.directory)
                if not items:
                    print(f"No checkpoints found in {args.directory}")
                    return 0
                for item in items:
                    status = "valid" if item.get("valid", True) else "INVALID"
                    print(f"{status:7}  {item['path']}")
                    if item.get("valid", True):
                        print(
                            f"         {item['config']['dimension']}D | "
                            f"{item['config']['layers']} layer(s) | "
                            f"{item['parameters']} parameters | "
                            f"{item['size_bytes']} bytes"
                        )
                    else:
                        print(f"         {item['error']}")
                return 0

            if args.checkpoint_command in {"inspect", "validate"}:
                info = (
                    inspect_checkpoint(args.path)
                    if args.checkpoint_command == "inspect"
                    else validate_checkpoint(args.path)
                )
                print(f"Checkpoint: {info['path']}")
                print(f"Format: {info['format']} v{info['format_version']}")
                print(f"Parameters: {info['parameters']}")
                print(f"Size: {info['size_bytes']} bytes")
                for key, value in info["config"].items():
                    print(f"{key}: {value}")
                if args.checkpoint_command == "validate":
                    print("Validation: OK")
                return 0

            parser.error("choose checkpoint list, checkpoint inspect, or checkpoint validate")
        except CheckpointManagerError as exc:
            parser.error(str(exc))

    if args.command == "generate":
        from memer.models.byte_tokenizer import ByteTokenizer
        from memer.models.transformer import MemerTransformer

        if not args.checkpoint.is_file():
            parser.error(f"checkpoint file does not exist: {args.checkpoint}")
        if args.max_new_tokens < 0:
            parser.error("--max-new-tokens cannot be negative")
        if args.temperature <= 0:
            parser.error("--temperature must be greater than zero")
        model = MemerTransformer.load(args.checkpoint)
        print(model.generate(
            args.prompt,
            tokenizer=ByteTokenizer(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        ))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
