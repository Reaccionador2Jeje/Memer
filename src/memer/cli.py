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

    models = commands.add_parser("models", help="manage named local Memer models")
    model_commands = models.add_subparsers(dest="models_command")

    model_list = model_commands.add_parser("list", help="list registered models")
    model_list.add_argument("--registry", type=Path)

    model_add = model_commands.add_parser("add", help="register a validated native checkpoint")
    model_add.add_argument("checkpoint", type=Path)
    model_add.add_argument("--name", required=True)
    model_add.add_argument("--version", default="1.0.0")
    model_add.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    model_add.add_argument("--api-platform", default="Memer")
    model_add.add_argument("--api-key-env", default="MEMER_API_KEYS")
    model_add.add_argument("--registry", type=Path)

    model_use = model_commands.add_parser("use", help="select the active model")
    model_use.add_argument("name")
    model_use.add_argument("--registry", type=Path)

    model_inspect = model_commands.add_parser("inspect", help="inspect a registered model")
    model_inspect.add_argument("name")
    model_inspect.add_argument("--registry", type=Path)

    model_remove = model_commands.add_parser("remove", help="remove a model from the registry (keeps checkpoint)")
    model_remove.add_argument("name")
    model_remove.add_argument("--registry", type=Path)

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
        if os.getenv("MEMER_PROVIDER", "mock").strip().lower() == "native" and not os.getenv("MEMER_CHECKPOINT_PATH"):
            from memer.models.registry import get_active_model
            active_model = get_active_model()
            if active_model:
                os.environ["MEMER_CHECKPOINT_PATH"] = active_model["checkpoint_path"]
                os.environ["MEMER_MODEL_NAME"] = active_model["name"]
                os.environ["MEMER_BASE_URL"] = active_model["base_url"] or ""
                os.environ["MEMER_API_KEY_PLATFORM"] = active_model["api_platform"]
                os.environ["MEMER_API_KEY_ENV"] = active_model["api_key_env"] or ""
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

    if args.command == "models":
        from memer.models.registry import (
            ModelRegistryError, add_model, get_model, list_models, remove_model, use_model,
        )
        registry_path = args.registry
        try:
            if args.models_command == "list":
                entries = list_models(registry_path) if registry_path else list_models()
                if not entries:
                    print("No models registered. Use 'memer models add' to register one.")
                    return 0
                for item in entries:
                    marker = "*" if item["active"] else " "
                    print(f"{marker} {item['name']}:{item['version']}  [{item['provider']}]")
                    print(f"    Base URL: {item['base_url'] or '(not configured)'}")
                    print(f"    API key platform: {item['api_platform']}")
                return 0
            if args.models_command == "add":
                options = {"name": args.name, "version": args.version,
                           "base_url": args.base_url, "api_platform": args.api_platform,
                           "api_key_env": args.api_key_env}
                if registry_path:
                    options["path"] = registry_path
                item = add_model(args.checkpoint, **options)
                print(f"Registered: {item['name']}:{item['version']}")
                print(f"Base URL: {item['base_url'] or '(not configured)'}")
                print(f"API key platform: {item['api_platform']}")
                print("API key secret: not stored; configure it in the environment.")
                if item["active"]:
                    print("Selected as active model.")
                return 0
            if args.models_command == "use":
                item = use_model(args.name, path=registry_path) if registry_path else use_model(args.name)
                print(f"Active model: {item['name']}:{item['version']}")
                return 0
            if args.models_command == "inspect":
                item = get_model(args.name, path=registry_path) if registry_path else get_model(args.name)
                print(f"Model: {item['name']}")
                print(f"Version: {item['version']}")
                print(f"Provider: {item['provider']}")
                print(f"Checkpoint: {item['checkpoint_path']}")
                print(f"Base URL: {item['base_url'] or '(not configured)'}")
                print(f"API key platform: {item['api_platform']}")
                print(f"API key environment variable: {item['api_key_env'] or '(none)'}")
                print(f"Active: {'yes' if item['active'] else 'no'}")
                print(f"Parameters: {item['parameters']}")
                return 0
            if args.models_command == "remove":
                if registry_path:
                    remove_model(args.name, path=registry_path)
                else:
                    remove_model(args.name)
                print(f"Removed '{args.name}' from registry. Checkpoint file was kept.")
                return 0
            parser.error("choose models list, add, use, inspect, or remove")
        except (ModelRegistryError, ValueError) as exc:
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
