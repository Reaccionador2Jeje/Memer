"""Utilities for discovering and validating local Memer checkpoints."""

from __future__ import annotations

from pathlib import Path

from memer.models.transformer import MemerTransformer, TransformerError

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


class CheckpointManagerError(ValueError):
    """Raised when a checkpoint cannot be inspected safely."""


def inspect_checkpoint(path: str | Path) -> dict[str, object]:
    """Load and validate a checkpoint, returning safe architecture metadata."""
    target = Path(path).expanduser()
    try:
        if target.is_symlink() or not target.is_file():
            raise CheckpointManagerError("checkpoint path must be a regular file")
        size = target.stat().st_size
        if size <= 0 or size > MAX_CHECKPOINT_BYTES:
            raise CheckpointManagerError(
                f"checkpoint size must be between 1 byte and {MAX_CHECKPOINT_BYTES} bytes"
            )
        model = MemerTransformer.load(target)
        modified = target.stat().st_mtime
    except CheckpointManagerError:
        raise
    except (OSError, TransformerError) as exc:
        raise CheckpointManagerError(str(exc)) from exc

    return {
        "path": str(target.resolve()),
        "size_bytes": size,
        "modified": modified,
        "format": "MEMER-TRANSFORMER",
        "format_version": 1,
        "config": {
            "vocab_size": model.vocab_size,
            "dimension": model.dimension,
            "heads": model.heads,
            "layers": model.layers,
            "feed_forward_dim": model.feed_forward_dim,
            "block_size": model.block_size,
            "seed": model.seed,
        },
        "parameters": model.parameter_count(),
    }


def validate_checkpoint(path: str | Path) -> dict[str, object]:
    """Validate a checkpoint and return its metadata."""
    return inspect_checkpoint(path)


def list_checkpoints(directory: str | Path = "checkpoints") -> list[dict[str, object]]:
    """List JSON checkpoints recursively, including validation errors."""
    root = Path(directory).expanduser()
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise CheckpointManagerError("checkpoint directory must be a real directory")

    results = []
    for candidate in sorted(root.rglob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            results.append(inspect_checkpoint(candidate))
        except CheckpointManagerError as exc:
            results.append({
                "path": str(candidate),
                "valid": False,
                "error": str(exc),
            })
        else:
            results[-1]["valid"] = True
    return results
