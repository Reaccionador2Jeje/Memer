# Memer Runtime 0.3 — local checkpoint management

This milestone adds CLI commands to discover, inspect, and validate local native Transformer checkpoints. It does not upload checkpoints, host a registry, or provide remote inference.

## Commands

List checkpoints recursively under the default `checkpoints/` directory:

```bash
memer checkpoint list
```

Choose another directory:

```bash
memer checkpoint list ./models
```

Inspect a checkpoint's architecture, parameter count, and file size:

```bash
memer checkpoint inspect ./checkpoints/memer-small.json
```

Validate the checkpoint format, configuration, tensor shapes, and finite numeric weights:

```bash
memer checkpoint validate ./checkpoints/memer-small.json
```

## Behavior

- Only regular JSON files are considered; symbolic links are skipped.
- Checkpoint files larger than 512 MiB are rejected before parsing.
- The list command reports invalid files individually and continues scanning.
- Inspect and validate use the existing strict `MemerTransformer.load()` implementation.
- Missing directories produce an empty list.
- Paths are expanded and displayed as filesystem paths.
- The checkpoint manager does not modify or delete checkpoint files.

## Example workflow

```bash
memer checkpoint list
memer checkpoint inspect checkpoints/memer-small.json
memer checkpoint validate checkpoints/memer-small.json
memer generate checkpoints/memer-small.json --prompt "Memer:" --max-new-tokens 40
```

## Limitations and next steps

Checkpoint management is local to the machine. Memer does not yet have a remote model registry, checkpoint upload/download API, checksum manifest, or model version promotion workflow. Memer Public continues to store model metadata only, not model weights.
