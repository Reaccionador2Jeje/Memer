"""Command-line trainer for the native Memer Transformer.

Usage:
    memer train corpus.txt --output memer-transformer.json
    python -m memer.models.train_cli corpus.txt --output memer-transformer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memer.models.byte_tokenizer import ByteTokenizer
from memer.models.data import encode_documents
from memer.models.training import Adam, evaluate_model, train_model
from memer.models.transformer import MemerTransformer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a small native Memer Transformer")
    parser.add_argument("dataset", type=Path, help="UTF-8 text corpus; one document per line")
    parser.add_argument("--output", type=Path, default=Path("memer-transformer.json"))
    parser.add_argument("--validation", type=Path, help="optional UTF-8 validation corpus")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--feed-forward-dim", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        parser.error(f"dataset file does not exist: {args.dataset}")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")

    text = args.dataset.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    documents = encode_documents(text, tokenizer)
    if not documents:
        parser.error("dataset contains no non-empty lines")

    model = MemerTransformer(
        dimension=args.dimension,
        heads=args.heads,
        layers=args.layers,
        feed_forward_dim=args.feed_forward_dim,
        block_size=args.block_size,
        seed=args.seed,
    )
    optimizer = Adam(learning_rate=args.learning_rate)
    history = train_model(
        model, documents, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, optimizer=optimizer,
    )
    if args.validation is not None:
        if not args.validation.is_file():
            parser.error(f"validation file does not exist: {args.validation}")
        validation_text = args.validation.read_text(encoding="utf-8")
        validation_documents = encode_documents(validation_text, tokenizer)
        if not validation_documents:
            parser.error("validation dataset contains no non-empty lines")
        validation_loss = evaluate_model(model, validation_documents, batch_size=args.batch_size)
        print(f"validation loss: {validation_loss:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    for epoch, loss in enumerate(history, start=1):
        print(f"epoch {epoch}/{len(history)} - loss: {loss:.6f}")
    print(f"checkpoint: {args.output}")
    print(f"parameters: {model.parameter_count():,}")
    print("Note: this is an experimental byte-level model; quality depends heavily on corpus size and training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
