"""Train the first native Memer character language model."""

from __future__ import annotations

import argparse
from pathlib import Path

from memer.models import TinyLanguageModel
from memer.models.tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Memer Models 0.1.0")
    parser.add_argument("--data", required=True, help="UTF-8 plain-text training corpus")
    parser.add_argument("--output", default="models/memer-tiny-0.1.json", help="output model file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    corpus_path = Path(args.data)
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    model = TinyLanguageModel(
        tokenizer,
        embedding_dim=args.embedding_dim,
        context_size=args.context_size,
        seed=args.seed,
    )
    metrics = model.train(
        text,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
    )
    output_path = model.save(args.output)

    print("Memer Models training complete")
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")
    print(f"Samples: {metrics['samples']}")
    print(f"Epochs: {metrics['total_epochs']}")
    print(f"Initial loss: {metrics['initial_loss']:.4f}")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Model saved: {output_path}")
    print()
    print("Sample generation:")
    print(model.generate(prompt="Memer", max_new_tokens=100, seed=args.seed))


if __name__ == "__main__":
    main()
