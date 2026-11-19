# Memer Models 0.1.0

## Scope

The first native Memer model is a deliberately small character-level neural language model. It is a research baseline for AtDev, not a foundation model and not a general-purpose conversational assistant.

## Network

1. Character tokenizer maps each Unicode character to an integer ID.
2. A fixed-size context window selects the most recent token IDs, padding with the beginning-of-sequence token.
3. The embedding layer maps context tokens to trainable vectors.
4. Context vectors are averaged.
5. A linear output layer produces one logit per vocabulary token.
6. Softmax converts logits into next-character probabilities.

Training uses next-character prediction, cross-entropy loss, and stochastic gradient descent. All parameters are implemented in Python using the standard library.

## Training

```bash
python scripts/train_memer_model.py \
  --data examples/memer-corpus.txt \
  --output models/memer-tiny-0.1.json \
  --epochs 20
```

The included corpus is only a smoke-test dataset. It is intentionally small and should not be interpreted as enough data to produce useful conversational behavior.

## Current limitations

- Character-level tokenization is inefficient compared with subword tokenizers.
- Context is short and represented by an average of token embeddings.
- The model predicts one character at a time.
- It has no instruction tuning, conversation memory, tool use, or safety tuning.
- Pure-Python training is intended for small experiments, not large-scale training.
- The current model has a maximum vocabulary of 256 tokens.

## Next research milestones

- Add dataset validation, train/validation split, and reproducible evaluation.
- Compare loss across training and validation data.
- Improve context representation and tokenization.
- Add a stable native provider to Memer Core after the model format is established.
- Keep model weights, datasets, and licenses documented.
