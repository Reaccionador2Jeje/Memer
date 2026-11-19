# Memer Models 0.2 — development notes

## Stage 1: stable byte tokenization

Memer 0.2 begins with a fixed UTF-8 byte tokenizer.

- IDs 0–255 represent raw UTF-8 bytes.
- ID 256 is BOS (beginning of sequence).
- ID 257 is EOS (end of sequence).
- ID 258 is PAD (padding).
- Vocabulary size is always 259, independent of the training corpus.
- Unicode text is encoded as UTF-8 and decoded with replacement for malformed byte sequences.

This makes token IDs reproducible across datasets and avoids building a character vocabulary for every training run. Byte tokenization is simple and robust, but sequences can be longer than with BPE or other learned subword tokenizers.

## Compatibility

The tokenizer is a separate module. The Memer 0.1 character model and its JSON checkpoints remain unchanged.

## Stage 2: causal next-token batches

`memer.models.data` now converts non-empty text lines into independent BOS...EOS documents and yields batches for next-token prediction. Inputs and targets are shifted by one token. Short rows are right-padded with PAD; their target positions use IGNORE_INDEX and their attention mask is zero. Shuffling is deterministic when a seed is supplied. This prepares data for a causal decoder; it does not implement attention or model training yet.

## Stage 3: decoder Transformer forward pass

`MemerTransformer` now implements a small decoder-only forward path: token and positional embeddings, pre-layer normalization, causal multi-head scaled dot-product self-attention, residual connections, a ReLU feed-forward network, final normalization, and tied token-embedding output projection. It accepts padded batches and an optional binary attention mask. The implementation uses only the Python standard library.

This milestone provides inference logits only. It does not yet include gradient computation, an optimizer, checkpoint persistence, or a training CLI. Pure-Python attention is intentionally for small experiments and will be slow for large sequence lengths.

## Next stages

1. Build causal training examples and padded batches.
2. Implement a small decoder-only Transformer with causal self-attention.
3. Add a memory-conscious training loop and validation metrics.
4. Add checkpoint save/load and a dedicated CLI.
5. Integrate the native model into Memer Core only after its format is stable.

The 0.2 model is experimental. It is not yet a conversational assistant or a production LLM.
