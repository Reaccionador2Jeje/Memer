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

## Stage 4: checkpoint persistence

`MemerTransformer.save()` stores configuration and all model tensors in a versioned JSON checkpoint. `MemerTransformer.load()` reconstructs the architecture and validates tensor names, dimensions, and finite numeric values before restoring weights. A save/load round trip should preserve forward logits exactly for the same inputs.

Checkpoints preserve initialized or modified weights, but this milestone does not add an optimizer or trainable gradient path. Training is still a future milestone.

## Stage 5: cross-entropy and output gradients

The new `memer.models.training.cross_entropy_with_gradients()` function calculates numerically stable mean next-token cross-entropy and its gradient with respect to the output logits. It supports batches, validates target IDs, and ignores padded targets through `IGNORE_INDEX`. This is the first training primitive; it does not yet propagate gradients through Transformer layers or update model parameters.

## Stage 6: backward primitives

The new `memer.models.backward` module implements analytical gradients for batched linear layers, LayerNorm, and ReLU. These functions are independently tested, including a finite-difference check for LayerNorm. They are building blocks only: full attention/block backward propagation and parameter updates are not implemented yet.

## Stage 7: causal attention backward

The `causal_attention_backward()` primitive now propagates gradients through the causal multi-head scaled dot-product attention operation and its query, key, value, and output projections. It returns input gradients plus gradients for all projection matrices and biases. The causal visibility rule and key-padding mask are applied during the backward pass. Tests compare selected analytical gradients with finite differences and verify masked positions receive no key/value gradients.

This function is an independently tested primitive. It is not yet wired into TransformerBlock or a full-model training loop.

## Stage 8: TransformerBlock backward

`TransformerBlock.backward()` now recomputes the block's forward activations and propagates gradients through both pre-LayerNorm residual branches: causal multi-head attention, the two LayerNorm operations, the ReLU feed-forward network, and both residual connections. It returns input gradients and a nested gradient dictionary for attention and feed-forward parameters. Tests include finite-difference checks for an input, a feed-forward weight, and an attention projection weight, plus padding-mask and invalid-shape checks.

This is a per-sequence block training primitive. The full model backward path, tied embedding gradients, optimizer, and training loop are still pending.

## Stage 9: full-model backward and tied embeddings

`MemerTransformer.backward()` now propagates gradients from logits through the final LayerNorm, every TransformerBlock, and the input token/position embeddings. Because the token embedding matrix is tied to the output projection, its returned gradient accumulates contributions from both paths. It accepts batches and per-row attention masks and returns a parameter-gradient tree matching the model checkpoint weight structure.

## Stage 10: Adam and training loop

The pure-Python `Adam` optimizer supports nested model parameter dictionaries/lists, bias correction, optional weight decay, and global gradient clipping. `train_batch()` runs forward, masked next-token cross-entropy, full backward, and one optimizer update. `train_model()` repeats this over shuffled batches and reports mean epoch losses.

## Stage 11: text generation and CLI

`MemerTransformer.generate()` provides seeded autoregressive sampling with a UTF-8 byte tokenizer. A small command-line trainer is available:

```bash
python -m memer.models.train_cli corpus.txt --output memer-transformer.json --epochs 3
```

Useful small-device settings can be supplied with `--dimension`, `--heads`, `--layers`, `--feed-forward-dim`, `--block-size`, and `--batch-size`. Pure-Python training is computationally expensive; start with a tiny corpus and small dimensions. A successful run proves the training pipeline works, not that the model has learned useful conversational behavior. Dataset quality and scale matter substantially.

## Next stages

1. Add validation metrics, optimizer-state checkpoint persistence, and more extensive training/evaluation tests.
2. Integrate the native model into Memer Core only after its format is stable.

The 0.2 model is experimental. It is not yet a conversational assistant or a production LLM.
