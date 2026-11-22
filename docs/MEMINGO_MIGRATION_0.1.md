# Memingo-native migration plan

Status: A0 foundation; no production runtime has been migrated yet.

## Goal

Make Memingo the primary implementation language for Memer while preserving the working Python Transformer 0.2 as a compatibility/reference implementation until the Memingo runtime reaches feature parity.

## Current verified capabilities

- Memer currently contains a byte-level decoder Transformer, training/evaluation code, checkpoint handling, a native provider, CLI, and FastAPI service implemented in Python.
- Memingo provides an interpreted language, project/file/JSON/HTTP helpers, and a dependency-free CPU engine for small dense neural networks.
- Memingo's current AI API does not implement tensor operations, automatic differentiation for arbitrary graphs, causal attention, Transformer blocks, or GPU execution.

## Migration rules

1. Do not delete or rewrite the Python model, checkpoints, training pipeline, API, or tests during A0.
2. Keep Python as a temporary compatibility backend; new orchestration and user-facing workflows should move to Memingo first.
3. Add Memingo numerical primitives only with tests and documented shape/precision behavior.
4. Port the model in small parity-tested layers: tensor primitives -> matrix operations -> activations/loss -> gradients -> attention -> Transformer block -> optimizer -> training loop -> checkpoint parity -> inference provider.
5. Keep checkpoint format versioned. Any format change must include a converter or a documented compatibility boundary.
6. TUI is a first-class product surface, but it must call the same Memer Core/model interfaces as the CLI and API; it must not implement a separate inference path.

## A0 deliverables

- Establish this migration contract and a dedicated feature branch.
- Add a Memingo-native AI smoke example to verify the currently supported dense-network API.
- Keep the example explicitly labelled as an API smoke test, not an LLM or Transformer.
- Define acceptance criteria for the next stages.

## Stage plan

### A1 — Memingo numerical foundation
- Define numeric array/tensor representation, shape validation, indexing, and deterministic serialization.
- Add tested matrix multiplication, transpose, vector operations, reductions, and stable softmax.
- Add deterministic random initialization and seeded reproducibility.
- Document memory and maximum-size limits for Android/Termux as well as desktop/server use.

### A2 — Autograd and Transformer primitives
- Implement reverse-mode automatic differentiation for the supported operations.
- Add embeddings, positional encoding, causal masking, scaled dot-product attention, layer normalization, feed-forward layers, residual connections, and cross-entropy.
- Compare forward values and gradients against the existing Python reference on tiny deterministic fixtures.

### A3 — Memer Models 0.3
- Port the byte-level decoder architecture and training loop to Memingo.
- Add batching, validation metrics, checkpoint save/load, sampling controls, and reproducible seeds.
- Require loss/gradient parity tests and checkpoint round-trip tests before making Memingo the default backend.

### A4 — Core and CLI migration
- Implement model loading and inference in Memingo.
- Preserve the existing HTTP contract and security limits.
- Keep the Python API available until Memingo Core passes API compatibility tests.

### A5 — Memer TUI
- Provide Chat, Models, Training, Checkpoints, Settings, and Help views.
- Support chat streaming where the runtime permits it, command history, keyboard navigation, and graceful terminal resize.
- Use the same model/provider interface as CLI and API; never silently fall back to a mock provider.

### A6 — Cutover
- Make Memingo the default implementation only after parity, test, checkpoint, and API compatibility gates pass.
- Retain a documented Python import/conversion path for existing checkpoints and users during the transition.

## Acceptance gates

- Existing Python test suite remains green throughout migration.
- Memingo examples pass the Memingo interpreter's check/test/run workflow.
- Numerical fixtures match the reference within documented floating-point tolerances.
- Training is deterministic for a fixed seed and configuration.
- Saved checkpoints reload with identical model configuration and equivalent outputs.
- API authentication, CORS, prompt limits, output limits, and concurrency protections remain intact.
- TUI and CLI share one inference implementation.
