# Memer

**The AI Model by AtDev**

Memer is AtDev's AI project: an open and accessible foundation for building a conversational assistant, model integrations, and developer tools.

> **Development status:** early development. Memer Core 0.2.0 provides the API and provider interface. Memer Models 0.1.0 adds a small, trainable native character-level neural language model. It is an experimental research model, not a foundation model or a capable general-purpose assistant.

## Project goals

- Build **Memer Core**, the service that coordinates conversations and model providers.
- Expose a documented HTTP API for applications in the AtDev ecosystem.
- Develop **Memer Chat**, the user-facing chat platform.
- Integrate **MemeCode** as AtDev's official coding agent.
- Research, adapt, evaluate, and eventually train Memer models.
- Keep essential functionality accessible and avoid mandatory paid features.

## Architecture

```text
Memer Chat / MemeCode / AtDev apps
                |
             Memer API
                |
            Memer Core
                |
        Model provider interface
          /             \
 Development mock   Native Memer model
                    (experimental)
                          |
                    Memer Models
```

Memer Core separates the API from model execution. A provider can be changed through configuration without changing the chat endpoint.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
uvicorn memer.api:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

## Train the native Memer model (0.1.0)

The first native model is a small character-level neural network implemented with the Python standard library. It learns next-character patterns using an embedding layer, context averaging, and a softmax output layer. It does not call an external model API and does not require Ollama.

Run from the repository root after installing the project:

```bash
python scripts/train_memer_model.py \\
  --data examples/memer-corpus.txt \\
  --output models/memer-tiny-0.1.json \\
  --epochs 20
```

The script prints training loss, saves the learned parameters, and generates a short sample. The included corpus is only a smoke-test example; useful language quality requires a much larger, carefully prepared, legally usable text corpus. Training speed and output quality depend on corpus size and hardware. The model is intentionally limited to a small vocabulary and short context.

To load and generate from a saved model:

```python
from memer.models import TinyLanguageModel

model = TinyLanguageModel.load("models/memer-tiny-0.1.json")
print(model.generate("Memer", max_new_tokens=100))
```

## Use a local Ollama model (optional integration)

First, install and start Ollama separately, then download a model supported by your device. For example, after choosing an appropriate model:

```bash
ollama pull llama3.2:1b
```

Configure Memer before starting the API:

```bash
MEMER_PROVIDER=ollama
MEMER_MODEL_NAME=llama3.2:1b
MEMER_OLLAMA_BASE_URL=http://127.0.0.1:11434
MEMER_REQUEST_TIMEOUT=120
```

On Windows PowerShell, set environment variables with `$env:NAME="value"`. On Linux/macOS, use `export NAME=value`.

Then run:

```bash
uvicorn memer.api:app --reload
```

The Ollama service must be reachable from the machine running Memer. Model memory and speed depend on the selected model and available hardware.

## API request

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, Memer!"}
    ]
  }'
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the native model's training, text generation, and save/load cycle, plus the API and provider adapters. They do not require an Ollama server or downloaded model.

## Repository layout

```text
src/memer/
  api.py
  config.py
  schemas.py
  providers/
    base.py
    mock.py
    ollama.py
  models/
    __init__.py
    tokenizer.py
    bigram.py
scripts/
  train_memer_model.py
examples/
  memer-corpus.txt
tests/
  test_provider.py
  test_ollama.py
```

## Roadmap

1. **Memer Core:** stable API contracts and provider architecture.
2. **Model runtime:** expand local and remote model adapters.
3. **Memer Chat:** web chat client and conversation management.
4. **MemeCode integration:** secure tool interface for the official coding agent.
5. **Memer Models:** first native experimental neural language model (0.1.0); improve tokenization, context, training efficiency, evaluation, and datasets.
6. **Public platform:** documented API, deployment guides, and developer ecosystem.

## Project principles

- Open development and transparent capabilities.
- Accessible software and no mandatory purchases for essential features.
- Clear separation between experimental components and production-ready features.
- Respect for data privacy, model licenses, and dataset provenance.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Developed by **AtDev**.
