# Memer

**The AI model and developer platform by AtDev**

Memer is AtDev's open project for building a native language model, a developer CLI, an HTTP API, and a public web playground.

> **Development status:** experimental. Memer Models 0.2 uses a small byte-level decoder Transformer implemented in pure Python. It can be trained and used for text generation, but it is not yet a capable general-purpose assistant or a production foundation model. Small test corpora will produce poor or incoherent output.

## Project components

- **Memer Models:** native byte-level Transformer, training, evaluation, checkpoints, and generation.
- **Memer CLI:** one command-line entry point for model training, text generation, and starting Memer Core.
- **Memer Core:** FastAPI service with a documented chat API and provider interface.
- **Memer Public:** browser-based public playground and project information.
- **MemeCode:** planned integration as AtDev's coding agent.

## Quick install

Requires Python 3.10 or newer.

### Linux, macOS, and Termux

From the repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
memer --help
```

Or run the installer:

```bash
sh install.sh
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
memer --help
```

Or run the PowerShell installer from the repository:

```powershell
.\install.ps1
```

The installers install the current GitHub main branch. A virtual environment is recommended, especially on Linux distributions that restrict system-wide pip installs. The CLI is installed as the `memer` command on Windows, Linux, macOS, and compatible Python environments such as Termux.

## Memer CLI

Show available commands:

```bash
memer --help
memer --version
memer info
```

Start Memer Core:

```bash
memer serve --host 127.0.0.1 --port 8000
```

Open the API documentation at http://127.0.0.1:8000/docs.

### Train a native model

Create a UTF-8 text corpus with one document per line, then train:

```bash
memer train corpus.txt \
  --output checkpoints/memer-small.json \
  --epochs 3 \
  --batch-size 2 \
  --dimension 16 \
  --heads 2 \
  --layers 1 \
  --feed-forward-dim 32 \
  --block-size 32
```

Optional validation data:

```bash
memer train train.txt --validation validation.txt --output checkpoints/memer.json
```

### Manage local checkpoints

List checkpoints in the default `checkpoints/` folder:

```bash
memer checkpoint list
```

Inspect architecture and parameter count:

```bash
memer checkpoint inspect checkpoints/memer-small.json
```

Validate a checkpoint before using it:

```bash
memer checkpoint validate checkpoints/memer-small.json
```

The list command marks invalid JSON or incompatible checkpoints instead of treating them as usable models. Inspection loads the checkpoint through Memer's format and tensor validation. Checkpoint files are local; this feature does not upload them to Memer Public.

### Generate text from a checkpoint

```bash
memer generate checkpoints/memer-small.json \
  --prompt "Memer:" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --seed 17
```

Generation quality depends on model size, training time, and the quality, quantity, and licensing of the dataset. The included tiny example corpus is only for checking that training works; it is not enough to teach the model conversational language.

## Memer Public

Memer Public is live at **https://memer-public-ity8fh.v2.appdeploy.ai/**. It includes the browser playground for connecting to a Memer Core API and an authenticated account workspace with a private model metadata library.

Signed-in users can register model cards (name, version, architecture, and description) and remove them later. Model records are isolated by account. **This library stores metadata only**: it does not upload checkpoint weights or provide hosted inference yet. The next infrastructure phase is a dedicated Memer runtime with authentication, resource limits, and usage controls.

The playground does not pretend to generate answers when no model server is connected. To use it, start a Memer API instance and enter its reachable base URL.

The API must allow the website's origin through `MEMER_CORS_ORIGINS`. For example:

```bash
MEMER_CORS_ORIGINS="http://localhost:5173,https://memer-public-ity8fh.v2.appdeploy.ai" memer serve
```

Use the exact deployed website origin in the list. Do not use a wildcard for a publicly exposed API unless you understand the access implications.

## Run the native model through Memer Core

After training and saving a checkpoint, configure the native provider:

```bash
export MEMER_PROVIDER=native
export MEMER_MODEL_NAME=memer-small
export MEMER_CHECKPOINT_PATH=checkpoints/memer-small.json
export MEMER_MAX_NEW_TOKENS=80
export MEMER_TEMPERATURE=0.8
export MEMER_CORS_ORIGINS="http://localhost:5173,https://memer-public-ity8fh.v2.appdeploy.ai"
memer serve --host 127.0.0.1 --port 8000
```

Then open [Memer Public](https://memer-public-ity8fh.v2.appdeploy.ai/), select Playground, and connect to `http://127.0.0.1:8000` when the site and API are being used on the same device. For a remote API, use its reachable HTTPS address and configure its CORS origin. The native provider is experimental; a tiny checkpoint trained on a few lines will still generate low-quality text.



## Memer Runtime security (0.1)

Memer Core now supports a production API-key gate and basic inference resource limits. For a hosted instance, configure `MEMER_ENV=production`, set `MEMER_API_KEYS` as a server-side secret, and review [the runtime security guide](docs/MEMER_RUNTIME_0.1.md). Production mode refuses chat requests without a valid Bearer key, limits prompt size and generation concurrency, and caps native output length. The concurrency guard is single-process only; distributed quotas and per-user keys are not implemented yet. Never put a server-wide API key in public browser code.

## Memer Runtime 0.3 — checkpoint management

Memer CLI can list, inspect, and validate local Transformer checkpoints. See [the checkpoint commands](#manage-local-checkpoints). This is local file management only; hosted uploads and model serving from a remote registry are not implemented yet.

## Memer Runtime 0.2 — Docker deployment

For Linux servers and compatible container hosts, Memer includes a Dockerfile and Compose setup:

```bash
cp .env.example .env
# Edit .env with your runtime configuration and private API key.
# Put a trained checkpoint at ./checkpoints/model.json when using the native provider.
docker compose build
docker compose up -d
docker compose logs -f memer
```

The container runs as a non-root user with a read-only filesystem, a read-only checkpoint mount, dropped Linux capabilities, and loopback-only host binding by default. Configure HTTPS and a reverse proxy before remote exposure. See [the Runtime 0.2 deployment guide](docs/MEMER_RUNTIME_0.2.md).

**Android/Termux:** Docker is not part of the supported Termux setup. Continue using the Python virtual environment and `memer serve` directly on Android.

## HTTP API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, Memer!"}
    ]
  }'
```

The response contains `content`, `model`, and `provider`. The default development provider is a mock and is not an AI model. Memer Core also supports the experimental native Transformer provider and the optional Ollama adapter.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Architecture

```text
Memer Public / Memer CLI / AtDev apps
                 |
              Memer API
                 |
              Memer Core
                 |
         Provider / Model runtime
             /          \
       Development     Native Memer
          mock          Transformer
                            |
                       Memer Models
```

## Repository layout

```text
src/memer/
  api.py
  cli.py
  config.py
  schemas.py
  providers/
  models/
    byte_tokenizer.py
    data.py
    transformer.py
    backward.py
    training.py
    train_cli.py
install.sh
install.ps1
web/
  index.html
  src/
    main.ts
    styles.css
tests/
docs/
```

## Roadmap

1. Stabilize the Memer Models training and checkpoint formats.
2. Expand and evaluate native Transformer inference in Memer Core.
3. Improve tokenizer efficiency, training speed, evaluation, and dataset tooling.
4. Expand Memer CLI with model inspection and checkpoint management.
5. Add secure checkpoint uploads and a managed, resource-limited Memer inference runtime to Memer Public.
6. Integrate MemeCode through a clearly permissioned tool interface.

## Project principles

- Open development and transparent capabilities.
- Accessible software and no mandatory purchases for essential features.
- Experimental features are clearly labeled.
- Respect for privacy, model licenses, and dataset provenance.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Developed by **AtDev**.
