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

Memer Public is live at **https://memer-public-ity8fh.v2.appdeploy.ai/**. It is the browser playground for connecting to a Memer Core API. It does not pretend to generate answers when no model server is connected. To use the playground, start a Memer API instance and enter its reachable base URL.

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
5. Grow Memer Public into the official Memer web platform.
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
