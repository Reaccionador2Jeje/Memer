# Memer

**The AI Model by AtDev**

Memer is AtDev's AI project: an open and accessible foundation for building a conversational assistant, model integrations, and developer tools.

> **Development status:** early development. Memer Core 0.1.0 establishes the API and provider architecture. The initial mock provider is for development only; it is not a trained language model.

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
   Development mock   Model runtime
                       (planned)
```

Memer Core separates the API from model execution. This allows model runtimes to be added or replaced without changing the public chat contract.

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

Open the interactive API documentation at http://127.0.0.1:8000/docs.

## First API request

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, Memer!"}
    ]
  }'
```

The default development provider returns a clearly identified placeholder response. It does not perform real language-model inference.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
src/memer/
  api.py             HTTP API
  config.py          Environment-based configuration
  schemas.py         Request and response contracts
  providers/
    base.py          Provider interface
    mock.py          Development-only mock provider
tests/
  test_provider.py
```

## Roadmap

1. **Memer Core 0.1.0:** API contracts, provider interface, health endpoint, tests.
2. **Model runtime:** connect a locally hosted/open model through a provider adapter.
3. **Memer Chat:** web chat client and conversation management.
4. **MemeCode integration:** secure tool interface for the official coding agent.
5. **Memer Models:** evaluation, fine-tuning experiments, datasets, and model lifecycle.
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
