# Memer — Ecosystem Integration

Status: planning document; the experimental model is not represented as a production assistant.

## Components
Keep model training/evaluation, CLI, HTTP service and public playground as separate components with explicit interfaces.

## Provider contract
- Define request/response schemas and bounded input/output sizes.
- Return clear errors for missing models, invalid requests and unavailable providers.
- Do not log secrets or private prompts by default.
- Add request timeouts and concurrency limits before public exposure.
- Document model limitations and evaluation results honestly.

## MemeCode integration
Use an explicit provider adapter and capability declaration. Tool execution remains controlled by MemeCode's own permission and approval system; model output must never grant itself additional permissions.

## Next tasks
1. Add API schema and health endpoint tests.
2. Add deterministic smoke tests using a tiny local test model.
3. Document model format and checkpoint compatibility.
4. Add evaluation fixtures and publish limitations.