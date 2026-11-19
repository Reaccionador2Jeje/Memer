"""FastAPI application exposing the Memer Core HTTP API."""

from uuid import uuid4
from fastapi import FastAPI, HTTPException

from memer import __version__
from memer.config import Settings
from memer.providers import MockProvider, OllamaProvider
from memer.schemas import ChatRequest, ChatResponse, HealthResponse

settings = Settings.from_env()


def build_provider():
    if settings.provider == "mock":
        return MockProvider(model_name=settings.model_name)
    if settings.provider == "ollama":
        return OllamaProvider(
            model_name=settings.model_name,
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout,
        )
    raise RuntimeError(
        f"Unsupported provider '{settings.provider}'. "
        "Available providers: mock, ollama."
    )


try:
    provider = build_provider()
except (RuntimeError, ValueError) as exc:
    provider = None
    provider_error = str(exc)
else:
    provider_error = None

app = FastAPI(
    title="Memer API",
    description=(
        "AtDev's experimental AI service. Supports a development mock "
        "and locally hosted models through Ollama."
    ),
    version=__version__,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if provider is None:
        raise HTTPException(status_code=503, detail=provider_error)
    return HealthResponse(
        status="ok",
        service="memer-core",
        version=__version__,
        provider=provider.name,
        model=provider.model_name,
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if provider is None:
        raise HTTPException(status_code=503, detail=provider_error)
    try:
        content = provider.generate(request.messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        id=f"memer-{uuid4().hex}",
        model=request.model or provider.model_name,
        provider=provider.name,
        content=content,
    )
