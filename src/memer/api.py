"""FastAPI application exposing the Memer Core HTTP API."""

from uuid import uuid4
from fastapi import FastAPI, HTTPException

from memer import __version__
from memer.config import Settings
from memer.providers import MockProvider
from memer.schemas import ChatRequest, ChatResponse, HealthResponse

settings = Settings.from_env()


def build_provider():
    if settings.provider == "mock":
        return MockProvider(model_name=settings.model_name)
    raise RuntimeError(
        f"Unsupported provider '{settings.provider}'. "
        "Only 'mock' is available in Memer Core 0.1.0."
    )


try:
    provider = build_provider()
except RuntimeError as exc:
    provider = None
    provider_error = str(exc)
else:
    provider_error = None

app = FastAPI(
    title="Memer API",
    description="AtDev's experimental AI service. The default provider is a development placeholder.",
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
        raise HTTPException(status_code=500, detail="Model provider failed") from exc

    return ChatResponse(
        id=f"memer-{uuid4().hex}",
        model=request.model or provider.model_name,
        provider=provider.name,
        content=content,
    )
