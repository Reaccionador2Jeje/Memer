"""FastAPI application exposing the Memer Core HTTP API."""

from uuid import uuid4
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from memer import __version__
from memer.config import Settings
from memer.providers import MockProvider, NativeMemerProvider, OllamaProvider
from memer.schemas import ChatRequest, ChatResponse, HealthResponse

settings = Settings.from_env()


def build_provider():
    if settings.provider == "mock":
        return MockProvider(model_name=settings.model_name)
    if settings.provider == "native":
        return NativeMemerProvider(
            model_name=settings.model_name,
            checkpoint_path=settings.native_checkpoint_path,
            max_new_tokens=settings.native_max_new_tokens,
            temperature=settings.native_temperature,
            seed=settings.native_seed,
        )
    if settings.provider == "ollama":
        return OllamaProvider(
            model_name=settings.model_name,
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout,
        )
    raise RuntimeError(
        f"Unsupported provider '{settings.provider}'. "
        "Available providers: mock, native, ollama."
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
        "AtDev's experimental AI service. Supports a development mock, "
        "the native experimental Transformer, and locally hosted models through Ollama."
    ),
    version=__version__,
)

# Browser clients can connect when the operator explicitly allows their origin.
# Keep the default limited to local development instead of opening a public API.
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "MEMER_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
    if request.model and request.model != provider.model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Configured model is '{provider.model_name}'. Model switching is not available yet.",
        )
    try:
        content = provider.generate(request.messages)
    except Exception as exc:
        # Keep internal exception details out of public API responses.
        raise HTTPException(status_code=502, detail="Model provider failed") from exc

    return ChatResponse(
        id=f"memer-{uuid4().hex}",
        model=request.model or provider.model_name,
        provider=provider.name,
        content=content,
    )
