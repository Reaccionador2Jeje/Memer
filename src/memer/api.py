"""FastAPI application exposing the Memer Core HTTP API."""

from uuid import uuid4
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from memer import __version__
from memer.config import Settings
from memer.providers import MockProvider, NativeMemerProvider, OllamaProvider
from memer.schemas import ChatRequest, ChatResponse, HealthResponse
from memer.security import RuntimeSecurity

settings = Settings.from_env()
security = RuntimeSecurity(
    settings.api_keys,
    require_api_key=settings.require_api_key,
    max_concurrent_generations=settings.max_concurrent_generations,
)


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

runtime_config_error = (
    "Production runtime requires at least one MEMER_API_KEYS entry"
    if settings.require_api_key and not settings.api_keys
    else None
)

app = FastAPI(
    title="Memer API",
    description=(
        "AtDev's experimental AI service. Supports a development mock, "
        "the native experimental Transformer, and locally hosted models through Ollama."
    ),
    version=__version__,
)

# Browser clients can connect when the operator explicitly allows their origin.
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
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if runtime_config_error:
        raise HTTPException(status_code=503, detail=runtime_config_error)
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
def chat(
    request: ChatRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    if runtime_config_error:
        raise HTTPException(status_code=503, detail=runtime_config_error)
    if not security.is_authorized(authorization):
        raise HTTPException(
            status_code=401,
            detail="A valid Memer API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if provider is None:
        raise HTTPException(status_code=503, detail=provider_error)
    input_size = sum(len(message.content) for message in request.messages)
    if input_size > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Request exceeds the {settings.max_input_chars}-character input limit",
        )
    if request.model and request.model != provider.model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Configured model is '{provider.model_name}'. Model switching is not available yet.",
        )
    if not security.acquire_generation_slot():
        raise HTTPException(
            status_code=429,
            detail="Memer runtime is at capacity; retry shortly",
            headers={"Retry-After": "2"},
        )

    try:
        try:
            content = provider.generate(request.messages)
        except Exception as exc:
            # Keep internal exception details out of public API responses.
            raise HTTPException(status_code=502, detail="Model provider failed") from exc
    finally:
        security.release_generation_slot()

    return ChatResponse(
        id=f"memer-{uuid4().hex}",
        model=request.model or provider.model_name,
        provider=provider.name,
        content=content,
    )
