"""Environment-based settings for Memer Core."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    model_name: str = "memer-dev-mock"
    host: str = "127.0.0.1"
    port: int = 8000
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout: float = 120.0
    native_checkpoint_path: str = "checkpoints/memer-transformer.json"
    native_max_new_tokens: int = 80
    native_temperature: float = 0.8
    native_seed: int = 17

    @classmethod
    def from_env(cls) -> "Settings":
        port_value = os.getenv("MEMER_PORT", "8000")
        timeout_value = os.getenv("MEMER_REQUEST_TIMEOUT", "120")
        try:
            port = int(port_value)
            request_timeout = float(timeout_value)
        except ValueError as exc:
            raise ValueError("MEMER_PORT and MEMER_REQUEST_TIMEOUT must be numeric") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MEMER_PORT must be between 1 and 65535")
        if not 0 < request_timeout <= 600:
            raise ValueError("MEMER_REQUEST_TIMEOUT must be greater than 0 and at most 600")

        provider = os.getenv("MEMER_PROVIDER", "mock").strip().lower()
        default_model_name = "memer-transformer" if provider == "native" else "memer-dev-mock"
        try:
            native_max_new_tokens = int(os.getenv("MEMER_MAX_NEW_TOKENS", "80"))
            native_temperature = float(os.getenv("MEMER_TEMPERATURE", "0.8"))
            native_seed = int(os.getenv("MEMER_SEED", "17"))
        except ValueError as exc:
            raise ValueError("MEMER_MAX_NEW_TOKENS, MEMER_TEMPERATURE, and MEMER_SEED must be numeric") from exc
        if not 1 <= native_max_new_tokens <= 4096:
            raise ValueError("MEMER_MAX_NEW_TOKENS must be between 1 and 4096")
        if not 0 < native_temperature <= 5:
            raise ValueError("MEMER_TEMPERATURE must be greater than 0 and at most 5")

        return cls(
            provider=provider,
            model_name=os.getenv("MEMER_MODEL_NAME", default_model_name).strip(),
            host=os.getenv("MEMER_HOST", "127.0.0.1").strip(),
            port=port,
            ollama_base_url=os.getenv(
                "MEMER_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            request_timeout=request_timeout,
            native_checkpoint_path=os.getenv("MEMER_CHECKPOINT_PATH", "checkpoints/memer-transformer.json").strip(),
            native_max_new_tokens=native_max_new_tokens,
            native_temperature=native_temperature,
            native_seed=native_seed,
        )
