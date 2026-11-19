"""Environment-based settings for Memer Core."""

from dataclasses import dataclass
import os


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


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
    api_keys: tuple[str, ...] = ()
    require_api_key: bool = False
    max_input_chars: int = 12000
    max_concurrent_generations: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            port = int(os.getenv("MEMER_PORT", "8000"))
            request_timeout = float(os.getenv("MEMER_REQUEST_TIMEOUT", "120"))
            native_max_new_tokens = int(os.getenv("MEMER_MAX_NEW_TOKENS", "80"))
            native_temperature = float(os.getenv("MEMER_TEMPERATURE", "0.8"))
            native_seed = int(os.getenv("MEMER_SEED", "17"))
            max_input_chars = int(os.getenv("MEMER_MAX_INPUT_CHARS", "12000"))
            max_concurrent_generations = int(os.getenv("MEMER_MAX_CONCURRENT_GENERATIONS", "1"))
        except ValueError as exc:
            raise ValueError("Memer numeric environment settings are invalid") from exc

        if not 1 <= port <= 65535:
            raise ValueError("MEMER_PORT must be between 1 and 65535")
        if not 0 < request_timeout <= 600:
            raise ValueError("MEMER_REQUEST_TIMEOUT must be greater than 0 and at most 600")
        if not 1 <= native_max_new_tokens <= 512:
            raise ValueError("MEMER_MAX_NEW_TOKENS must be between 1 and 512 for bounded runtime usage")
        if not 0 < native_temperature <= 5:
            raise ValueError("MEMER_TEMPERATURE must be greater than 0 and at most 5")
        if not 1024 <= max_input_chars <= 100000:
            raise ValueError("MEMER_MAX_INPUT_CHARS must be between 1024 and 100000")
        if not 1 <= max_concurrent_generations <= 8:
            raise ValueError("MEMER_MAX_CONCURRENT_GENERATIONS must be between 1 and 8")

        provider = os.getenv("MEMER_PROVIDER", "mock").strip().lower()
        default_model_name = "memer-transformer" if provider == "native" else "memer-dev-mock"
        api_keys = tuple(
            key.strip()
            for key in os.getenv("MEMER_API_KEYS", "").split(",")
            if key.strip()
        )
        if any(len(key) < 32 for key in api_keys):
            raise ValueError("Every MEMER_API_KEYS entry must contain at least 32 characters")

        production = os.getenv("MEMER_ENV", "development").strip().lower() == "production"
        require_api_key = production or _read_bool("MEMER_REQUIRE_API_KEY", False)

        return cls(
            provider=provider,
            model_name=os.getenv("MEMER_MODEL_NAME", default_model_name).strip(),
            host=os.getenv("MEMER_HOST", "127.0.0.1").strip(),
            port=port,
            ollama_base_url=os.getenv(
                "MEMER_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            request_timeout=request_timeout,
            native_checkpoint_path=os.getenv(
                "MEMER_CHECKPOINT_PATH", "checkpoints/memer-transformer.json"
            ).strip(),
            native_max_new_tokens=native_max_new_tokens,
            native_temperature=native_temperature,
            native_seed=native_seed,
            api_keys=api_keys,
            require_api_key=require_api_key,
            max_input_chars=max_input_chars,
            max_concurrent_generations=max_concurrent_generations,
        )
