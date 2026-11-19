"""Local registry for named Memer model configurations."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from memer.models.checkpoints import CheckpointManagerError, inspect_checkpoint

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_REGISTRY = Path(os.getenv(
    "MEMER_REGISTRY_PATH",
    str(Path.home() / ".config" / "memer" / "models.json"),
)).expanduser()


class ModelRegistryError(ValueError):
    """Raised when a model registry operation is invalid."""


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ModelRegistryError(
            "model name must be 1-64 characters: letters, numbers, dot, underscore or hyphen"
        )
    return name


def _validate_base_url(base_url: str | None) -> str | None:
    if base_url in (None, ""):
        return None
    if not isinstance(base_url, str):
        raise ModelRegistryError("base URL must be text")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelRegistryError("base URL must be a valid http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ModelRegistryError("base URL must not contain credentials")
    return base_url.rstrip("/")


def _read(path: str | Path = DEFAULT_REGISTRY) -> dict:
    target = Path(path).expanduser()
    if not target.exists():
        return {"format": "MEMER-MODEL-REGISTRY", "version": 1, "active": None, "models": {}}
    if target.is_symlink() or not target.is_file():
        raise ModelRegistryError("registry path must be a regular file")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"cannot read model registry: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("format") != "MEMER-MODEL-REGISTRY"
        or data.get("version") != 1
        or not isinstance(data.get("models"), dict)
        or data.get("active") is not None and not isinstance(data.get("active"), str)
    ):
        raise ModelRegistryError("model registry format is invalid")
    return data


def _write(data: dict, path: str | Path = DEFAULT_REGISTRY) -> None:
    target = Path(path).expanduser()
    if target.is_symlink():
        raise ModelRegistryError("registry path cannot be a symbolic link")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".models-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        raise ModelRegistryError(f"cannot save model registry: {exc}") from exc


def list_models(path: str | Path = DEFAULT_REGISTRY) -> list[dict]:
    data = _read(path)
    return [
        {**entry, "active": name == data["active"]}
        for name, entry in sorted(data["models"].items())
    ]


def add_model(
    checkpoint: str | Path,
    *,
    name: str,
    version: str = "1.0.0",
    base_url: str | None = None,
    api_platform: str = "none",
    api_key_env: str | None = None,
    path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    name = _validate_name(name)
    if not isinstance(version, str) or not version.strip() or len(version) > 64:
        raise ModelRegistryError("version must contain 1-64 characters")
    if not isinstance(api_platform, str) or not api_platform.strip() or len(api_platform) > 64:
        raise ModelRegistryError("API key platform must contain 1-64 characters")
    if api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", api_key_env):
        raise ModelRegistryError("API key environment variable name is invalid")
    info = inspect_checkpoint(checkpoint)
    data = _read(path)
    if name in data["models"]:
        raise ModelRegistryError(f"model '{name}' is already registered")
    entry = {
        "name": name,
        "version": version.strip(),
        "provider": "native",
        "checkpoint_path": info["path"],
        "base_url": _validate_base_url(base_url),
        "api_platform": api_platform.strip(),
        "api_key_env": api_key_env,
        "architecture": info["config"],
        "parameters": info["parameters"],
    }
    data["models"][name] = entry
    if data["active"] is None:
        data["active"] = name
    _write(data, path)
    return {**entry, "active": data["active"] == name}


def get_model(name: str, path: str | Path = DEFAULT_REGISTRY) -> dict:
    name = _validate_name(name)
    data = _read(path)
    try:
        return {**data["models"][name], "active": data["active"] == name}
    except KeyError as exc:
        raise ModelRegistryError(f"model '{name}' is not registered") from exc


def use_model(name: str, path: str | Path = DEFAULT_REGISTRY) -> dict:
    name = _validate_name(name)
    data = _read(path)
    if name not in data["models"]:
        raise ModelRegistryError(f"model '{name}' is not registered")
    data["active"] = name
    _write(data, path)
    return {**data["models"][name], "active": True}


def remove_model(name: str, path: str | Path = DEFAULT_REGISTRY) -> None:
    name = _validate_name(name)
    data = _read(path)
    if name not in data["models"]:
        raise ModelRegistryError(f"model '{name}' is not registered")
    del data["models"][name]
    if data["active"] == name:
        data["active"] = next(iter(sorted(data["models"])), None)
    _write(data, path)


def get_active_model(path: str | Path = DEFAULT_REGISTRY) -> dict | None:
    data = _read(path)
    active = data["active"]
    if active is None:
        return None
    if active not in data["models"]:
        raise ModelRegistryError("active model entry is missing from registry")
    return {**data["models"][active], "active": True}
