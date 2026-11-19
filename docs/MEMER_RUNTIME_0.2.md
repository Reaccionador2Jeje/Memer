# Memer Runtime 0.2 — reproducible container deployment

This milestone adds a Docker image and Docker Compose deployment for Memer Core. It is intended for Linux servers and compatible container hosts. It does not make Docker available inside Android/Termux, and it does not provide hosted model training or checkpoint uploads.

## Requirements

- Docker Engine with the Compose plugin
- A trained Memer checkpoint if using the native provider
- A private production API key

## Prepare configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` on the server. For a native production runtime, configure:

```dotenv
MEMER_ENV=production
MEMER_PROVIDER=native
MEMER_MODEL_NAME=memer-small
MEMER_CHECKPOINT_PATH=/app/checkpoints/model.json
MEMER_API_KEYS=replace-with-a-random-secret-of-at-least-32-characters
MEMER_MAX_NEW_TOKENS=80
MEMER_MAX_INPUT_CHARS=12000
MEMER_MAX_CONCURRENT_GENERATIONS=1
MEMER_CORS_ORIGINS=https://memer-public-ity8fh.v2.appdeploy.ai
```

Generate a key locally:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Never commit `.env` or share its contents. Keep the key in the host's secret-management facility where available.

## Add the checkpoint

Place the trained checkpoint at:

```text
./checkpoints/model.json
```

The directory is mounted read-only in the container. If using the mock provider for an initial API smoke test, a checkpoint is not required, but production mode still requires an API key.

## Build and start

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f memer
```

By default, the service binds to `127.0.0.1:8000` on the host. This avoids exposing the API directly to the public internet. Put a trusted HTTPS reverse proxy in front of it before allowing remote access.

To change the host bind address or external port, set these in the shell before starting Compose:

```bash
export MEMER_BIND_ADDRESS=127.0.0.1
export MEMER_PUBLIC_PORT=8000
docker compose up -d
```

Do not change the bind address to `0.0.0.0` unless the host firewall and network exposure are intentionally configured.

## Verify

```bash
curl http://127.0.0.1:8000/health
```

A production chat request must include the configured Bearer key. Do not place a server-wide key in frontend code.

## Stop and update

```bash
docker compose down
docker compose build --pull
docker compose up -d
```

The checkpoint directory remains on the host and is not deleted by `docker compose down`.

## Security notes

- The image runs as an unprivileged `memer` user.
- The container filesystem is read-only; `/tmp` is a small temporary filesystem.
- Linux capabilities are dropped and privilege escalation is disabled.
- The checkpoint mount is read-only.
- Compose binds to loopback by default.
- Configure HTTPS, firewall rules, request limits, monitoring, and backups at the host/reverse-proxy layer.
- The in-process concurrency limit remains process-local. Run a single worker per model process.
- Docker packaging does not make the experimental byte-level Transformer suitable for production-quality assistant responses.

## Android and Termux

Docker Engine is not part of the supported Termux workflow. On Android, continue running Memer in the Python virtual environment:

```bash
cd ~/Memer
source .venv/bin/activate
memer serve --host 127.0.0.1 --port 8000
```

Use the container deployment on a separate Linux server or compatible host.
