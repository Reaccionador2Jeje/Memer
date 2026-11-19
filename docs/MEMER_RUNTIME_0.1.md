# Memer Runtime 0.1 — security foundation

Memer Runtime is the hosted inference boundary for Memer Core. This milestone adds basic API-key authentication and bounded generation concurrency. It does not deploy a server, upload checkpoints, or provide account-linked keys from Memer Public yet.

## Production configuration

Set these environment variables in the hosting provider's secret/configuration panel:

```bash
MEMER_ENV=production
MEMER_PROVIDER=native
MEMER_MODEL_NAME=memer-small
MEMER_CHECKPOINT_PATH=/opt/memer/checkpoints/memer-small.json
MEMER_API_KEYS=replace-with-a-random-secret-of-at-least-32-characters
MEMER_MAX_NEW_TOKENS=80
MEMER_MAX_INPUT_CHARS=12000
MEMER_MAX_CONCURRENT_GENERATIONS=1
MEMER_CORS_ORIGINS=https://memer-public-ity8fh.v2.appdeploy.ai
```

Generate a key locally with Python:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep the key private. Do not commit it to Git, paste it into a public issue, or embed a server-wide key in browser JavaScript. Configure it only as a server-side secret. Multiple keys may be supplied as a comma-separated list to support rotation; each key must contain at least 32 characters.

## Request behavior

- In development, API-key authentication is disabled by default.
- `MEMER_ENV=production` always requires a valid `Authorization: Bearer <key>` header.
- `MEMER_REQUIRE_API_KEY=true` enables the same requirement outside production.
- Production startup reports unhealthy if API-key authentication is required but no keys are configured.
- Prompt text is capped by `MEMER_MAX_INPUT_CHARS` (default 12,000 characters).
- Native generation is capped at 512 new tokens; the default is 80.
- Concurrent model generations are limited to one by default. Requests arriving at capacity receive HTTP 429 and `Retry-After: 2`.
- Invalid or missing credentials receive HTTP 401.
- Oversized prompts receive HTTP 413.

## Operational limitations

The concurrency guard is process-local. It is a safety baseline for a single worker, not a distributed rate limiter. Run one application worker per model process until a shared queue/limiter is implemented. A reverse proxy or hosting platform should also enforce network-level request/body limits and HTTPS.

The API-key list is shared by the runtime instance. Per-user account tokens, usage accounting, key revocation UI, checkpoint upload/storage, durable queues, and autoscaling are future milestones. Memer Public's current model library stores metadata only and does not expose runtime secrets to browsers.

The native model itself remains experimental. Authentication and resource controls do not make a small checkpoint a capable assistant.
