# Memer Runtime 0.4 — named model registry and API identity

Runtime 0.4 adds a local registry for named native models. A registry entry stores model identity and connection metadata:

- model name and version;
- provider (native in this milestone);
- validated local checkpoint path and architecture summary;
- base URL for the Memer API;
- API-key platform label;
- the environment-variable name used to configure the API key.

**Secrets are never stored in the registry.** The registry stores only the name of an environment variable, not its value. The API key itself must remain in the server environment or secret manager.

## Register a model

After training a native checkpoint:

~~~bash
memer models add checkpoints/memer-small.json \
  --name memer-small \
  --version 0.4.0 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-platform Memer \
  --api-key-env MEMER_API_KEYS
~~~

The first registered model becomes active automatically.

## List and inspect

~~~bash
memer models list
memer models inspect memer-small
~~~

The active model is marked with * in the list.

## Switch active model

~~~bash
memer models use memer-small
~~~

## Start Memer using the active model

On the same terminal session:

~~~bash
export MEMER_PROVIDER=native
memer serve --host 127.0.0.1 --port 8000
~~~

If MEMER_CHECKPOINT_PATH is not explicitly set, the CLI loads the active registry entry and supplies its checkpoint path and model name to Memer Core. An explicit MEMER_CHECKPOINT_PATH always takes precedence.

The health endpoint reports the configured model name, base URL, and API-key platform label:

~~~bash
curl http://127.0.0.1:8000/health
~~~

## Remove an entry

~~~bash
memer models remove memer-small
~~~

This removes only the registry entry. The checkpoint file is never deleted. If the active model is removed, the next registered model (alphabetically) becomes active.

## Registry location

By default, the registry is stored at:

~~~text
~/.config/memer/models.json
~~~

Set MEMER_REGISTRY_PATH to use another location. CLI commands also accept --registry PATH, which is useful for testing or maintaining separate registries.

## API key security

For the built-in Memer API, configure server-side authentication with a random secret of at least 32 characters:

~~~bash
export MEMER_ENV=production
export MEMER_API_KEYS="your-long-random-secret"
export MEMER_REQUIRE_API_KEY=true
~~~

Clients send that key in the Authorization: Bearer ... header. Never paste the secret into a public frontend, commit it to Git, or include it in a model registry file. The --api-key-env option is descriptive metadata; it does not read or reveal the secret.

## Scope of this milestone

The base URL and API-key platform are model connection metadata, and the Memer Core health endpoint reports them. Runtime 0.4 still runs native checkpoints locally. It does not yet call arbitrary third-party OpenAI-compatible APIs, provision API keys at external platforms, or host remote model weights. Those require a separate remote-provider integration.
