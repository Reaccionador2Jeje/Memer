# AtDev Billing — Hybrid plans (Memer)

**Status:** proposal only. Memer's open-source CLI, model code, local experimentation, and self-hosting remain free. No inference API billing is active.

## Hybrid model

Charge only for optional AtDev-operated inference capacity, hosted API usage, managed deployments, or additional hosted conveniences. Users must be able to run supported models locally without a subscription. Do not promise model quality, uptime, or throughput beyond measured capabilities.

## Proposed plans (CLP/month; exploratory)

| Plan | Reference price | Scope |
| --- | ---: | --- |
| Local / Community | $0 | Local models, CLI, training experiments, and self-hosting |
| API Starter | $490 | Small metered hosted API allowance, if hosted inference is available |
| API Plus | $990 | Larger metered allowance and usage dashboard |
| Developer | $1,990 | Higher allowance and additional API projects |

These are planning figures, not sale prices. Model inference has variable compute costs: use hard token/request limits, rate limits, per-account spending ceilings, and transparent usage metering. Never offer unlimited inference at a fixed low price.

## Suggested entitlements

- `memer.api.requests.monthly`
- `memer.api.input_tokens.monthly`
- `memer.api.output_tokens.monthly`
- `memer.api.projects.max`
- `memer.hosted.models.available`

The server must calculate usage and resolve entitlements. Do not trust client-reported token counts or plan IDs. Keep API keys hashed or securely managed; never commit provider secrets or payment data.

## Launch gates

Publish measured model limitations, privacy/retention terms, rate limits, abuse controls, service status, and cancellation/expiry behavior. Keep checkout disabled until the service is costed and an adult account owner reviews payment-provider eligibility and legal/tax obligations.
