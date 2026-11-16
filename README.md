# cartesia-voice-agent

A multi-tenant conversational voice agent on the Cartesia Line SDK.
One deployed agent serves every tenant. On each call it asks a platform API to
resolve the dialed number, then uses that tenant's prompt, tools, and voice.

The agent handles natural conversation, language selection, idle prompts,
live transfer, and a callback-bridge flow. Tool calls (booking, transfer,
transcripts, and so on) are HTTP POSTs to the platform; this repo is the
voice side only.

## Stack

Python 3.10+, Cartesia Line (`cartesia-line`), httpx, loguru. Install and run
with `uv`.

## Setup

```bash
cp .env.example .env   # fill in PLATFORM_BASE_URL and provider keys
uv sync
uv run python main.py
```

Run the offline tests:

```bash
uv run python test_jev.py
# test_idle.py also loads main.py; set PLATFORM_BASE_URL first
PLATFORM_BASE_URL=http://127.0.0.1:9 uv run python test_idle.py
```

## Config

See `.env.example`. `PLATFORM_BASE_URL` is required (no baked-in host).
`PLATFORM_WEBHOOK_SECRET` must match the platform. Tool URLs are built from
`PLATFORM_TOOL_PATH_PREFIX` (default `/api/voice`):

- resolve: `{PLATFORM_BASE_URL}{prefix}/resolve`
- tools: `{PLATFORM_BASE_URL}{prefix}/{business_id}/tools/{slug}`

## Licence

MIT, Copyright (c) 2026 Emil Senkel.
