# Pipecat Memcode

`pipecat-memcode` gives a Pipecat voice agent durable, personal long-term
memory backed by [Memcode](https://memcode.in). It retrieves relevant memories
before inference and stores only finalized user/assistant turns after assistant
aggregation.

This is a community-maintained Pipecat integration, maintained by Memcode. It
does not modify or ship as part of Pipecat core.

## Why there are two processors

One `MemcodeMemoryService` exposes two coordinated processors:

```text
transport.input() -> STT -> user_aggregator
                            -> memory.recall_processor()
                            -> LLM -> TTS -> transport.output()
                            -> assistant_aggregator
                            -> memory.capture_processor()
```

- Recall belongs **after the user aggregator and before the LLM**. It sees a
  finalized user message and can enrich that inference.
- Capture belongs **after the assistant aggregator**. It receives Pipecat's
  finalized `LLMContextAssistantTurnFrame`, pairs it with the finalized user
  turn, and queues exactly that delta for ingestion.

Interim transcripts, speculative contexts, raw TTS text frames, old history,
and Memcode's injected context are never ingested.

## Installation

```bash
uv add pipecat-memcode
```

Install the optional dependencies used by the runnable WebRTC example with:

```bash
uv add "pipecat-memcode[example]"
```

The first release targets Python 3.11-3.14, `pipecat-ai>=1.10,<1.11`, and
`memcode-sdk>=2.4,<3`. Pipecat releases outside the 1.10 line are not yet claimed
compatible. Source, issues, and release history live in the
[`pipecat-memcode` repository](https://github.com/vivekgupta-memcode/pipecat-memcode).

## OAuth 2.1 connection

Production applications should connect each participant to Memcode using
Authorization Code with S256 PKCE and dynamic client registration:

1. Discover Memcode's authorization-server and protected-resource metadata.
2. Register the Pipecat application's exact callback URI once per deployment.
3. Generate a new `state`, PKCE verifier, and S256 challenge for each account
   connection.
4. Send the user to Memcode's authorization and consent page, requesting the
   Memory API resource and `memory:read memory:write` scopes.
5. Validate `state`, exchange the code with the verifier, and store the access
   and rotating refresh tokens encrypted under the application's user record.
6. Give this package that user's `AsyncAccessTokenProvider`. The SDK resolves a
   token for every request and performs one coordinated refresh/retry after a
   401.

Dynamic registration is deployment setup, not per-call or per-conversation
work. A `MemcodeMemoryService` instance is per authenticated participant. The
OAuth token subject selects the personal memory scope, so this integration
never accepts or transmits a `user_id`.

`AsyncMemcodeOAuthClient` from `memcode-sdk>=2.4.0` implements discovery,
dynamic registration, PKCE, token exchange, rotation, and the access-token
provider interface consumed here.

Never put a refresh token, authorization code, or PKCE verifier in frontend
storage, logs, frame metadata, or LLM context.

## Run the foundational example

The [single-file example](examples/foundational/memcode_memory.py) is a complete
Small WebRTC voice bot using Deepgram STT, OpenAI, Cartesia TTS, and Memcode. Its
account-connection commands are separate from the real-time bot command, and it
never opens a browser automatically.

For a source checkout, install the package, development tools, and example
dependencies, then create the local environment file:

```bash
uv sync --group dev --extra example
cp .env.example .env
uv run python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Put the generated value in `MEMCODE_TOKEN_ENCRYPTION_KEY` in `.env`, then fill
the three voice-provider keys. These are the example variables:

| Variable | Required | Purpose |
|---|---|---|
| `DEEPGRAM_API_KEY` | bot | Speech-to-text |
| `OPENAI_API_KEY` | bot | Language model |
| `OPENAI_MODEL` | no | Defaults to `gpt-4.1-mini` |
| `CARTESIA_API_KEY` | bot | Text-to-speech |
| `CARTESIA_VOICE_ID` | no | Defaults to the voice in `.env.example` |
| `MEMCODE_CLIENT_ID` | connect and bot | Public client ID created during registration |
| `MEMCODE_REDIRECT_URI` | no | Defaults to `http://127.0.0.1:8765/callback` |
| `MEMCODE_TOKEN_KEY` | no | Stable local grant lookup key |
| `MEMCODE_TOKEN_ENCRYPTION_KEY` | connect and bot | Fernet key protecting the local token file |
| `MEMCODE_TOKEN_PATH` | no | Defaults to `.memcode-oauth.enc` |

Register the local public client once:

```bash
uv run python examples/foundational/memcode_memory.py --register
```

Copy the printed, non-secret client ID into `MEMCODE_CLIENT_ID` in `.env`. Then
start an explicit account connection:

```bash
uv run python examples/foundational/memcode_memory.py --connect
```

Open the printed authorization URL yourself, approve access, and paste the full
redirected callback URL into the hidden terminal prompt. A browser may show an
unreachable loopback page; the address bar still contains the callback URL.
The example validates OAuth state and writes access and rotating refresh tokens
only to the encrypted, gitignored token file.

To disconnect the locally stored account before authorizing a different one:

```bash
uv run python examples/foundational/memcode_memory.py --disconnect
uv run python examples/foundational/memcode_memory.py --connect
```

`--disconnect` removes this example's local token record; it does not delete
the Memcode account or its memories.

After connection, run the bot:

```bash
uv run python examples/foundational/memcode_memory.py -t webrtc
```

Open the Pipecat runner URL printed in the terminal and connect your microphone.
The local encrypted store and its process-local refresh lock are intentionally
limited to this one-process example. Production deployments must use an
encrypted server-side `AsyncOAuthTokenStore` with an atomic save and a
distributed refresh lease covering every worker.

## Pipeline usage

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat_memcode import MemcodeMemoryConfig, MemcodeMemoryService

# `token_provider` is scoped to the signed-in participant and implements
# memcode_sdk.AsyncAccessTokenProvider.
memory = MemcodeMemoryService(
    access_token_provider=token_provider,
    api_url="https://memory.memcode.in",
    session_id=call_id,  # use a stable room/call ID for retry idempotency
    config=MemcodeMemoryConfig(
        search_top_k=5,
        search_timeout_seconds=5.0,
    ),
)

context = LLMContext([{"role": "developer", "content": "You are a concise, helpful assistant."}])
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        memory.recall_processor(),
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
        memory.capture_processor(),
    ]
)
```

### Graceful disconnects are required for final-turn capture

End a normal Pipecat 1.10 session with `stop_when_done()` so `EndFrame` drains
through the pipeline and the final completed user/assistant turn obtains a
durable Memcode receipt:

```python
@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    await runner.stop_when_done()
```

Do not map a normal client disconnect to `runner.cancel()`: cancellation is an
urgent path that intentionally discards the active turn. Do not substitute
`runner.end()` on Pipecat 1.10 either; runner cleanup can cancel the worker
before its queued `EndFrame` finishes. A participant should also let the
assistant finish before disconnecting because no shutdown path can recover an
upstream turn that was never finalized.

Applications that already own a per-user SDK client can inject it instead:

```python
from memcode_sdk import AsyncMemcodeClient
from pipecat_memcode import MemcodeMemoryService

client = AsyncMemcodeClient(
    api_url="https://memory.memcode.in",
    access_token_provider=token_provider,
)
memory = MemcodeMemoryService(
    client=client,
    session_id=call_id,
    close_client=False,  # the application retains lifecycle ownership
)
```

When the service constructs the SDK client, it closes that client during
processor cleanup. The application still owns the injected token provider and
must close an `AsyncMemcodeOAuthClient` or `DelegatingMemoryTokenProvider` from
its own connection/session lifecycle. When an existing client is injected, the
application owns it unless `close_client=True` is explicitly requested.

See [`examples/foundational/memcode_memory.py`](examples/foundational/memcode_memory.py)
for the complete runnable integration.

## Runtime contract

Recall uses `AsyncMemcodeClient.search_v2`, not `retrieve_v2`. It always asks
for extracted memories only (`mode="memories"`,
`include_original_chunks=False`), bounds the resulting block, marks it as
reference-only data, and fails open if Memcode is slow or unavailable. Each
record retains a safe domain label. The default context guidance prefers
current profile records for identity and preferences over conflicting summary
records, places profile records first when the context budget is tight, and
tells the model to acknowledge other conflicts instead of guessing.

Capture uses `AsyncMemcodeClient.ingest_v2` in a Pipecat-managed background
task. Assistant-turn frames are staged because Pipecat can emit one at both a
tool preamble and the post-tool answer. All segments remain attached to the
same user turn and are written once when the next finalized user turn arrives,
or when graceful `EndFrame` or cleanup finalizes the session. An
`InterruptionFrame` discards the interrupted partial assistant turn so it
cannot be captured by the next user turn. Urgent `CancelFrame` discards the
active turn, signals any owned writes to stop, and propagates immediately
without waiting on Memcode. Each write carries a deterministic SHA-256
idempotency key derived from the stable session ID and combined finalized turn.
Graceful shutdown work is bounded by `shutdown_timeout_seconds`; cleanup and
client closing are cancellation-safe and idempotent.

The ingestion call returns a durable receipt, and the integration logs only its
safe operational fields (job/status/creation state/indexing estimate/request
ID). It never logs conversation text or OAuth credentials. Memory extraction
continues in Memcode asynchronously; this package intentionally does not hold
up the voice pipeline by polling that job. Respect the receipt's
`estimated_available_in_seconds` value before testing recall in a new session;
the estimate can be around 60 seconds, so immediate cross-session recall is not
guaranteed.

## Configuration

| Field | Default | Meaning |
|---|---:|---|
| `search_top_k` | `5` | Maximum memories requested per memory domain |
| `search_minimum_score` | `0.0` | Minimum relevance score |
| `search_mode` | `"default"` | Memcode routing mode (`default` or `global`) |
| `search_timeout_seconds` | `5.0` | Recall latency budget before fail-open |
| `ingest_timeout_seconds` | `10.0` | Budget for a durable ingest receipt |
| `shutdown_timeout_seconds` | `12.0` | Graceful EndFrame and cleanup budget |
| `max_context_characters` | `4000` | Maximum complete injected context block |
| `context_role` | `"developer"` | Injected universal-context role |
| `context_header` | reference-only warning | Boundary between data and instructions |
| `effort_level` | `"low"` | Memcode ingest effort (`low` or `high`) |

Use a stable, non-secret `session_id` from the Pipecat call or room. If omitted,
the service creates a random ID, which preserves in-process retry safety but
cannot deduplicate the same turn after a process restart.

## Failure behavior

- Search timeout or error: the unchanged context continues to the LLM.
- Partial search response: available results are used; failed domains do not
  erase valid hits.
- Ingest timeout or error: the voice response is never blocked or failed.
- Interruption or cancellation: partial active turns are discarded; urgent
  cancellation never waits for Memcode and can abandon unreceipted background
  writes. Reserve it for genuinely urgent shutdown.
- Normal disconnect: use `runner.stop_when_done()` so the final completed turn
  is captured before cleanup.
- Accepted ingest: extraction is asynchronous; use the receipt's availability
  estimate rather than assuming immediate recall.
- Duplicate context frames: recall is cached per finalized conversational
  prefix and only one pending capture turn is created.
- Duplicate write attempt: the same finalized turn receives the same
  idempotency key.

## Development

```bash
uv sync --group dev --extra example
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

No browser is required for the unit suite. A release candidate should also be
verified against a real OAuth account in an explicitly authorized staging run,
including a session long enough to rotate an access token.

## License and attribution

This integration is released under the BSD 2-Clause License. Pipecat is an
open-source project maintained by Daily; Memcode maintains this community
package and its Memcode-specific behavior.
