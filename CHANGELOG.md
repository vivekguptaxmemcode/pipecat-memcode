# Changelog

All notable changes to `pipecat-memcode` will be documented here.

## 0.1.0 - Unreleased

### Added

- Coordinated recall and capture processors for the tested Pipecat 1.10 line.
- Personal Memcode v2 search and durable background ingest.
- OAuth access-token-provider support through `memcode-sdk`.
- Bounded, fail-open recall and bounded terminal cleanup.
- Deterministic per-turn idempotency keys and injected-context filtering.
- Tool preamble and post-tool assistant segments are staged into one ingest.
- Urgent cancellation and interrupted partial turns are discarded without ingest.
- Cleanup fallback writes and owned-client closure share a bounded, cancellation-safe lifecycle.
- Runnable WebRTC voice example with separate OAuth registration and account-connection modes.
- Encrypted local token persistence for the foundational example.
