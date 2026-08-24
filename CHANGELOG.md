# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The IR
JSON representation is covered by semver (see `docs/design.md` § 3).

## [0.1.0] - 2026-08-24

Initial release.

- Unified intermediate representation (IR) for LLM chat requests,
  responses and streams, designed for faithful round-trips of agent
  history (thinking signatures, tool calls, provider-specific data).
- Bidirectional conversion for four formats: OpenAI Chat Completions
  (incl. dialects such as DeepSeek), OpenAI Responses, Anthropic Messages
  and Google `generateContent` — non-streaming and SSE streaming.
- Nothing is silently dropped: every conversion loss carries a warning
  with a stable code, fixed severity and JSON-Pointer location; strict
  mode turns semantic build-side losses into errors.
- Format-namespaced `extra` (RFC 7396 merge) plus request hooks for
  anything the IR does not model.
- Pluggable HTTP transport (`HttpClient` trait) with a feature-gated
  reqwest default; `default-features = false` gives a pure data layer.
- Model listing (with bounded auto-pagination) and provider-side token
  counting; structured retry material (`Error::is_retryable`,
  `TruncatedStream`, `Retry-After` incl. HTTP-date).

[0.1.0]: https://github.com/orzogc/llm-api/releases/tag/v0.1.0
