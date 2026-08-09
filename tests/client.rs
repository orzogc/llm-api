//! Client-layer integration tests: a mock `ApiFormat` plus either a fully
//! scripted `HttpClient` (deterministic control over chunks and failures)
//! or wiremock with the reqwest transport (end-to-end over real HTTP).

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{Value, json};

use llm_api::client::{CallOptions, Client, EndpointConfig, Limits, Override, ProviderConfig};
use llm_api::http::{
    ApiKey, AuthHeader, BodyStream, HttpClient, HttpError, HttpErrorKind, SseEvent,
};
use llm_api::ir::MergeLog;
use llm_api::{
    Accumulator, ApiErrorKind, ApiFormat, AuthScheme, BodyKind, BuildCtx, BuiltRequest, CallMode,
    ContentBlock, ConversionDirection, ConversionWarning, ConvertOptions, EndpointUrl, Error,
    Message, Model, Request, RequestHooks, Response, ResponseMeta, Role, StreamEvent, StreamItem,
    StreamParser, TokenCount, WarningCode, build_url, finalize_request,
};

// ---------------------------------------------------------------------------
// Mock format
// ---------------------------------------------------------------------------

/// Deserializes a unified stream event from its IR JSON representation
/// (`StreamEvent` variants are `#[non_exhaustive]` and cannot be built with
/// struct literals outside the crate).
fn sev(v: Value) -> StreamEvent {
    serde_json::from_value(v).expect("valid stream event JSON")
}

fn parse_err(message: &'static str) -> Error {
    // `Error::Parse` cannot be constructed outside the crate; the mock uses
    // `NotSupported` as a stand-in error carrier.
    Error::NotSupported(message)
}

/// A minimal `ApiFormat` speaking a private JSON protocol.
///
/// Chat body: `{"model", "strict", "stream", "messages", ...}` (extra merge
/// and hooks applied via the core `finalize_request`). Build warnings:
/// `top_k` set → cosmetic `SamplingParameterDropped`; `stop_sequences` set →
/// semantic `StopSequencesDropped` (drives strict-gate tests). Responses:
/// `{"text", "warn"}`; `warn: true` adds one parse-side warning.
#[derive(Clone)]
struct MockFormat {
    id: &'static str,
    auth: Option<AuthScheme>,
}

impl MockFormat {
    fn new(id: &'static str) -> Self {
        Self {
            id,
            auth: Some(AuthScheme::bearer()),
        }
    }

    fn with_auth_scheme(mut self, auth: Option<AuthScheme>) -> Self {
        self.auth = auth;
        self
    }

    fn arc(self) -> Arc<dyn ApiFormat> {
        Arc::new(self)
    }

    fn chat_body(
        &self,
        req: &Request,
        ctx: &BuildCtx,
    ) -> llm_api::Result<(Value, Vec<ConversionWarning>)> {
        let messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| {
                json!({
                    "role": serde_json::to_value(m.role).expect("role serializes"),
                    "content": m.text(),
                })
            })
            .collect();
        let mut body = json!({
            "model": ctx.model,
            "strict": ctx.convert.strict,
            "stream": ctx.mode == CallMode::Streaming,
            "messages": messages,
        });
        let mut warnings = Vec::new();
        if req.top_k.is_some() {
            warnings.push(ConversionWarning::to_format(
                WarningCode::SamplingParameterDropped,
                self.id,
                "/top_k",
                "top_k dropped",
            ));
        }
        if req.stop_sequences.is_some() {
            warnings.push(ConversionWarning::to_format(
                WarningCode::StopSequencesDropped,
                self.id,
                "/stop",
                "stop_sequences dropped",
            ));
        }
        let mut log = MergeLog::new();
        req.extra.merge_into(self.id, &mut body, "", &mut log);
        let pointers: Vec<(String, Role)> = req
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| (format!("/messages/{i}"), m.role))
            .collect();
        finalize_request(
            &mut body,
            &mut warnings,
            &log,
            ctx.convert.strict,
            &ctx.hooks,
            &pointers,
        )?;
        Ok((body, warnings))
    }

    fn built_post(
        &self,
        ctx: &BuildCtx,
        path: &str,
        body: &Value,
        warnings: Vec<ConversionWarning>,
    ) -> llm_api::Result<BuiltRequest> {
        let url = build_url(&ctx.url, path, &ctx.model, None, &[], &ctx.extra_query)?;
        let mut built = BuiltRequest::post_json(url, body);
        built
            .headers
            .insert("x-a", http::HeaderValue::from_static("f"));
        built
            .headers
            .insert("x-fmt", http::HeaderValue::from_static("f"));
        built.auth = self.auth.clone();
        built.warnings = warnings;
        Ok(built)
    }
}

impl ApiFormat for MockFormat {
    fn id(&self) -> &str {
        self.id
    }

    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> llm_api::Result<BuiltRequest> {
        let (body, warnings) = self.chat_body(req, ctx)?;
        self.built_post(ctx, "chat", &body, warnings)
    }

    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> llm_api::Result<Response> {
        let v: Value =
            serde_json::from_slice(body).map_err(|_| parse_err("mock: bad response body"))?;
        let text = v["text"].as_str().unwrap_or_default();
        let mut response = Response::new(Message::assistant(vec![ContentBlock::text(text)]));
        response.status = meta.status;
        response.headers = meta.headers.clone();
        response.raw = Some(v.clone());
        if v["warn"].as_bool() == Some(true) {
            response.warnings.push(ConversionWarning::from_format(
                WarningCode::MalformedField,
                self.id,
                "/text",
                "parse warning",
            ));
        }
        Ok(response)
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(MockStreamParser {
            id: self.id,
            seen_stop: false,
            started: HashSet::new(),
            held: None,
        })
    }

    fn parse_request(&self, _body: &[u8]) -> llm_api::Result<(Request, Vec<ConversionWarning>)> {
        Err(Error::NotSupported("mock: parse_request"))
    }

    fn build_models_request(
        &self,
        ctx: &BuildCtx,
        cursor: Option<&str>,
    ) -> llm_api::Result<BuiltRequest> {
        let mut query = ctx.extra_query.clone();
        if let Some(cursor) = cursor {
            query.push(("cursor".to_owned(), cursor.to_owned()));
        }
        let url = build_url(&ctx.url, "models", &ctx.model, None, &[], &query)?;
        let mut built = BuiltRequest::get(url);
        built.auth = self.auth.clone();
        Ok(built)
    }

    fn parse_models_response(&self, body: &[u8]) -> llm_api::Result<(Vec<Model>, Option<String>)> {
        let v: Value =
            serde_json::from_slice(body).map_err(|_| parse_err("mock: bad models body"))?;
        let mut models = Vec::new();
        if let Some(entries) = v["data"].as_array() {
            for entry in entries {
                models.push(Model::new(
                    entry["id"].as_str().unwrap_or_default(),
                    entry.clone(),
                ));
            }
        }
        Ok((models, v["next"].as_str().map(str::to_owned)))
    }

    fn build_count_tokens_request(
        &self,
        req: &Request,
        ctx: &BuildCtx,
    ) -> llm_api::Result<BuiltRequest> {
        let (mut body, warnings) = self.chat_body(req, ctx)?;
        body["count"] = json!(true);
        self.built_post(ctx, "count", &body, warnings)
    }

    fn parse_count_tokens_response(&self, body: &[u8]) -> llm_api::Result<TokenCount> {
        let v: Value =
            serde_json::from_slice(body).map_err(|_| parse_err("mock: bad count body"))?;
        let mut count = TokenCount::new(v["input_tokens"].as_u64().unwrap_or(0));
        count.raw = Some(v.clone());
        if v["warn"].as_bool() == Some(true) {
            count.warnings.push(ConversionWarning::from_format(
                WarningCode::MalformedField,
                self.id,
                "/input_tokens",
                "count parse warning",
            ));
        }
        Ok(count)
    }
}

/// Stream protocol (one JSON object per SSE event):
/// - `{"k":"start","id":…}` → `MessageStart`
/// - `{"k":"text","i":n,"t":…,"warn":bool}` → `BlockStart` (first time for
///   the index) + `BlockDelta`; `warn` attaches one warning to the call
/// - `{"k":"warn","m":…}` → no events, one warning (held by the client)
/// - `{"k":"hold","t":…}` → nothing now; `finish()` emits a `BlockStart`
///   with that text plus one warning
/// - `{"k":"stop"}` → `MessageDelta` (stop reason + usage) + `MessageStop`
/// - `{"k":"bad"}` → parser error
/// - `{"k":"apierr","own_headers":bool}` → an in-stream `Error::Api` built
///   from the event body alone (empty headers, no `retry_after`), the way
///   the Anthropic/Responses stream parsers construct theirs;
///   `own_headers` sets a parser-supplied `x-parser` header instead
/// - anything else → `Unknown` + `UnknownStreamEvent` warning
struct MockStreamParser {
    id: &'static str,
    seen_stop: bool,
    started: HashSet<usize>,
    held: Option<String>,
}

impl MockStreamParser {
    fn warn(&self, message: &str) -> ConversionWarning {
        ConversionWarning::from_format(WarningCode::MalformedField, self.id, "/stream", message)
    }
}

impl StreamParser for MockStreamParser {
    fn parse(
        &mut self,
        event: &SseEvent,
    ) -> llm_api::Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        let Ok(v) = serde_json::from_str::<Value>(&event.data) else {
            return Ok((
                vec![StreamEvent::Unknown],
                vec![ConversionWarning::from_format(
                    WarningCode::UnknownStreamEvent,
                    self.id,
                    "/stream",
                    "unparseable event",
                )],
            ));
        };
        match v["k"].as_str() {
            Some("start") => Ok((
                vec![sev(json!({"type": "message_start", "id": v["id"]}))],
                Vec::new(),
            )),
            Some("text") => {
                let index = v["i"].as_u64().unwrap_or(0);
                let text = v["t"].as_str().unwrap_or_default();
                let mut events = Vec::new();
                if self.started.insert(index as usize) {
                    events.push(sev(json!({
                        "type": "block_start",
                        "index": index,
                        "block": {"type": "text", "text": ""},
                    })));
                }
                events.push(sev(json!({
                    "type": "block_delta",
                    "index": index,
                    "delta": {"type": "text", "value": text},
                })));
                let warnings = if v["warn"].as_bool() == Some(true) {
                    vec![self.warn("attached")]
                } else {
                    Vec::new()
                };
                Ok((events, warnings))
            }
            Some("warn") => Ok((
                Vec::new(),
                vec![self.warn(v["m"].as_str().unwrap_or("held"))],
            )),
            Some("hold") => {
                self.held = Some(v["t"].as_str().unwrap_or_default().to_owned());
                Ok((Vec::new(), Vec::new()))
            }
            Some("stop") => {
                self.seen_stop = true;
                Ok((
                    vec![
                        sev(json!({
                            "type": "message_delta",
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 3, "output_tokens": 5},
                        })),
                        sev(json!({"type": "message_stop"})),
                    ],
                    Vec::new(),
                ))
            }
            Some("bad") => Err(parse_err("mock: bad stream event")),
            Some("apierr") => {
                let mut headers = http::HeaderMap::new();
                if v["own_headers"].as_bool() == Some(true) {
                    headers.insert("x-parser", http::HeaderValue::from_static("own"));
                }
                Err(Error::api(
                    429,
                    ApiErrorKind::RateLimit,
                    "in-stream error",
                    event.data.clone(),
                    headers,
                ))
            }
            _ => Ok((
                vec![StreamEvent::Unknown],
                vec![ConversionWarning::from_format(
                    WarningCode::UnknownStreamEvent,
                    self.id,
                    "/stream",
                    "unknown event kind",
                )],
            )),
        }
    }

    fn finish(&mut self) -> llm_api::Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        if !self.seen_stop {
            return Err(parse_err("mock: truncated stream"));
        }
        if let Some(text) = self.held.take() {
            return Ok((
                vec![sev(json!({
                    "type": "block_start",
                    "index": 90,
                    "block": {"type": "text", "text": text},
                }))],
                vec![self.warn("finish warning")],
            ));
        }
        Ok((Vec::new(), Vec::new()))
    }
}

// ---------------------------------------------------------------------------
// Scripted transport
// ---------------------------------------------------------------------------

struct CapturedCall {
    method: http::Method,
    uri: http::Uri,
    headers: http::HeaderMap,
    body: Bytes,
    auth: Option<AuthHeader>,
}

type Captured = Arc<Mutex<Vec<CapturedCall>>>;

enum Scripted {
    Respond {
        status: u16,
        headers: Vec<(&'static str, &'static str)>,
        chunks: Vec<Result<Bytes, HttpError>>,
    },
    Fail(HttpErrorKind),
}

/// A transport driven by a pre-recorded script; captures every call.
struct ScriptedHttp {
    captured: Captured,
    script: Mutex<VecDeque<Scripted>>,
}

impl HttpClient for ScriptedHttp {
    fn send(
        &self,
        request: http::Request<Bytes>,
        auth: Option<AuthHeader>,
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>
    {
        let (parts, body) = request.into_parts();
        self.captured.lock().unwrap().push(CapturedCall {
            method: parts.method,
            uri: parts.uri,
            headers: parts.headers,
            body,
            auth,
        });
        let next = self.script.lock().unwrap().pop_front();
        Box::pin(async move {
            match next {
                Some(Scripted::Respond {
                    status,
                    headers,
                    chunks,
                }) => {
                    let stream: BodyStream = Box::pin(futures_util::stream::iter(chunks));
                    let mut builder = http::Response::builder().status(status);
                    for (name, value) in headers {
                        builder = builder.header(name, value);
                    }
                    Ok(builder.body(stream).expect("valid scripted response"))
                }
                Some(Scripted::Fail(kind)) => Err(HttpError::new(kind)),
                None => panic!("scripted transport exhausted: unexpected extra call"),
            }
        })
    }
}

fn scripted(script: Vec<Scripted>) -> (Client, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let http = ScriptedHttp {
        captured: captured.clone(),
        script: Mutex::new(script.into()),
    };
    (Client::new(http), captured)
}

fn respond(status: u16, headers: &[(&'static str, &'static str)], body: &[u8]) -> Scripted {
    Scripted::Respond {
        status,
        headers: headers.to_vec(),
        chunks: vec![Ok(Bytes::copy_from_slice(body))],
    }
}

fn respond_json(v: &Value) -> Scripted {
    respond(
        200,
        &[("content-type", "application/json")],
        v.to_string().as_bytes(),
    )
}

fn sse_chunk(v: &Value) -> Result<Bytes, HttpError> {
    Ok(Bytes::from(format!("data: {v}\n\n")))
}

fn respond_sse(events: &[Value]) -> Scripted {
    Scripted::Respond {
        status: 200,
        headers: vec![("content-type", "text/event-stream"), ("x-sse", "yes")],
        chunks: events.iter().map(sse_chunk).collect(),
    }
}

fn mock_provider() -> ProviderConfig {
    ProviderConfig::new(
        MockFormat::new("mock").arc(),
        "http://mock.local/v1",
        "base-model",
    )
    .unwrap()
    .with_auth("secret-key")
}

fn user_request() -> Request {
    Request::with_messages(vec![Message::user_text("hi")])
}

async fn drain(mut handle: llm_api::client::StreamHandle) -> Vec<Result<StreamItem, Error>> {
    let mut items = Vec::new();
    while let Some(item) = handle.next().await {
        items.push(item);
    }
    items
}

fn ok_item(items: &[Result<StreamItem, Error>], index: usize) -> &StreamItem {
    match &items[index] {
        Ok(item) => item,
        Err(e) => panic!("item {index}: expected Ok, got error: {e}"),
    }
}

// ---------------------------------------------------------------------------
// CallOptions merging
// ---------------------------------------------------------------------------

#[tokio::test]
async fn call_options_per_call_wins_wholesale() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider.convert = ConvertOptions::default().strict(true);
    provider.hooks = RequestHooks::new().with_on_request(|v| {
        v["hook"] = json!("provider");
        Ok(())
    });
    provider.extra_query = vec![
        ("a".to_owned(), "1".to_owned()),
        ("b".to_owned(), "2".to_owned()),
    ];

    let opts = CallOptions::new()
        .with_model("per-call-model")
        // Whole replacement: strict goes back to false.
        .with_convert(ConvertOptions::default())
        // Whole replacement: the provider hook must not run.
        .with_hooks(RequestHooks::new().with_on_request(|v| {
            v["hook"] = json!("call");
            Ok(())
        }))
        .with_query("b", "9")
        .with_query("c", "3");

    client
        .send(&provider, &user_request(), &opts)
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(body["model"], json!("per-call-model"));
    assert_eq!(body["strict"], json!(false));
    assert_eq!(body["hook"], json!("call"));
    // extra_query stacks: provider first, per-call same-name keys win.
    assert_eq!(
        calls[0].uri.to_string(),
        "http://mock.local/v1/chat?a=1&b=9&c=3"
    );
    assert_eq!(calls[0].method, http::Method::POST);
}

#[tokio::test]
async fn call_options_defaults_inherit_provider() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider.hooks = RequestHooks::new()
        .with_on_message(|index, role, value| {
            assert_eq!(index, 0);
            assert_eq!(*role, Role::User);
            value["hooked"] = json!(true);
            Ok(())
        })
        .with_on_request(|v| {
            v["hook"] = json!("provider");
            Ok(())
        });
    provider.extra_query = vec![("a".to_owned(), "1".to_owned())];

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(body["model"], json!("base-model"));
    assert_eq!(body["strict"], json!(false));
    assert_eq!(body["hook"], json!("provider"));
    assert_eq!(body["messages"][0]["hooked"], json!(true));
    assert_eq!(calls[0].uri.to_string(), "http://mock.local/v1/chat?a=1");
}

#[tokio::test]
async fn strict_mode_fails_before_any_io() {
    let (client, captured) = scripted(vec![]);
    let mut provider = mock_provider();
    provider.convert = ConvertOptions::default().strict(true);
    let mut request = user_request();
    request.stop_sequences = Some(vec!["s".to_owned()]); // semantic warning

    let err = client
        .send(&provider, &request, &CallOptions::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Conversion(_)));
    assert!(
        captured.lock().unwrap().is_empty(),
        "no HTTP call may happen"
    );
}

// ---------------------------------------------------------------------------
// Header layering and auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn header_layering_full_chain() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    // Format defaults: content-type json, x-a: f, x-fmt: f.
    provider
        .extra_headers
        .insert("x-a", http::HeaderValue::from_static("p"));
    provider
        .extra_headers
        .insert("x-b", http::HeaderValue::from_static("p"));
    let mut endpoint_headers = http::HeaderMap::new();
    endpoint_headers.insert("x-b", http::HeaderValue::from_static("e"));
    endpoint_headers.insert("x-c", http::HeaderValue::from_static("e"));
    provider.chat.headers = Override::Set(endpoint_headers);
    let opts = CallOptions::new()
        .with_header(
            http::HeaderName::from_static("x-c"),
            http::HeaderValue::from_static("call"),
        )
        .with_header(
            http::HeaderName::from_static("x-d"),
            http::HeaderValue::from_static("call"),
        );

    client
        .send(&provider, &user_request(), &opts)
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    let headers = &calls[0].headers;
    assert_eq!(headers.get("content-type").unwrap(), "application/json");
    assert_eq!(
        headers.get("x-fmt").unwrap(),
        "f",
        "untouched format default survives"
    );
    assert_eq!(
        headers.get("x-a").unwrap(),
        "p",
        "provider overrides format default"
    );
    assert_eq!(
        headers.get("x-b").unwrap(),
        "e",
        "endpoint Set overrides provider"
    );
    assert_eq!(
        headers.get("x-c").unwrap(),
        "call",
        "per-call overrides endpoint"
    );
    assert_eq!(
        headers.get("x-d").unwrap(),
        "call",
        "per-call adds new names"
    );
}

#[tokio::test]
async fn header_layering_disable_drops_provider_layer_only() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider
        .extra_headers
        .insert("x-a", http::HeaderValue::from_static("p"));
    provider
        .extra_headers
        .insert("x-b", http::HeaderValue::from_static("p"));
    provider.chat.headers = Override::Disable;
    let opts = CallOptions::new().with_header(
        http::HeaderName::from_static("x-d"),
        http::HeaderValue::from_static("call"),
    );

    client
        .send(&provider, &user_request(), &opts)
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    let headers = &calls[0].headers;
    assert_eq!(
        headers.get("x-a").unwrap(),
        "f",
        "format default not removable"
    );
    assert!(headers.get("x-b").is_none(), "provider layer dropped");
    assert_eq!(
        headers.get("x-d").unwrap(),
        "call",
        "per-call still applies"
    );
}

#[tokio::test]
async fn auth_inherit_combines_scheme_and_key_outside_headers() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let provider = mock_provider();

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    // The key never sits in the request headers; it travels separately.
    assert!(calls[0].headers.get(http::header::AUTHORIZATION).is_none());
    let auth = calls[0].auth.as_ref().expect("auth resolved");
    assert_eq!(auth.name, http::header::AUTHORIZATION);
    assert_eq!(auth.prefix.as_deref(), Some("Bearer "));
    assert_eq!(auth.value.expose(), "secret-key");
    // The rendered header value is marked sensitive.
    let value = auth.header_value().unwrap();
    assert!(value.is_sensitive());
    assert_eq!(value.to_str().unwrap(), "Bearer secret-key");
    // The redacted Debug never leaks the key.
    assert!(!format!("{auth:?}").contains("secret-key"));
}

#[tokio::test]
async fn auth_inherit_without_provider_key_sends_none() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider.auth = None;

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    assert!(captured.lock().unwrap()[0].auth.is_none());
}

#[tokio::test]
async fn auth_inherit_without_format_scheme_sends_none() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let format = MockFormat::new("mock").with_auth_scheme(None).arc();
    let provider = ProviderConfig::new(format, "http://mock.local/v1", "m")
        .unwrap()
        .with_auth("secret-key");

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    assert!(captured.lock().unwrap()[0].auth.is_none());
}

#[tokio::test]
async fn auth_endpoint_set_used_verbatim() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider.auth = None; // Set wins even without a provider key.
    provider.chat.auth = Override::Set(AuthHeader {
        name: http::HeaderName::from_static("x-api-key"),
        prefix: None,
        value: ApiKey::new("endpoint-key"),
    });

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    let auth = calls[0].auth.as_ref().expect("auth set");
    assert_eq!(auth.name, "x-api-key");
    assert_eq!(auth.prefix, None);
    assert_eq!(auth.value.expose(), "endpoint-key");
}

#[tokio::test]
async fn auth_endpoint_disable_sends_none() {
    let (client, captured) = scripted(vec![respond_json(&json!({"text": "ok"}))]);
    let mut provider = mock_provider();
    provider.chat.auth = Override::Disable;

    client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    assert!(captured.lock().unwrap()[0].auth.is_none());
}

// ---------------------------------------------------------------------------
// send: success / errors / caps
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_success_prepends_build_warnings() {
    let (client, _) = scripted(vec![respond(
        200,
        &[("content-type", "application/json"), ("x-rate", "42")],
        json!({"text": "hello", "warn": true})
            .to_string()
            .as_bytes(),
    )]);
    let provider = mock_provider();
    let mut request = user_request();
    request.top_k = Some(7); // build-side warning

    let response = client
        .send(&provider, &request, &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(response.text(), "hello");
    assert_eq!(response.status, 200);
    assert_eq!(response.headers.get("x-rate").unwrap(), "42");
    assert!(response.raw.is_some());
    // Order: build warnings first, then parse warnings.
    assert_eq!(response.warnings.len(), 2);
    assert_eq!(
        response.warnings[0].code,
        WarningCode::SamplingParameterDropped
    );
    assert_eq!(
        response.warnings[0].direction,
        ConversionDirection::ToFormat
    );
    assert_eq!(response.warnings[1].code, WarningCode::MalformedField);
    assert_eq!(
        response.warnings[1].direction,
        ConversionDirection::FromFormat
    );
}

#[tokio::test]
async fn send_transport_failure_is_transport_error() {
    let (client, _) = scripted(vec![Scripted::Fail(HttpErrorKind::Connect)]);
    let err = client
        .send(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Transport(e) if e.kind == HttpErrorKind::Connect));
}

#[tokio::test]
async fn send_success_body_over_cap_fails_with_metadata() {
    let (client, _) = scripted(vec![respond(
        200,
        &[("content-type", "application/json")],
        &[b'x'; 50],
    )]);
    let mut provider = mock_provider();
    provider.limits = Limits::new().with_max_response_body(8);

    let err = client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    match err {
        Error::BodyTooLarge {
            kind,
            limit,
            status,
            headers,
            prefix,
            ..
        } => {
            assert_eq!(kind, BodyKind::SuccessBody);
            assert_eq!(limit, 8);
            assert_eq!(status, Some(200));
            assert_eq!(
                headers.unwrap().get("content-type").unwrap(),
                "application/json"
            );
            assert_eq!(prefix.len(), 8);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn send_success_body_transport_failure_mid_read() {
    let (client, _) = scripted(vec![Scripted::Respond {
        status: 200,
        headers: vec![("content-type", "application/json")],
        chunks: vec![
            Ok(Bytes::from_static(b"{\"text\":")),
            Err(HttpError::new(HttpErrorKind::Body)),
        ],
    }]);
    let err = client
        .send(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Transport(e) if e.kind == HttpErrorKind::Body));
}

#[tokio::test]
async fn send_api_error_full_body() {
    let (client, _) = scripted(vec![respond(
        429,
        &[("content-type", "application/json"), ("retry-after", "7")],
        json!({"error": {"message": "slow down"}})
            .to_string()
            .as_bytes(),
    )]);
    let err = client
        .send(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            message,
            truncated,
            parsed,
            retry_after,
            headers,
            ..
        } => {
            assert_eq!(status, 429);
            assert_eq!(kind, ApiErrorKind::RateLimit);
            assert_eq!(message, "slow down");
            assert!(!truncated);
            assert!(parsed.is_some());
            assert_eq!(retry_after, Some(Duration::from_secs(7)));
            assert_eq!(headers.get("retry-after").unwrap(), "7");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn send_api_error_body_over_cap_truncates_but_stays_api() {
    let (client, _) = scripted(vec![respond(500, &[("retry-after", "3")], &[b'e'; 50])]);
    let mut provider = mock_provider();
    provider.limits = Limits::new().with_max_error_body(10);

    let err = client
        .send(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            truncated,
            raw,
            retry_after,
            headers,
            ..
        } => {
            assert_eq!(status, 500);
            assert_eq!(kind, ApiErrorKind::ServerError);
            assert!(truncated);
            assert_eq!(raw.len(), 10, "raw is exactly the read prefix");
            assert_eq!(retry_after, Some(Duration::from_secs(3)));
            assert_eq!(headers.get("retry-after").unwrap(), "3");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn send_api_error_transport_failure_mid_body_keeps_classification() {
    let (client, _) = scripted(vec![Scripted::Respond {
        status: 503,
        headers: vec![],
        chunks: vec![
            Ok(Bytes::from_static(b"par")),
            Err(HttpError::new(HttpErrorKind::Body)),
        ],
    }]);
    let err = client
        .send(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            truncated,
            raw,
            ..
        } => {
            assert_eq!(status, 503);
            assert_eq!(kind, ApiErrorKind::Overloaded);
            assert!(truncated);
            assert_eq!(&raw[..], b"par");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_pump_exposes_meta_and_items() {
    let (client, captured) = scripted(vec![respond_sse(&[
        json!({"k": "start", "id": "s1"}),
        json!({"k": "text", "i": 0, "t": "he"}),
        json!({"k": "text", "i": 0, "t": "llo"}),
        json!({"k": "stop"}),
    ])]);
    let provider = mock_provider();
    let mut request = user_request();
    request.top_k = Some(1);

    let handle = client
        .stream(&provider, &request, &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(handle.status(), 200);
    assert_eq!(handle.headers().get("x-sse").unwrap(), "yes");
    assert_eq!(handle.warnings().len(), 1);
    assert_eq!(
        handle.warnings()[0].code,
        WarningCode::SamplingParameterDropped
    );

    // The build side saw the streaming mode.
    {
        let calls = captured.lock().unwrap();
        let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
        assert_eq!(body["stream"], json!(true));
    }

    let items = drain(handle).await;
    assert_eq!(items.len(), 6);
    assert!(matches!(
        ok_item(&items, 0).event,
        StreamEvent::MessageStart { .. }
    ));
    assert!(matches!(
        ok_item(&items, 1).event,
        StreamEvent::BlockStart { .. }
    ));
    assert!(matches!(
        ok_item(&items, 2).event,
        StreamEvent::BlockDelta { .. }
    ));
    assert!(matches!(
        ok_item(&items, 3).event,
        StreamEvent::BlockDelta { .. }
    ));
    assert!(matches!(
        ok_item(&items, 4).event,
        StreamEvent::MessageDelta { .. }
    ));
    assert!(matches!(ok_item(&items, 5).event, StreamEvent::MessageStop));
    // Without include_raw, known events carry no raw payload.
    assert!(
        items
            .iter()
            .all(|i| ok_item(std::slice::from_ref(i), 0).raw.is_none())
    );
}

#[tokio::test]
async fn stream_collect_matches_manual_accumulator() {
    let session = [
        json!({"k": "start", "id": "s1"}),
        json!({"k": "text", "i": 0, "t": "hel"}),
        json!({"k": "text", "i": 0, "t": "lo"}),
        json!({"k": "stop"}),
    ];
    let (client, _) = scripted(vec![respond_sse(&session), respond_sse(&session)]);
    let provider = mock_provider();
    let mut request = user_request();
    request.top_k = Some(1);

    let collected = client
        .stream(&provider, &request, &CallOptions::new())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let mut handle = client
        .stream(&provider, &request, &CallOptions::new())
        .await
        .unwrap();
    let mut acc = Accumulator::new();
    acc.set_status(handle.status());
    acc.set_headers(handle.headers().clone());
    acc.extend_warnings(handle.warnings().to_vec());
    while let Some(item) = handle.next().await {
        acc.push(&item.unwrap()).unwrap();
    }
    let manual = acc.finish().unwrap();

    assert_eq!(collected, manual);
    assert_eq!(collected.text(), "hello");
    assert_eq!(collected.id.as_deref(), Some("s1"));
    assert_eq!(collected.status, 200);
    assert_eq!(collected.usage.as_ref().unwrap().output_tokens, 5);
    assert!(
        collected.raw.is_none(),
        "accumulated responses carry no raw body"
    );
    assert_eq!(
        collected.warnings[0].code,
        WarningCode::SamplingParameterDropped
    );
}

#[tokio::test]
async fn stream_warnings_attach_to_first_item_of_batch() {
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        // One parse call producing two events + one warning.
        json!({"k": "text", "i": 0, "t": "a", "warn": true}),
        json!({"k": "stop"}),
    ])]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    assert!(ok_item(&items, 0).warnings.is_empty());
    let first_of_batch = ok_item(&items, 1);
    assert!(matches!(
        first_of_batch.event,
        StreamEvent::BlockStart { .. }
    ));
    assert_eq!(first_of_batch.warnings.len(), 1);
    assert_eq!(first_of_batch.warnings[0].message, "attached");
    assert!(
        ok_item(&items, 2).warnings.is_empty(),
        "second item of the batch carries none"
    );
}

#[tokio::test]
async fn stream_held_warnings_attach_to_next_item() {
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        // No events: the warning is held...
        json!({"k": "warn", "m": "held"}),
        // ...and attaches before this call's own warning on its first item.
        json!({"k": "text", "i": 0, "t": "x", "warn": true}),
        json!({"k": "stop"}),
    ])]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    let carrier = ok_item(&items, 1);
    assert!(matches!(carrier.event, StreamEvent::BlockStart { .. }));
    let messages: Vec<&str> = carrier
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .collect();
    assert_eq!(messages, ["held", "attached"]);
}

#[tokio::test]
async fn stream_synthesizes_carrier_for_warnings_at_eof() {
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        json!({"k": "text", "i": 0, "t": "x"}),
        json!({"k": "stop"}),
        // Held warning with no later item to carry it.
        json!({"k": "warn", "m": "tail"}),
    ])]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    let last = ok_item(&items, items.len() - 1);
    assert!(matches!(last.event, StreamEvent::Unknown));
    assert!(last.raw.is_none(), "synthetic carrier has no raw payload");
    assert_eq!(last.warnings.len(), 1);
    assert_eq!(last.warnings[0].message, "tail");
}

#[tokio::test]
async fn stream_finish_events_follow_the_same_rules() {
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        json!({"k": "hold", "t": "tail-block"}),
        json!({"k": "stop"}),
    ])]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    let last = ok_item(&items, items.len() - 1);
    match &last.event {
        StreamEvent::BlockStart { index, block, .. } => {
            assert_eq!(*index, 90);
            assert!(matches!(block, ContentBlock::Text { text, .. } if text == "tail-block"));
        }
        other => panic!("unexpected final event: {other:?}"),
    }
    assert_eq!(last.warnings.len(), 1);
    assert_eq!(last.warnings[0].message, "finish warning");
    assert!(
        last.raw.is_none(),
        "finish-produced events have no source SSE event"
    );
}

#[tokio::test]
async fn stream_include_raw_populates_every_item() {
    let events = [
        json!({"k": "start", "id": "s1"}),
        json!({"k": "text", "i": 0, "t": "x"}),
        json!({"k": "stop"}),
    ];
    let (client, _) = scripted(vec![respond_sse(&events)]);
    let opts = CallOptions::new().with_include_raw(true);
    let handle = client
        .stream(&mock_provider(), &user_request(), &opts)
        .await
        .unwrap();
    let items = drain(handle).await;

    assert_eq!(items.len(), 5);
    let expected = [
        events[0].to_string(),
        events[1].to_string(), // BlockStart
        events[1].to_string(), // BlockDelta of the same SSE event
        events[2].to_string(), // MessageDelta
        events[2].to_string(), // MessageStop
    ];
    for (i, expected) in expected.iter().enumerate() {
        assert_eq!(
            ok_item(&items, i).raw.as_deref(),
            Some(expected.as_str()),
            "item {i}"
        );
    }
}

#[tokio::test]
async fn stream_unknown_event_always_carries_raw() {
    let unknown = json!({"k": "mystery"});
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        unknown.clone(),
        json!({"k": "stop"}),
    ])]);
    // include_raw is off; Unknown still gets its payload.
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    let item = ok_item(&items, 1);
    assert!(matches!(item.event, StreamEvent::Unknown));
    assert_eq!(item.raw.as_deref(), Some(unknown.to_string().as_str()));
    assert_eq!(item.warnings[0].code, WarningCode::UnknownStreamEvent);
    assert!(ok_item(&items, 0).raw.is_none());
}

#[tokio::test]
async fn stream_sse_event_over_cap_fails_with_meta() {
    let big = json!({"k": "text", "i": 0, "t": "X".repeat(100)});
    let (client, _) = scripted(vec![respond_sse(&[json!({"k": "start"}), big])]);
    let mut provider = mock_provider();
    provider.limits = Limits::new().with_max_sse_event(40);

    let handle = client
        .stream(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    assert!(matches!(
        ok_item(&items, 0).event,
        StreamEvent::MessageStart { .. }
    ));
    match &items[1] {
        Err(Error::BodyTooLarge {
            kind,
            limit,
            status,
            headers,
            ..
        }) => {
            assert_eq!(*kind, BodyKind::SseEvent);
            assert_eq!(*limit, 40);
            assert_eq!(*status, Some(200), "status filled in by the client");
            let headers = headers.as_ref().expect("headers filled in by the client");
            assert_eq!(headers.get("x-sse").unwrap(), "yes");
        }
        other => panic!("unexpected item: {other:?}"),
    }
    assert_eq!(items.len(), 2, "the stream terminates after the error");
}

#[tokio::test]
async fn stream_transport_error_mid_stream_terminates() {
    let (client, _) = scripted(vec![Scripted::Respond {
        status: 200,
        headers: vec![("content-type", "text/event-stream")],
        chunks: vec![
            sse_chunk(&json!({"k": "start"})),
            Err(HttpError::new(HttpErrorKind::Body)),
        ],
    }]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    assert_eq!(items.len(), 2);
    assert!(matches!(
        ok_item(&items, 0).event,
        StreamEvent::MessageStart { .. }
    ));
    assert!(matches!(&items[1], Err(Error::Transport(e)) if e.kind == HttpErrorKind::Body));
}

#[tokio::test]
async fn stream_parser_error_terminates() {
    let (client, _) = scripted(vec![respond_sse(&[
        json!({"k": "start"}),
        json!({"k": "bad"}),
        // Never reaches the parser: the pump stops after a fatal error.
        json!({"k": "text", "i": 0, "t": "x"}),
    ])]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;

    assert_eq!(items.len(), 2);
    assert!(matches!(&items[1], Err(Error::NotSupported(m)) if m.contains("bad stream event")));
}

#[tokio::test]
async fn stream_truncated_at_eof_surfaces_finish_error() {
    let session = [
        json!({"k": "start"}),
        json!({"k": "text", "i": 0, "t": "x"}),
    ];
    let (client, _) = scripted(vec![respond_sse(&session), respond_sse(&session)]);
    let provider = mock_provider();

    let handle = client
        .stream(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;
    assert!(matches!(
        items.last().unwrap(),
        Err(Error::NotSupported(m)) if m.contains("truncated")
    ));

    // collect() propagates the same failure.
    let err = client
        .stream(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported(m) if m.contains("truncated")));
}

#[tokio::test]
async fn stream_non_2xx_fails_like_send() {
    let (client, _) = scripted(vec![respond(
        401,
        &[("content-type", "application/json")],
        json!({"error": {"message": "bad key"}})
            .to_string()
            .as_bytes(),
    )]);
    let err = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            message,
            ..
        } => {
            assert_eq!(status, 401);
            assert_eq!(kind, ApiErrorKind::Auth);
            assert_eq!(message, "bad key");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn stream_in_stream_api_error_gains_response_headers() {
    // Parser-built in-stream API errors carry no HTTP metadata; the handle
    // fills in the real response headers and derives `retry_after`.
    let session = [json!({"k": "start"}), json!({"k": "apierr"})];
    let scripted_response = || Scripted::Respond {
        status: 200,
        headers: vec![
            ("content-type", "text/event-stream"),
            ("request-id", "req-1"),
            ("retry-after", "9"),
        ],
        chunks: session.iter().map(sse_chunk).collect(),
    };
    let (client, _) = scripted(vec![scripted_response(), scripted_response()]);
    let provider = mock_provider();

    let handle = client
        .stream(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;
    assert_eq!(items.len(), 2, "the stream terminates after the error");
    match &items[1] {
        Err(Error::Api {
            status,
            kind,
            headers,
            retry_after,
            ..
        }) => {
            assert_eq!(*status, 429, "parser-reported status survives");
            assert_eq!(*kind, ApiErrorKind::RateLimit);
            assert_eq!(
                headers.get("request-id").unwrap(),
                "req-1",
                "response headers filled in"
            );
            assert_eq!(*retry_after, Some(Duration::from_secs(9)));
        }
        other => panic!("unexpected item: {other:?}"),
    }

    // collect() consumes the handle (and its headers accessor); the error
    // itself must still carry the response headers.
    let err = client
        .stream(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    match err {
        Error::Api {
            headers,
            retry_after,
            ..
        } => {
            assert_eq!(headers.get("request-id").unwrap(), "req-1");
            assert_eq!(retry_after, Some(Duration::from_secs(9)));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn stream_in_stream_api_error_preserves_parser_headers() {
    let (client, _) = scripted(vec![Scripted::Respond {
        status: 200,
        headers: vec![
            ("content-type", "text/event-stream"),
            ("request-id", "req-1"),
            ("retry-after", "9"),
        ],
        chunks: vec![sse_chunk(&json!({"k": "apierr", "own_headers": true}))],
    }]);
    let handle = client
        .stream(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap();
    let items = drain(handle).await;
    match &items[0] {
        Err(Error::Api {
            headers,
            retry_after,
            ..
        }) => {
            // Headers a parser set deliberately survive untouched…
            assert_eq!(headers.get("x-parser").unwrap(), "own");
            assert!(headers.get("request-id").is_none());
            // …while a missing retry_after is still derived from the real
            // response headers.
            assert_eq!(*retry_after, Some(Duration::from_secs(9)));
        }
        other => panic!("unexpected item: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// list_models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_models_paginates_via_derived_endpoint() {
    let (client, captured) = scripted(vec![
        respond_json(&json!({"data": [{"id": "a"}, {"id": "b"}], "next": "c1"})),
        respond_json(&json!({"data": [{"id": "c"}]})),
    ]);
    let provider = mock_provider(); // models endpoint unset -> derived from chat

    let models = client.list_models(&provider).await.unwrap();
    assert_eq!(
        models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    assert_eq!(models[0].raw, json!({"id": "a"}));

    let calls = captured.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].method, http::Method::GET);
    assert_eq!(calls[0].uri.to_string(), "http://mock.local/v1/models");
    assert_eq!(
        calls[1].uri.to_string(),
        "http://mock.local/v1/models?cursor=c1"
    );
    // Derived endpoint inherits provider auth.
    assert_eq!(calls[0].auth.as_ref().unwrap().value.expose(), "secret-key");
}

#[tokio::test]
async fn derived_endpoints_keep_chat_auth_and_header_overrides() {
    // Unset models/count_tokens derive from `chat` — including its Set
    // auth and header overrides, not just format and URL.
    let (client, captured) = scripted(vec![
        respond_json(&json!({"data": [{"id": "a"}]})),
        respond_json(&json!({"input_tokens": 3})),
    ]);
    let mut provider = mock_provider();
    provider.auth = None; // prove the endpoint Set auth is what travels
    provider.chat.auth = Override::Set(AuthHeader {
        name: http::HeaderName::from_static("x-chat-key"),
        prefix: None,
        value: ApiKey::new("chat-secret"),
    });
    let mut endpoint_headers = http::HeaderMap::new();
    endpoint_headers.insert("x-endpoint", http::HeaderValue::from_static("e"));
    provider.chat.headers = Override::Set(endpoint_headers);

    client.list_models(&provider).await.unwrap();
    client
        .count_tokens(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    assert_eq!(calls.len(), 2);
    for (i, call) in calls.iter().enumerate() {
        let auth = call
            .auth
            .as_ref()
            .unwrap_or_else(|| panic!("call {i}: derived endpoint must send the chat Set auth"));
        assert_eq!(auth.name, "x-chat-key", "call {i}");
        assert_eq!(auth.value.expose(), "chat-secret", "call {i}");
        assert_eq!(
            call.headers.get("x-endpoint").unwrap(),
            "e",
            "call {i}: derived endpoint must apply the chat Set headers"
        );
    }
}

#[tokio::test]
async fn derived_endpoints_keep_chat_auth_disable() {
    let (client, captured) = scripted(vec![
        respond_json(&json!({"data": []})),
        respond_json(&json!({"input_tokens": 3})),
    ]);
    let mut provider = mock_provider(); // provider key is set
    provider.chat.auth = Override::Disable;

    client.list_models(&provider).await.unwrap();
    client
        .count_tokens(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();

    let calls = captured.lock().unwrap();
    assert!(
        calls.iter().all(|c| c.auth.is_none()),
        "chat auth Disable must apply to derived endpoints"
    );
}

#[tokio::test]
async fn list_models_repeated_cursor_is_malformed_pagination() {
    let (client, _) = scripted(vec![
        respond_json(&json!({"data": [{"id": "a"}], "next": "c1"})),
        respond_json(&json!({"data": [{"id": "b"}], "next": "c1"})),
    ]);
    let err = client.list_models(&mock_provider()).await.unwrap_err();
    match err {
        Error::Parse { message, .. } => assert!(message.contains("malformed pagination")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn list_models_full_chat_url_without_config_not_supported() {
    let (client, captured) = scripted(vec![]);
    let provider = ProviderConfig::from_endpoint(
        EndpointConfig::new(
            MockFormat::new("mock").arc(),
            EndpointUrl::full("http://h/chat"),
        ),
        "m",
    );
    let err = client.list_models(&provider).await.unwrap_err();
    assert!(matches!(err, Error::NotSupported(_)));
    assert!(captured.lock().unwrap().is_empty());
}

#[tokio::test]
async fn list_models_explicit_endpoint_overrides() {
    let (client, captured) = scripted(vec![respond_json(&json!({"data": [{"id": "a"}]}))]);
    // Full chat URL, but models has its own endpoint with its own auth.
    let provider = ProviderConfig::from_endpoint(
        EndpointConfig::new(
            MockFormat::new("mock").arc(),
            EndpointUrl::full("http://h/chat"),
        ),
        "m",
    )
    .with_auth("chat-key")
    .with_models_endpoint(
        EndpointConfig::new(
            MockFormat::new("mock").arc(),
            EndpointUrl::base("http://models.local/api").unwrap(),
        )
        .with_auth(Override::Set(AuthHeader {
            name: http::HeaderName::from_static("x-models-key"),
            prefix: None,
            value: ApiKey::new("mk"),
        })),
    );

    let models = client.list_models(&provider).await.unwrap();
    assert_eq!(models.len(), 1);

    let calls = captured.lock().unwrap();
    assert_eq!(calls[0].uri.to_string(), "http://models.local/api/models");
    let auth = calls[0].auth.as_ref().unwrap();
    assert_eq!(auth.name, "x-models-key");
    assert_eq!(auth.value.expose(), "mk");
}

#[tokio::test]
async fn list_models_non_2xx_is_api_error() {
    let (client, _) = scripted(vec![respond(
        403,
        &[],
        json!({"error": {"message": "forbidden"}})
            .to_string()
            .as_bytes(),
    )]);
    let err = client.list_models(&mock_provider()).await.unwrap_err();
    assert!(matches!(err, Error::Api { status: 403, .. }));
}

// ---------------------------------------------------------------------------
// count_tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn count_tokens_derived_endpoint_and_warning_order() {
    let (client, captured) = scripted(vec![respond_json(
        &json!({"input_tokens": 42, "warn": true}),
    )]);
    let provider = mock_provider();
    let mut request = user_request();
    request.top_k = Some(7); // build warning

    let count = client
        .count_tokens(&provider, &request, &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(count.input_tokens, 42);
    assert!(count.raw.is_some());
    // Same-format counting: build warning then parse warning, no
    // approximation marker.
    let codes: Vec<&WarningCode> = count.warnings.iter().map(|w| &w.code).collect();
    assert_eq!(
        codes,
        [
            &WarningCode::SamplingParameterDropped,
            &WarningCode::MalformedField
        ]
    );

    let calls = captured.lock().unwrap();
    assert_eq!(calls[0].uri.to_string(), "http://mock.local/v1/count");
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(body["count"], json!(true));
    assert_eq!(body["model"], json!("base-model"));
}

#[tokio::test]
async fn count_tokens_call_options_apply_and_include_raw_is_ignored() {
    let response = json!({"input_tokens": 5});
    let (client, captured) = scripted(vec![respond_json(&response), respond_json(&response)]);
    let provider = mock_provider();
    let request = user_request();

    let with_raw = CallOptions::new()
        .with_model("count-model")
        .with_include_raw(true);
    let without_raw = CallOptions::new().with_model("count-model");
    let a = client
        .count_tokens(&provider, &request, &with_raw)
        .await
        .unwrap();
    let b = client
        .count_tokens(&provider, &request, &without_raw)
        .await
        .unwrap();

    assert_eq!(a, b, "include_raw must have no effect on token counting");
    let calls = captured.lock().unwrap();
    assert_eq!(calls[0].body, calls[1].body);
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert_eq!(
        body["model"],
        json!("count-model"),
        "per-call model applies"
    );
}

#[tokio::test]
async fn count_tokens_decoupled_format_adds_approximation_warning() {
    let (client, _) = scripted(vec![respond_json(
        &json!({"input_tokens": 9, "warn": true}),
    )]);
    let mut provider = mock_provider();
    provider.count_tokens = Some(EndpointConfig::new(
        MockFormat::new("mock_b").arc(), // different instance and id
        EndpointUrl::base("http://mock.local/v1").unwrap(),
    ));
    let mut request = user_request();
    request.top_k = Some(7);

    let count = client
        .count_tokens(&provider, &request, &CallOptions::new())
        .await
        .unwrap();
    let codes: Vec<&WarningCode> = count.warnings.iter().map(|w| &w.code).collect();
    assert_eq!(
        codes,
        [
            &WarningCode::SamplingParameterDropped,
            &WarningCode::CountTokensApproximate,
            &WarningCode::MalformedField,
        ],
        "order: build warnings, approximation marker, parse warnings"
    );
    let approx = &count.warnings[1];
    assert_eq!(approx.format, "mock_b");
    assert_eq!(approx.direction, ConversionDirection::ToFormat);
}

#[tokio::test]
async fn count_tokens_same_id_different_instance_is_not_decoupled() {
    let (client, _) = scripted(vec![respond_json(&json!({"input_tokens": 9}))]);
    let mut provider = mock_provider();
    // A different Arc but the same format id: not decoupled.
    provider.count_tokens = Some(EndpointConfig::new(
        MockFormat::new("mock").arc(),
        EndpointUrl::base("http://mock.local/v1").unwrap(),
    ));

    let count = client
        .count_tokens(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap();
    assert!(
        !count
            .warnings
            .iter()
            .any(|w| w.code == WarningCode::CountTokensApproximate),
        "same-id formats must not be marked approximate"
    );
}

#[tokio::test]
async fn count_tokens_full_chat_url_without_config_not_supported() {
    let (client, _) = scripted(vec![]);
    let provider = ProviderConfig::from_endpoint(
        EndpointConfig::new(
            MockFormat::new("mock").arc(),
            EndpointUrl::full("http://h/chat"),
        ),
        "m",
    );
    let err = client
        .count_tokens(&provider, &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported(_)));
}

#[tokio::test]
async fn count_tokens_non_2xx_is_api_error() {
    let (client, _) = scripted(vec![respond(
        400,
        &[],
        json!({"error": {"message": "nope"}}).to_string().as_bytes(),
    )]);
    let err = client
        .count_tokens(&mock_provider(), &user_request(), &CallOptions::new())
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Api {
            status: 400,
            kind: ApiErrorKind::InvalidRequest,
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// End-to-end over real HTTP: wiremock + the reqwest transport
// ---------------------------------------------------------------------------

#[cfg(feature = "reqwest")]
mod wire {
    use super::*;
    use wiremock::matchers::{
        body_partial_json, header, method, path, query_param, query_param_is_missing,
    };
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn wire_provider(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new(MockFormat::new("mock").arc(), &server.uri(), "m-wire")
            .unwrap()
            .with_auth("real-key")
    }

    #[tokio::test]
    async fn send_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(header("authorization", "Bearer real-key"))
            .and(header("content-type", "application/json"))
            .and(body_partial_json(
                json!({"model": "m-wire", "stream": false}),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-rate", "42")
                    .set_body_json(json!({"text": "hello", "warn": true})),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let mut request = user_request();
        request.top_k = Some(2);

        let response = client
            .send(&provider, &request, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(response.text(), "hello");
        assert_eq!(response.status, 200);
        assert_eq!(response.headers.get("x-rate").unwrap(), "42");
        assert_eq!(response.warnings.len(), 2);
        assert_eq!(
            response.warnings[0].code,
            WarningCode::SamplingParameterDropped
        );
        assert_eq!(response.warnings[1].code, WarningCode::MalformedField);
    }

    #[tokio::test]
    async fn auth_injection_beats_all_header_layers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(header("authorization", "Bearer real-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": "ok"})))
            .expect(1)
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        // A per-call attempt to spoof the auth header loses to injection.
        let opts = CallOptions::new().with_header(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_static("Bearer spoofed"),
        );
        let client = Client::new(reqwest::Client::new());
        client
            .send(&provider, &user_request(), &opts)
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        let values: Vec<_> = requests[0]
            .headers
            .get_all("authorization")
            .iter()
            .collect();
        assert_eq!(values, ["Bearer real-key"], "exactly one, injected value");
    }

    #[tokio::test]
    async fn api_error_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "7")
                    .set_body_json(json!({"error": {"message": "slow down"}})),
            )
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let err = client
            .send(&provider, &user_request(), &CallOptions::new())
            .await
            .unwrap_err();
        match err {
            Error::Api {
                status,
                kind,
                message,
                retry_after,
                truncated,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(kind, ApiErrorKind::RateLimit);
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(Duration::from_secs(7)));
                assert!(!truncated);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn connection_failure_is_transport_error() {
        // Bind and drop a listener to get a port that refuses connections.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let provider = ProviderConfig::new(
            MockFormat::new("mock").arc(),
            &format!("http://127.0.0.1:{port}"),
            "m",
        )
        .unwrap();
        let client = Client::new(reqwest::Client::new());
        let err = client
            .send(&provider, &user_request(), &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[tokio::test]
    async fn stream_end_to_end_collect() {
        let events = [
            json!({"k": "start", "id": "wire-1"}),
            json!({"k": "text", "i": 0, "t": "str"}),
            json!({"k": "text", "i": 0, "t": "eam"}),
            json!({"k": "stop"}),
        ];
        let body: String = events.iter().map(|v| format!("data: {v}\n\n")).collect();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .and(header("authorization", "Bearer real-key"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let handle = client
            .stream(&provider, &user_request(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(handle.status(), 200);
        let response = handle.collect().await.unwrap();
        assert_eq!(response.text(), "stream");
        assert_eq!(response.id.as_deref(), Some("wire-1"));
        assert_eq!(response.usage.as_ref().unwrap().input_tokens, 3);
    }

    #[tokio::test]
    async fn stream_non_2xx_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat"))
            .respond_with(
                ResponseTemplate::new(401).set_body_json(json!({"error": {"message": "bad key"}})),
            )
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let err = client
            .stream(&provider, &user_request(), &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Api {
                status: 401,
                kind: ApiErrorKind::Auth,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn list_models_pagination_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header("authorization", "Bearer real-key"))
            .and(query_param_is_missing("cursor"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"data": [{"id": "a"}], "next": "c1"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param("cursor", "c1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "b"}]})))
            .expect(1)
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let models = client.list_models(&provider).await.unwrap();
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[tokio::test]
    async fn count_tokens_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/count"))
            .and(header("authorization", "Bearer real-key"))
            .and(body_partial_json(json!({"count": true, "model": "m-wire"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 11})))
            .expect(1)
            .mount(&server)
            .await;

        let provider = wire_provider(&server).await;
        let client = Client::new(reqwest::Client::new());
        let count = client
            .count_tokens(&provider, &user_request(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(count.input_tokens, 11);
        assert!(count.warnings.is_empty());
    }
}
