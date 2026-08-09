//! Feature-gated default transport: `impl HttpClient for reqwest::Client`.
//!
//! The library does not wrap reqwest configuration — build your own client
//! (proxy, timeouts, TLS) and pass it in.

use std::future::Future;
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;

use super::{AuthHeader, BodyStream, HttpClient, HttpError, HttpErrorKind};

fn classify(e: &reqwest::Error) -> HttpErrorKind {
    if e.is_timeout() {
        HttpErrorKind::Timeout
    } else if e.is_connect() {
        HttpErrorKind::Connect
    } else if e.is_body() || e.is_decode() {
        HttpErrorKind::Body
    } else {
        HttpErrorKind::Other
    }
}

impl HttpClient for reqwest::Client {
    fn send(
        &self,
        request: http::Request<Bytes>,
        auth: Option<AuthHeader>,
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>
    {
        Box::pin(async move {
            let (parts, body) = request.into_parts();
            let url = reqwest::Url::parse(&parts.uri.to_string())
                .map_err(|e| HttpError::with_source(HttpErrorKind::Other, e))?;
            let mut headers = parts.headers;
            if let Some(auth) = auth {
                // Injected last; `header_value` marks the value sensitive.
                let value = auth
                    .header_value()
                    .map_err(|e| HttpError::with_source(HttpErrorKind::Other, e))?;
                headers.insert(auth.name, value);
            }
            let request = self
                .request(parts.method, url)
                .headers(headers)
                .body(body)
                .build()
                .map_err(|e| HttpError::with_source(HttpErrorKind::Other, e))?;
            let response = self
                .execute(request)
                .await
                .map_err(|e| HttpError::with_source(classify(&e), e))?;

            let status = response.status();
            let headers = response.headers().clone();
            let stream = response.bytes_stream();
            let body: BodyStream = Box::pin(MapErrStream { inner: stream });

            let mut builder = http::Response::builder().status(status);
            if let Some(h) = builder.headers_mut() {
                *h = headers;
            }
            builder.body(body).map_err(|e| HttpError::with_source(HttpErrorKind::Other, e))
        })
    }
}

/// Adapts reqwest's byte stream to `BodyStream` without pulling in
/// `futures-util`.
struct MapErrStream<S> {
    inner: S,
}

impl<S> Stream for MapErrStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Bytes, HttpError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx).map(|opt| {
            opt.map(|res| res.map_err(|e| HttpError::with_source(classify(&e), e)))
        })
    }
}
