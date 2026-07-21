// fauxx-desktop: Fauxx Desktop Companion
// Copyright (C) 2026 Digital Grease
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! LM Studio sidecar: [`LlmConfig`], the [`LlmTransport`] seam
//! ([`LmStudioTransport`] for the real localhost server, [`MockTransport`] for
//! tests), and [`LmStudioAssistant`], which implements
//! [`SemanticAssistant`](crate::persona_engine::sidecar::SemanticAssistant) on
//! top of it.
//!
//! # Wire format
//!
//! LM Studio exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint
//! over plain HTTP on loopback (no TLS: it is a local developer tool, not a
//! network service, and `deny.toml` bans the native-TLS stacks anyway). Rather
//! than pull in a full HTTP client crate, [`LmStudioTransport`] speaks just
//! enough HTTP/1.1 by hand over a `tokio::net::TcpStream`, matching the house
//! preference for dependency-light infrastructure. It sends `Connection:
//! close`, but does NOT rely on the peer actually closing the socket to know
//! the response is complete: LM Studio's local server has been observed to
//! keep connections open (`Connection: keep-alive`, a several-second idle
//! timeout) regardless of what the client requests, which would make a naive
//! read-to-EOF client block for the full request timeout on every call. The
//! read loop instead parses the response's `Content-Length` header and stops
//! as soon as that many body bytes have arrived, falling back to read-to-EOF
//! only if no `Content-Length` is present (e.g. a chunked response), which
//! keeps the common case fast and correct without adding chunked-encoding
//! support.
//!
//! # Sync bridge
//!
//! [`SemanticAssistant`](crate::persona_engine::sidecar::SemanticAssistant) is
//! a plain synchronous trait (the whole goal/planner/appraisal pipeline is
//! synchronous by design, so tests stay `#[test]`, not `#[tokio::test]`, and a
//! `DisabledAssistant` never needs a runtime at all). [`LmStudioAssistant`]'s
//! trait methods therefore bridge into the async transport with
//! `tokio::task::block_in_place` + `Handle::current().block_on`, which
//! requires being called from within a MULTI-THREADED Tokio runtime (exactly
//! what `apps/cli`'s `#[tokio::main]` provides). This is an intentional,
//! documented constraint of an opt-in, advanced feature; it is never exercised
//! unless the operator explicitly enables the LLM sidecar.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::persona::CategoryPool;
use crate::persona_engine::policy::PersonaPolicy;
use crate::persona_engine::sidecar::SemanticAssistant;
use crate::querybank::QueryBlocklist;

/// Configuration for the optional local LLM sidecar. **Disabled by default**;
/// `fauxx-core` ships with no cloud provider and no bundled model. Enabling
/// this is an explicit, local-only operator opt-in (there is no config path
/// that reaches a non-loopback host by default; `endpoint` is operator-supplied).
#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    /// Whether the sidecar may be consulted at all. `false` by default.
    pub enabled: bool,
    /// The LM Studio server's `host:port` (no scheme), e.g. `127.0.0.1:1234`.
    pub endpoint: String,
    /// The model identifier LM Studio has loaded (its `/v1/models` id).
    pub model: String,
    /// Per-request timeout. A timeout is treated exactly like any other
    /// failure: the deterministic fallback is used.
    pub timeout_ms: u64,
    /// Bearer token to send as `Authorization: Bearer <token>`, for an LM
    /// Studio server with its optional "Require Authentication" server
    /// setting turned on (LM Studio 0.4+). `None` (the default) sends no
    /// `Authorization` header at all, which is what an unauthenticated local
    /// server (LM Studio's own default) expects.
    pub api_key: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "127.0.0.1:1234".to_string(),
            model: "local-model".to_string(),
            timeout_ms: 4_000,
            api_key: None,
        }
    }
}

/// A minimal chat-completion transport, injected so tests can swap in a
/// deterministic [`MockTransport`] instead of a real socket.
#[async_trait]
pub trait LlmTransport: Send + Sync {
    /// Send a system + user message pair; return the assistant's raw text
    /// reply, or an error describing what went wrong (network, non-2xx status,
    /// or a malformed response). Callers treat any `Err` as "unavailable".
    async fn chat(&self, system: &str, user: &str) -> Result<String, String>;
}

/// The real transport: a hand-rolled HTTP/1.1 POST to LM Studio's local
/// OpenAI-compatible endpoint. See the module doc for the wire-format notes.
#[derive(Debug, Clone)]
pub struct LmStudioTransport {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl LmStudioTransport {
    /// Build a transport from `config`. Does not connect until [`chat`](Self::chat)
    /// is called.
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            api_key: config.api_key.clone(),
            timeout: Duration::from_millis(config.timeout_ms),
        }
    }
}

/// The maximum number of header bytes read while looking for the `\r\n\r\n`
/// separator, before giving up. A guard against an endless or malicious
/// response, not a real-world limit (LM Studio's headers are tiny).
const MAX_HEADER_BYTES: usize = 32 * 1024;
/// The maximum total response size (headers + body) accepted. `max_tokens` is
/// bounded to 200 per request, so a real reply is a few KB at most; this is a
/// guard against an endless stream on a connection that never closes.
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Read one HTTP/1.1 response from `stream`: read until the header block is
/// complete, then read exactly `Content-Length` more body bytes (falling back
/// to reading until EOF if no `Content-Length` header is present, e.g. a
/// chunked response). Does NOT wait for the peer to close the connection,
/// which LM Studio's server may not do promptly even after `Connection:
/// close` (see the module doc).
async fn read_http_response(stream: &mut TcpStream) -> Result<String, String> {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];

    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err("response headers exceeded the size limit".to_string());
        }
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|e| format!("read response headers: {e}"))?;
        if n == 0 {
            return Err("connection closed before headers completed".to_string());
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let header_text = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let content_length = header_text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.trim()
            .eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<usize>().ok())
            .flatten()
    });
    let body_start = header_end + 4;

    match content_length {
        Some(len) => {
            let target = body_start
                .saturating_add(len)
                .min(body_start.saturating_add(MAX_RESPONSE_BYTES));
            while buf.len() < target {
                let n = stream
                    .read(&mut chunk)
                    .await
                    .map_err(|e| format!("read response body: {e}"))?;
                if n == 0 {
                    break; // peer closed early; return whatever body arrived.
                }
                let take = n.min(target - buf.len());
                buf.extend_from_slice(&chunk[..take]);
            }
        }
        None => loop {
            if buf.len() > MAX_RESPONSE_BYTES {
                return Err("response body exceeded the size limit".to_string());
            }
            let n = stream
                .read(&mut chunk)
                .await
                .map_err(|e| format!("read response body: {e}"))?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        },
    }

    String::from_utf8(buf).map_err(|e| format!("response is not UTF-8: {e}"))
}

/// The byte offset of the first occurrence of `needle` in `haystack`, if any.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[async_trait]
impl LlmTransport for LmStudioTransport {
    async fn chat(&self, system: &str, user: &str) -> Result<String, String> {
        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
            "stream": false,
        })
        .to_string();

        let mut headers = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: {host}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Connection: close\r\n",
            host = self.endpoint,
            len = request_body.len(),
        );
        if let Some(key) = &self.api_key {
            headers.push_str(&format!("Authorization: Bearer {key}\r\n"));
        }
        headers.push_str("\r\n");
        let http_request = format!("{headers}{request_body}");

        let call = async {
            let mut stream = TcpStream::connect(&self.endpoint)
                .await
                .map_err(|e| format!("connect to {}: {e}", self.endpoint))?;
            stream
                .write_all(http_request.as_bytes())
                .await
                .map_err(|e| format!("write request: {e}"))?;
            read_http_response(&mut stream).await
        };

        let raw_response = tokio::time::timeout(self.timeout, call)
            .await
            .map_err(|_| "request timed out".to_string())??;

        parse_http_chat_response(&raw_response)
    }
}

/// Parse a raw HTTP/1.1 response into the chat completion's message content.
/// Fails closed (an `Err`) on a non-2xx status, an unparseable header/body
/// split, or JSON that does not match the expected `choices[0].message.content`
/// shape.
fn parse_http_chat_response(raw: &str) -> Result<String, String> {
    let status_line = raw
        .lines()
        .next()
        .ok_or_else(|| "empty HTTP response".to_string())?;
    let status_ok = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .is_some_and(|code| (200..300).contains(&code));
    if !status_ok {
        return Err(format!("non-2xx HTTP response: {status_line}"));
    }

    let split_at = raw
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response: no header/body separator".to_string())?;
    let body = &raw[split_at + 4..];

    let parsed: ChatResponse =
        serde_json::from_str(body).map_err(|e| format!("invalid chat-completion JSON: {e}"))?;
    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "chat-completion response had no choices".to_string())
}

/// A deterministic, canned-response transport for tests. Never touches the
/// network. Configure it with [`MockTransport::ok`] or [`MockTransport::err`].
pub struct MockTransport {
    response: std::sync::Mutex<Result<String, String>>,
}

impl MockTransport {
    /// A transport whose `chat` always succeeds with `text`.
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            response: std::sync::Mutex::new(Ok(text.into())),
        }
    }

    /// A transport whose `chat` always fails with `message`.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            response: std::sync::Mutex::new(Err(message.into())),
        }
    }
}

#[async_trait]
impl LlmTransport for MockTransport {
    async fn chat(&self, _system: &str, _user: &str) -> Result<String, String> {
        self.response
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// The maximum accepted length (chars) of a proposed subseed, before any
/// downstream blocklist/Venn check even runs. A guard against a runaway or
/// misbehaving model response, not the safety check itself.
const MAX_PROPOSED_SEED_LEN: usize = 120;

/// An LLM-backed [`SemanticAssistant`], bounded to augmenting (never
/// replacing) the deterministic pipeline. See the module doc for the sync
/// bridge and its multi-threaded-runtime requirement.
pub struct LmStudioAssistant<T: LlmTransport> {
    config: LlmConfig,
    transport: T,
}

impl<T: LlmTransport> LmStudioAssistant<T> {
    /// Build an assistant from `config` and an injected `transport` (the real
    /// [`LmStudioTransport`] or, in tests, a [`MockTransport`]).
    pub fn new(config: LlmConfig, transport: T) -> Self {
        Self { config, transport }
    }

    /// Run one bounded chat call and return the raw text, or `None` on any
    /// failure (disabled, network error, or timeout). The ONLY place that
    /// bridges into the async transport.
    fn chat(&self, system: &str, user: &str) -> Option<String> {
        if !self.config.enabled {
            return None;
        }
        let fut = self.transport.chat(system, user);
        let result =
            tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(fut));
        result.ok()
    }
}

impl<T: LlmTransport> SemanticAssistant for LmStudioAssistant<T> {
    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    fn propose_subseed(
        &self,
        persona: &PersonaPolicy,
        category: &str,
        context: &str,
    ) -> Option<String> {
        // Schema guard: only propose within a category the persona is
        // actually allowed to have (belt and suspenders; appraisal re-checks
        // this against the full Venn regardless).
        CategoryPool::from_name(category)?;
        let system = format!(
            "You help a fictional, harmless decoy persona named {} explore a new, \
             closely related search topic. Persona's existing interests in this \
             category ({category}): {}. Given a short context, respond with ONLY \
             one new topic, three to eight words, no explanation, no quotes, no \
             punctuation besides spaces and hyphens. Never propose anything about \
             accounts, logins, purchases, medical, legal, financial, political, or \
             adult subjects.",
            persona.display_name,
            persona
                .topic_seeds
                .get(category)
                .map(|s| s.join(", "))
                .unwrap_or_default(),
        );
        let raw = self.chat(&system, context)?;
        let cleaned = raw.trim().trim_matches('"').trim();
        if cleaned.is_empty()
            || cleaned.chars().count() > MAX_PROPOSED_SEED_LEN
            || cleaned.contains('\n')
            || QueryBlocklist::bundled().is_blocked(cleaned)
        {
            return None;
        }
        Some(cleaned.to_string())
    }

    fn appraise_salience(&self, stimulus_text: &str, category: Option<&str>) -> Option<f64> {
        let system = "You rate, from 0.0 to 1.0, how much a specific fictional decoy \
                       persona would plausibly notice and care about a short, mundane \
                       observation. Respond with ONLY a single decimal number between \
                       0.0 and 1.0, nothing else."
            .to_string();
        let user = match category {
            Some(c) => format!("Category: {c}\nObservation: {stimulus_text}"),
            None => format!("Observation: {stimulus_text}"),
        };
        let raw = self.chat(&system, &user)?;
        raw.trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
    }

    fn summarize_memory(&self, recent: &[String]) -> Option<String> {
        if recent.is_empty() {
            return None;
        }
        let system = "You write ONE short, boring, third-person reflective sentence \
                       (under 25 words) summarizing a recurring theme in a list of \
                       recent, mundane observations about a fictional decoy persona. \
                       Respond with ONLY that sentence."
            .to_string();
        let user = recent.join("\n");
        let raw = self.chat(&system, &user)?;
        let cleaned = raw.trim().to_string();
        if cleaned.is_empty() || cleaned.chars().count() > 200 || cleaned.contains('\n') {
            return None;
        }
        Some(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona_engine::builtins;

    fn elias() -> PersonaPolicy {
        match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        }
    }

    fn enabled_config() -> LlmConfig {
        LlmConfig {
            enabled: true,
            ..LlmConfig::default()
        }
    }

    fn chat_json(content: &str) -> String {
        serde_json::json!({
            "choices": [{"message": {"content": content}}]
        })
        .to_string()
    }

    // --- parse_http_chat_response: pure, no runtime needed ---

    #[test]
    fn parses_a_well_formed_http_chat_response() {
        let body = chat_json("blotting paper technique");
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let parsed = parse_http_chat_response(&raw);
        assert_eq!(parsed, Ok("blotting paper technique".to_string()));
    }

    #[test]
    fn rejects_non_2xx_status() {
        let raw = "HTTP/1.1 500 Internal Server Error\r\n\r\n{}";
        assert!(parse_http_chat_response(raw).is_err());
    }

    #[test]
    fn rejects_malformed_json_body() {
        let raw = "HTTP/1.1 200 OK\r\n\r\nnot json";
        assert!(parse_http_chat_response(raw).is_err());
    }

    #[test]
    fn rejects_missing_header_body_separator() {
        assert!(parse_http_chat_response("garbage").is_err());
    }

    // --- MockTransport wire-through, via a real (local) TCP loopback server ---
    // for LmStudioTransport itself, so the hand-rolled HTTP client is exercised
    // end to end without needing a real LM Studio process.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lmstudio_transport_round_trips_over_a_loopback_server() {
        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(l) => l,
            Err(e) => panic!("failed to bind loopback listener: {e}"),
        };
        let addr = match listener.local_addr() {
            Ok(a) => a,
            Err(e) => panic!("failed to read local addr: {e}"),
        };

        let server = tokio::spawn(async move {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => panic!("failed to accept loopback connection: {e}"),
            };
            let mut buf = vec![0u8; 4096];
            let _ = socket.read(&mut buf).await; // discard the request
            let body = chat_json("mercury barometer tube replacement");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        });

        let config = LlmConfig {
            enabled: true,
            endpoint: addr.to_string(),
            model: "test-model".to_string(),
            timeout_ms: 2_000,
            api_key: None,
        };
        let transport = LmStudioTransport::new(&config);
        let reply = transport.chat("system", "user").await;
        if let Err(e) = server.await {
            panic!("server task failed: {e}");
        }
        assert_eq!(reply, Ok("mercury barometer tube replacement".to_string()));
    }

    /// Reproduces LM Studio's observed real-world behavior: the server does
    /// NOT necessarily close the socket after responding, even though the
    /// client sent `Connection: close` (it has been seen replying
    /// `Connection: keep-alive` with a several-second idle timeout). A client
    /// that waited for EOF to know the response was complete would block for
    /// the full request timeout on every single call. The `Content-Length`
    /// based reader must return as soon as the body has fully arrived,
    /// regardless of whether the peer ever closes the connection.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lmstudio_transport_returns_promptly_when_the_peer_holds_the_connection_open() {
        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(l) => l,
            Err(e) => panic!("failed to bind loopback listener: {e}"),
        };
        let addr = match listener.local_addr() {
            Ok(a) => a,
            Err(e) => panic!("failed to read local addr: {e}"),
        };

        let server = tokio::spawn(async move {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => panic!("failed to accept loopback connection: {e}"),
            };
            let mut buf = vec![0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let body = chat_json("mercury barometer tube replacement");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
            // Deliberately hold the socket open well past the client's
            // timeout instead of closing it, simulating an uncooperative
            // keep-alive server.
            tokio::time::sleep(Duration::from_millis(1_500)).await;
        });

        let config = LlmConfig {
            enabled: true,
            endpoint: addr.to_string(),
            model: "test-model".to_string(),
            timeout_ms: 400,
            api_key: None,
        };
        let transport = LmStudioTransport::new(&config);
        let started = std::time::Instant::now();
        let reply = transport.chat("system", "user").await;
        let elapsed = started.elapsed();
        assert_eq!(reply, Ok("mercury barometer tube replacement".to_string()));
        assert!(
            elapsed < Duration::from_millis(400),
            "expected a prompt return well under the timeout, took {elapsed:?}"
        );
        let _ = server.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lmstudio_transport_fails_closed_on_connection_refused() {
        // Nothing listens on this port; the connect must fail, not hang or panic.
        let config = LlmConfig {
            enabled: true,
            endpoint: "127.0.0.1:1".to_string(), // reserved, nothing listens
            model: "test-model".to_string(),
            timeout_ms: 500,
            api_key: None,
        };
        let transport = LmStudioTransport::new(&config);
        assert!(transport.chat("system", "user").await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lmstudio_transport_sends_the_bearer_token_when_configured() {
        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(l) => l,
            Err(e) => panic!("failed to bind loopback listener: {e}"),
        };
        let addr = match listener.local_addr() {
            Ok(a) => a,
            Err(e) => panic!("failed to read local addr: {e}"),
        };

        let server = tokio::spawn(async move {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => panic!("failed to accept loopback connection: {e}"),
            };
            let mut buf = vec![0u8; 4096];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).into_owned();
            let body = chat_json(if request.contains("Authorization: Bearer secret-token") {
                "authorized"
            } else {
                "unauthorized"
            });
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        });

        let config = LlmConfig {
            enabled: true,
            endpoint: addr.to_string(),
            model: "test-model".to_string(),
            timeout_ms: 2_000,
            api_key: Some("secret-token".to_string()),
        };
        let transport = LmStudioTransport::new(&config);
        let reply = transport.chat("system", "user").await;
        if let Err(e) = server.await {
            panic!("server task failed: {e}");
        }
        assert_eq!(reply, Ok("authorized".to_string()));
    }

    // --- LmStudioAssistant, via MockTransport (no network) ---

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disabled_assistant_never_calls_the_transport() {
        let assistant = LmStudioAssistant::new(
            LlmConfig::default(),
            MockTransport::err("should not be called"),
        );
        assert!(!assistant.is_enabled());
        let policy = elias();
        assert!(assistant
            .propose_subseed(&policy, "CRAFTS", "context")
            .is_none());
        assert!(assistant.appraise_salience("text", None).is_none());
        assert!(assistant.summarize_memory(&["x".to_string()]).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn enabled_assistant_proposes_a_clean_subseed() {
        let assistant = LmStudioAssistant::new(
            enabled_config(),
            MockTransport::ok("  \"nib grinding basics\"  "),
        );
        let policy = elias();
        let proposal = assistant.propose_subseed(&policy, "CRAFTS", "came across ink");
        assert_eq!(proposal.as_deref(), Some("nib grinding basics"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn proposal_is_rejected_when_blocklisted() {
        let assistant = LmStudioAssistant::new(enabled_config(), MockTransport::ok("call 988 now"));
        let policy = elias();
        assert!(assistant
            .propose_subseed(&policy, "CRAFTS", "context")
            .is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn proposal_is_rejected_for_an_unknown_category() {
        let assistant = LmStudioAssistant::new(enabled_config(), MockTransport::ok("some topic"));
        let policy = elias();
        assert!(assistant
            .propose_subseed(&policy, "NOT_A_REAL_CATEGORY", "context")
            .is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_error_falls_back_to_none() {
        let assistant =
            LmStudioAssistant::new(enabled_config(), MockTransport::err("connection refused"));
        let policy = elias();
        assert!(assistant
            .propose_subseed(&policy, "CRAFTS", "context")
            .is_none());
        assert!(assistant.appraise_salience("text", None).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn appraise_salience_parses_and_clamps() {
        let in_range = LmStudioAssistant::new(enabled_config(), MockTransport::ok("0.73"));
        assert_eq!(in_range.appraise_salience("text", None), Some(0.73));

        let over_range = LmStudioAssistant::new(enabled_config(), MockTransport::ok("5.0"));
        assert_eq!(over_range.appraise_salience("text", None), Some(1.0));

        let not_a_number = LmStudioAssistant::new(enabled_config(), MockTransport::ok("maybe?"));
        assert_eq!(not_a_number.appraise_salience("text", None), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn summarize_memory_rejects_empty_and_oversized() {
        let assistant =
            LmStudioAssistant::new(enabled_config(), MockTransport::ok("a".repeat(500)));
        assert!(assistant.summarize_memory(&["x".to_string()]).is_none());

        let empty_input = LmStudioAssistant::new(enabled_config(), MockTransport::ok("anything"));
        assert!(empty_input.summarize_memory(&[]).is_none());

        let good = LmStudioAssistant::new(
            enabled_config(),
            MockTransport::ok("keeps circling back to railway history"),
        );
        assert_eq!(
            good.summarize_memory(&["a".to_string(), "b".to_string()]),
            Some("keeps circling back to railway history".to_string())
        );
    }
}
