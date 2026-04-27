//! Fuzzy Sprint Phase 11 — MCP HTTP round-trip tests for the 14 fuzzy
//! tools (`fuzzy_probability` from Phase 10 + 13 Phase-11 additions).
//!
//! Every test uses the `one_shot_capture()` harness copied from
//! [`crate::mcp::http::tests`] so these tests stay standalone and don't
//! need the `server` feature at runtime (no axum router). Each test
//! asserts that:
//!
//! * The method + path match the documented REST envelope.
//! * The body / query-string round-trip through serde without drift.
//! * `None`-valued optional args produce a URL bit-identical to the
//!   pre-sprint shape (regression guard for the Phase 11 backward-compat
//!   invariant).
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
//!        [duboisprade1989fuzzyallen] [novak2008quantifiers]
//!        [murinovanovak2014peterson] [belohlavek2004fuzzyfca]
//!        [mamdani1975mamdani]
//!        [flaminio2026fsta] [faginhalpern1994fuzzyprob].

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::mcp::backend::McpBackend;
use crate::mcp::http::{append_fuzzy_query, HttpBackend};

/// Minimal one-shot HTTP responder used for Phase 11 HTTP-backend
/// proxy tests. Captures the first request as `(method, path, body)`
/// and replies with a canned JSON `{ "ok": true }` 200 response.
///
/// Mirrors the Phase 10 `one_shot_capture` helper in
/// [`crate::mcp::http::tests`]; we reproduce it here to keep the test
/// module standalone (the tests module in http.rs is non-`pub`).
pub(super) async fn one_shot_capture() -> (
    String,
    tokio::sync::oneshot::Receiver<(String, String, Vec<u8>)>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{}", addr);
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let mut total = 0usize;
        loop {
            let n = sock.read(&mut buf[total..]).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            total += n;
            if let Some(pos) = buf[..total]
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
            {
                let header_end = pos + 4;
                let header_owned: String =
                    String::from_utf8_lossy(&buf[..header_end]).into_owned();
                let mut content_len = 0usize;
                for line in header_owned.split("\r\n") {
                    if let Some(rest) = line.strip_prefix("Content-Length: ") {
                        content_len = rest.parse().unwrap_or(0);
                    } else if let Some(rest) = line.strip_prefix("content-length: ") {
                        content_len = rest.parse().unwrap_or(0);
                    }
                }
                while total - header_end < content_len {
                    let n = sock.read(&mut buf[total..]).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    total += n;
                }
                let req_line = header_owned.lines().next().unwrap_or("");
                let mut parts = req_line.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                let body = buf[header_end..header_end + content_len].to_vec();
                let _ = tx.send((method, path, body));
                break;
            }
        }
        let body = b"{\"ok\":true}";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.write_all(body).await;
        let _ = sock.shutdown().await;
    });
    (base, rx)
}

async fn capture(
    rx: tokio::sync::oneshot::Receiver<(String, String, Vec<u8>)>,
) -> (String, String, Vec<u8>) {
    tokio::time::timeout(std::time::Duration::from_secs(2), rx)
        .await
        .expect("listener closed before request arrived")
        .expect("no request received")
}

// ── T1: fuzzy_probability (Phase 10) ─────────────────────────────────

#[tokio::test]
async fn test_fuzzy_probability_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let event = serde_json::json!({"predicate_kind": "custom", "predicate_payload": {}});
    let distribution = serde_json::json!({"kind": "discrete", "outcomes": []});
    let _ = backend
        .fuzzy_probability("hamlet", event.clone(), distribution.clone(), Some("godel"))
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/hybrid/probability");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["narrative_id"], "hamlet");
    assert_eq!(parsed["tnorm"], "godel");
    assert_eq!(parsed["event"], event);
    assert_eq!(parsed["distribution"], distribution);
}

// ── T2: fuzzy_list_tnorms ────────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_list_tnorms_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.fuzzy_list_tnorms().await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    assert_eq!(path, "/fuzzy/tnorms");
}

// ── T3: fuzzy_list_aggregators ───────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_list_aggregators_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.fuzzy_list_aggregators().await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    assert_eq!(path, "/fuzzy/aggregators");
}

// ── T4: fuzzy_get_config ─────────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_get_config_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.fuzzy_get_config().await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    assert_eq!(path, "/fuzzy/config");
}

// ── T5: fuzzy_set_config ─────────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_set_config_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_set_config(Some("goguen"), Some("median"), None, false)
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "PUT");
    assert_eq!(path, "/fuzzy/config");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["tnorm"], "goguen");
    assert_eq!(parsed["aggregator"], "median");
    // reset omitted → not present or false
    assert!(parsed.get("reset").is_none());
}

#[tokio::test]
async fn test_fuzzy_set_config_reset_path() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.fuzzy_set_config(None, None, None, true).await;
    let (_, _, body) = capture(rx).await;
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["reset"], true);
}

// ── T6: fuzzy_create_measure ─────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_create_measure_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_create_measure("my-measure", 2, vec![0.0, 0.3, 0.5, 1.0])
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/measures");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["name"], "my-measure");
    assert_eq!(parsed["n"], 2);
    assert_eq!(parsed["values"], serde_json::json!([0.0, 0.3, 0.5, 1.0]));
}

// ── T7: fuzzy_list_measures ──────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_list_measures_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.fuzzy_list_measures().await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    assert_eq!(path, "/fuzzy/measures");
}

// ── T8: fuzzy_aggregate ──────────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_aggregate_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_aggregate(
            vec![0.1, 0.2, 0.3],
            "mean",
            None,
            None,
            None,
            None,
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/aggregate");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["aggregator"], "mean");
    assert_eq!(parsed["xs"], serde_json::json!([0.1, 0.2, 0.3]));
    // Optional fields omitted
    assert!(parsed.get("tnorm").is_none());
    assert!(parsed.get("owa_weights").is_none());
}

// ── T9: fuzzy_allen_gradation ────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_allen_gradation_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let a_id = uuid::Uuid::now_v7().to_string();
    let b_id = uuid::Uuid::now_v7().to_string();
    let _ = backend
        .fuzzy_allen_gradation("hamlet", &a_id, &b_id)
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/analysis/fuzzy-allen");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["narrative_id"], "hamlet");
    assert_eq!(parsed["a_id"], a_id);
    assert_eq!(parsed["b_id"], b_id);
}

// ── T10: fuzzy_quantify ──────────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_quantify_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_quantify(
            "hamlet",
            "most",
            Some("Actor"),
            Some("confidence>0.5"),
            Some("label-1"),
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/quantify");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["quantifier"], "most");
    assert_eq!(parsed["where"], "confidence>0.5");
    assert_eq!(parsed["entity_type"], "Actor");
    assert_eq!(parsed["label"], "label-1");
}

// ── T11: fuzzy_verify_syllogism ──────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_verify_syllogism_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_verify_syllogism(
            "hamlet",
            "ALL type:Actor IS type:Actor",
            "ALL * IS type:Actor",
            "ALL type:Actor IS *",
            Some(0.7),
            Some("godel"),
            Some("I"),
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/syllogism/verify");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["threshold"], 0.7);
    assert_eq!(parsed["tnorm"], "godel");
    assert_eq!(parsed["figure_hint"], "I");
}

// ── T12: fuzzy_build_lattice ─────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_build_lattice_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .fuzzy_build_lattice(
            "hamlet",
            Some("Actor"),
            Some(vec!["tag1".into(), "tag2".into()]),
            Some(2),
            Some("lukasiewicz"),
            false,
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/fca/lattice");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["narrative_id"], "hamlet");
    assert_eq!(parsed["entity_type"], "Actor");
    assert_eq!(parsed["threshold"], 2);
    assert_eq!(parsed["tnorm"], "lukasiewicz");
    assert_eq!(
        parsed["attribute_allowlist"],
        serde_json::json!(["tag1", "tag2"])
    );
    // large_context=false omitted
    assert!(parsed.get("large_context").is_none());
}

// ── T13: fuzzy_create_rule ───────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_create_rule_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let antecedent = serde_json::json!([
        {
            "variable_path": "entity.confidence",
            "membership": {"Triangular": {"a": 0.0, "b": 0.5, "c": 1.0}},
            "linguistic_term": "high"
        }
    ]);
    let consequent = serde_json::json!({
        "variable": "risk",
        "membership": {"Triangular": {"a": 0.0, "b": 0.5, "c": 1.0}},
        "linguistic_term": "medium"
    });
    let _ = backend
        .fuzzy_create_rule(
            "r1",
            "hamlet",
            antecedent.clone(),
            consequent.clone(),
            None,
            Some(true),
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/rules");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["name"], "r1");
    assert_eq!(parsed["narrative_id"], "hamlet");
    assert_eq!(parsed["antecedent"], antecedent);
    assert_eq!(parsed["consequent"], consequent);
    assert_eq!(parsed["enabled"], true);
}

// ── T14: fuzzy_evaluate_rules ────────────────────────────────────────

#[tokio::test]
async fn test_fuzzy_evaluate_rules_http_roundtrip() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let entity_id = uuid::Uuid::now_v7().to_string();
    let _ = backend
        .fuzzy_evaluate_rules(
            "hamlet",
            &entity_id,
            Some(vec!["rule-a".into(), "rule-b".into()]),
            Some(crate::fuzzy::aggregation::AggregatorKind::Mean),
        )
        .await;
    let (method, path, body) = capture(rx).await;
    assert_eq!(method, "POST");
    assert_eq!(path, "/fuzzy/rules/hamlet/evaluate");
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["entity_id"], entity_id);
    assert_eq!(
        parsed["rule_ids"],
        serde_json::json!(["rule-a", "rule-b"])
    );
    assert_eq!(parsed["firing_aggregator"], "Mean");
}

// ── Existing-tool extension: None path bit-identical + Some path
//    appends query string correctly (the Phase 11 backward-compat
//    acceptance gate) ────────────────────────────────────────────────

#[test]
fn test_append_fuzzy_query_none_is_bit_identical() {
    // Sprint-invariant guard: when both knobs are None/empty, the URL
    // must be byte-for-byte the pre-sprint path.
    assert_eq!(append_fuzzy_query("/narratives/x/stats", None, None), "/narratives/x/stats");
    assert_eq!(
        append_fuzzy_query("/narratives/x/stats", Some(""), Some("  ")),
        "/narratives/x/stats"
    );
}

#[test]
fn test_append_fuzzy_query_some_path_appends() {
    assert_eq!(
        append_fuzzy_query("/narratives/x/stats", Some("godel"), None),
        "/narratives/x/stats?tnorm=godel"
    );
    assert_eq!(
        append_fuzzy_query("/narratives/x/stats", None, Some("mean")),
        "/narratives/x/stats?aggregator=mean"
    );
    assert_eq!(
        append_fuzzy_query("/narratives/x/stats", Some("godel"), Some("mean")),
        "/narratives/x/stats?tnorm=godel&aggregator=mean"
    );
    // Pre-existing query string → use `&` separator.
    assert_eq!(
        append_fuzzy_query("/foo?a=1", Some("godel"), None),
        "/foo?a=1&tnorm=godel"
    );
}

#[tokio::test]
async fn test_get_narrative_stats_none_is_pre_sprint_identical() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend.get_narrative_stats("hamlet", None, None).await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    // Must be the pre-Phase-11 URL exactly — no trailing `?` or `&`.
    assert_eq!(path, "/narratives/hamlet/stats");
}

#[tokio::test]
async fn test_get_narrative_stats_some_path_appends_query_string() {
    let (base, rx) = one_shot_capture().await;
    let backend = HttpBackend::new(base);
    let _ = backend
        .get_narrative_stats("hamlet", Some("goguen"), Some("choquet"))
        .await;
    let (method, path, _) = capture(rx).await;
    assert_eq!(method, "GET");
    assert_eq!(path, "/narratives/hamlet/stats?tnorm=goguen&aggregator=choquet");
}
