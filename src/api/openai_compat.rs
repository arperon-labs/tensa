//! OpenAI-compatible API endpoints.
//!
//! Provides `POST /v1/chat/completions` and `GET /v1/models` so that
//! tools expecting the OpenAI chat API can talk to TENSA's RAG engine.

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::{AppState, LlmConfig};

/// OpenAI ChatCompletion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use (ignored — always uses configured TENSA model).
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Whether to stream the response. Currently ignored (always non-streaming).
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature (forwarded to LLM if applicable).
    pub temperature: Option<f64>,
    /// Maximum tokens in the response.
    pub max_tokens: Option<usize>,
}

/// A single chat message in the OpenAI format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant".
    pub role: String,
    /// Message content.
    pub content: String,
}

/// OpenAI ChatCompletion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique response ID.
    pub id: String,
    /// Object type, always "chat.completion".
    pub object: String,
    /// Unix timestamp of creation.
    pub created: i64,
    /// Model name.
    pub model: String,
    /// Response choices (always one for non-streaming).
    pub choices: Vec<Choice>,
    /// Token usage estimates.
    pub usage: Usage,
}

/// A single completion choice.
#[derive(Debug, Serialize)]
pub struct Choice {
    /// Choice index (always 0).
    pub index: usize,
    /// The assistant message.
    pub message: ResponseMessage,
    /// Reason the model stopped generating.
    pub finish_reason: String,
}

/// Output message in a choice.
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    /// Always "assistant".
    pub role: String,
    /// Generated content.
    pub content: String,
}

/// Token usage information.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Estimated prompt tokens.
    pub prompt_tokens: usize,
    /// Estimated completion tokens.
    pub completion_tokens: usize,
    /// Total tokens.
    pub total_tokens: usize,
}

/// Response for GET /v1/models.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Always "list".
    pub object: String,
    /// Available models.
    pub data: Vec<ModelInfo>,
}

/// A single model entry.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Always "model".
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Owner name.
    pub owned_by: String,
}

/// Extract the last user message from a messages list.
fn extract_question(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
}

/// Extract the system message (if any) from messages.
fn extract_system_message(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
}

/// Return the model name from the current LLM config.
fn model_name_from_config(config: &LlmConfig) -> String {
    match config {
        LlmConfig::Anthropic { model, .. } => model.clone(),
        LlmConfig::OpenRouter { model, .. } => model.clone(),
        LlmConfig::Local { model, .. } => model.clone(),
        LlmConfig::Gemini { model, .. } => model.clone(),
        LlmConfig::Bedrock { model_id, .. } => model_id.clone(),
        LlmConfig::None => "tensa-rag".to_string(),
    }
}

/// Check if a string looks like a TensaQL query.
fn looks_like_tensaql(text: &str) -> bool {
    let trimmed = text.trim_start().to_uppercase();
    trimmed.starts_with("MATCH ")
        || trimmed.starts_with("INFER ")
        || trimmed.starts_with("ASK ")
        || trimmed.starts_with("EXPLAIN ")
        || trimmed.starts_with("DISCOVER ")
        || trimmed.starts_with("EXPORT ")
}

/// POST /v1/chat/completions — OpenAI-compatible chat completion.
///
/// Extracts the last user message, runs it through the RAG pipeline,
/// and returns the answer in OpenAI's response format.
/// Streaming is not yet supported; `stream: true` is silently ignored.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let question = match extract_question(&body.messages) {
        Some(q) if !q.is_empty() => q.to_string(),
        _ => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "No user message found", "type": "invalid_request_error"}})),
            ).into_response()
        }
    };

    let _system_ctx = extract_system_message(&body.messages);

    // If it looks like TensaQL, execute as query
    if looks_like_tensaql(&question) {
        let it_guard = state
            .interval_tree
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let inf_guard = state.inference_extractor.read().unwrap();
        let ext_guard = state.extractor.read().unwrap();
        let vi_guard = state.vector_index.as_ref().and_then(|vi| vi.read().ok());
        let vi_ref = vi_guard.as_deref();
        let emb_guard = state.embedder.read().unwrap();
        let emb_ref = emb_guard.as_deref();
        let ext_ref = inf_guard.as_deref().or(ext_guard.as_deref());

        match crate::query::parser::parse_query(&question) {
            Ok(query) => match crate::query::planner::plan_query(&query) {
                Ok(plan) => {
                    match crate::query::executor::execute_full_with_extractor(
                        &plan,
                        &state.hypergraph,
                        &it_guard,
                        false,
                        vi_ref,
                        emb_ref,
                        ext_ref,
                    ) {
                        Ok(rows) => {
                            let content = serde_json::to_string_pretty(&rows)
                                .unwrap_or_else(|_| "[]".to_string());
                            return build_completion_response(&state, &question, &content);
                        }
                        Err(e) => {
                            let content = format!("Query execution error: {}", e);
                            return build_completion_response(&state, &question, &content);
                        }
                    }
                }
                Err(e) => {
                    let content = format!("Query planning error: {}", e);
                    return build_completion_response(&state, &question, &content);
                }
            },
            Err(e) => {
                let content = format!("Parse error: {}", e);
                return build_completion_response(&state, &question, &content);
            }
        }
    }

    // Otherwise, use RAG pipeline
    let mode = crate::query::rag_config::RetrievalMode::Hybrid;
    let budget = state.rag_config.read().unwrap().budget.clone();
    let inf_guard = state.inference_extractor.read().unwrap();
    let ext_guard = state.extractor.read().unwrap();
    let extractor = match inf_guard.as_deref().or(ext_guard.as_deref()) {
        Some(e) => e,
        None => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {"message": "No LLM configured", "type": "server_error"}})),
            ).into_response()
        }
    };
    let vi_guard = state.vector_index.as_ref().and_then(|vi| vi.read().ok());
    let vi_ref = vi_guard.as_deref();
    let emb_guard = state.embedder.read().unwrap();
    let emb_ref = emb_guard.as_deref();
    let rr_ref = state.reranker.as_deref();

    match crate::query::rag::execute_ask(
        &question,
        None,
        &mode,
        &budget,
        &state.hypergraph,
        vi_ref,
        emb_ref,
        extractor,
        rr_ref,
        None,
        false,
        None,
    ) {
        Ok(answer) => build_completion_response(&state, &question, &answer.answer),
        Err(e) => error_response(e).into_response(),
    }
}

/// Build a ChatCompletionResponse as an axum response.
fn build_completion_response(
    state: &AppState,
    prompt: &str,
    content: &str,
) -> axum::response::Response {
    let config = state.llm_config.read().unwrap();
    let model = model_name_from_config(&config);
    let prompt_tokens = crate::query::token_budget::TokenBudget::estimate_tokens(prompt);
    let completion_tokens = crate::query::token_budget::TokenBudget::estimate_tokens(content);

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::now_v7()),
        object: "chat.completion".to_string(),
        created: Utc::now().timestamp(),
        model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: content.to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    json_ok(&response)
}

/// GET /v1/models — list available models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state.llm_config.read().unwrap();
    let now = Utc::now().timestamp();

    let mut models = vec![ModelInfo {
        id: "tensa-rag".to_string(),
        object: "model".to_string(),
        created: now,
        owned_by: "tensa".to_string(),
    }];

    let model_name = model_name_from_config(&config);
    if model_name != "tensa-rag" {
        models.push(ModelInfo {
            id: model_name,
            object: "model".to_string(),
            created: now,
            owned_by: "tensa".to_string(),
        });
    }

    json_ok(&ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_response_serializes() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "tensa-rag".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: "Hello world".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_parse_chat_request() {
        let json_str = r#"{
            "model": "tensa-rag",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Who is Alice?"}
            ],
            "stream": false,
            "temperature": 0.7
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.model.as_deref(), Some("tensa-rag"));
        assert!(!req.stream);
        assert!((req.temperature.unwrap() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_question_from_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "First question".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Answer".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Follow-up question".to_string(),
            },
        ];
        assert_eq!(extract_question(&messages), Some("Follow-up question"));
    }

    #[test]
    fn test_extract_question_no_user() {
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "Be helpful.".to_string(),
        }];
        assert_eq!(extract_question(&messages), None);
    }

    #[test]
    fn test_looks_like_tensaql() {
        assert!(looks_like_tensaql("MATCH (e:Actor) RETURN e"));
        assert!(looks_like_tensaql("  INFER CENTRALITY(\"test\")"));
        assert!(looks_like_tensaql("ASK \"question\" OVER \"nar\""));
        assert!(!looks_like_tensaql("Who is Alice?"));
        assert!(!looks_like_tensaql("Tell me about the matching algorithm"));
    }

    #[test]
    fn test_models_response_serializes() {
        let resp = ModelsResponse {
            object: "list".to_string(),
            data: vec![ModelInfo {
                id: "tensa-rag".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "tensa".to_string(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "tensa-rag");
    }

    #[test]
    fn test_extract_system_message() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "System prompt".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Question".to_string(),
            },
        ];
        assert_eq!(extract_system_message(&messages), Some("System prompt"));
    }
}
