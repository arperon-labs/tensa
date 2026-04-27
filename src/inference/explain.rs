//! LLM-powered explanation generation for inference results.
//!
//! After an inference engine produces a result, an `ExplainProvider`
//! can generate a natural-language explanation of the findings.

use crate::error::Result;
use crate::types::InferenceResult;

/// Trait for generating natural-language explanations of inference results.
///
/// Implementations call an LLM with the inference result context and
/// return a human-readable explanation string.
pub trait ExplainProvider: Send + Sync {
    /// Generate a natural-language explanation for an inference result.
    fn explain(&self, result: &InferenceResult) -> Result<String>;
}

/// Build a prompt for the LLM to explain an inference result.
pub fn build_explain_prompt(result: &InferenceResult) -> String {
    let result_json = serde_json::to_string_pretty(&result.result).unwrap_or_default();
    format!(
        "You are an analyst explaining inference results from a narrative intelligence system.\n\
         \n\
         Job type: {:?}\n\
         Target entity/situation ID: {}\n\
         Confidence: {:.2}\n\
         \n\
         Result data:\n\
         {}\n\
         \n\
         Explain this result in 2-3 concise sentences. Focus on what it means narratively \
         and what actionable insights it provides. Be specific about the entities and \
         relationships involved.",
        result.job_type, result.target_id, result.confidence, result_json
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    struct MockExplainProvider;

    impl ExplainProvider for MockExplainProvider {
        fn explain(&self, result: &InferenceResult) -> Result<String> {
            Ok(format!(
                "Mock explanation for job {} with confidence {:.2}",
                result.job_id, result.confidence
            ))
        }
    }

    fn test_result() -> InferenceResult {
        InferenceResult {
            job_id: "test-001".into(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: Uuid::now_v7(),
            result: serde_json::json!({"links": [{"from": "a", "to": "b", "strength": 0.9}]}),
            confidence: 0.85,
            explanation: None,
            status: JobStatus::Completed,
            created_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }
    }

    #[test]
    fn test_explain_provider_returns_string() {
        let provider = MockExplainProvider;
        let result = test_result();
        let explanation = provider.explain(&result).unwrap();
        assert!(explanation.contains("test-001"));
        assert!(explanation.contains("0.85"));
    }

    #[test]
    fn test_build_explain_prompt_includes_context() {
        let result = test_result();
        let prompt = build_explain_prompt(&result);
        assert!(prompt.contains("CausalDiscovery"));
        assert!(prompt.contains("0.85"));
        assert!(prompt.contains("strength"));
    }

    #[test]
    fn test_inference_result_with_explanation_roundtrip() {
        let mut result = test_result();
        result.explanation = Some("The causal chain shows A directly causes B.".into());
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: InferenceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.explanation.unwrap(),
            "The causal chain shows A directly causes B."
        );
    }

    #[test]
    fn test_inference_result_without_explanation_roundtrip() {
        let result = test_result();
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: InferenceResult = serde_json::from_str(&json).unwrap();
        assert!(deserialized.explanation.is_none());
    }

    #[test]
    fn test_backward_compat_no_explanation_field() {
        let json = serde_json::json!({
            "job_id": "old-001",
            "job_type": "CausalDiscovery",
            "target_id": Uuid::now_v7(),
            "result": {},
            "confidence": 0.7,
            "status": "Completed",
            "created_at": "2025-01-01T00:00:00Z",
            "completed_at": "2025-01-01T00:00:00Z"
        });
        let result: InferenceResult = serde_json::from_value(json).unwrap();
        assert!(result.explanation.is_none());
    }
}
