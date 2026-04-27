//! Error mapping from TensaError to MCP error responses.

use rmcp::model::CallToolResult;
use rmcp::model::Content;

use crate::error::TensaError;

/// Convert a TensaError into an MCP CallToolResult with `is_error: true`.
///
/// We return tool-level errors (not protocol-level McpError) so the
/// MCP session stays alive and the client can see the error message.
pub fn error_result(err: TensaError) -> CallToolResult {
    CallToolResult::error(vec![Content::text(format!("Error: {}", err))])
}
