//! TENSA MCP server binary entry point.
//!
//! Reads configuration from environment variables:
//! - `TENSA_MCP_MODE`: "embedded" (default) or "http"
//! - `TENSA_DATA_DIR`: Data directory for embedded mode (default: "tensa_server_data")
//! - `TENSA_API_URL`: REST API URL for http mode (default: "http://localhost:3000")
//!
//! Logging goes to stderr since stdout is the MCP JSON-RPC transport.

use std::sync::Arc;

use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // CRITICAL: logging to stderr — stdout is MCP transport
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tensa=info".parse().unwrap()),
        )
        .init();

    tracing::info!("TENSA MCP server v{}", env!("CARGO_PKG_VERSION"));

    let mode = std::env::var("TENSA_MCP_MODE").unwrap_or_else(|_| "embedded".to_string());

    match mode.as_str() {
        "embedded" => {
            let data_dir =
                std::env::var("TENSA_DATA_DIR").unwrap_or_else(|_| "tensa_server_data".to_string());

            let store: Arc<dyn tensa::KVStore> = {
                #[cfg(feature = "rocksdb")]
                {
                    tracing::info!("Opening RocksDB store at {}", data_dir);
                    Arc::new(tensa::store::rocks::RocksDBStore::open(&data_dir)?)
                }
                #[cfg(not(feature = "rocksdb"))]
                {
                    tracing::warn!("RocksDB not available, using in-memory store");
                    let _ = data_dir; // suppress unused warning
                    Arc::new(tensa::store::memory::MemoryStore::new())
                }
            };

            let backend = Arc::new(tensa::mcp::embedded::EmbeddedBackend::from_env(store));
            let server = tensa::mcp::server::TensaMcp::new(backend);

            tracing::info!("Starting MCP server in embedded mode (stdio transport)");
            let service = server.serve(rmcp::transport::io::stdio()).await?;
            service.waiting().await?;
        }
        "http" => {
            let api_url = std::env::var("TENSA_API_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string());

            tracing::info!("Starting MCP server in HTTP mode (API: {})", api_url);

            let backend = Arc::new(tensa::mcp::http::HttpBackend::new(api_url));
            let server = tensa::mcp::server::TensaMcp::new(backend);

            let service = server.serve(rmcp::transport::io::stdio()).await?;
            service.waiting().await?;
        }
        other => {
            anyhow::bail!(
                "Unknown TENSA_MCP_MODE: '{}'. Use 'embedded' or 'http'.",
                other
            );
        }
    }

    Ok(())
}
