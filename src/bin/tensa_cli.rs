//! TENSA CLI — command-line interface for querying and managing TENSA.
//!
//! Connects to a running TENSA API server (default: http://localhost:3000).
//!
//! # Usage
//! ```sh
//! tensa-cli query "MATCH (e:Actor) RETURN e LIMIT 5"
//! tensa-cli status
//! tensa-cli export my-narrative --format csv
//! tensa-cli job abc-123
//! tensa-cli ingest --file report.txt --narrative case-alpha
//! ```

use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(
    name = "tensa-cli",
    about = "TENSA CLI — narrative intelligence engine"
)]
struct Cli {
    /// Base URL of the TENSA API server.
    #[arg(long, default_value = "http://localhost:3000", global = true)]
    url: String,

    /// Output raw JSON (no formatting).
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a TensaQL query.
    Query {
        /// The TensaQL query string.
        query: String,
    },
    /// Check server health and status.
    Status,
    /// Export a narrative in the specified format.
    Export {
        /// Narrative ID to export.
        narrative_id: String,
        /// Export format (csv, graphml, json, manuscript, report).
        #[arg(long, default_value = "json")]
        format: String,
    },
    /// Get an inference job result.
    Job {
        /// Job ID.
        id: String,
    },
    /// Ingest text into TENSA.
    Ingest {
        /// Text to ingest (reads from stdin if not provided and --file not set).
        text: Option<String>,
        /// File to ingest.
        #[arg(long)]
        file: Option<String>,
        /// Narrative ID to associate with.
        #[arg(long)]
        narrative: Option<String>,
        /// Source name for provenance.
        #[arg(long)]
        source_name: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    let client = reqwest::blocking::Client::new();
    let base = cli.url.trim_end_matches('/');

    let result = match cli.command {
        Commands::Status => {
            let resp = client.get(format!("{}/health", base)).send();
            handle_response(resp, cli.json)
        }
        Commands::Query { query } => {
            let resp = client
                .post(format!("{}/query", base))
                .json(&serde_json::json!({"query": query}))
                .send();
            handle_response(resp, cli.json)
        }
        Commands::Export {
            narrative_id,
            format,
        } => {
            let resp = client
                .get(format!(
                    "{}/narratives/{}/export?format={}",
                    base, narrative_id, format
                ))
                .send();
            handle_response(resp, cli.json)
        }
        Commands::Job { id } => {
            let resp = client.get(format!("{}/jobs/{}/result", base, id)).send();
            handle_response(resp, cli.json)
        }
        Commands::Ingest {
            text,
            file,
            narrative,
            source_name,
        } => {
            let content = if let Some(path) = file {
                match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error reading file '{}': {}", path, e);
                        process::exit(1);
                    }
                }
            } else if let Some(t) = text {
                t
            } else {
                // Read from stdin
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin()
                    .read_to_string(&mut buf)
                    .unwrap_or_default();
                buf
            };

            let mut body = serde_json::json!({"text": content});
            if let Some(n) = narrative {
                body["narrative_id"] = serde_json::json!(n);
            }
            if let Some(s) = source_name {
                body["source_name"] = serde_json::json!(s);
            }

            let resp = client.post(format!("{}/ingest", base)).json(&body).send();
            handle_response(resp, cli.json)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn handle_response(
    resp: std::result::Result<reqwest::blocking::Response, reqwest::Error>,
    raw_json: bool,
) -> std::result::Result<(), String> {
    match resp {
        Ok(response) => {
            let status = response.status();
            let body = response.text().unwrap_or_else(|_| "<no body>".to_string());

            if !status.is_success() {
                eprintln!("HTTP {}: {}", status.as_u16(), body);
                return Err(format!("Request failed with status {}", status.as_u16()));
            }

            if raw_json {
                println!("{}", body);
            } else {
                // Try to pretty-print JSON
                match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(val) => {
                        println!("{}", serde_json::to_string_pretty(&val).unwrap_or(body));
                    }
                    Err(_) => println!("{}", body),
                }
            }
            Ok(())
        }
        Err(e) => Err(format!("Connection failed: {}", e)),
    }
}
