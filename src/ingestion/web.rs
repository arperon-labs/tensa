//! URL and RSS/Atom feed ingestion.
//!
//! Fetches web pages, strips HTML to extract text content,
//! and parses RSS/Atom feeds for batch ingestion.

use crate::error::{Result, TensaError};

/// A parsed RSS/Atom feed item.
#[derive(Debug, Clone)]
pub struct RssItem {
    pub title: String,
    pub link: Option<String>,
    pub content: String,
    pub published: Option<chrono::DateTime<chrono::Utc>>,
}

/// Fetch a URL and extract text content by stripping HTML tags.
pub async fn fetch_and_extract_text(url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .header("User-Agent", "TENSA/0.1 (narrative-ingestion)")
        .send()
        .await
        .map_err(|e| TensaError::LlmError(format!("Failed to fetch URL: {}", e)))?;

    if !resp.status().is_success() {
        return Err(TensaError::LlmError(format!(
            "HTTP {} fetching {}",
            resp.status(),
            url
        )));
    }

    let html = resp
        .text()
        .await
        .map_err(|e| TensaError::LlmError(format!("Failed to read response: {}", e)))?;

    Ok(strip_html(&html))
}

/// Strip HTML tags and extract readable text content.
#[cfg(feature = "web-ingest")]
pub fn strip_html(html: &str) -> String {
    use scraper::{Html, Selector};
    let document = Html::parse_document(html);

    // Try to find article/main content first
    let selectors = [
        "article",
        "main",
        "[role=main]",
        ".content",
        "#content",
        "body",
    ];

    for sel_str in selectors {
        if let Ok(selector) = Selector::parse(sel_str) {
            let elements: Vec<_> = document.select(&selector).collect();
            if !elements.is_empty() {
                let text: String = elements
                    .iter()
                    .flat_map(|el| el.text())
                    .collect::<Vec<_>>()
                    .join(" ");
                let cleaned = clean_whitespace(&text);
                if !cleaned.is_empty() {
                    return cleaned;
                }
            }
        }
    }

    // Fallback: extract all text from body
    let text: String = document.root_element().text().collect::<Vec<_>>().join(" ");
    clean_whitespace(&text)
}

/// Fallback strip_html when web-ingest feature is not enabled.
#[cfg(not(feature = "web-ingest"))]
pub fn strip_html(html: &str) -> String {
    use std::sync::OnceLock;
    static HTML_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = HTML_RE.get_or_init(|| regex::Regex::new(r"<[^>]+>").expect("valid regex"));
    let text = re.replace_all(html, " ");
    clean_whitespace(&text)
}

/// Fetch and parse an RSS/Atom feed, returning up to `max_items` entries.
#[cfg(feature = "web-ingest")]
pub async fn fetch_rss_items(feed_url: &str, max_items: usize) -> Result<Vec<RssItem>> {
    let client = reqwest::Client::new();
    let resp = client
        .get(feed_url)
        .header("User-Agent", "TENSA/0.1 (narrative-ingestion)")
        .send()
        .await
        .map_err(|e| TensaError::LlmError(format!("Failed to fetch feed: {}", e)))?;

    let body = resp
        .bytes()
        .await
        .map_err(|e| TensaError::LlmError(format!("Failed to read feed: {}", e)))?;

    let feed = feed_rs::parser::parse(&body[..])
        .map_err(|e| TensaError::LlmError(format!("Failed to parse feed: {}", e)))?;

    let items: Vec<RssItem> = feed
        .entries
        .into_iter()
        .take(max_items)
        .map(|entry| {
            let title = entry.title.map(|t| t.content).unwrap_or_default();
            let link = entry.links.first().map(|l| l.href.clone());
            let content = entry
                .content
                .and_then(|c| c.body)
                .or_else(|| entry.summary.map(|s| s.content))
                .unwrap_or_default();
            let published = entry
                .published
                .map(|dt| chrono::DateTime::<chrono::Utc>::from(dt));
            RssItem {
                title,
                link,
                content: strip_html(&content),
                published,
            }
        })
        .collect();

    Ok(items)
}

/// Collapse whitespace and trim.
fn clean_whitespace(text: &str) -> String {
    use std::sync::OnceLock;
    static WS_RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = WS_RE.get_or_init(|| regex::Regex::new(r"\s+").expect("valid regex"));
    re.replace_all(text.trim(), " ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_simple() {
        let html = "<p>Hello <b>world</b></p>";
        let text = strip_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<p>"));
    }

    #[test]
    fn test_strip_html_preserves_text() {
        let html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Content here"));
    }

    #[test]
    fn test_clean_whitespace() {
        assert_eq!(clean_whitespace("  hello   world  "), "hello world");
        assert_eq!(clean_whitespace("\n\thello\n\tworld\n"), "hello world");
    }
}
