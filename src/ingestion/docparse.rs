//! Document parsing for PDF and DOCX files (Sprint P3.6 — F-IG2).
//!
//! Extracts plain text from PDF and DOCX documents for ingestion.
//! Feature-gated behind `docparse` to avoid mandatory heavy dependencies.

use crate::error::{Result, TensaError};

/// Supported document types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    Pdf,
    Docx,
}

/// Default maximum document size (10 MB).
pub const DEFAULT_MAX_DOC_SIZE: usize = 10 * 1024 * 1024;

/// Maximum document size (configurable, defaults to 10 MB).
pub static MAX_DOC_SIZE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_MAX_DOC_SIZE);

/// Set the maximum document size for parsing.
pub fn set_max_doc_size(size: usize) {
    MAX_DOC_SIZE.store(size, std::sync::atomic::Ordering::Relaxed);
}

/// Get the current maximum document size.
pub fn max_doc_size() -> usize {
    MAX_DOC_SIZE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Detect document type from magic bytes.
///
/// - PDF: starts with `%PDF`
/// - DOCX: starts with `PK` (ZIP archive)
pub fn detect_doc_type(bytes: &[u8]) -> Option<DocType> {
    if bytes.len() < 4 {
        return None;
    }
    if &bytes[..4] == b"%PDF" {
        Some(DocType::Pdf)
    } else if &bytes[..2] == b"PK" {
        Some(DocType::Docx)
    } else {
        None
    }
}

/// Validate that document size is within limits.
pub fn validate_doc_size(bytes: &[u8]) -> Result<()> {
    let limit = max_doc_size();
    if bytes.len() > limit {
        Err(TensaError::DocParseError(format!(
            "Document too large: {} bytes (max {})",
            bytes.len(),
            limit
        )))
    } else {
        Ok(())
    }
}

/// Extract plain text from a PDF byte buffer.
#[cfg(feature = "docparse")]
pub fn extract_pdf_text(bytes: &[u8]) -> Result<String> {
    validate_doc_size(bytes)?;

    let doc = lopdf::Document::load_mem(bytes)
        .map_err(|e| TensaError::DocParseError(format!("PDF load error: {e}")))?;

    let mut text = String::new();
    let pages = doc.get_pages();

    for (&page_num, _) in &pages {
        if let Ok(page_text) = doc.extract_text(&[page_num]) {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&page_text);
        }
    }

    if text.trim().is_empty() {
        return Err(TensaError::DocParseError(
            "No text could be extracted from PDF".into(),
        ));
    }

    Ok(text)
}

/// Extract plain text from a DOCX byte buffer.
#[cfg(feature = "docparse")]
pub fn extract_docx_text(bytes: &[u8]) -> Result<String> {
    validate_doc_size(bytes)?;

    let docx = docx_rs::read_docx(bytes)
        .map_err(|e| TensaError::DocParseError(format!("DOCX read error: {e}")))?;

    let mut text = String::new();
    for child in docx.document.children {
        if let docx_rs::DocumentChild::Paragraph(para) = child {
            let mut para_text = String::new();
            for child in &para.children {
                if let docx_rs::ParagraphChild::Run(run) = child {
                    for child in &run.children {
                        if let docx_rs::RunChild::Text(t) = child {
                            para_text.push_str(&t.text);
                        }
                    }
                }
            }
            if !para_text.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&para_text);
            }
        }
    }

    if text.trim().is_empty() {
        return Err(TensaError::DocParseError(
            "No text could be extracted from DOCX".into(),
        ));
    }

    Ok(text)
}

/// Extract text from document bytes, auto-detecting type.
#[cfg(feature = "docparse")]
pub fn extract_text(bytes: &[u8]) -> Result<String> {
    match detect_doc_type(bytes) {
        Some(DocType::Pdf) => extract_pdf_text(bytes),
        Some(DocType::Docx) => extract_docx_text(bytes),
        None => Err(TensaError::DocParseError(
            "Unknown document format (expected PDF or DOCX)".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_pdf_magic_bytes() {
        let pdf_bytes = b"%PDF-1.4 some content here";
        assert_eq!(detect_doc_type(pdf_bytes), Some(DocType::Pdf));
    }

    #[test]
    fn test_detect_docx_magic_bytes() {
        let docx_bytes = b"PK\x03\x04some zip content";
        assert_eq!(detect_doc_type(docx_bytes), Some(DocType::Docx));
    }

    #[test]
    fn test_detect_unknown_format() {
        let unknown = b"Hello, world!";
        assert_eq!(detect_doc_type(unknown), None);

        let too_short = b"AB";
        assert_eq!(detect_doc_type(too_short), None);

        let empty: &[u8] = b"";
        assert_eq!(detect_doc_type(empty), None);
    }

    #[test]
    fn test_validate_doc_size_ok() {
        let small = vec![0u8; 1000];
        assert!(validate_doc_size(&small).is_ok());
    }

    #[test]
    fn test_validate_doc_size_too_large() {
        let big = vec![0u8; max_doc_size() + 1];
        let result = validate_doc_size(&big);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too large"));
    }

    // Feature-gated tests for actual PDF/DOCX extraction
    #[cfg(feature = "docparse")]
    mod docparse_tests {
        use super::*;

        #[test]
        fn test_extract_pdf_text() {
            // lopdf can only extract text from real PDFs; test error handling for minimal/invalid
            let fake_pdf = b"%PDF-1.4 not a real pdf";
            let result = extract_pdf_text(fake_pdf);
            // Should error because it's not a valid PDF structure
            assert!(result.is_err());
        }

        #[test]
        fn test_extract_docx_text() {
            // Create a minimal DOCX in memory using docx-rs
            let docx = docx_rs::Docx::new().add_paragraph(
                docx_rs::Paragraph::new().add_run(docx_rs::Run::new().add_text("Hello from DOCX")),
            );
            let mut buf = Vec::new();
            docx.build()
                .pack(&mut std::io::Cursor::new(&mut buf))
                .unwrap();

            let text = extract_docx_text(&buf).unwrap();
            assert!(text.contains("Hello from DOCX"));
        }

        #[test]
        fn test_extract_text_auto_detect() {
            // Test with DOCX (since we can create one easily)
            let docx = docx_rs::Docx::new().add_paragraph(
                docx_rs::Paragraph::new().add_run(docx_rs::Run::new().add_text("Auto-detected")),
            );
            let mut buf = Vec::new();
            docx.build()
                .pack(&mut std::io::Cursor::new(&mut buf))
                .unwrap();

            let text = extract_text(&buf).unwrap();
            assert!(text.contains("Auto-detected"));

            // Test unknown format
            let result = extract_text(b"unknown format");
            assert!(result.is_err());
        }
    }
}
