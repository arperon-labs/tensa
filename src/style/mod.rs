//! Style embedding, encoder infrastructure, and LoRA management (Sprint D9.5).
//!
//! Provides dense voice embeddings from author corpora, style blending,
//! and LoRA adapter training/merging infrastructure for voice-faithful
//! text generation.
//!
//! Feature-gated behind `generation`.

pub mod embedding;
pub mod encoder;
pub mod lora;
