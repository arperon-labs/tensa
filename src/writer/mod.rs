//! Writer-facing features that sit on top of the hypergraph.
//!
//! Writer-tool gap sprints live here:
//! - [`scene`] — W7 scene-schema helpers (parent-cycle validation, manuscript order, word count).
//! - [`reorder`] — W8 batch binder reorder helper.

pub mod annotation;
pub mod cited_generation;
pub mod collection;
pub mod factcheck;
pub mod reorder;
pub mod research;
pub mod scene;
