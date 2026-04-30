#[cfg(feature = "server")]
pub mod adaptation_routes;
#[cfg(feature = "server")]
pub mod analysis;
#[cfg(feature = "server")]
pub mod analysis_status_routes;
#[cfg(feature = "server")]
pub mod analytics_readback_routes;
#[cfg(feature = "server")]
pub mod annotation_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod archetype_routes;
#[cfg(feature = "server")]
pub mod architecture_routes;
#[cfg(feature = "server")]
pub mod argumentation_gradual;
#[cfg(feature = "server")]
pub mod bulk_routes;
#[cfg(feature = "server")]
pub mod chunk_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod cib_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod claims_routes;
#[cfg(feature = "server")]
pub mod collection_routes;
#[cfg(feature = "server")]
pub mod compile_routes;
#[cfg(feature = "server")]
pub mod continuity_routes;
#[cfg(feature = "server")]
pub mod cost_ledger_routes;
#[cfg(feature = "server")]
pub mod debug_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod disinfo_routes;
#[cfg(feature = "server")]
pub mod editing_routes;
#[cfg(feature = "server")]
pub mod fuzzy;
#[cfg(feature = "server")]
pub mod generation_routes;
#[cfg(feature = "server")]
pub mod import_routes;
#[cfg(feature = "server")]
pub mod inference;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod monitor_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod multilingual_routes;
#[cfg(feature = "server")]
pub mod openai_compat;
#[cfg(feature = "server")]
pub mod openapi;
#[cfg(feature = "server")]
pub mod plan_routes;
#[cfg(feature = "server")]
pub mod project_routes;
#[cfg(feature = "server")]
pub mod research_routes;
#[cfg(feature = "server")]
pub mod revision_routes;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod scheduler_routes;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub mod settings_routes;
#[cfg(feature = "server")]
pub mod source_routes;
#[cfg(all(feature = "server", feature = "disinfo"))]
pub mod spread_routes;
#[cfg(feature = "server")]
pub mod storywriting_routes;
#[cfg(feature = "server")]
pub mod style_routes;
#[cfg(feature = "server")]
pub mod synth;
#[cfg(feature = "server")]
pub mod template_routes;
#[cfg(feature = "server")]
pub mod temporal_ordhorn;
#[cfg(all(feature = "server", feature = "adversarial"))]
pub mod wargame_routes;
#[cfg(feature = "server")]
pub mod workshop_routes;
#[cfg(feature = "server")]
pub mod workspace_routes;
