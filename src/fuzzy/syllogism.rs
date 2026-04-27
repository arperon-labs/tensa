//! Graded fuzzy syllogisms — Peterson's square of opposition extended
//! with intermediate quantifiers.
//!
//! **Prototype status.** Phase 7 ships a working verifier over the
//! 5 Peterson canonical figures under Gödel semantics (configurable
//! t-norm combination). Formal soundness / completeness vs the Ł-BL*
//! fuzzy logic is **deferred to a future Phase 7.5** when Novák's Ł-BL*
//! library becomes trivially linkable. Callers should treat
//! [`GradedValidity.degree`] as a graded confidence score, NOT a
//! formal proof obligation.
//!
//! ## Supported figures
//!
//! | Figure | Major | Minor | Conclusion | Validity |
//! |--------|-------|-------|------------|----------|
//! | I      | All S–M  | All M–P  | All S–P  | classical-valid |
//! | I*     | Most S–M | All M–P  | Most S–P | graded-valid |
//! | II     | All S–M  | Most M–P | Most/??–P | **INVALID in Peterson** |
//! | III    | Almost-all S–M | All M–P | Almost-all S–P | graded-valid |
//! | IV     | Most M–S | Most M–P | Many S–P | graded-valid |
//!
//! Figures that don't match any of the 5 canonical triples are reported
//! as `figure = "non-canonical"` and `valid = false` regardless of
//! degree — consistent with Peterson's taxonomy.
//!
//! ## Evaluation semantics
//!
//! Each premise + conclusion is a triple `(Quantifier, subject_pred_id,
//! object_pred_id)`. Predicate ids resolve to indicator functions
//! `μ: Entity → [0,1]` via a pluggable [`PredicateResolver`]. The
//! verifier evaluates each of the three quantified statements against
//! the narrative using [`crate::fuzzy::quantifier::evaluate_over_entities`]
//! to get three degrees `d_major, d_minor, d_conclusion ∈ [0,1]`; the
//! graded validity is
//! ```text
//!     degree = T(d_major, T(d_minor, d_conclusion))
//! ```
//! under the configured t-norm (default Gödel → `min`). A syllogism is
//! considered `valid` iff `degree >= threshold AND figure != "non-canonical"
//! AND figure != "II"` (Peterson-invalid figure).
//!
//! ## KV persistence
//!
//! Proofs persist at `fz/syllog/{narrative_id}/{proof_id_v7_BE_16}`
//! via [`save_syllogism_proof`] / [`load_syllogism_proof`] /
//! [`delete_syllogism_proof`] / [`list_syllogism_proofs_for_narrative`].
//!
//! Cites: [murinovanovak2013syllogisms] [murinovanovak2014peterson]
//!        [novak2008quantifiers].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::quantifier::{self, Quantifier};
use crate::fuzzy::tnorm::{tnorm_for, TNormKind};
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::{Entity, EntityType};

/// A single premise or conclusion inside a [`Syllogism`]. Shape is
/// `<quantifier> <subject_predicate_id> IS <object_predicate_id>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyllogismStatement {
    /// One of the four Phase 6 built-in quantifiers, plus the classical
    /// `All` treated here as a ramp that saturates at `r = 1.0`.
    pub quantifier: Quantifier,
    /// Identifier of the subject-side predicate (resolves to an
    /// [`Entity`]-valued indicator via [`PredicateResolver`]).
    pub subject_pred_id: String,
    /// Identifier of the object-side predicate.
    pub object_pred_id: String,
}

/// A syllogism = major premise + minor premise + conclusion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Syllogism {
    pub major: SyllogismStatement,
    pub minor: SyllogismStatement,
    pub conclusion: SyllogismStatement,
    /// Optional caller-supplied figure hint. When `None` the verifier
    /// auto-detects via [`classify_figure`]. Valid values: `"I"`, `"I*"`,
    /// `"II"`, `"III"`, `"IV"`, `"non-canonical"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub figure_hint: Option<String>,
}

/// Result of [`verify`]: graded validity degree, classified figure,
/// Peterson-taxonomy validity flag, and the threshold used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradedValidity {
    /// Combined graded truth value in `[0,1]` under the configured t-norm.
    pub degree: f64,
    /// Peterson figure label: `"I"`, `"I*"`, `"II"`, `"III"`, `"IV"`,
    /// or `"non-canonical"`.
    pub figure: String,
    /// `true` iff `degree >= threshold AND figure is Peterson-valid`.
    /// Figure II is Peterson-invalid by taxonomy; non-canonical figures
    /// likewise receive `valid = false` regardless of degree.
    pub valid: bool,
    /// Threshold that was compared against `degree`.
    pub threshold: f64,
}

/// Pluggable resolver: `predicate_id → (Entity → [0,1] indicator)`.
///
/// Default resolver is [`TypePredicateResolver`], which interprets ids
/// of the form `"type:Actor"` / `"type:Location"` / etc. as EntityType
/// filters. Custom resolvers (e.g. TextFilterResolver, PropPathResolver)
/// plug in via this trait so test fixtures and the writer workshop can
/// wire domain-specific predicates without touching the core verifier.
pub trait PredicateResolver: Send + Sync {
    /// Resolve a predicate id into a boxed indicator function. Returns
    /// `Err(TensaError::InvalidInput)` on unknown ids so the verifier
    /// can surface a clean 400 instead of silently returning `μ = 0`.
    fn resolve(&self, predicate_id: &str)
        -> Result<Box<dyn Fn(&Entity) -> f64 + Send + Sync>>;
}

/// Default resolver: interprets `"type:<EntityType>"` ids. Any other id
/// (and plain `"entity"` as a catch-all "everything counts") is
/// supported; unknown ids return an [`TensaError::InvalidInput`].
#[derive(Debug, Default, Clone, Copy)]
pub struct TypePredicateResolver;

impl PredicateResolver for TypePredicateResolver {
    fn resolve(
        &self,
        predicate_id: &str,
    ) -> Result<Box<dyn Fn(&Entity) -> f64 + Send + Sync>> {
        let trimmed = predicate_id.trim();
        if trimmed.eq_ignore_ascii_case("entity") || trimmed == "*" {
            return Ok(Box::new(|_e: &Entity| 1.0));
        }
        if let Some(rest) = trimmed.strip_prefix("type:") {
            let et: EntityType =
                std::str::FromStr::from_str(rest.trim()).map_err(|_| {
                    TensaError::InvalidInput(format!(
                        "syllogism predicate '{predicate_id}' has unknown EntityType \
                         '{rest}'; expected Actor/Location/Artifact/Concept/Organization",
                    ))
                })?;
            return Ok(Box::new(move |e: &Entity| -> f64 {
                if e.entity_type == et {
                    1.0
                } else {
                    0.0
                }
            }));
        }
        Err(TensaError::InvalidInput(format!(
            "unknown predicate id '{predicate_id}'; default resolver supports \
             'type:<EntityType>' or 'entity'/'*'. Plug a custom PredicateResolver \
             for domain-specific predicates.",
        )))
    }
}

/// Classify a syllogism into one of the 5 Peterson canonical figures
/// based on the triple of quantifiers. Returns `"non-canonical"` when
/// no match is found.
///
/// The subject/object-term role structure is not checked here — the
/// caller is responsible for ordering the three statements so that the
/// standard syllogism structure (major→M→P, minor→S→M, conclusion→S→P
/// for Figures I/I*/II/III; major→M→S, minor→M→P, conclusion→S→P for
/// Figure IV) is followed. [`verify`] auto-detects by quantifier
/// triple, which is sufficient for the 5 canonical figures.
pub fn classify_figure(s: &Syllogism) -> &'static str {
    // Note: the classifier treats `AlmostAll` and the classical "All"
    // uniformly (both map to the AlmostAll ramp at parse time), so
    // Figures I and III share a quantifier triple. Callers that know
    // they mean III set `figure_hint = Some("III")` on [`Syllogism`];
    // the verifier honours the hint over this classifier.
    use Quantifier::*;
    match (s.major.quantifier, s.minor.quantifier, s.conclusion.quantifier) {
        (AlmostAll, AlmostAll, AlmostAll) => "I",
        (Most, AlmostAll, Most) => "I*",
        // Figure II is Peterson-invalid regardless of the conclusion.
        (AlmostAll, Most, _) => "II",
        (Most, Most, Many) => "IV",
        _ => "non-canonical",
    }
}

/// Verify a syllogism against a narrative. Returns graded validity.
///
/// * `hg` — hypergraph to query entities from.
/// * `narrative_id` — narrative scope for the three quantifier evaluations.
/// * `syllogism` — the (major, minor, conclusion) triple.
/// * `tnorm` — how to combine the three degrees; Gödel (min) is the
///   default choice per the Phase 7 spec.
/// * `threshold` — degree at or above which the syllogism is considered
///   `valid`. `0.5` is the standard Peterson threshold.
/// * `resolver` — how to resolve predicate ids into indicator functions.
///
/// The figure is auto-detected via [`classify_figure`]; a user-supplied
/// `figure_hint` (when present on the `Syllogism` struct) overrides the
/// classifier. Figure "II" and "non-canonical" always yield `valid =
/// false` regardless of degree (Peterson taxonomy).
pub fn verify(
    hg: &Hypergraph,
    narrative_id: &str,
    syllogism: &Syllogism,
    tnorm: TNormKind,
    threshold: f64,
    resolver: &dyn PredicateResolver,
) -> Result<GradedValidity> {
    if narrative_id.trim().is_empty() {
        return Err(TensaError::InvalidInput(
            "syllogism narrative_id is empty".into(),
        ));
    }
    let threshold = threshold.clamp(0.0, 1.0);

    // Fetch narrative entities once — three statement evaluations reuse
    // the same domain. Avoids the 3× `list_entities_by_narrative` cost.
    let entities = hg.list_entities_by_narrative(narrative_id)?;

    let d_major = evaluate_statement(&entities, &syllogism.major, resolver)?;
    let d_minor = evaluate_statement(&entities, &syllogism.minor, resolver)?;
    let d_conclusion = evaluate_statement(&entities, &syllogism.conclusion, resolver)?;

    let t = tnorm_for(tnorm);
    let degree = t.combine(d_major, t.combine(d_minor, d_conclusion));

    let figure = syllogism
        .figure_hint
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| classify_figure(syllogism).to_string());

    let peterson_valid = !matches!(figure.as_str(), "II" | "non-canonical");
    let valid = peterson_valid && degree >= threshold;

    tracing::debug!(
        narrative_id = %narrative_id,
        figure = %figure,
        d_major, d_minor, d_conclusion, degree, threshold,
        "syllogism verified"
    );

    Ok(GradedValidity {
        degree,
        figure,
        valid,
        threshold,
    })
}

/// Evaluate `Q(r)` for one quantifier statement over a pre-fetched
/// entity slice.
///
/// Subject/object asymmetry: the subject predicate restricts the
/// domain to entities satisfying `μ_S(e) ≥ 0.5`, and the object
/// predicate is the graded indicator summed into the cardinality
/// ratio `r = (Σ μ_O(e)) / |restricted|`. Mirrors the classical "All
/// S are P → for every s ∈ S, P(s) holds" reading. Entity-level
/// semantics only — situation-level syllogisms are Phase 7.5.
fn evaluate_statement(
    entities: &[Entity],
    stmt: &SyllogismStatement,
    resolver: &dyn PredicateResolver,
) -> Result<f64> {
    let subject = resolver.resolve(&stmt.subject_pred_id)?;
    let object = resolver.resolve(&stmt.object_pred_id)?;

    let (n, sum) = entities
        .iter()
        .filter(|e| subject(e).clamp(0.0, 1.0) >= 0.5)
        .fold((0_usize, 0.0_f64), |(n, s), e| {
            (n + 1, s + object(e).clamp(0.0, 1.0))
        });
    let r = if n == 0 { 0.0 } else { sum / (n as f64) };
    Ok(quantifier::evaluate(stmt.quantifier, r))
}

/// Persisted proof record — the syllogism, the validity result, the
/// proof id, and a created-at timestamp. No verifier state is
/// retained beyond what's needed to replay the proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyllogismProof {
    pub id: Uuid,
    pub syllogism: Syllogism,
    pub graded_validity: GradedValidity,
    pub created_at: DateTime<Utc>,
}

impl SyllogismProof {
    /// Construct a new proof record with a v7 UUID + current timestamp.
    pub fn new(syllogism: Syllogism, graded_validity: GradedValidity) -> Self {
        Self {
            id: Uuid::now_v7(),
            syllogism,
            graded_validity,
            created_at: Utc::now(),
        }
    }
}

/// Persist a proof at `fz/syllog/{narrative_id}/{proof_id_v7_BE_16}`.
pub fn save_syllogism_proof(
    store: &dyn KVStore,
    narrative_id: &str,
    proof: &SyllogismProof,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_syllog(narrative_id, &proof.id);
    let bytes = serde_json::to_vec(proof)
        .map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(&key, &bytes)
}

/// Load a persisted proof, or `None` when the key does not exist.
pub fn load_syllogism_proof(
    store: &dyn KVStore,
    narrative_id: &str,
    proof_id: &Uuid,
) -> Result<Option<SyllogismProof>> {
    let key = crate::fuzzy::key_fuzzy_syllog(narrative_id, proof_id);
    match store.get(&key)? {
        Some(bytes) => {
            let p: SyllogismProof = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(p))
        }
        None => Ok(None),
    }
}

/// Delete a persisted proof (idempotent).
pub fn delete_syllogism_proof(
    store: &dyn KVStore,
    narrative_id: &str,
    proof_id: &Uuid,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_syllog(narrative_id, proof_id);
    store.delete(&key)
}

/// List every persisted syllogism proof for a narrative, newest-first.
/// Entries whose values fail to deserialize are skipped with a warn!.
pub fn list_syllogism_proofs_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<SyllogismProof>> {
    let mut prefix = crate::fuzzy::FUZZY_SYLLOG_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, v) in pairs {
        match serde_json::from_slice::<SyllogismProof>(&v) {
            Ok(p) => out.push(p),
            Err(e) => tracing::warn!(
                narrative_id = %narrative_id,
                "syllogism proof deserialize failed ({e}); skipping"
            ),
        }
    }
    // Newest first: v7 UUIDs sort chronologically, so reverse.
    out.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(out)
}

// ── Tiny DSL for TensaQL / REST surfaces ─────────────────────────────────────
//
// The grammar keeps the three statements as free-form strings of the
// shape "<QUANTIFIER> <subj_pred_id> IS <obj_pred_id>". Parsing lives
// here so the parser + REST handler share one implementation; the
// pest grammar stays simple.

/// Parse a tiny-DSL statement into [`SyllogismStatement`].
/// Accepts: `"MOST actors IS humans"`, `"ALL type:Actor IS type:Location"`, ...
pub fn parse_statement(spec: &str) -> Result<SyllogismStatement> {
    let raw = spec.trim();
    if raw.is_empty() {
        return Err(TensaError::InvalidInput(
            "syllogism statement is empty".into(),
        ));
    }
    // Split on " IS " (case-insensitive) — the subject/object separator.
    let upper = raw.to_ascii_uppercase();
    let is_idx = upper.find(" IS ").ok_or_else(|| {
        TensaError::InvalidInput(format!(
            "syllogism statement '{raw}' missing ' IS ' separator; \
             expected '<QUANTIFIER> <subject> IS <object>'",
        ))
    })?;
    let (lhs, rhs_with_is) = raw.split_at(is_idx);
    let rhs = &rhs_with_is[4..]; // strip " IS "

    // Split LHS on first whitespace → (quantifier, subject).
    let mut lhs_iter = lhs.trim().splitn(2, char::is_whitespace);
    let q_raw = lhs_iter.next().unwrap_or("").trim();
    let subj = lhs_iter.next().unwrap_or("").trim();
    let obj = rhs.trim();

    if q_raw.is_empty() || subj.is_empty() || obj.is_empty() {
        return Err(TensaError::InvalidInput(format!(
            "syllogism statement '{raw}' malformed; expected \
             '<QUANTIFIER> <subject> IS <object>'",
        )));
    }

    // Accept "ALL" as AlmostAll (the ramp that saturates at r=1.0 most
    // closely approximates the classical universal quantifier at the
    // graded layer; see Novák 2008 §III).
    let quantifier = parse_quantifier_token(q_raw)?;

    Ok(SyllogismStatement {
        quantifier,
        subject_pred_id: subj.to_string(),
        object_pred_id: obj.to_string(),
    })
}

/// Map a DSL quantifier keyword to the canonical [`Quantifier`]. Accepts
/// `"ALL"` as a synonym for `AlmostAll` (see module docs / Novák 2008).
fn parse_quantifier_token(token: &str) -> Result<Quantifier> {
    let t = token.trim().to_ascii_uppercase();
    match t.as_str() {
        "ALL" | "EVERY" | "ALMOST_ALL" | "ALMOSTALL" | "ALMOST-ALL" => {
            Ok(Quantifier::AlmostAll)
        }
        "MOST" => Ok(Quantifier::Most),
        "MANY" => Ok(Quantifier::Many),
        "FEW" => Ok(Quantifier::Few),
        _ => Err(TensaError::InvalidInput(format!(
            "unknown quantifier token '{token}'; expected one of \
             ALL, MOST, MANY, ALMOST_ALL, FEW",
        ))),
    }
}
