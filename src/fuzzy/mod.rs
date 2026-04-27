//! Fuzzy logic layer — t-norms, aggregation operators, fuzzy temporal
//! reasoning, intermediate quantifiers, graded syllogisms, fuzzy formal
//! concept analysis, Mamdani rule systems, hybrid fuzzy-probabilistic
//! inference.
//!
//! This module is a **layer parallel to `analysis/`, `inference/`, and
//! `synth/`**: it depends only on `hypergraph/`, `store/`, `error/`, and
//! `types/`, and is never depended on by anything below itself. Fuzzy
//! reasoning sits above the canonical temporal-hypergraph model and
//! below the query / API / MCP layers.
//!
//! ## Why a trait-first layer
//!
//! TNorm / TConorm / Aggregator are abstractions, not a fixed enum of
//! operators. The TensaQL grammar and REST / MCP surfaces accept the
//! operator name as a **string parameter** — new families (Schweizer–
//! Sklar, Frank, Dombi, …) plug in via [`registry::TNormRegistry`] and
//! [`registry::AggregatorRegistry`] without parser or API migration.
//!
//! ## Backward-compatibility invariant
//!
//! Before Phase 1 wires the TNorm machinery into existing confidence-
//! aggregation call sites, every downstream numeric path stays
//! bit-identical. Phase 0 only adds stubs — `combine` bodies all return
//! `0.0` until Phase 1 fills them in. Phase 1 ships with default
//! [`tnorm::TNormKind::Godel`] and site-specific defaults (Goguen for
//! DS product, arithmetic mean for weighted composites) that replay the
//! pre-sprint numerics exactly.
//!
//! ## KV layout
//!
//! Seven prefixes under `fz/`. **UUIDs in keys use 16-byte big-endian
//! binary** (`Uuid::as_bytes()` — NOT hex strings), so `prefix_scan` on
//! a v7-UUID-sorted slice returns rows in chronological order.
//!
//! ```text
//!   fz/tn/{narrative_id_utf8}/{config_hash_hex}            → TNorm config record
//!   fz/agg/{narrative_id_utf8}/{target_id_v7_BE_BIN_16}    → cached aggregation result
//!   fz/allen/{narrative_id_utf8}/{a_id_BE_16}/{b_id_BE_16} → fuzzy Allen 13-vector for a pair
//!   fz/quant/{narrative_id_utf8}/{predicate_hash_hex}      → intermediate-quantifier eval cache
//!   fz/fca/{narrative_id_utf8}/{lattice_id_BE_16}          → fuzzy formal concept lattice
//!   fz/rules/{narrative_id_utf8}/{rule_id_BE_16}           → Mamdani rule record
//!   fz/syllog/{narrative_id_utf8}/{proof_id_BE_16}         → graded syllogism proof trace
//! ```
//!
//! Always go through the `key_fuzzy_*` helpers below — they are the
//! single source of truth for the encoding contract.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
//!        [duboisprade1989fuzzyallen] [novak2008quantifiers]
//!        [murinovanovak2014peterson] [belohlavek2004fuzzyfca]
//!        [mamdani1975mamdani] [flaminio2026fsta].

pub mod aggregation;
pub mod aggregation_choquet;
pub mod aggregation_learn;
pub mod aggregation_measure;
pub mod aggregation_owa;
pub mod allen;
pub mod allen_store;
pub mod fca;
pub mod fca_store;
pub mod hashing;
pub mod hybrid;
pub mod quantifier;
pub mod registry;
pub mod rules;
pub mod rules_eval;
pub mod rules_store;
pub mod rules_types;
pub mod syllogism;
pub mod synthetic_cib_dataset;
pub mod tnorm;

#[cfg(test)]
mod aggregation_tests;
#[cfg(test)]
mod allen_tests;
#[cfg(test)]
mod backward_compat_tests;
#[cfg(test)]
mod fca_tests;
#[cfg(test)]
mod hybrid_tests;
#[cfg(test)]
pub(crate) mod quantifier_tests;
#[cfg(test)]
mod rules_integration_tests;
#[cfg(test)]
mod rules_tests;
#[cfg(test)]
mod syllogism_tests;
#[cfg(test)]
mod tnorm_tests;

use uuid::Uuid;

// ── KV prefix constants (single source of truth) ─────────────────────────────

pub const FUZZY_TN_PREFIX: &[u8] = b"fz/tn/";
pub const FUZZY_AGG_PREFIX: &[u8] = b"fz/agg/";
pub const FUZZY_ALLEN_PREFIX: &[u8] = b"fz/allen/";
pub const FUZZY_QUANT_PREFIX: &[u8] = b"fz/quant/";
pub const FUZZY_FCA_PREFIX: &[u8] = b"fz/fca/";
pub const FUZZY_RULE_PREFIX: &[u8] = b"fz/rules/";
pub const FUZZY_SYLLOG_PREFIX: &[u8] = b"fz/syllog/";
/// Phase 10 — persisted fuzzy-probability query reports under
/// `fz/hybrid/{narrative_id_utf8}/{query_id_v7_BE_BIN_16}`.
pub const FUZZY_HYBRID_PREFIX: &[u8] = b"fz/hybrid/";

// ── Key builders (NEVER assemble keys manually) ──────────────────────────────

/// `fz/tn/{narrative_id}/{config_hash}` — cached t-norm configuration record.
pub fn key_fuzzy_config(narrative_id: &str, config_hash: &str) -> Vec<u8> {
    let mut k =
        Vec::with_capacity(FUZZY_TN_PREFIX.len() + narrative_id.len() + 1 + config_hash.len());
    k.extend_from_slice(FUZZY_TN_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(config_hash.as_bytes());
    k
}

/// `fz/agg/{narrative_id}/{target_id_v7_BE_BIN_16}` — cached aggregation result.
pub fn key_fuzzy_agg(narrative_id: &str, target_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_AGG_PREFIX.len() + narrative_id.len() + 1 + 16);
    k.extend_from_slice(FUZZY_AGG_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(target_id.as_bytes());
    k
}

/// `fz/allen/{narrative_id}/{a_id_BE_16}/{b_id_BE_16}` — fuzzy Allen 13-vector.
pub fn key_fuzzy_allen(narrative_id: &str, a_id: &Uuid, b_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_ALLEN_PREFIX.len() + narrative_id.len() + 2 + 32);
    k.extend_from_slice(FUZZY_ALLEN_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(a_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(b_id.as_bytes());
    k
}

/// `fz/quant/{narrative_id}/{predicate_hash}` — intermediate-quantifier cache.
pub fn key_fuzzy_quant(narrative_id: &str, predicate_hash: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(
        FUZZY_QUANT_PREFIX.len() + narrative_id.len() + 1 + predicate_hash.len(),
    );
    k.extend_from_slice(FUZZY_QUANT_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(predicate_hash.as_bytes());
    k
}

/// `fz/fca/{narrative_id}/{lattice_id_v7_BE_BIN_16}` — fuzzy concept lattice.
pub fn key_fuzzy_fca(narrative_id: &str, lattice_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_FCA_PREFIX.len() + narrative_id.len() + 1 + 16);
    k.extend_from_slice(FUZZY_FCA_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(lattice_id.as_bytes());
    k
}

/// `fz/rules/{narrative_id}/{rule_id_v7_BE_BIN_16}` — Mamdani rule record.
pub fn key_fuzzy_rule(narrative_id: &str, rule_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_RULE_PREFIX.len() + narrative_id.len() + 1 + 16);
    k.extend_from_slice(FUZZY_RULE_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(rule_id.as_bytes());
    k
}

/// `fz/syllog/{narrative_id}/{proof_id_v7_BE_BIN_16}` — graded syllogism trace.
pub fn key_fuzzy_syllog(narrative_id: &str, proof_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_SYLLOG_PREFIX.len() + narrative_id.len() + 1 + 16);
    k.extend_from_slice(FUZZY_SYLLOG_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(proof_id.as_bytes());
    k
}

/// `fz/hybrid/{narrative_id}/{query_id_v7_BE_BIN_16}` — Phase 10
/// fuzzy-probability query report. Chronological key ordering mirrors
/// the other persistence slices.
pub fn key_fuzzy_hybrid(narrative_id: &str, query_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_HYBRID_PREFIX.len() + narrative_id.len() + 1 + 16);
    k.extend_from_slice(FUZZY_HYBRID_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(query_id.as_bytes());
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzy::aggregation::AggregatorKind;
    use crate::fuzzy::registry::{AggregatorRegistry, TNormRegistry};
    use crate::fuzzy::tnorm::TNormKind;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_registry_lists_four_tnorms() {
        let reg = TNormRegistry::default();
        let mut names = reg.list();
        names.sort();
        assert_eq!(names, vec!["godel", "goguen", "hamacher", "lukasiewicz"]);
    }

    #[test]
    fn test_registry_lists_six_aggregators() {
        let reg = AggregatorRegistry::default();
        let mut names = reg.list();
        names.sort();
        assert_eq!(
            names,
            vec![
                "choquet",
                "mean",
                "median",
                "owa",
                "tconorm_reduce",
                "tnorm_reduce"
            ]
        );
    }

    #[test]
    fn test_registry_get_resolves_each_tnorm_name() {
        let reg = TNormRegistry::default();
        assert!(matches!(reg.get("godel").unwrap(), TNormKind::Godel));
        assert!(matches!(reg.get("goguen").unwrap(), TNormKind::Goguen));
        assert!(matches!(
            reg.get("lukasiewicz").unwrap(),
            TNormKind::Lukasiewicz
        ));
        assert!(matches!(reg.get("hamacher").unwrap(), TNormKind::Hamacher(_)));
    }

    #[test]
    fn test_registry_get_rejects_unknown_name() {
        let reg = TNormRegistry::default();
        assert!(reg.get("einstein").is_err());
    }

    #[test]
    fn test_aggregator_registry_get_resolves_each_name() {
        let reg = AggregatorRegistry::default();
        assert!(matches!(reg.get("mean").unwrap(), AggregatorKind::Mean));
        assert!(matches!(reg.get("median").unwrap(), AggregatorKind::Median));
        // Parameterized ones default to an empty / trivial param.
        assert!(matches!(reg.get("owa").unwrap(), AggregatorKind::Owa(_)));
        assert!(matches!(
            reg.get("choquet").unwrap(),
            AggregatorKind::Choquet(_)
        ));
        assert!(matches!(
            reg.get("tnorm_reduce").unwrap(),
            AggregatorKind::TNormReduce(_)
        ));
        assert!(matches!(
            reg.get("tconorm_reduce").unwrap(),
            AggregatorKind::TConormReduce(_)
        ));
    }

    #[test]
    fn test_key_builder_uuid_big_endian() {
        // Two v7 UUIDs minted ~1ms apart must sort bytewise in chronological
        // order because v7 UUIDs encode the timestamp in the high bits and
        // `Uuid::as_bytes()` returns them big-endian.
        let nid = "chrono-narrative";
        let first = Uuid::now_v7();
        sleep(Duration::from_millis(2));
        let second = Uuid::now_v7();

        let k1 = key_fuzzy_agg(nid, &first);
        let k2 = key_fuzzy_agg(nid, &second);
        assert!(k1 < k2, "chronological key ordering must match v7 timeline");
    }

    #[test]
    fn test_key_builder_length() {
        // Each key = prefix + narrative_id + '/' + uuid (16 bytes BE binary).
        let nid = "narr";
        let id = Uuid::now_v7();

        let k_agg = key_fuzzy_agg(nid, &id);
        assert_eq!(k_agg.len(), FUZZY_AGG_PREFIX.len() + nid.len() + 1 + 16);

        let k_fca = key_fuzzy_fca(nid, &id);
        assert_eq!(k_fca.len(), FUZZY_FCA_PREFIX.len() + nid.len() + 1 + 16);

        let k_rule = key_fuzzy_rule(nid, &id);
        assert_eq!(k_rule.len(), FUZZY_RULE_PREFIX.len() + nid.len() + 1 + 16);

        let k_syl = key_fuzzy_syllog(nid, &id);
        assert_eq!(k_syl.len(), FUZZY_SYLLOG_PREFIX.len() + nid.len() + 1 + 16);

        let k_hyb = key_fuzzy_hybrid(nid, &id);
        assert_eq!(k_hyb.len(), FUZZY_HYBRID_PREFIX.len() + nid.len() + 1 + 16);

        // Two UUIDs in the Allen key.
        let id2 = Uuid::now_v7();
        let k_all = key_fuzzy_allen(nid, &id, &id2);
        assert_eq!(k_all.len(), FUZZY_ALLEN_PREFIX.len() + nid.len() + 2 + 32);

        // String-hash keys store a hex / hash string after the separator.
        let hash = "deadbeef";
        let k_tn = key_fuzzy_config(nid, hash);
        assert_eq!(k_tn.len(), FUZZY_TN_PREFIX.len() + nid.len() + 1 + hash.len());
        let k_q = key_fuzzy_quant(nid, hash);
        assert_eq!(
            k_q.len(),
            FUZZY_QUANT_PREFIX.len() + nid.len() + 1 + hash.len()
        );
    }

    #[test]
    fn test_prefix_disjointness() {
        // Every prefix starts with `fz/` but no prefix is a prefix of another.
        let all: &[&[u8]] = &[
            FUZZY_TN_PREFIX,
            FUZZY_AGG_PREFIX,
            FUZZY_ALLEN_PREFIX,
            FUZZY_QUANT_PREFIX,
            FUZZY_FCA_PREFIX,
            FUZZY_RULE_PREFIX,
            FUZZY_SYLLOG_PREFIX,
            FUZZY_HYBRID_PREFIX,
        ];
        for (i, a) in all.iter().enumerate() {
            assert!(a.starts_with(b"fz/"), "prefix {:?} not under fz/", a);
            for (j, b) in all.iter().enumerate() {
                if i == j {
                    continue;
                }
                assert!(
                    !a.starts_with(b) && !b.starts_with(a),
                    "prefix overlap: {:?} vs {:?}",
                    a,
                    b
                );
            }
        }
    }
}
