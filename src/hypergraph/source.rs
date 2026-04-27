//! Source intelligence operations: Source CRUD, SourceAttribution
//! dual-index, ContentionLink dual-index, and corroboration scoring.
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! [`Hypergraph::compute_corroboration`] computes
//! `1 - Π(1 - trust_i)` — probability that at least one source is right.
//! Iteratively this is the **Goguen t-conorm** (probabilistic OR). Phase 1
//! exposes the choice via [`Hypergraph::compute_corroboration_with_tconorm`];
//! the existing `compute_corroboration` delegates to it with
//! `TNormKind::Goguen` to preserve pre-sprint numerics.
//!
//! The Bayesian confidence path in [`Hypergraph::recompute_confidence`] is
//! conjugate-prior arithmetic — **not** a t-norm — and the Phase 2
//! aggregator selector owns that call site; Phase 1 leaves it classical.
//!
//! Cites: [klement2000].

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::{reduce_tconorm, TNormKind};
use crate::hypergraph::{keys, Hypergraph};
use crate::source::*;
use crate::store::TxnOp;

impl Hypergraph {
    // ─── Source CRUD ─────────────────────────────────────────────

    /// Create a new source record.
    pub fn create_source(&self, source: Source) -> Result<Uuid> {
        let key = keys::source_key(&source.id);
        if self.store.get(&key)?.is_some() {
            return Err(TensaError::SourceExists(source.id));
        }
        let bytes = serde_json::to_vec(&source)?;
        self.store.put(&key, &bytes)?;
        Ok(source.id)
    }

    /// Get a source by UUID.
    pub fn get_source(&self, id: &Uuid) -> Result<Source> {
        let key = keys::source_key(id);
        let bytes = self
            .store
            .get(&key)?
            .ok_or(TensaError::SourceNotFound(*id))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Update a source by applying a closure.
    pub fn update_source(&self, id: &Uuid, updater: impl FnOnce(&mut Source)) -> Result<Source> {
        let key = keys::source_key(id);
        let bytes = self
            .store
            .get(&key)?
            .ok_or(TensaError::SourceNotFound(*id))?;
        let mut source: Source = serde_json::from_slice(&bytes)?;
        updater(&mut source);
        source.updated_at = Utc::now();
        let new_bytes = serde_json::to_vec(&source)?;
        self.store.put(&key, &new_bytes)?;
        Ok(source)
    }

    /// Delete a source and all its attributions.
    pub fn delete_source(&self, id: &Uuid) -> Result<()> {
        let key = keys::source_key(id);
        if self.store.get(&key)?.is_none() {
            return Err(TensaError::SourceNotFound(*id));
        }
        // Remove all attributions from this source.
        // Key layout: "sa/" (3) + source_uuid (16) + "/" (1) + target_uuid (16) = 36 bytes.
        // Extract target UUID directly from key bytes to avoid deserializing values.
        let attr_prefix = keys::source_attribution_prefix(id);
        let pairs = self.store.prefix_scan(&attr_prefix)?;
        let mut ops: Vec<TxnOp> = Vec::new();
        for (fwd_key, _value) in &pairs {
            if fwd_key.len() >= 36 {
                if let Ok(target_id) = Uuid::from_slice(&fwd_key[fwd_key.len() - 16..]) {
                    let rev_key = keys::source_attribution_reverse_key(&target_id, id);
                    ops.push(TxnOp::Delete(fwd_key.clone()));
                    ops.push(TxnOp::Delete(rev_key));
                }
            }
        }
        ops.push(TxnOp::Delete(key));
        self.store.transaction(ops)
    }

    /// List all registered sources.
    pub fn list_sources(&self) -> Result<Vec<Source>> {
        let prefix = keys::source_prefix();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut sources = Vec::new();
        for (_key, value) in pairs {
            let source: Source = serde_json::from_slice(&value)?;
            sources.push(source);
        }
        Ok(sources)
    }

    /// List sources with cursor-based pagination.
    pub fn list_sources_paginated(
        &self,
        limit: usize,
        after: Option<&Uuid>,
    ) -> Result<(Vec<Source>, Option<Uuid>)> {
        let prefix = keys::source_prefix();
        let start = match after {
            Some(cursor) => {
                let mut k = keys::source_key(cursor);
                k.push(0);
                k
            }
            None => prefix.clone(),
        };
        let mut end = prefix;
        end.push(0xFF);

        let pairs = self.store.range(&start, &end)?;
        let mut result = Vec::with_capacity(limit + 1);
        for (_key, value) in pairs.iter().take(limit + 1) {
            result.push(serde_json::from_slice::<Source>(value)?);
        }

        let next_cursor = if result.len() > limit {
            result.pop();
            result.last().map(|s| s.id)
        } else {
            None
        };

        Ok((result, next_cursor))
    }

    /// List sources that have at least one attribution pointing at an
    /// entity or situation in the given narrative. Sources themselves are
    /// global — this walks the reverse attribution index for every target
    /// in the narrative, collects unique `source_id`s, then dereferences
    /// each. Source records that are missing (deleted since the
    /// attribution was written) are skipped; other errors propagate.
    pub fn list_sources_for_narrative(&self, narrative_id: &str) -> Result<Vec<Source>> {
        use std::collections::HashSet;
        let target_ids = self
            .list_entities_by_narrative(narrative_id)?
            .into_iter()
            .map(|e| e.id)
            .chain(
                self.list_situations_by_narrative(narrative_id)?
                    .into_iter()
                    .map(|s| s.id),
            );
        let mut source_ids: HashSet<Uuid> = HashSet::new();
        for tid in target_ids {
            source_ids.extend(self.get_source_ids_for_target(&tid)?);
        }
        let mut sources: Vec<Source> = Vec::with_capacity(source_ids.len());
        for sid in source_ids {
            match self.get_source(&sid) {
                Ok(s) => sources.push(s),
                Err(TensaError::SourceNotFound(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        sources.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(sources)
    }

    // ─── Source Attribution (dual-index) ─────────────────────────

    /// Add a source attribution link.
    /// Writes dual index: `sa/{source}/{target}` and `sar/{target}/{source}`.
    /// After writing, reactively recomputes confidence on the target (best-effort).
    pub fn add_attribution(&self, attr: SourceAttribution) -> Result<()> {
        // Validate source exists
        self.get_source(&attr.source_id)?;

        let fwd_key = keys::source_attribution_key(&attr.source_id, &attr.target_id);
        if self.store.get(&fwd_key)?.is_some() {
            return Err(TensaError::AttributionExists {
                source_id: attr.source_id,
                target_id: attr.target_id,
            });
        }

        let rev_key = keys::source_attribution_reverse_key(&attr.target_id, &attr.source_id);
        let bytes = serde_json::to_vec(&attr)?;
        self.store.transaction(vec![
            TxnOp::Put(fwd_key, bytes.clone()),
            TxnOp::Put(rev_key, bytes),
        ])?;

        // Best-effort: recompute confidence for the target
        let _ = self.recompute_and_write_confidence(&attr.target_id, attr.target_kind);
        Ok(())
    }

    /// Remove a source attribution link.
    /// Reads the attribution before deleting to determine target kind,
    /// then reactively recomputes confidence on the target (best-effort).
    pub fn remove_attribution(&self, source_id: &Uuid, target_id: &Uuid) -> Result<()> {
        let fwd_key = keys::source_attribution_key(source_id, target_id);
        let bytes = self
            .store
            .get(&fwd_key)?
            .ok_or(TensaError::NotFound(format!(
                "Attribution {} -> {}",
                source_id, target_id
            )))?;
        let attr: SourceAttribution = serde_json::from_slice(&bytes)?;
        let rev_key = keys::source_attribution_reverse_key(target_id, source_id);
        self.store
            .transaction(vec![TxnOp::Delete(fwd_key), TxnOp::Delete(rev_key)])?;

        // Best-effort: recompute confidence for the target
        let _ = self.recompute_and_write_confidence(&attr.target_id, attr.target_kind);
        Ok(())
    }

    /// Get all attributions from a given source.
    pub fn get_attributions_for_source(&self, source_id: &Uuid) -> Result<Vec<SourceAttribution>> {
        let prefix = keys::source_attribution_prefix(source_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let attr: SourceAttribution = serde_json::from_slice(&value)?;
            result.push(attr);
        }
        Ok(result)
    }

    /// Get all source attributions for a given target (entity or situation).
    pub fn get_attributions_for_target(&self, target_id: &Uuid) -> Result<Vec<SourceAttribution>> {
        let prefix = keys::source_attribution_reverse_prefix(target_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let attr: SourceAttribution = serde_json::from_slice(&value)?;
            result.push(attr);
        }
        Ok(result)
    }

    /// Get only the `source_id`s attributing a given target. Faster than
    /// `get_attributions_for_target` when the full record isn't needed —
    /// reads the source UUID from the last 16 bytes of the `sar/` key and
    /// skips JSON deserialization entirely.
    pub fn get_source_ids_for_target(&self, target_id: &Uuid) -> Result<Vec<Uuid>> {
        let prefix = keys::source_attribution_reverse_prefix(target_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut out = Vec::with_capacity(pairs.len());
        for (key, _value) in pairs {
            if key.len() >= 16 {
                if let Ok(sid) = Uuid::from_slice(&key[key.len() - 16..]) {
                    out.push(sid);
                }
            }
        }
        Ok(out)
    }

    // ─── Contention Links (dual-index) ──────────────────────────

    /// Add a contention link between two situations.
    /// Validates both situations exist. Writes dual index.
    pub fn add_contention(&self, link: ContentionLink) -> Result<()> {
        // Validate both situations exist
        self.get_situation(&link.situation_a)?;
        self.get_situation(&link.situation_b)?;

        let fwd_key = keys::contention_key(&link.situation_a, &link.situation_b);
        if self.store.get(&fwd_key)?.is_some() {
            return Err(TensaError::ContentionExists {
                situation_a: link.situation_a,
                situation_b: link.situation_b,
            });
        }

        let rev_key = keys::contention_reverse_key(&link.situation_b, &link.situation_a);
        let bytes = serde_json::to_vec(&link)?;
        self.store.transaction(vec![
            TxnOp::Put(fwd_key, bytes.clone()),
            TxnOp::Put(rev_key, bytes),
        ])
    }

    /// Remove a contention link.
    pub fn remove_contention(&self, a: &Uuid, b: &Uuid) -> Result<()> {
        let fwd_key = keys::contention_key(a, b);
        if self.store.get(&fwd_key)?.is_none() {
            return Err(TensaError::ContentionNotFound {
                situation_a: *a,
                situation_b: *b,
            });
        }
        let rev_key = keys::contention_reverse_key(b, a);
        self.store
            .transaction(vec![TxnOp::Delete(fwd_key), TxnOp::Delete(rev_key)])
    }

    /// Get all contentions involving a situation (both forward and reverse).
    pub fn get_contentions_for_situation(
        &self,
        situation_id: &Uuid,
    ) -> Result<Vec<ContentionLink>> {
        let mut result = Vec::new();
        // Forward: this situation is situation_a
        let fwd_prefix = keys::contention_prefix(situation_id);
        for (_key, value) in self.store.prefix_scan(&fwd_prefix)? {
            let link: ContentionLink = serde_json::from_slice(&value)?;
            result.push(link);
        }
        // Reverse: this situation is situation_b
        let rev_prefix = keys::contention_reverse_prefix(situation_id);
        for (_key, value) in self.store.prefix_scan(&rev_prefix)? {
            let link: ContentionLink = serde_json::from_slice(&value)?;
            result.push(link);
        }
        Ok(result)
    }

    /// List every contention touching a situation in the given narrative.
    /// Dedupes the forward/reverse indexes so each pair appears once.
    pub fn list_contentions_for_narrative(
        &self,
        narrative_id: &str,
    ) -> Result<Vec<ContentionLink>> {
        use std::collections::HashSet;
        let sit_ids: HashSet<Uuid> = self
            .list_situations_by_narrative(narrative_id)?
            .into_iter()
            .map(|s| s.id)
            .collect();
        let mut seen: HashSet<(Uuid, Uuid)> = HashSet::new();
        let mut out = Vec::new();
        for sid in &sit_ids {
            for link in self.get_contentions_for_situation(sid)? {
                // Canonical ordering so forward/reverse pairs collapse to one entry.
                let (a, b) = if link.situation_a < link.situation_b {
                    (link.situation_a, link.situation_b)
                } else {
                    (link.situation_b, link.situation_a)
                };
                if !sit_ids.contains(&a) || !sit_ids.contains(&b) {
                    continue;
                }
                if seen.insert((a, b)) {
                    out.push(link);
                }
            }
        }
        Ok(out)
    }

    /// Resolve a contention by marking it resolved with analyst notes.
    pub fn resolve_contention(
        &self,
        a: &Uuid,
        b: &Uuid,
        resolution: String,
    ) -> Result<ContentionLink> {
        let fwd_key = keys::contention_key(a, b);
        let bytes = self
            .store
            .get(&fwd_key)?
            .ok_or(TensaError::ContentionNotFound {
                situation_a: *a,
                situation_b: *b,
            })?;
        let mut link: ContentionLink = serde_json::from_slice(&bytes)?;
        link.resolved = true;
        link.resolution = Some(resolution);

        let rev_key = keys::contention_reverse_key(b, a);
        let new_bytes = serde_json::to_vec(&link)?;
        self.store.transaction(vec![
            TxnOp::Put(fwd_key, new_bytes.clone()),
            TxnOp::Put(rev_key, new_bytes),
        ])?;
        Ok(link)
    }

    // ─── Corroboration Scoring ───────────────────────────────────

    /// Compute corroboration score for a target based on independent sources.
    ///
    /// Defaults to the **Goguen t-conorm** (probabilistic OR) via
    /// `1 - Π(1 - trust_i)` — bit-identical to the pre-sprint formula. Use
    /// [`Self::compute_corroboration_with_tconorm`] to select a different
    /// t-conorm family (e.g. Gödel `max` for the "strongest-source wins"
    /// interpretation).
    pub fn compute_corroboration(&self, target_id: &Uuid) -> Result<f32> {
        self.compute_corroboration_with_tconorm(target_id, TNormKind::Goguen)
    }

    /// Variant of [`Self::compute_corroboration`] that accepts a t-conorm
    /// kind. `TNormKind::Goguen` returns the probabilistic-OR corroboration
    /// (default); `TNormKind::Godel` returns `max` of trust scores;
    /// `TNormKind::Lukasiewicz` returns bounded sum.
    pub fn compute_corroboration_with_tconorm(
        &self,
        target_id: &Uuid,
        tconorm: TNormKind,
    ) -> Result<f32> {
        let attrs = self.get_attributions_for_target(target_id)?;
        if attrs.is_empty() {
            return Ok(0.0);
        }
        let mut trusts: Vec<f64> = Vec::with_capacity(attrs.len());
        for attr in &attrs {
            if let Ok(source) = self.get_source(&attr.source_id) {
                trusts.push(source.trust_score.clamp(0.0, 1.0) as f64);
            }
        }
        let reduced = reduce_tconorm(tconorm, &trusts) as f32;
        Ok(reduced.clamp(0.0, 1.0))
    }

    /// Recompute confidence for an attribution target and write it back.
    /// Uses `update_entity_no_snapshot` for entities (derived changes shouldn't
    /// create state history noise) and `update_situation` for situations.
    fn recompute_and_write_confidence(
        &self,
        target_id: &Uuid,
        target_kind: AttributionTarget,
    ) -> Result<()> {
        match target_kind {
            AttributionTarget::Entity => {
                let entity = self.get_entity(target_id)?;
                // Use the original extraction confidence if a breakdown exists,
                // otherwise fall back to the current stored confidence.
                let extraction = entity
                    .confidence_breakdown
                    .as_ref()
                    .map(|bd| bd.extraction)
                    .unwrap_or(entity.confidence);
                let breakdown = self.recompute_confidence(target_id, extraction)?;
                let composite = breakdown.composite();
                self.update_entity_no_snapshot(target_id, |e| {
                    e.confidence = composite;
                    e.confidence_breakdown = Some(breakdown.clone());
                })?;
            }
            AttributionTarget::Situation => {
                let situation = self.get_situation(target_id)?;
                let extraction = situation
                    .confidence_breakdown
                    .as_ref()
                    .map(|bd| bd.extraction)
                    .unwrap_or(situation.confidence);
                let breakdown = self.recompute_confidence(target_id, extraction)?;
                let composite = breakdown.composite();
                self.update_situation(target_id, |s| {
                    s.confidence = composite;
                    s.confidence_breakdown = Some(breakdown.clone());
                })?;
            }
        }
        Ok(())
    }

    /// Recompute confidence for all targets attributed to a given source.
    /// Call this after changing a source's `trust_score` to propagate the change
    /// to every entity and situation that source is attributed to.
    pub fn propagate_source_trust_change(&self, source_id: &Uuid) -> Result<()> {
        let attributions = self.get_attributions_for_source(source_id)?;
        for attr in &attributions {
            let _ = self.recompute_and_write_confidence(&attr.target_id, attr.target_kind);
        }
        Ok(())
    }

    /// Recompute full confidence breakdown for a target entity or situation.
    ///
    /// Uses Bayesian updating with a Beta distribution prior:
    /// - Prior: `Beta(alpha_0, beta_0)` initialized from extraction confidence
    /// - Each source attribution is treated as a Bernoulli observation with
    ///   likelihood proportional to the source's trust score
    /// - Evidence weight decays with recency (sigmoid with 1-week half-life)
    /// - Posterior mean `alpha / (alpha + beta)` becomes the new confidence
    ///
    /// See [`Self::recompute_confidence_with_aggregator`] for the Phase 2
    /// aggregator-selectable variant. The default entry point delegates
    /// with `None` and stays bit-identical to the pre-sprint numerics.
    ///
    /// Cites: [yager1988owa] [grabisch1996choquet].
    pub fn recompute_confidence(
        &self,
        target_id: &Uuid,
        current_confidence: f32,
    ) -> Result<ConfidenceBreakdown> {
        self.recompute_confidence_with_aggregator(target_id, current_confidence, None)
    }

    /// Phase 2 variant of [`Self::recompute_confidence`] accepting an
    /// optional aggregator. When `None` the Bayesian Beta-posterior path
    /// runs unchanged (backward-compat). When `Some(agg)` the per-source
    /// trust-weighted Bernoulli contributions are routed through the
    /// aggregator to produce the posterior mean, with the Beta α / β
    /// book-keeping derived from the aggregated signal so downstream
    /// components (source_credibility, corroboration, recency) keep the
    /// same shape.
    pub fn recompute_confidence_with_aggregator(
        &self,
        target_id: &Uuid,
        current_confidence: f32,
        aggregator: Option<crate::fuzzy::aggregation::AggregatorKind>,
    ) -> Result<ConfidenceBreakdown> {
        const PRIOR_STRENGTH: f32 = 2.0;

        let attrs = self.get_attributions_for_target(target_id)?;
        let extraction = current_confidence;

        // Step 1: Initialize Beta prior from extraction confidence
        let prior_alpha = (extraction * PRIOR_STRENGTH).max(0.01);
        let prior_beta = ((1.0 - extraction) * PRIOR_STRENGTH).max(0.01);
        let mut alpha = prior_alpha;
        let mut beta = prior_beta;

        // Per-source trust-weighted signals — captured so the aggregator
        // branch can fold them under a caller-chosen operator without
        // re-iterating the attribution list.
        let mut total_trust = 0.0_f32;
        let mut source_count = 0u32;
        let mut trust_signals: Vec<f64> = Vec::new();

        for attr in &attrs {
            if let Ok(source) = self.get_source(&attr.source_id) {
                let trust = source.trust_score.clamp(0.0, 1.0);
                total_trust += trust;
                source_count += 1;

                // Recency-based evidence weight (sigmoid decay, 1-week half-life)
                let age_hours = (Utc::now() - attr.retrieved_at).num_hours() as f32;
                let evidence_weight = 1.0 / (1.0 + (age_hours / 168.0));

                // Bayesian update: treat source as Bernoulli observation
                // P(evidence | H=true) = trust, P(evidence | H=false) = 1 - trust
                alpha += trust * evidence_weight;
                beta += (1.0 - trust) * evidence_weight;

                // Trust-weighted signal ∈ [0, 1] for the aggregator fold.
                trust_signals.push((trust * evidence_weight).clamp(0.0, 1.0) as f64);
            }
        }

        // Phase 2 aggregator branch: when Some, rewrite the posterior mean
        // to the aggregator's view of the per-source trust signals. The
        // Beta α / β book-keeping is re-derived so the posterior mean
        // matches `alpha / (alpha + beta)` under the new signal.
        if let Some(agg) = aggregator {
            if !trust_signals.is_empty() {
                let aggregator_boxed = crate::fuzzy::aggregation::aggregator_for(agg);
                let agg_mean = aggregator_boxed.aggregate(&trust_signals)?.clamp(0.0, 1.0) as f32;
                // Re-derive α, β so the posterior mean matches agg_mean.
                // Preserve the prior's concentration by scaling to the
                // concentration already accumulated (strong-prior invariant).
                let concentration = (alpha + beta).max(f32::EPSILON);
                alpha = (agg_mean * concentration).max(0.01);
                beta = ((1.0 - agg_mean) * concentration).max(0.01);
            }
        }

        // Step 3: Derive legacy component scores for backward compatibility
        let source_credibility = if source_count > 0 {
            total_trust / source_count as f32
        } else {
            current_confidence
        };

        // Corroboration derived from posterior concentration
        let concentration = alpha + beta;
        let corroboration = if attrs.is_empty() {
            0.0
        } else {
            (1.0 - 1.0 / concentration).clamp(0.0, 1.0)
        };

        // Recency: sigmoid decay based on newest attribution
        let recency = if let Some(newest) = attrs.iter().map(|a| a.retrieved_at).max() {
            let age_hours = (Utc::now() - newest).num_hours() as f32;
            1.0 / (1.0 + (age_hours / 168.0))
        } else {
            0.5
        };

        Ok(ConfidenceBreakdown {
            extraction,
            source_credibility,
            corroboration,
            recency,
            prior_alpha: Some(prior_alpha),
            prior_beta: Some(prior_beta),
            posterior_alpha: Some(alpha),
            posterior_beta: Some(beta),
        })
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
