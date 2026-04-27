//! Per-field fitter helpers for [`super::calibrate::fit_params_from_narrative`].
//!
//! Pulled out of `calibrate.rs` so that file stays under the 500-line cap.
//! Every function here is `pub(super)` — used only by the calibrator's
//! orchestration body. No external callers.
//!
//! See `calibrate.rs` module-level docs for the algorithm and field list.

use std::collections::HashMap;

use uuid::Uuid;

use crate::types::Situation;

use super::types::EathParams;

// ── Tunable constants (named so /simplify doesn't flag them as magic) ────────

/// Bucket count cap for the `lambda_schedule`. Set to 100 so the schedule is
/// always bounded regardless of how many situations the narrative contains.
pub(super) const LAMBDA_SCHEDULE_BUCKET_CAP: usize = 100;

/// Lambda clamp range. Per Mancastroppa, `Λ_t` is a unitless modulator;
/// extreme values destabilise the phase Markov chain. `[0.01, 100.0]` keeps
/// the multiplier bounded while still allowing a 10000× dynamic range.
pub(super) const LAMBDA_CLAMP_MIN: f32 = 0.01;
pub(super) const LAMBDA_CLAMP_MAX: f32 = 100.0;

/// Rate-scale clamp range for `rho_low` / `rho_high`. They feed directly into
/// per-tick transition probabilities, so > 1.0 makes no physical sense and
/// < 0.01 means "transition essentially never happens" — the code handles
/// both edges but clamping makes the resulting params readable.
pub(super) const RHO_CLAMP_MIN: f32 = 0.01;
pub(super) const RHO_CLAMP_MAX: f32 = 1.0;

/// Continuation-overlap threshold for `p_from_scratch` estimation. Two
/// consecutive situations are considered a "continuation" if at least this
/// fraction of their joint participants overlap.
pub(super) const CONTINUATION_OVERLAP_THRESHOLD: f32 = 0.5;

// ── Sort + per-entity stats ─────────────────────────────────────────────────

/// Return indices into `situations` sorted chronologically. `temporal.start`
/// when present orders the situation; rows with `start = None` (typically
/// `TimeGranularity::Unknown`) fall back to insertion order, broken by `id`
/// (UUID compare) so the sort is total and stable across runs.
pub(super) fn sort_situations_chronologically(situations: &[Situation]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..situations.len()).collect();
    order.sort_by(|&i, &j| {
        let a = situations[i].temporal.start;
        let b = situations[j].temporal.start;
        match (a, b) {
            (Some(a), Some(b)) => a.cmp(&b).then(situations[i].id.cmp(&situations[j].id)),
            // None < Some — but stable: when both are None, fall back to
            // `created_at`, then `id` for total ordering across replays.
            (None, None) => situations[i]
                .created_at
                .cmp(&situations[j].created_at)
                .then(situations[i].id.cmp(&situations[j].id)),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
        }
    });
    order
}

/// Step 4 — per-entity activity vectors aT (fraction of situations attended)
/// and ah (mean group size when attending).
pub(super) fn compute_per_entity_activity(
    n: usize,
    num_situations: usize,
    sit_count_per_entity: &[u32],
    group_size_sum_per_entity: &[u64],
) -> (Vec<f32>, Vec<f32>) {
    let mut a_t = Vec::with_capacity(n);
    let mut a_h = Vec::with_capacity(n);
    for i in 0..n {
        let count = sit_count_per_entity[i];
        a_t.push(if num_situations == 0 {
            0.0
        } else {
            count as f32 / num_situations as f32
        });
        a_h.push(if count == 0 {
            0.0
        } else {
            group_size_sum_per_entity[i] as f32 / count as f32
        });
    }
    (a_t, a_h)
}

// ── Group-size histogram ────────────────────────────────────────────────────

/// Step 7 — empirical group-size histogram.
///
/// `group_size_distribution[k] = count of situations whose participant set
/// had exactly `k + 2` members`. Bins past `max_group_size` are dropped.
/// Empty input ⇒ uniform `[1, 1, 1]` fallback so `sample_group_size`
/// downstream still has a non-empty distribution to draw from.
pub(super) fn build_group_size_histogram(
    sorted_situations: &[&Situation],
    participants_by_sit: &HashMap<Uuid, Vec<usize>>,
    max_group_size: usize,
) -> Vec<u32> {
    let bins = max_group_size.saturating_sub(1);
    let mut hist = vec![0u32; bins.max(1)];
    let mut any_observed = false;
    for sit in sorted_situations {
        let group_size = participants_by_sit
            .get(&sit.id)
            .map(|m| m.len())
            .unwrap_or(0);
        if group_size < 2 {
            // Solo or empty groups don't contribute to a hyperedge size
            // distribution — EATH only emits groups of size >= 2.
            continue;
        }
        let bin = group_size.saturating_sub(2);
        if bin < hist.len() {
            hist[bin] = hist[bin].saturating_add(1);
            any_observed = true;
        }
    }
    if !any_observed {
        // Fallback: at least dyads/triads/quads with weight 1 each so the
        // empirical inverse-CDF in `sample_group_size` still works.
        return vec![1, 1, 1];
    }
    hist
}

// ── Lambda schedule ─────────────────────────────────────────────────────────

/// Step 5 — bucket the chronologically-sorted situations into at most
/// `LAMBDA_SCHEDULE_BUCKET_CAP` evenly-sized chunks and emit the per-bucket
/// activity multiplier (`bucket_count / mean_bucket_count`). Returns the
/// schedule and the mean situations-per-bucket value (re-used for `xi`).
///
/// Caller passes the count rather than the sorted slice — bucket-membership
/// only depends on positional index after the chronological sort, not the
/// situation contents themselves.
pub(super) fn build_lambda_schedule(num_situations: usize) -> (Vec<f32>, f32) {
    if num_situations == 0 {
        return (Vec::new(), 0.0);
    }
    let buckets = num_situations.clamp(1, LAMBDA_SCHEDULE_BUCKET_CAP);
    let mut counts = vec![0u32; buckets];
    // Even-split: situation index `i` lands in bucket `(i * buckets) / N`.
    // Using integer math here (not floor of float division) avoids a corner
    // case where the last bucket gets one extra situation due to rounding.
    for i in 0..num_situations {
        let b = ((i * buckets) / num_situations).min(buckets - 1);
        counts[b] = counts[b].saturating_add(1);
    }
    let mean: f32 = num_situations as f32 / buckets as f32;
    let safe_mean = if mean > 0.0 { mean } else { 1.0 };
    let mut schedule = Vec::with_capacity(buckets);
    for &c in &counts {
        let v = c as f32 / safe_mean;
        schedule.push(v.clamp(LAMBDA_CLAMP_MIN, LAMBDA_CLAMP_MAX));
    }
    (schedule, safe_mean)
}

// ── p_from_scratch ──────────────────────────────────────────────────────────

/// Step 6 — `p_from_scratch ≈ 1 - fraction_of_continuations`. A "continuation"
/// is a consecutive (i, i+1) situation pair whose participant overlap is at
/// least `CONTINUATION_OVERLAP_THRESHOLD` of the joint set size.
///
/// Edge case: < 2 situations ⇒ no consecutive pairs exist ⇒ assume "always
/// from scratch" (1.0). Same for the case where every situation has zero
/// participants.
pub(super) fn estimate_p_from_scratch(
    sorted_situations: &[&Situation],
    participants_by_sit: &HashMap<Uuid, Vec<usize>>,
) -> f32 {
    if sorted_situations.len() < 2 {
        return 1.0;
    }
    let mut continuations = 0u32;
    let mut comparable_pairs = 0u32;
    for window in sorted_situations.windows(2) {
        let a = match participants_by_sit.get(&window[0].id) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let b = match participants_by_sit.get(&window[1].id) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        comparable_pairs += 1;
        let overlap = pair_overlap_fraction(a, b);
        if overlap >= CONTINUATION_OVERLAP_THRESHOLD {
            continuations += 1;
        }
    }
    if comparable_pairs == 0 {
        return 1.0;
    }
    let cont_frac = continuations as f32 / comparable_pairs as f32;
    (1.0 - cont_frac).clamp(0.0, 1.0)
}

/// Helper for `estimate_p_from_scratch`. Computes |A ∩ B| / |A ∪ B|. Uses a
/// sorted-vec linear scan since participant lists are small (typically ≤ 10)
/// and one HashSet alloc per pair would dominate the cost.
fn pair_overlap_fraction(a: &[usize], b: &[usize]) -> f32 {
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_unstable();
    a_sorted.dedup();
    b_sorted.sort_unstable();
    b_sorted.dedup();
    let mut intersection = 0u32;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a_sorted.len() && j < b_sorted.len() {
        match a_sorted[i].cmp(&b_sorted[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let union = (a_sorted.len() + b_sorted.len()) as u32 - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

// ── Burstiness → rho pair ───────────────────────────────────────────────────

/// Step 11 — estimate `(rho_low, rho_high)` from per-entity activity bursts.
///
/// For each entity we walk the sorted situation timeline marking ticks the
/// entity was active. Run-length-encode active vs quiet streaks. The rate
/// scales become `1 / mean_quiet_run` and `1 / mean_active_run` respectively
/// (each clamped to `[RHO_CLAMP_MIN, RHO_CLAMP_MAX]`).
///
/// Edge cases:
/// - Entity never active → contributes nothing to `rho_low` (treated as "no
///   evidence"); when ALL entities never participated we fall back to
///   `(RHO_CLAMP_MIN, RHO_CLAMP_MAX)`.
/// - Entity always active (no quiet runs) → contributes `1.0` to `rho_low`
///   (saturated transition rate; entity wakes immediately).
pub(super) fn estimate_rho_pair(
    sorted_situations: &[&Situation],
    participants_by_sit: &HashMap<Uuid, Vec<usize>>,
    n: usize,
) -> (f32, f32) {
    if n == 0 || sorted_situations.is_empty() {
        return (RHO_CLAMP_MIN, RHO_CLAMP_MAX);
    }

    // Per-entity timeline as a flat row-major bitfield: `active_at[i*S + t]`.
    // One contiguous allocation of N*S booleans avoids per-row Vec overhead at
    // 10k+ situations. Bool ⇒ 1 byte; for N=10k S=10k that's 100MB, which is
    // the practical ceiling — calibrations beyond that should bucket the
    // timeline first (Phase 4+ optimization).
    let num_situations = sorted_situations.len();
    let mut active_at = vec![false; n * num_situations];
    for (t, sit) in sorted_situations.iter().enumerate() {
        if let Some(members) = participants_by_sit.get(&sit.id) {
            for &idx in members {
                if idx < n {
                    active_at[idx * num_situations + t] = true;
                }
            }
        }
    }

    let mut quiet_run_total: u64 = 0;
    let mut quiet_run_count: u64 = 0;
    let mut active_run_total: u64 = 0;
    let mut active_run_count: u64 = 0;

    for row_start in (0..n).map(|i| i * num_situations) {
        let row = &active_at[row_start..row_start + num_situations];
        let mut cur_quiet: u32 = 0;
        let mut cur_active: u32 = 0;
        for &is_active in row {
            if is_active {
                if cur_quiet > 0 {
                    quiet_run_total += cur_quiet as u64;
                    quiet_run_count += 1;
                    cur_quiet = 0;
                }
                cur_active += 1;
            } else {
                if cur_active > 0 {
                    active_run_total += cur_active as u64;
                    active_run_count += 1;
                    cur_active = 0;
                }
                cur_quiet += 1;
            }
        }
        if cur_quiet > 0 {
            quiet_run_total += cur_quiet as u64;
            quiet_run_count += 1;
        }
        if cur_active > 0 {
            active_run_total += cur_active as u64;
            active_run_count += 1;
        }
    }

    let rho_low = derive_rate(quiet_run_total, quiet_run_count);
    let rho_high = derive_rate(active_run_total, active_run_count);
    (rho_low, rho_high)
}

/// Common rate derivation: `1 / mean_run_length` clamped to the rho range.
/// Empty input (no runs of either type) saturates to `RHO_CLAMP_MAX` —
/// downstream `validate_params` accepts this as a usable default.
fn derive_rate(total: u64, count: u64) -> f32 {
    if count == 0 {
        return RHO_CLAMP_MAX;
    }
    let mean = total as f32 / count as f32;
    if mean <= 0.0 {
        return RHO_CLAMP_MAX;
    }
    (1.0 / mean).clamp(RHO_CLAMP_MIN, RHO_CLAMP_MAX)
}

// ── NaN / inf guard ─────────────────────────────────────────────────────────

/// Walks every numeric field on `params` and errors with `SynthFailure` if
/// any is non-finite. (Integer fields can't be NaN; only the f32 fields need
/// checking.) Returns the formatted error message via the caller's
/// `Result`-bearing macro path so this module stays self-contained.
pub(super) fn first_non_finite(params: &EathParams) -> Option<String> {
    for (i, &v) in params.a_t_distribution.iter().enumerate() {
        if !v.is_finite() {
            return Some(format!("a_t at index {i}: {v}"));
        }
    }
    for (i, &v) in params.a_h_distribution.iter().enumerate() {
        if !v.is_finite() {
            return Some(format!("a_h at index {i}: {v}"));
        }
    }
    for (i, &v) in params.lambda_schedule.iter().enumerate() {
        if !v.is_finite() {
            return Some(format!("lambda at index {i}: {v}"));
        }
    }
    for (name, v) in [
        ("p_from_scratch", params.p_from_scratch),
        ("omega_decay", params.omega_decay),
        ("rho_low", params.rho_low),
        ("rho_high", params.rho_high),
        ("xi", params.xi),
    ] {
        if !v.is_finite() {
            return Some(format!("{name}: {v}"));
        }
    }
    None
}
