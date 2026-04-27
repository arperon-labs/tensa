//! Activity-phase sampling for EATH (Phase 1).
//!
//! Mancastroppa-Cencetti-Barrat model entities as Markov-chain "stations" with
//! two phases:
//!
//! - **Low** — instantaneous activity is suppressed by `LOW_PHASE_GAMMA` (≈ 0).
//! - **High** — instantaneous activity equals `a_h(i)` from the calibrated
//!   distribution.
//!
//! Each tick, every entity samples a phase transition driven by `Λ_t` (global
//! modulation), `a_T(i)` (persistence), and the rate scales `rho_low` /
//! `rho_high`. After all transitions resolve, this module also computes the
//! per-tick group count via Bernoulli-remainder rounding and the empirical
//! group-size sample.
//!
//! See `docs/synth_eath_algorithm.md` §2 + §4.1 for the full algorithmic spec.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::types::EathParams;

/// Low-phase activity suppression. Paper §III.1 fixes γ ≈ 1e-3 — the low
/// phase still contributes a tiny amount of activity (so total activity is
/// never exactly zero even when every entity is "off") but is dominated by
/// the high-phase mass when any entity is active.
pub const LOW_PHASE_GAMMA: f32 = 1e-3;

/// Which activity phase an entity is currently in.
///
/// Not serialized — `PhaseState` lives only during a single `generate()` call.
/// Reconstructing from seed is cheaper than KV round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityPhase {
    Low,
    High,
}

/// Per-entity phase state maintained across ticks.
#[derive(Debug, Clone, Copy)]
pub struct PhaseState {
    pub phase: ActivityPhase,
    /// How many consecutive ticks the entity has been in its current phase.
    /// Used for diagnostics (mean dwell time) but not by the core algorithm.
    pub steps_in_phase: u32,
}

impl Default for PhaseState {
    fn default() -> Self {
        Self {
            phase: ActivityPhase::Low,
            steps_in_phase: 0,
        }
    }
}

/// All entities' phase states + a scratch buffer for instantaneous activity.
///
/// Owns the two parallel arrays so the inner loop never re-allocates them.
pub struct ActivityField {
    pub phases: Vec<PhaseState>,
    /// Instantaneous activity per entity at the current tick. Mutated in
    /// place by `step_transition` after every transition resolves; readers
    /// (group recruitment) consume this buffer directly.
    pub current_activity: Vec<f32>,
}

impl ActivityField {
    /// Construct a fresh field of `n` entities, all starting in `Low` phase.
    pub fn new(n: usize) -> Self {
        Self {
            phases: vec![PhaseState::default(); n],
            current_activity: vec![0.0; n],
        }
    }

    /// Aggregate activity over all entities. Returns `total_activity` (`A_t`
    /// in the paper). Cheap — one O(N) summation.
    pub fn total_activity(&self) -> f32 {
        self.current_activity.iter().sum()
    }

    /// Run Step 1 of the algorithm for the current tick. Returns `lambda_t`
    /// for the caller to pass through to `draw_num_groups`.
    ///
    /// Mutates `self.phases` (transitions, dwell counters) and
    /// `self.current_activity` in place.
    pub fn step_transition(
        &mut self,
        rng: &mut ChaCha8Rng,
        params: &EathParams,
        tick: u64,
        mean_a_t: f32,
    ) -> f32 {
        let lambda_t = lambda_at(params, tick);

        // mean_a_t == 0 ⇒ rho_low effectively zero ⇒ no Low→High transitions.
        // Guard against div-by-zero in `r_lh` calculation.
        let safe_mean = if mean_a_t > 0.0 { mean_a_t } else { 1.0 };

        for i in 0..self.phases.len() {
            let a_t_i = params.a_t_distribution.get(i).copied().unwrap_or(0.0);
            let a_h_i = params.a_h_distribution.get(i).copied().unwrap_or(0.0);

            // Always consume one f32 draw per entity per tick — this preserves
            // determinism even if `current_phase` short-circuits below.
            let coin: f32 = rng.gen();

            let cur = self.phases[i];
            match cur.phase {
                ActivityPhase::Low => {
                    let r_lh = (lambda_t * params.rho_low * a_t_i / safe_mean).clamp(0.0, 1.0);
                    if coin < r_lh {
                        self.phases[i] = PhaseState {
                            phase: ActivityPhase::High,
                            steps_in_phase: 0,
                        };
                    } else {
                        self.phases[i].steps_in_phase = cur.steps_in_phase.saturating_add(1);
                    }
                }
                ActivityPhase::High => {
                    let r_hl = ((1.0 - lambda_t) * params.rho_high).clamp(0.0, 1.0);
                    if coin < r_hl {
                        self.phases[i] = PhaseState {
                            phase: ActivityPhase::Low,
                            steps_in_phase: 0,
                        };
                    } else {
                        self.phases[i].steps_in_phase = cur.steps_in_phase.saturating_add(1);
                    }
                }
            }

            self.current_activity[i] = match self.phases[i].phase {
                ActivityPhase::High => a_h_i,
                ActivityPhase::Low => LOW_PHASE_GAMMA * a_h_i,
            };
        }

        lambda_t
    }
}

/// Read `lambda_t` for tick `t`, wrapping modulo the schedule length so a
/// short schedule repeats (intentional — daily / weekly cycles).
pub(crate) fn lambda_at(params: &EathParams, tick: u64) -> f32 {
    if params.lambda_schedule.is_empty() {
        1.0
    } else {
        let idx = (tick as usize) % params.lambda_schedule.len();
        params.lambda_schedule[idx]
    }
}

/// Draw the number of groups to form at this tick.
///
/// Discretization strategy: round-to-nearest with Bernoulli on the fractional
/// remainder (Option B from the design doc §9-Q1). Avoids a `rand_distr`
/// dependency for the Poisson alternative; sufficient burstiness comes from
/// the phase dynamics + Λ_t schedule rather than from the discretization.
///
/// Always consumes exactly one f32 draw — keeps RNG advancement deterministic.
pub(crate) fn draw_num_groups(
    params: &EathParams,
    total_activity: f32,
    mean_total_activity: f32,
    n: usize,
    rng: &mut ChaCha8Rng,
) -> usize {
    // Always advance the RNG so the post-condition holds even when we
    // short-circuit to zero.
    let coin: f32 = rng.gen();

    if mean_total_activity <= 0.0 || total_activity <= 0.0 {
        return 0;
    }

    let raw = params.xi * total_activity / mean_total_activity;
    if !raw.is_finite() || raw <= 0.0 {
        return 0;
    }

    let floor = raw.floor();
    let frac = raw - floor;
    let mut count = floor as usize;
    if coin < frac {
        count += 1;
    }
    // Cap at N/2 — a single tick cannot produce more groups than disjoint
    // pairs the entity set could support.
    count.min(n / 2)
}

/// Sample a group size from the empirical distribution.
///
/// Always consumes exactly one f64 draw (`gen::<f64>()`) — match the
/// determinism contract.
pub(crate) fn sample_group_size(params: &EathParams, rng: &mut ChaCha8Rng) -> usize {
    if params.group_size_distribution.is_empty() {
        // Always consume the draw to keep determinism alignment.
        let _: f64 = rng.gen();
        return 2;
    }
    let total: f64 = params.group_size_distribution.iter().map(|&c| c as f64).sum();
    let r: f64 = rng.gen::<f64>() * total;
    let mut cum = 0.0_f64;
    for (k, &count) in params.group_size_distribution.iter().enumerate() {
        cum += count as f64;
        if r < cum {
            // Index 0 = dyads (size 2), index 1 = triads, ...
            let size = k + 2;
            return size.min(params.max_group_size.max(2));
        }
    }
    // Fallback: max bin (handles floating-point edge case where r ≥ total).
    (params.group_size_distribution.len() + 1).min(params.max_group_size.max(2))
}

/// Returns the order-propensity weight φ_i(m) for entity `i` and group size
/// `m`. When `params.order_propensity` is empty (the Phase-1 default), falls
/// back to the empirical `group_size_distribution` probability of size m.
pub(crate) fn order_propensity_for(
    params: &EathParams,
    entity_idx: usize,
    group_size: usize,
) -> f32 {
    if group_size < 2 || params.max_group_size < 2 {
        return 0.0;
    }
    let max_g = params.max_group_size;
    let row_len = max_g.saturating_sub(1); // sizes 2..=max_g
    let bin = group_size - 2;
    if bin >= row_len {
        return 0.0;
    }

    if !params.order_propensity.is_empty() {
        let off = entity_idx * row_len + bin;
        if let Some(&v) = params.order_propensity.get(off) {
            return v;
        }
    }

    // Default: uniform over empirical group_size_distribution.
    if params.group_size_distribution.is_empty() {
        return 1.0;
    }
    let total: u32 = params.group_size_distribution.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let count = params
        .group_size_distribution
        .get(bin)
        .copied()
        .unwrap_or(0);
    count as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn base_params() -> EathParams {
        EathParams {
            a_t_distribution: vec![0.5, 0.5, 0.5],
            a_h_distribution: vec![1.0, 1.0, 1.0],
            lambda_schedule: vec![],
            p_from_scratch: 0.5,
            omega_decay: 0.95,
            group_size_distribution: vec![1, 1, 1],
            rho_low: 0.5,
            rho_high: 0.3,
            xi: 1.0,
            order_propensity: vec![],
            max_group_size: 4,
            stm_capacity: 7,
            num_entities: 3,
        }
    }

    #[test]
    fn test_low_to_high_transition_with_high_rate() {
        let mut field = ActivityField::new(3);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut p = base_params();
        // Force a guaranteed transition: r_lh = 1.0 * 1.0 * 1.0 / 1.0 = 1.0.
        p.rho_low = 1.0;
        p.a_t_distribution = vec![1.0; 3];
        // Mean a_T = 1.0 — single transition probability per entity.
        let _ = field.step_transition(&mut rng, &p, 0, 1.0);
        for ph in &field.phases {
            assert_eq!(ph.phase, ActivityPhase::High);
        }
    }

    #[test]
    fn test_step_transition_is_deterministic() {
        let p = base_params();
        let mut field_a = ActivityField::new(3);
        let mut field_b = ActivityField::new(3);
        let mut rng_a = ChaCha8Rng::seed_from_u64(42);
        let mut rng_b = ChaCha8Rng::seed_from_u64(42);
        let mean = 0.5;
        for t in 0..50 {
            field_a.step_transition(&mut rng_a, &p, t, mean);
            field_b.step_transition(&mut rng_b, &p, t, mean);
        }
        let phases_a: Vec<_> = field_a.phases.iter().map(|p| p.phase).collect();
        let phases_b: Vec<_> = field_b.phases.iter().map(|p| p.phase).collect();
        assert_eq!(phases_a, phases_b);
    }

    #[test]
    fn test_lambda_at_wraps_modulo() {
        let mut p = base_params();
        p.lambda_schedule = vec![0.1, 0.2, 0.3];
        assert!((lambda_at(&p, 0) - 0.1).abs() < 1e-6);
        assert!((lambda_at(&p, 3) - 0.1).abs() < 1e-6);
        assert!((lambda_at(&p, 5) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_lambda_default_is_one_when_schedule_empty() {
        let p = base_params();
        assert!((lambda_at(&p, 0) - 1.0).abs() < 1e-6);
        assert!((lambda_at(&p, 999) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_draw_num_groups_zero_when_no_activity() {
        let p = base_params();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        assert_eq!(draw_num_groups(&p, 0.0, 1.0, 100, &mut rng), 0);
    }

    #[test]
    fn test_draw_num_groups_caps_at_n_over_2() {
        let mut p = base_params();
        p.xi = 1000.0;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 10;
        let count = draw_num_groups(&p, 100.0, 1.0, n, &mut rng);
        assert!(count <= n / 2);
    }

    #[test]
    fn test_sample_group_size_in_range() {
        let p = base_params();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..100 {
            let s = sample_group_size(&p, &mut rng);
            assert!((2..=p.max_group_size).contains(&s));
        }
    }

    #[test]
    fn test_order_propensity_uniform_default() {
        let p = base_params();
        let phi_2 = order_propensity_for(&p, 0, 2);
        // group_size_distribution = [1, 1, 1] → fraction is 1/3 for size 2.
        assert!((phi_2 - 1.0 / 3.0).abs() < 1e-6);
    }
}
