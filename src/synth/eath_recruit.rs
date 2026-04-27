//! Group-recruitment helpers for EATH (Phase 1).
//!
//! `recruit_from_scratch` builds a brand-new group weighted by current
//! activity × order propensity × normalized LTM weight. `recruit_from_memory`
//! picks a seed group from STM and grows / shrinks / passes-through to size
//! `m` per the paper's continuation dynamics.
//!
//! Pulled out of `eath.rs` to keep that file under the 500-line cap. No
//! behavioural change — the inner `pick_top_k_weighted` and both recruitment
//! helpers are package-private and called only from `EathSurrogate::generate`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::activity::order_propensity_for;
use super::memory::{LongTermMemory, ShortTermMemory};
use super::types::EathParams;

/// Hard cap on the omega-decay exponent (matches the constant in `memory.rs`).
const MAX_AGE_EXPONENT: i32 = 1024;

/// Weighted sampling without replacement of `k` indices into `out`.
///
/// Algorithm: O(n*k) "pick max, zero out, repeat" — fast for `k <= 10` and
/// `n <= a few thousand`, the EATH operating point. For very large N a
/// reservoir sampler would be cheaper but isn't necessary at Phase-1 sizes.
///
/// `weight_buf` MUST already contain the per-entity weights (caller fills).
/// On return, `weight_buf` is restored to all-non-negative — caller still
/// re-fills before each group, so callers don't depend on the exact state.
pub(super) fn pick_top_k_weighted(
    weight_buf: &mut [f32],
    k: usize,
    rng: &mut ChaCha8Rng,
    out: &mut Vec<usize>,
) {
    out.clear();
    let n = weight_buf.len();
    let want = k.min(n);
    for _ in 0..want {
        let total: f32 = weight_buf.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            // Always advance the RNG so two callers diverging on this branch
            // still consume the same number of f32 draws.
            let _: f32 = rng.gen();
            // Fall back to uniform-without-replacement: pick any unpicked.
            let unpicked: Vec<usize> = (0..n).filter(|&i| weight_buf[i] >= 0.0).collect();
            if unpicked.is_empty() {
                return;
            }
            let coin: f32 = rng.gen();
            let idx =
                unpicked[((coin * unpicked.len() as f32) as usize).min(unpicked.len() - 1)];
            out.push(idx);
            weight_buf[idx] = -1.0;
            continue;
        }
        let r: f32 = rng.gen::<f32>() * total;
        let mut cum = 0.0_f32;
        let mut picked = 0_usize;
        for (i, &w) in weight_buf.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            cum += w;
            if r < cum {
                picked = i;
                break;
            }
        }
        out.push(picked);
        weight_buf[picked] = -1.0;
    }
    // Restore non-negative defaults so the caller's "fill weights" pass is
    // idempotent (no special-case for already-marked positions).
    for w in weight_buf.iter_mut() {
        if *w < 0.0 {
            *w = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recruit_from_scratch(
    m: usize,
    current_activity: &[f32],
    params: &EathParams,
    ltm: &LongTermMemory,
    tick: u64,
    n: usize,
    weight_buf: &mut Vec<f32>,
    out: &mut Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    if weight_buf.len() != n {
        weight_buf.resize(n, 0.0);
    }
    for i in 0..n {
        let phi = order_propensity_for(params, i, m);
        let omega = ltm.mean_normalized_weight(i, n, tick, params.omega_decay);
        weight_buf[i] = current_activity[i] * phi * omega;
    }
    pick_top_k_weighted(weight_buf, m, rng, out);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recruit_from_memory(
    m: usize,
    stm: &ShortTermMemory,
    current_activity: &[f32],
    params: &EathParams,
    ltm: &LongTermMemory,
    tick: u64,
    n: usize,
    weight_buf: &mut Vec<f32>,
    out: &mut Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    // Pick a seed group (recency-weighted within the size window).
    let mut candidates = stm.candidates_for_size(m);
    if candidates.is_empty() {
        candidates = (0..stm.len()).collect();
    }
    if candidates.is_empty() {
        // Empty STM — degrade gracefully to from-scratch.
        recruit_from_scratch(
            m,
            current_activity,
            params,
            ltm,
            tick,
            n,
            weight_buf,
            out,
            rng,
        );
        return;
    }

    let total_w: f32 = candidates
        .iter()
        .map(|&idx| {
            stm.get(idx)
                .map(|g| params.omega_decay.powi((g.age as i32).min(MAX_AGE_EXPONENT)))
                .unwrap_or(0.0)
        })
        .sum();
    let coin: f32 = rng.gen();
    let mut seed_idx = candidates[0];
    if total_w > 0.0 {
        let r = coin * total_w;
        let mut cum = 0.0_f32;
        for &c_idx in &candidates {
            let w = stm
                .get(c_idx)
                .map(|g| params.omega_decay.powi((g.age as i32).min(MAX_AGE_EXPONENT)))
                .unwrap_or(0.0);
            cum += w;
            if r < cum {
                seed_idx = c_idx;
                break;
            }
        }
    } else {
        let pick = ((coin * candidates.len() as f32) as usize).min(candidates.len() - 1);
        seed_idx = candidates[pick];
    }

    let seed_members: Vec<usize> = stm
        .get(seed_idx)
        .map(|g| g.members.clone())
        .unwrap_or_default();

    out.clear();
    if seed_members.len() == m {
        out.extend(seed_members);
        return;
    }

    if seed_members.len() < m {
        // Grow: recruit (m - seed.len()) new entities by from-scratch weights,
        // with seed members masked out so they're not duplicated.
        out.extend(seed_members.iter().copied());
        if weight_buf.len() != n {
            weight_buf.resize(n, 0.0);
        }
        for i in 0..n {
            if out.contains(&i) {
                weight_buf[i] = 0.0;
                continue;
            }
            let phi = order_propensity_for(params, i, m);
            let omega = ltm.mean_normalized_weight(i, n, tick, params.omega_decay);
            weight_buf[i] = current_activity[i] * phi * omega;
        }
        let need = m - out.len();
        let mut additions: Vec<usize> = Vec::with_capacity(need);
        pick_top_k_weighted(weight_buf, need, rng, &mut additions);
        out.extend(additions);
        return;
    }

    // Shrink: drop (seed.len() - m) entities uniformly at random via partial
    // Fisher-Yates. Paper §III.2 — aggregation/disaggregation are symmetric;
    // no preference for dropping low-activity members.
    let mut pool = seed_members;
    for i in 0..m {
        let coin: f32 = rng.gen();
        let j = i + ((coin * (pool.len() - i) as f32) as usize).min(pool.len() - i - 1);
        pool.swap(i, j);
    }
    out.extend(pool.into_iter().take(m));
}
