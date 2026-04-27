//! Memory kernel for active-group reactivation (Phase 1).
//!
//! Two memory structures power the EATH continuation dynamics:
//!
//! - **Short-term memory (STM)** — a bounded ring buffer of the last
//!   `stm_capacity` groups formed. Used by `RECRUIT_FROM_MEMORY` to pick a
//!   "seed" group to mutate when the dice say "continue, don't start fresh".
//! - **Long-term memory (LTM)** — a sparse pairwise co-participation matrix
//!   with lazy geometric decay (`omega_decay^Δt`). Modulates the recruitment
//!   weights so familiar pairs cluster again.
//!
//! Both store entity *indices* (positions in the pre-built `entity_ids`
//! vector), not UUIDs — UUID lookup is deferred until emit time so the hot
//! path stays index-only.
//!
//! See `docs/synth_eath_algorithm.md` §4.2 / §4.3 for the full design.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};

/// Hard ceiling on the decay exponent. `omega_decay.powi(MAX_DECAY_EXPONENT)`
/// is already so close to zero that further work is wasted, and clamping
/// guards against pathological `current_tick - last_updated` overflow when
/// LTM weights are queried at very large ticks.
const MAX_DECAY_EXPONENT: i32 = 1024;

/// One recently-formed group, retained for possible continuation.
///
/// Members are entity indices (0-based position in `entity_ids`), NOT UUIDs.
/// UUID resolution is deferred to emit time so the hot path stays index-only.
#[derive(Debug, Clone)]
pub struct RecentGroup {
    pub members: Vec<usize>,
    /// Ticks since this group was formed. Incremented by [`ShortTermMemory::age_all`].
    pub age: u32,
}

/// Bounded ring buffer of recent groups for short-term memory continuation.
///
/// Capacity is `EathParams.stm_capacity` (default 7). When the buffer is full,
/// the oldest group is evicted. `VecDeque` front-is-newest convention so
/// `push_front` + `pop_back` is the eviction primitive.
///
/// Memory: O(capacity * max_group_size). With capacity=7, max_group_size=10,
/// that's ≈70 `usize` values + struct overhead ≈ <1 KB.
pub struct ShortTermMemory {
    groups: VecDeque<RecentGroup>,
    capacity: usize,
}

impl ShortTermMemory {
    /// New STM with the given capacity. Capacity is clamped to ≥ 1 — a
    /// zero-capacity STM defeats the continuation mechanism entirely.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            groups: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Add a newly-formed group at the front. Evicts the oldest if at capacity.
    pub fn push(&mut self, group: RecentGroup) {
        if self.groups.len() == self.capacity {
            self.groups.pop_back();
        }
        self.groups.push_front(group);
    }

    /// Increment every retained group's age by one tick (saturating).
    pub fn age_all(&mut self) {
        for g in &mut self.groups {
            g.age = g.age.saturating_add(1);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Pick a uniformly-random retained group's index. Returns `None` when
    /// empty.
    ///
    /// Always consumes exactly one f32 draw — keeps determinism alignment
    /// even when the caller doesn't end up using the result.
    pub fn pick_random_index(&self, rng: &mut ChaCha8Rng) -> Option<usize> {
        let coin: f32 = rng.gen();
        if self.groups.is_empty() {
            return None;
        }
        let idx = (coin * self.groups.len() as f32) as usize;
        Some(idx.min(self.groups.len() - 1))
    }

    /// Indices of groups whose member count is in `{m-1, m, m+1}` (the
    /// paper's continuation-size window). Returns indices into the internal
    /// deque so callers can read by `get`.
    pub fn candidates_for_size(&self, m: usize) -> Vec<usize> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                let s = g.members.len();
                s == m || s + 1 == m || (m > 0 && s == m + 1) || (m >= 1 && s == m.saturating_sub(1))
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn get(&self, idx: usize) -> Option<&RecentGroup> {
        self.groups.get(idx)
    }

    /// Iterate over retained groups in newest-first order.
    pub fn iter(&self) -> impl Iterator<Item = &RecentGroup> {
        self.groups.iter()
    }
}

// ── Long-term memory ────────────────────────────────────────────────────────

/// One LTM cell: accumulated co-participation weight + the tick of the last
/// touch. Used together to compute lazily-decayed effective weight on read.
#[derive(Debug, Clone, Copy)]
pub struct LtmEntry {
    pub weight: f32,
    pub last_updated_tick: u64,
}

/// Sparse long-term memory: pairwise co-participation weights with lazy decay.
///
/// Key: `(min_idx, max_idx)` — pairs are stored canonically so lookup never
/// has to consider both orderings.
///
/// **Why HashMap, not dense matrix:** for a narrative with `N` entities a
/// dense ω_0 matrix is O(N²) — at N=5000 that's 100 MB just for the matrix.
/// Most entity pairs in real narratives never co-participate; HashMap stores
/// only observed pairs (O(|observed_pairs|)).
///
/// **Decay formula:** weight at tick `t` for a pair last touched at `t_last`
/// is `stored_weight * omega_decay.powi(min(t - t_last, MAX_DECAY_EXPONENT))`.
/// Decay is applied only on read, never proactively iterated.
pub struct LongTermMemory {
    cells: HashMap<(u32, u32), LtmEntry>,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    /// Number of distinct pairs observed so far (no decay applied — counts
    /// non-zero stored weights).
    pub fn num_pairs(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Record every pair `(i, j)` in `members` (with i < j) as having
    /// co-participated at `tick`. Increments stored weight by 1.0 per pair.
    pub fn record_group(&mut self, members: &[usize], tick: u64) {
        for (a_pos, &a) in members.iter().enumerate() {
            for &b in &members[a_pos + 1..] {
                let key = canonical_pair(a, b);
                let entry = self.cells.entry(key).or_insert(LtmEntry {
                    weight: 0.0,
                    last_updated_tick: tick,
                });
                entry.weight += 1.0;
                entry.last_updated_tick = tick;
            }
        }
    }

    /// Record one pair-with-weight at a given tick. Pair canonicalization
    /// (`min`, `max`) happens here. Pairs with `a == b` are silently ignored.
    /// Repeated calls on the same canonical pair update the stored
    /// `last_updated_tick` (overwriting older ticks) and ADD `weight` to the
    /// running total — so the most recent caller "wins" the timestamp while
    /// the weight integrates over all observations.
    pub fn record_pair(&mut self, a: usize, b: usize, weight: f32, tick: u64) {
        if a == b {
            return;
        }
        let key = canonical_pair(a, b);
        let entry = self.cells.entry(key).or_insert(LtmEntry {
            weight: 0.0,
            last_updated_tick: tick,
        });
        entry.weight += weight;
        entry.last_updated_tick = tick;
    }

    /// Build an `LongTermMemory` from a stream of observed pairs.
    ///
    /// Used by Phase 2 calibration as a Phase 4 hook — the calibrator may
    /// pre-load LTM from the source narrative's co-participation graph so
    /// `recruit_from_scratch`'s `ω̃₀(i, ⋅)` term reflects observed pairwise
    /// affinity from tick 0 of generation. Pair canonicalization happens
    /// inside `record_pair`; `(a, a)` self-pairs are silently skipped.
    ///
    /// All pairs are stamped with `current_tick` so they decay uniformly
    /// from the start of the run (no ancient pair is treated as "fresher"
    /// than another by accident).
    pub fn from_observed_pairs(
        pairs: impl Iterator<Item = (usize, usize, f32)>,
        current_tick: u64,
    ) -> Self {
        let mut ltm = Self::new();
        for (a, b, weight) in pairs {
            ltm.record_pair(a, b, weight, current_tick);
        }
        ltm
    }

    /// Effective decayed weight for pair `(a, b)` at `current_tick`.
    /// Returns 0.0 for pairs never observed.
    pub fn get_decayed(&self, a: usize, b: usize, current_tick: u64, omega_decay: f32) -> f32 {
        let key = canonical_pair(a, b);
        match self.cells.get(&key) {
            Some(entry) => {
                let elapsed = current_tick.saturating_sub(entry.last_updated_tick);
                let exp = (elapsed as i32).min(MAX_DECAY_EXPONENT);
                entry.weight * omega_decay.powi(exp)
            }
            None => 0.0,
        }
    }

    /// Mean normalized weight ω̃₀(a, ⋅) for entity `a`. When LTM is empty,
    /// returns `1.0 / n` (uniform). `n` is the total entity count, passed in
    /// because LTM doesn't track it.
    pub fn mean_normalized_weight(
        &self,
        a: usize,
        n: usize,
        current_tick: u64,
        omega_decay: f32,
    ) -> f32 {
        if self.cells.is_empty() {
            return 1.0 / n.max(1) as f32;
        }
        let mut total = 0.0_f32;
        for (&(x, y), entry) in &self.cells {
            if x as usize != a && y as usize != a {
                continue;
            }
            let elapsed = current_tick.saturating_sub(entry.last_updated_tick);
            let exp = (elapsed as i32).min(MAX_DECAY_EXPONENT);
            total += entry.weight * omega_decay.powi(exp);
        }
        if total <= 0.0 {
            1.0 / n.max(1) as f32
        } else {
            total / n as f32
        }
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn canonical_pair(a: usize, b: usize) -> (u32, u32) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (lo as u32, hi as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_stm_evicts_oldest() {
        let mut stm = ShortTermMemory::new(3);
        stm.push(RecentGroup { members: vec![1], age: 0 });
        stm.push(RecentGroup { members: vec![2], age: 0 });
        stm.push(RecentGroup { members: vec![3], age: 0 });
        stm.push(RecentGroup { members: vec![4], age: 0 });
        assert_eq!(stm.len(), 3);
        let members: Vec<Vec<usize>> = stm.iter().map(|g| g.members.clone()).collect();
        // Newest-first: 4, 3, 2 (1 was evicted).
        assert_eq!(members, vec![vec![4], vec![3], vec![2]]);
    }

    #[test]
    fn test_stm_age_all_increments() {
        let mut stm = ShortTermMemory::new(3);
        stm.push(RecentGroup { members: vec![1], age: 0 });
        stm.age_all();
        stm.age_all();
        assert_eq!(stm.get(0).unwrap().age, 2);
    }

    #[test]
    fn test_stm_candidates_for_size() {
        let mut stm = ShortTermMemory::new(4);
        stm.push(RecentGroup { members: vec![1, 2], age: 0 });
        stm.push(RecentGroup { members: vec![1, 2, 3], age: 0 });
        stm.push(RecentGroup { members: vec![1, 2, 3, 4], age: 0 });
        stm.push(RecentGroup { members: vec![1, 2, 3, 4, 5], age: 0 });
        // m = 3 → candidates have size 2, 3, or 4.
        let cs = stm.candidates_for_size(3);
        assert_eq!(cs.len(), 3);
    }

    #[test]
    fn test_stm_pick_random_index_is_deterministic_in_bounds() {
        let mut stm = ShortTermMemory::new(3);
        stm.push(RecentGroup { members: vec![1], age: 0 });
        stm.push(RecentGroup { members: vec![2], age: 0 });
        stm.push(RecentGroup { members: vec![3], age: 0 });
        let mut rng_a = ChaCha8Rng::seed_from_u64(99);
        let mut rng_b = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..20 {
            let a = stm.pick_random_index(&mut rng_a).unwrap();
            let b = stm.pick_random_index(&mut rng_b).unwrap();
            assert_eq!(a, b);
            assert!(a < stm.len());
        }
    }

    #[test]
    fn test_ltm_record_and_decay() {
        let mut ltm = LongTermMemory::new();
        ltm.record_group(&[0, 1, 2], 0);
        // Pair (0,1) observed once; with omega=1.0 returns 1.0.
        assert!((ltm.get_decayed(0, 1, 0, 1.0) - 1.0).abs() < 1e-6);
        // With omega=0.5 and elapsed=2 ticks: 1.0 * 0.5^2 = 0.25.
        assert!((ltm.get_decayed(0, 1, 2, 0.5) - 0.25).abs() < 1e-6);
        // Missing pair returns 0.
        assert_eq!(ltm.get_decayed(5, 6, 0, 1.0), 0.0);
    }

    #[test]
    fn test_ltm_canonical_pair_is_symmetric() {
        let mut ltm = LongTermMemory::new();
        ltm.record_group(&[0, 1], 0);
        let w_ab = ltm.get_decayed(0, 1, 0, 1.0);
        let w_ba = ltm.get_decayed(1, 0, 0, 1.0);
        assert!((w_ab - w_ba).abs() < 1e-6);
    }

    #[test]
    fn test_ltm_mean_normalized_weight_empty_is_uniform() {
        let ltm = LongTermMemory::new();
        let w = ltm.mean_normalized_weight(0, 10, 0, 0.95);
        assert!((w - 0.1).abs() < 1e-6);
    }

    /// Phase 2: `LongTermMemory::from_observed_pairs` must canonicalize pairs
    /// (`(b, a)` and `(a, b)` collapse), skip self-pairs (`(a, a)`), and
    /// integrate weights on the canonicalized key. The returned LTM should
    /// then answer `get_decayed` symmetrically for the most-recent weight.
    #[test]
    fn test_long_term_memory_from_observed_pairs() {
        // 5 pairs, two with the same canonical form: (0, 1, 1.0) and (1, 0, 2.0)
        // collapse to a single (0, 1) entry with weight 3.0. A self-pair
        // (3, 3, 9.0) is silently skipped.
        let pairs = vec![
            (0, 1, 1.0),
            (1, 0, 2.0), // canonicalizes to (0, 1), adds to weight
            (2, 3, 0.5),
            (4, 2, 0.75),
            (3, 3, 9.0), // self-pair — skipped
        ];
        let ltm = LongTermMemory::from_observed_pairs(pairs.into_iter(), 100);

        // Self-pair skipped → three distinct unordered pairs retained.
        assert_eq!(ltm.num_pairs(), 3);

        // At the same tick (no decay) with omega=1.0 the merged (0, 1) weight
        // is 1.0 + 2.0 = 3.0, queried symmetrically.
        let w_ab = ltm.get_decayed(0, 1, 100, 1.0);
        let w_ba = ltm.get_decayed(1, 0, 100, 1.0);
        assert!((w_ab - 3.0).abs() < 1e-6);
        assert!((w_ba - 3.0).abs() < 1e-6);

        // Sanity on the other pairs.
        assert!((ltm.get_decayed(2, 3, 100, 1.0) - 0.5).abs() < 1e-6);
        assert!((ltm.get_decayed(2, 4, 100, 1.0) - 0.75).abs() < 1e-6);

        // The skipped self-pair left no trace.
        assert_eq!(ltm.get_decayed(3, 3, 100, 1.0), 0.0);
    }

    #[test]
    fn test_record_pair_is_symmetric_and_accumulates() {
        let mut ltm = LongTermMemory::new();
        ltm.record_pair(5, 2, 1.5, 10);
        ltm.record_pair(2, 5, 0.5, 20); // same canonical pair; ADDS weight
        // Latest tick wins for timestamp; weight = 1.5 + 0.5 = 2.0.
        let w = ltm.get_decayed(2, 5, 20, 1.0);
        assert!((w - 2.0).abs() < 1e-6);
        // Self-pair is a no-op.
        ltm.record_pair(7, 7, 99.0, 30);
        assert_eq!(ltm.get_decayed(7, 7, 30, 1.0), 0.0);
    }
}
