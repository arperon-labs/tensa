use crate::error::{Result, TensaError};
use crate::types::{AllenInterval, AllenRelation};

/// Compute the Allen relation between two intervals from their timestamps.
/// Both intervals must have start and end times.
pub fn relation_between(a: &AllenInterval, b: &AllenInterval) -> Result<AllenRelation> {
    let a_start = a
        .start
        .ok_or_else(|| TensaError::InvalidInterval("Interval A missing start".into()))?;
    let a_end = a
        .end
        .ok_or_else(|| TensaError::InvalidInterval("Interval A missing end".into()))?;
    let b_start = b
        .start
        .ok_or_else(|| TensaError::InvalidInterval("Interval B missing start".into()))?;
    let b_end = b
        .end
        .ok_or_else(|| TensaError::InvalidInterval("Interval B missing end".into()))?;

    if a_end < b_start {
        Ok(AllenRelation::Before)
    } else if a_end == b_start {
        Ok(AllenRelation::Meets)
    } else if a_start < b_start && a_end > b_start && a_end < b_end {
        Ok(AllenRelation::Overlaps)
    } else if a_start == b_start && a_end < b_end {
        Ok(AllenRelation::Starts)
    } else if a_start > b_start && a_end < b_end {
        Ok(AllenRelation::During)
    } else if a_start > b_start && a_end == b_end {
        Ok(AllenRelation::Finishes)
    } else if a_start == b_start && a_end == b_end {
        Ok(AllenRelation::Equals)
    } else if a_start == b_start && a_end > b_end {
        Ok(AllenRelation::StartedBy)
    } else if a_start < b_start && a_end == b_end {
        Ok(AllenRelation::FinishedBy)
    } else if a_start < b_start && a_end > b_end {
        Ok(AllenRelation::Contains)
    } else if a_start > b_start && a_start < b_end && a_end > b_end {
        Ok(AllenRelation::OverlappedBy)
    } else if a_start == b_end {
        Ok(AllenRelation::MetBy)
    } else {
        // a_start > b_end
        Ok(AllenRelation::After)
    }
}

/// Compose two Allen relations: if A r1 B and B r2 C, what are the possible relations for A ? C?
/// Returns a set of possible relations.
pub fn compose(r1: AllenRelation, r2: AllenRelation) -> Vec<AllenRelation> {
    let idx1 = relation_index(r1);
    let idx2 = relation_index(r2);
    COMPOSITION_TABLE[idx1][idx2].to_vec()
}

fn relation_index(r: AllenRelation) -> usize {
    use AllenRelation::*;
    match r {
        Before => 0,
        Meets => 1,
        Overlaps => 2,
        Starts => 3,
        During => 4,
        Finishes => 5,
        Equals => 6,
        FinishedBy => 7,
        Contains => 8,
        StartedBy => 9,
        OverlappedBy => 10,
        MetBy => 11,
        After => 12,
    }
}

// Allen's 13x13 composition table.
// Each cell is a slice of AllenRelations that are possible when composing row relation with column relation.
// Order: Before, Meets, Overlaps, Starts, During, Finishes, Equals, FinishedBy, Contains, StartedBy, OverlappedBy, MetBy, After
//
// Abbreviations used in comments:
// b=Before, m=Meets, o=Overlaps, s=Starts, d=During, f=Finishes,
// eq=Equals, fb=FinishedBy, cn=Contains, sb=StartedBy, ob=OverlappedBy, mb=MetBy, a=After
// ALL = all 13 relations

use AllenRelation::*;

pub(crate) const ALL: &[AllenRelation] = &[
    Before,
    Meets,
    Overlaps,
    Starts,
    During,
    Finishes,
    Equals,
    FinishedBy,
    Contains,
    StartedBy,
    OverlappedBy,
    MetBy,
    After,
];

// The full composition table, following the standard Allen 1983 definitions.
// Row = first relation, Column = second relation
static COMPOSITION_TABLE: [[&[AllenRelation]; 13]; 13] = [
    // Before ∘ X
    [
        /* b∘b  */ &[Before],
        /* b∘m  */ &[Before],
        /* b∘o  */ &[Before],
        /* b∘s  */ &[Before],
        /* b∘d  */ &[Before, Meets, Overlaps, Starts, During],
        /* b∘f  */ &[Before, Meets, Overlaps, Starts, During],
        /* b∘eq */ &[Before],
        /* b∘fb */ &[Before],
        /* b∘cn */ &[Before],
        /* b∘sb */ &[Before],
        /* b∘ob */ &[Before, Meets, Overlaps, Starts, During],
        /* b∘mb */ &[Before, Meets, Overlaps, Starts, During],
        /* b∘a  */ ALL,
    ],
    // Meets ∘ X
    [
        /* m∘b  */ &[Before],
        /* m∘m  */ &[Before],
        /* m∘o  */ &[Before],
        /* m∘s  */ &[Meets],
        /* m∘d  */ &[Overlaps, Starts, During],
        /* m∘f  */ &[Overlaps, Starts, During],
        /* m∘eq */ &[Meets],
        /* m∘fb */ &[Before],
        /* m∘cn */ &[Before],
        /* m∘sb */ &[Meets],
        /* m∘ob */ &[Overlaps, Starts, During],
        /* m∘mb */ &[Overlaps, Starts, During],
        /* m∘a  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
    ],
    // Overlaps ∘ X
    [
        /* o∘b  */ &[Before],
        /* o∘m  */ &[Before],
        /* o∘o  */ &[Before, Meets, Overlaps],
        /* o∘s  */ &[Overlaps],
        /* o∘d  */ &[Overlaps, Starts, During],
        /* o∘f  */ &[Overlaps, Starts, During],
        /* o∘eq */ &[Overlaps],
        /* o∘fb */ &[Before, Meets, Overlaps],
        /* o∘cn */ &[Before, Meets, Overlaps],
        /* o∘sb */ &[Overlaps],
        /* o∘ob */
        &[
            Before,
            Meets,
            Overlaps,
            FinishedBy,
            Contains,
            Overlaps,
            Starts,
            During,
            Equals,
            OverlappedBy,
        ],
        /* o∘mb */ &[Overlaps, Starts, During, FinishedBy, Equals],
        /* o∘a  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
    ],
    // Starts ∘ X
    [
        /* s∘b  */ &[Before],
        /* s∘m  */ &[Before],
        /* s∘o  */ &[Before, Meets, Overlaps],
        /* s∘s  */ &[Starts],
        /* s∘d  */ &[During],
        /* s∘f  */ &[During],
        /* s∘eq */ &[Starts],
        /* s∘fb */ &[Before, Meets, Overlaps],
        /* s∘cn */ &[Before, Meets, Overlaps],
        /* s∘sb */ &[Starts, StartedBy, Equals],
        /* s∘ob */ &[During, Finishes, OverlappedBy],
        /* s∘mb */ &[During, Finishes, OverlappedBy],
        /* s∘a  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
    ],
    // During ∘ X
    [
        /* d∘b  */ &[Before],
        /* d∘m  */ &[Before],
        /* d∘o  */ &[Before, Meets, Overlaps, Starts, During],
        /* d∘s  */ &[During],
        /* d∘d  */ &[During],
        /* d∘f  */ &[During],
        /* d∘eq */ &[During],
        /* d∘fb */ &[Before, Meets, Overlaps, Starts, During],
        /* d∘cn */ &[Before, Meets, Overlaps, Starts, During],
        /* d∘sb */ &[During],
        /* d∘ob */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* d∘mb */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* d∘a  */ ALL,
    ],
    // Finishes ∘ X
    [
        /* f∘b  */ &[Before],
        /* f∘m  */ &[Meets],
        /* f∘o  */ &[Overlaps, Starts, During],
        /* f∘s  */ &[During],
        /* f∘d  */ &[During],
        /* f∘f  */ &[During],
        /* f∘eq */ &[Finishes],
        /* f∘fb */ &[Overlaps, Starts, During, FinishedBy, Equals],
        /* f∘cn */ &[Before, Meets, Overlaps, Starts, During],
        /* f∘sb */ &[During, Finishes, OverlappedBy],
        /* f∘ob */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* f∘mb */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* f∘a  */ ALL,
    ],
    // Equals ∘ X
    [
        /* eq∘b  */ &[Before],
        /* eq∘m  */ &[Meets],
        /* eq∘o  */ &[Overlaps],
        /* eq∘s  */ &[Starts],
        /* eq∘d  */ &[During],
        /* eq∘f  */ &[Finishes],
        /* eq∘eq */ &[Equals],
        /* eq∘fb */ &[FinishedBy],
        /* eq∘cn */ &[Contains],
        /* eq∘sb */ &[StartedBy],
        /* eq∘ob */ &[OverlappedBy],
        /* eq∘mb */ &[MetBy],
        /* eq∘a  */ &[After],
    ],
    // FinishedBy ∘ X
    [
        /* fb∘b  */ &[Before],
        /* fb∘m  */ &[Meets],
        /* fb∘o  */ &[Overlaps],
        /* fb∘s  */ &[Overlaps],
        /* fb∘d  */ &[Overlaps, Starts, During],
        /* fb∘f  */ &[Finishes, FinishedBy, Equals],
        /* fb∘eq */ &[FinishedBy],
        /* fb∘fb */ &[FinishedBy],
        /* fb∘cn */ &[Contains],
        /* fb∘sb */
        &[
            Contains,
            FinishedBy,
            Overlaps,
            StartedBy,
            OverlappedBy,
            Equals,
        ],
        /* fb∘ob */ &[OverlappedBy],
        /* fb∘mb */ &[MetBy],
        /* fb∘a  */ &[After],
    ],
    // Contains ∘ X
    [
        /* cn∘b  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
        /* cn∘m  */ &[Overlaps, FinishedBy, Contains],
        /* cn∘o  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
        /* cn∘s  */ &[Overlaps, FinishedBy, Contains],
        /* cn∘d  */ ALL,
        /* cn∘f  */
        &[
            Overlaps,
            Starts,
            During,
            FinishedBy,
            Contains,
            StartedBy,
            OverlappedBy,
            Equals,
        ],
        /* cn∘eq */ &[Contains],
        /* cn∘fb */ &[Contains, FinishedBy, Overlaps],
        /* cn∘cn */ &[Contains],
        /* cn∘sb */
        &[
            Contains,
            FinishedBy,
            Overlaps,
            StartedBy,
            OverlappedBy,
            Equals,
        ],
        /* cn∘ob */ &[OverlappedBy, MetBy, After, Contains, StartedBy],
        /* cn∘mb */ &[OverlappedBy, MetBy, After],
        /* cn∘a  */ &[OverlappedBy, MetBy, After],
    ],
    // StartedBy ∘ X
    [
        /* sb∘b  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
        /* sb∘m  */ &[Overlaps, FinishedBy, Contains],
        /* sb∘o  */ &[Overlaps, FinishedBy, Contains],
        /* sb∘s  */ &[Starts, StartedBy, Equals],
        /* sb∘d  */ &[During, Finishes, OverlappedBy],
        /* sb∘f  */ &[During, Finishes, OverlappedBy],
        /* sb∘eq */ &[StartedBy],
        /* sb∘fb */
        &[
            Contains,
            FinishedBy,
            Overlaps,
            StartedBy,
            OverlappedBy,
            Equals,
        ],
        /* sb∘cn */ &[Contains],
        /* sb∘sb */ &[StartedBy],
        /* sb∘ob */ &[OverlappedBy],
        /* sb∘mb */ &[MetBy],
        /* sb∘a  */ &[OverlappedBy, MetBy, After],
    ],
    // OverlappedBy ∘ X
    [
        /* ob∘b  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
        /* ob∘m  */ &[Overlaps, FinishedBy, Contains],
        /* ob∘o  */
        &[
            Before,
            Meets,
            Overlaps,
            FinishedBy,
            Contains,
            Overlaps,
            Starts,
            During,
            Equals,
            OverlappedBy,
        ],
        /* ob∘s  */ &[During, Finishes, OverlappedBy],
        /* ob∘d  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* ob∘f  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* ob∘eq */ &[OverlappedBy],
        /* ob∘fb */ &[OverlappedBy, MetBy, After],
        /* ob∘cn */ &[OverlappedBy, MetBy, After],
        /* ob∘sb */ &[OverlappedBy],
        /* ob∘ob */
        &[
            Before,
            Meets,
            Overlaps,
            FinishedBy,
            Contains,
            Overlaps,
            Starts,
            During,
            Equals,
            OverlappedBy,
        ],
        /* ob∘mb */ &[Overlaps, Starts, During, FinishedBy, Equals],
        /* ob∘a  */ &[OverlappedBy, MetBy, After],
    ],
    // MetBy ∘ X
    [
        /* mb∘b  */ &[Before, Meets, Overlaps, FinishedBy, Contains],
        /* mb∘m  */ &[Overlaps, FinishedBy, Contains],
        /* mb∘o  */ &[Overlaps, FinishedBy, Contains],
        /* mb∘s  */ &[During, Finishes, OverlappedBy],
        /* mb∘d  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* mb∘f  */ &[MetBy],
        /* mb∘eq */ &[MetBy],
        /* mb∘fb */ &[OverlappedBy, MetBy, After],
        /* mb∘cn */ &[OverlappedBy, MetBy, After],
        /* mb∘sb */ &[MetBy],
        /* mb∘ob */ &[Overlaps, FinishedBy, Contains],
        /* mb∘mb */ &[After],
        /* mb∘a  */ &[After],
    ],
    // After ∘ X
    [
        /* a∘b  */ ALL,
        /* a∘m  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* a∘o  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* a∘s  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* a∘d  */ &[During, Finishes, OverlappedBy, MetBy, After],
        /* a∘f  */ &[After],
        /* a∘eq */ &[After],
        /* a∘fb */ &[OverlappedBy, MetBy, After],
        /* a∘cn */ &[OverlappedBy, MetBy, After],
        /* a∘sb */ &[OverlappedBy, MetBy, After],
        /* a∘ob */ &[OverlappedBy, MetBy, After],
        /* a∘mb */ &[After],
        /* a∘a  */ &[After],
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeGranularity;
    use chrono::{DateTime, Duration, Utc};

    fn interval(start_offset: i64, end_offset: i64) -> AllenInterval {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        AllenInterval {
            start: Some(base + Duration::hours(start_offset)),
            end: Some(base + Duration::hours(end_offset)),
            granularity: TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        }
    }

    #[test]
    fn test_allen_before() {
        let a = interval(0, 1);
        let b = interval(2, 3);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Before);
    }

    #[test]
    fn test_allen_after() {
        let a = interval(4, 5);
        let b = interval(1, 2);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::After);
    }

    #[test]
    fn test_allen_meets() {
        let a = interval(0, 2);
        let b = interval(2, 4);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Meets);
    }

    #[test]
    fn test_allen_met_by() {
        let a = interval(2, 4);
        let b = interval(0, 2);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::MetBy);
    }

    #[test]
    fn test_allen_overlaps() {
        let a = interval(0, 3);
        let b = interval(2, 5);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Overlaps);
    }

    #[test]
    fn test_allen_overlapped_by() {
        let a = interval(2, 5);
        let b = interval(0, 3);
        assert_eq!(
            relation_between(&a, &b).unwrap(),
            AllenRelation::OverlappedBy
        );
    }

    #[test]
    fn test_allen_during() {
        let a = interval(2, 3);
        let b = interval(1, 4);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::During);
    }

    #[test]
    fn test_allen_contains() {
        let a = interval(1, 4);
        let b = interval(2, 3);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Contains);
    }

    #[test]
    fn test_allen_starts() {
        let a = interval(1, 3);
        let b = interval(1, 5);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Starts);
    }

    #[test]
    fn test_allen_started_by() {
        let a = interval(1, 5);
        let b = interval(1, 3);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::StartedBy);
    }

    #[test]
    fn test_allen_finishes() {
        let a = interval(3, 5);
        let b = interval(1, 5);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Finishes);
    }

    #[test]
    fn test_allen_finished_by() {
        let a = interval(1, 5);
        let b = interval(3, 5);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::FinishedBy);
    }

    #[test]
    fn test_allen_equals() {
        let a = interval(1, 5);
        let b = interval(1, 5);
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Equals);
    }

    #[test]
    fn test_allen_inverse_symmetry() {
        let pairs = [
            (AllenRelation::Before, AllenRelation::After),
            (AllenRelation::Meets, AllenRelation::MetBy),
            (AllenRelation::Overlaps, AllenRelation::OverlappedBy),
            (AllenRelation::Starts, AllenRelation::StartedBy),
            (AllenRelation::During, AllenRelation::Contains),
            (AllenRelation::Finishes, AllenRelation::FinishedBy),
        ];
        for (r, inv) in &pairs {
            assert_eq!(r.inverse(), *inv);
            assert_eq!(inv.inverse(), *r);
        }
        assert_eq!(AllenRelation::Equals.inverse(), AllenRelation::Equals);
    }

    #[test]
    fn test_allen_missing_timestamps() {
        let a = AllenInterval {
            start: None,
            end: Some(Utc::now()),
            granularity: TimeGranularity::Unknown,
            relations: vec![],
            fuzzy_endpoints: None,
        };
        let b = interval(0, 1);
        assert!(relation_between(&a, &b).is_err());
    }

    #[test]
    fn test_composition_before_before() {
        let result = compose(AllenRelation::Before, AllenRelation::Before);
        assert_eq!(result, vec![AllenRelation::Before]);
    }

    #[test]
    fn test_composition_before_meets() {
        let result = compose(AllenRelation::Before, AllenRelation::Meets);
        assert_eq!(result, vec![AllenRelation::Before]);
    }

    #[test]
    fn test_composition_meets_before() {
        let result = compose(AllenRelation::Meets, AllenRelation::Before);
        assert_eq!(result, vec![AllenRelation::Before]);
    }

    #[test]
    fn test_composition_equals_any() {
        // Equals composed with anything yields that same relation
        let relations = [
            Before,
            Meets,
            Overlaps,
            Starts,
            During,
            Finishes,
            Equals,
            FinishedBy,
            Contains,
            StartedBy,
            OverlappedBy,
            MetBy,
            After,
        ];
        for r in &relations {
            let result = compose(AllenRelation::Equals, *r);
            assert_eq!(result, vec![*r], "Equals ∘ {:?} should be [{:?}]", r, r);
        }
    }

    #[test]
    fn test_composition_before_after_is_all() {
        let result = compose(AllenRelation::Before, AllenRelation::After);
        assert_eq!(result.len(), 13);
    }

    #[test]
    fn test_composition_after_after() {
        let result = compose(AllenRelation::After, AllenRelation::After);
        assert_eq!(result, vec![AllenRelation::After]);
    }

    #[test]
    fn test_composition_table_completeness() {
        // All 169 entries should be non-empty
        let relations = [
            Before,
            Meets,
            Overlaps,
            Starts,
            During,
            Finishes,
            Equals,
            FinishedBy,
            Contains,
            StartedBy,
            OverlappedBy,
            MetBy,
            After,
        ];
        for r1 in &relations {
            for r2 in &relations {
                let result = compose(*r1, *r2);
                assert!(
                    !result.is_empty(),
                    "compose({:?}, {:?}) should not be empty",
                    r1,
                    r2
                );
            }
        }
    }

    #[test]
    fn test_composition_overlaps_during() {
        let result = compose(AllenRelation::Overlaps, AllenRelation::During);
        assert!(result.contains(&AllenRelation::Overlaps));
        assert!(result.contains(&AllenRelation::Starts));
        assert!(result.contains(&AllenRelation::During));
    }

    #[test]
    fn test_interval_with_approximate_granularity() {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let a = AllenInterval {
            start: Some(base),
            end: Some(base + Duration::hours(1)),
            granularity: TimeGranularity::Approximate,
            relations: vec![],
            fuzzy_endpoints: None,
        };
        let b = AllenInterval {
            start: Some(base + Duration::hours(2)),
            end: Some(base + Duration::hours(3)),
            granularity: TimeGranularity::Approximate,
            relations: vec![],
            fuzzy_endpoints: None,
        };
        // Should still compute correctly based on timestamps
        assert_eq!(relation_between(&a, &b).unwrap(), AllenRelation::Before);
    }
}
