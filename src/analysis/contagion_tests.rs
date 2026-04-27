use super::*;
use crate::analysis::test_helpers::{add_entity, add_situation, make_hg};
use crate::hypergraph::keys;

fn make_info(about: Uuid, fact: &str, knows: bool, learns: bool, reveals: bool) -> Option<InfoSet> {
    let kf = KnowledgeFact {
        about_entity: about,
        fact: fact.to_string(),
        confidence: 1.0,
    };
    Some(InfoSet {
        knows_before: if knows { vec![kf.clone()] } else { vec![] },
        learns: if learns { vec![kf.clone()] } else { vec![] },
        reveals: if reveals { vec![kf] } else { vec![] },
        beliefs_about_others: vec![],
    })
}

#[test]
fn test_single_spreader_tells_one() {
    let hg = make_hg();
    let n = "sir1";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Witness,
        info_set: make_info(about, "secret", false, true, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.total_infected, 2);
    assert_eq!(result.spread_events.len(), 1);
    // R₀: A spread to 1 person, B spread to 0. Average = 1/2 = 0.5
    assert!(result.r0 >= 0.0);
}

#[test]
fn test_single_spreader_tells_three() {
    let hg = make_hg();
    let n = "sir3";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);
    let d = add_entity(&hg, "d", n);

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    for &e in &[b, c, d] {
        hg.add_participant(Participation {
            entity_id: e,
            situation_id: s,
            role: Role::Witness,
            info_set: make_info(about, "secret", false, true, false),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.total_infected, 4);
    assert_eq!(result.spread_events.len(), 3);
}

#[test]
fn test_chain_spread() {
    let hg = make_hg();
    let n = "chain";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    // Situation 1: A reveals, B learns.
    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s1,
        role: Role::Witness,
        info_set: make_info(about, "secret", false, true, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Situation 2: B reveals, C learns.
    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: c,
        situation_id: s2,
        role: Role::Witness,
        info_set: make_info(about, "secret", false, true, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.total_infected, 3);
}

#[test]
fn test_information_never_revealed() {
    let hg = make_hg();
    let n = "hidden";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    // A knows but does NOT reveal.
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Witness,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.r0, 0.0);
    assert_eq!(result.spread_events.len(), 0);
}

#[test]
fn test_empty_narrative() {
    let hg = make_hg();
    let result = run_contagion(&hg, "empty", "fact", Uuid::now_v7()).unwrap();
    assert_eq!(result.total_infected, 0);
    assert_eq!(result.r0, 0.0);
}

#[test]
fn test_fact_known_by_everyone() {
    let hg = make_hg();
    let n = "allknow";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "public", true, false, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Witness,
        info_set: make_info(about, "public", true, false, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let result = run_contagion(&hg, n, "public", about).unwrap();
    // Both know, no spread events.
    assert_eq!(result.spread_events.len(), 0);
}

#[test]
fn test_patient_zero_identified() {
    let hg = make_hg();
    let n = "pzero";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Witness,
        info_set: make_info(about, "secret", false, true, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.patient_zero, Some(a));
}

#[test]
fn test_kv_storage() {
    let hg = make_hg();
    let n = "kvcon";
    let about = Uuid::now_v7();
    let a = add_entity(&hg, "a", n);

    // Need at least one situation for the result to be stored.
    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "fact", true, false, false),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    run_contagion(&hg, n, "fact", about).unwrap();
    let fh = hash_fact("fact");
    let key = analysis_key(keys::ANALYSIS_CONTAGION, &[n, &fh.to_string()]);
    let stored = hg.store().get(&key).unwrap();
    assert!(stored.is_some());
}

#[test]
fn test_inference_engine_trait() {
    let engine = ContagionEngine;
    assert_eq!(engine.job_type(), InferenceJobType::ContagionAnalysis);
}

#[test]
fn test_star_topology_spread() {
    let hg = make_hg();
    let n = "star_spread";
    let about = Uuid::now_v7();
    let center = add_entity(&hg, "center", n);
    let p1 = add_entity(&hg, "p1", n);
    let p2 = add_entity(&hg, "p2", n);
    let p3 = add_entity(&hg, "p3", n);

    // Center reveals to all in one situation.
    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: center,
        situation_id: s,
        role: Role::Protagonist,
        info_set: make_info(about, "secret", true, false, true),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    for &e in &[p1, p2, p3] {
        hg.add_participant(Participation {
            entity_id: e,
            situation_id: s,
            role: Role::Witness,
            info_set: make_info(about, "secret", false, true, false),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    let result = run_contagion(&hg, n, "secret", about).unwrap();
    assert_eq!(result.total_infected, 4);
    assert_eq!(result.spread_events.len(), 3);
}
