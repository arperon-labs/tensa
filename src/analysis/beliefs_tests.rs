use super::*;
use crate::analysis::analysis_key;
use crate::analysis::test_helpers::{add_entity, add_situation, make_hg};
use crate::hypergraph::keys;

fn kf(about: Uuid, fact: &str) -> KnowledgeFact {
    KnowledgeFact {
        about_entity: about,
        fact: fact.to_string(),
        confidence: 1.0,
    }
}

// ─── The Spy Scenario ───────────────────────────────────
// A tells B a secret. B tells C (without A present). A meets C.
// A's model of C's knowledge should NOT include the secret.

#[test]
fn test_spy_scenario() {
    let hg = make_hg();
    let n = "spy";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let c = add_entity(&hg, "C", n);
    let target = Uuid::now_v7(); // the entity the secret is about

    // Situation 1: A reveals secret to B.
    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "is a spy")],
            learns: vec![],
            reveals: vec![kf(target, "is a spy")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s1,
        role: Role::Confidant,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "is a spy")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Situation 2: B reveals secret to C (A NOT present).
    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Informant,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "is a spy")],
            learns: vec![],
            reveals: vec![kf(target, "is a spy")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: c,
        situation_id: s2,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "is a spy")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Situation 3: A meets C (no secrets revealed).
    let s3 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s3,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "is a spy")],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: c,
        situation_id: s3,
        role: Role::Witness,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "is a spy")],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();

    // A's model of C at s3: A was NOT present when B told C,
    // so A should NOT know that C knows the secret.
    let gap = analysis
        .gaps
        .iter()
        .find(|g| g.entity_a == a && g.entity_b == c && g.at_situation == s3);
    assert!(gap.is_some(), "Expected a belief gap for A about C at s3");
    let gap = gap.unwrap();
    let secret_key = FactKey {
        about_entity: target,
        fact: "is a spy".to_string(),
    };
    assert!(
        gap.unknown_to_a.contains(&secret_key),
        "A should not know that C knows the secret"
    );
}

// ─── Always Co-Present ──────────────────────────────────

#[test]
fn test_always_copresent_synchronized() {
    let hg = make_hg();
    let n = "copresent";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![],
            reveals: vec![kf(target, "fact1")],
            beliefs_about_others: vec![],
        }),
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
            learns: vec![kf(target, "fact1")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();
    // A's model of B should include fact1 (A was present when B learned it).
    let a_about_b = analysis
        .snapshots
        .iter()
        .find(|s| s.entity_a == a && s.entity_b == b)
        .unwrap();
    let fact_key = FactKey {
        about_entity: target,
        fact: "fact1".to_string(),
    };
    assert!(a_about_b.believed_knowledge.contains(&fact_key));
    assert!(a_about_b.actual_knowledge.contains(&fact_key));
}

// ─── No Shared Situations ───────────────────────────────

#[test]
fn test_no_shared_situations() {
    let hg = make_hg();
    let n = "noshare";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);

    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();
    // No co-presence → no snapshots recorded.
    assert!(analysis.snapshots.is_empty());
}

// ─── Chain: A→B→C→D ────────────────────────────────────

#[test]
fn test_chain_a_model_of_d_empty() {
    let hg = make_hg();
    let n = "chain";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let c = add_entity(&hg, "C", n);
    let d = add_entity(&hg, "D", n);
    let target = Uuid::now_v7();

    // A reveals to B.
    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret")],
            learns: vec![],
            reveals: vec![kf(target, "secret")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s1,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "secret")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // B reveals to C (A not present).
    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Informant,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret")],
            learns: vec![],
            reveals: vec![kf(target, "secret")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: c,
        situation_id: s2,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "secret")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // C reveals to D (A not present).
    let s3 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: c,
        situation_id: s3,
        role: Role::Informant,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret")],
            learns: vec![],
            reveals: vec![kf(target, "secret")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: d,
        situation_id: s3,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "secret")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();

    // A was never co-present with D, so no snapshot for A about D.
    let a_about_d = analysis
        .snapshots
        .iter()
        .find(|s| s.entity_a == a && s.entity_b == d);
    assert!(
        a_about_d.is_none(),
        "A has no model of D (never co-present)"
    );
}

// ─── Multiple Secrets ───────────────────────────────────

#[test]
fn test_multiple_secrets_partial_reveal() {
    let hg = make_hg();
    let n = "multi";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    // A reveals only secret1 to B (knows both secret1 and secret2).
    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret1"), kf(target, "secret2")],
            learns: vec![],
            reveals: vec![kf(target, "secret1")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "secret1")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();

    // A's model of B should include secret1 but NOT secret2.
    let a_about_b = analysis
        .snapshots
        .iter()
        .find(|s| s.entity_a == a && s.entity_b == b)
        .unwrap();

    let key1 = FactKey {
        about_entity: target,
        fact: "secret1".to_string(),
    };
    let key2 = FactKey {
        about_entity: target,
        fact: "secret2".to_string(),
    };
    assert!(a_about_b.believed_knowledge.contains(&key1));
    assert!(!a_about_b.believed_knowledge.contains(&key2));
}

// ─── Belief Update on Re-meeting ────────────────────────

#[test]
fn test_belief_update_on_remeet() {
    let hg = make_hg();
    let n = "remeet";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    // S1: A and B meet, no knowledge exchange.
    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s1,
        role: Role::Protagonist,
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
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s1,
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

    // S2: A and B meet again, A reveals a fact.
    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s2,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "new_info")],
            learns: vec![],
            reveals: vec![kf(target, "new_info")],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Recipient,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "new_info")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();

    // A's model of B at s1: empty knowledge.
    let a_b_s1 = analysis
        .snapshots
        .iter()
        .find(|s| s.entity_a == a && s.entity_b == b && s.situation_id == s1)
        .unwrap();
    assert!(a_b_s1.believed_knowledge.is_empty());

    // A's model of B at s2: includes new_info.
    let a_b_s2 = analysis
        .snapshots
        .iter()
        .find(|s| s.entity_a == a && s.entity_b == b && s.situation_id == s2)
        .unwrap();
    let key = FactKey {
        about_entity: target,
        fact: "new_info".to_string(),
    };
    assert!(a_b_s2.believed_knowledge.contains(&key));
}

// ─── Edge Cases ─────────────────────────────────────────

#[test]
fn test_empty_narrative() {
    let hg = make_hg();
    let analysis = run_beliefs(&hg, "empty").unwrap();
    assert!(analysis.snapshots.is_empty());
    assert!(analysis.gaps.is_empty());
}

#[test]
fn test_kv_storage() {
    let hg = make_hg();
    let n = "kvbelief";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);

    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s,
        role: Role::Witness,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    run_beliefs(&hg, n).unwrap();

    let key = analysis_key(
        keys::ANALYSIS_BELIEF,
        &[n, &a.to_string(), &b.to_string(), &s.to_string()],
    );
    let stored = hg.store().get(&key).unwrap();
    assert!(stored.is_some());
}

#[test]
fn test_belief_gap_detection() {
    // B knows something A doesn't know B knows.
    let hg = make_hg();
    let n = "gaptest";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    // B learns a fact alone (A not present).
    let s1 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![kf(target, "hidden")],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // A meets B (no reveals).
    let s2 = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s2,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: b,
        situation_id: s2,
        role: Role::Witness,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "hidden")],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let analysis = run_beliefs(&hg, n).unwrap();

    // A should have a gap: B knows "hidden" but A doesn't know B knows it.
    let gap = analysis
        .gaps
        .iter()
        .find(|g| g.entity_a == a && g.entity_b == b && g.at_situation == s2);
    assert!(gap.is_some());
    let gap = gap.unwrap();
    let key = FactKey {
        about_entity: target,
        fact: "hidden".to_string(),
    };
    assert!(gap.unknown_to_a.contains(&key));
}

// ─── SymbolicToM Tests ─────────────────────────────────

#[test]
fn test_symbolic_tom_parses() {
    let hg = make_hg();
    let n = "tom_parse";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    // A's first situation has beliefs_about_others populated.
    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret")],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![RecursiveBelief {
                about_entity: b,
                believed_knowledge: vec![kf(target, "is an ally")],
                confidence: 0.8,
                last_updated_at: s,
            }],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let tom = parse_symbolic_tom(&hg, n).unwrap();

    // A's initial knowledge should include "secret".
    let a_knowledge = tom.initial_knowledge.get(&a).unwrap();
    assert!(a_knowledge.contains(&FactKey {
        about_entity: target,
        fact: "secret".to_string()
    }));

    // A's initial belief about B should include "is an ally".
    let a_about_b = tom.initial_beliefs.get(&(a, b)).unwrap();
    assert!(a_about_b.contains(&FactKey {
        about_entity: target,
        fact: "is an ally".to_string()
    }));
}

#[test]
fn test_symbolic_tom_pipeline() {
    // SymbolicToM beliefs should be present in the pipeline output.
    let hg = make_hg();
    let n = "tom_pipeline";
    let a = add_entity(&hg, "A", n);
    let b = add_entity(&hg, "B", n);
    let target = Uuid::now_v7();

    // A has a pre-existing belief that B knows a secret (from LLM/text).
    let s = add_situation(&hg, n);
    hg.add_participant(Participation {
        entity_id: a,
        situation_id: s,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![kf(target, "secret")],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![RecursiveBelief {
                about_entity: b,
                believed_knowledge: vec![kf(target, "secret")],
                confidence: 0.9,
                last_updated_at: s,
            }],
        }),
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

    let analysis = run_beliefs(&hg, n).unwrap();

    // A's model of B at situation s should include the ToM-seeded belief.
    let a_about_b = analysis
        .snapshots
        .iter()
        .find(|snap| snap.entity_a == a && snap.entity_b == b && snap.situation_id == s)
        .unwrap();
    let secret_key = FactKey {
        about_entity: target,
        fact: "secret".to_string(),
    };
    assert!(
        a_about_b.believed_knowledge.contains(&secret_key),
        "ToM-seeded belief should appear in pipeline output"
    );

    // B doesn't actually know the secret → gap should exist.
    let gap = analysis
        .gaps
        .iter()
        .find(|g| g.entity_a == a && g.entity_b == b && g.at_situation == s);
    assert!(gap.is_some(), "False belief from ToM should produce a gap");
    let gap = gap.unwrap();
    assert!(
        gap.false_beliefs.contains(&secret_key),
        "A falsely believes B knows the secret"
    );
}

#[test]
fn test_inference_engine_trait() {
    let engine = BeliefEngine;
    assert_eq!(engine.job_type(), InferenceJobType::BeliefModeling);
}
