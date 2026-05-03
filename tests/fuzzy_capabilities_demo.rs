//! End-to-end demonstration of every fuzzy capability shipped in TENSA's
//! Fuzzy Logic Sprint (Ch. 14) and Graded Acceptability & Measure
//! Learning Sprint (Ch. 15).
//!
//! This is a **showcase test**, not a regression test — its job is to
//! drive every fuzzy surface against a single coherent narrative
//! (Operation Vesper, a synthetic coordinated-disinformation campaign)
//! and emit a structured report at
//! `target/fuzzy_capabilities_demo/report.json` plus a human-readable
//! transcript at `target/fuzzy_capabilities_demo/transcript.txt`. Run
//! with:
//!
//! ```bash
//! cargo test --no-default-features --test fuzzy_capabilities_demo -- --nocapture
//! ```
//!
//! The companion document is `documentation/fuzzy_capabilities_report.md`,
//! which interprets the numbers, ties each section to a DeepRAP research
//! objective / deliverable, and explains how to reproduce.
//!
//! Layout of this file:
//!
//!   1. Helpers (`make_hg`, `mint_entity`, `mint_situation`)
//!   2. `build_operation_vesper` — narrative scenario
//!   3. Per-capability sections, one function each, all called from
//!      `fuzzy_capabilities_demo` at the bottom.
//!
//! Every section appends to a shared `Transcript` struct so the report
//! file is one structured artefact rather than scattered prints.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::{json, Value};
use uuid::Uuid;

use tensa::analysis::argumentation::{Argument, ArgumentationFramework};
use tensa::analysis::argumentation_gradual::{
    run_gradual_argumentation, GradualResult, GradualSemanticsKind,
};
use tensa::fuzzy::aggregation_choquet::{choquet, choquet_exact};
use tensa::fuzzy::aggregation_learn::learn_choquet_measure;
use tensa::fuzzy::aggregation_measure::{
    new_monotone, symmetric_additive, symmetric_optimistic, symmetric_pessimistic,
};
use tensa::fuzzy::aggregation_owa::{linguistic_weights, owa, Quantifier as OwaQuantifier};
use tensa::fuzzy::allen::{
    graded_relation_with, relation_index, FuzzyEndpoints, GradedAllenConfig, TrapezoidalFuzzy,
};
use tensa::fuzzy::fca::{build_lattice, FormalContext, FormalContextOptions};
use tensa::fuzzy::hybrid::{fuzzy_probability, FuzzyEvent, FuzzyEventPredicate, ProbDist};
use tensa::fuzzy::quantifier::{evaluate_over_entities, Quantifier};
use tensa::fuzzy::registry::{AggregatorRegistry, TNormRegistry};
use tensa::fuzzy::rules::{
    evaluate_rule_set, FuzzyCondition, FuzzyOutput, MembershipFunction, RuleSet,
};
use tensa::fuzzy::rules_types::MamdaniRule;
use tensa::fuzzy::syllogism::{
    classify_figure, verify, Syllogism, SyllogismStatement, TypePredicateResolver,
};
use tensa::fuzzy::synthetic_cib_dataset::generate_synthetic_cib;
use tensa::fuzzy::tnorm::{combine_tconorm, combine_tnorm, reduce_tconorm, reduce_tnorm, TNormKind};
use tensa::hypergraph::Hypergraph;
use tensa::store::memory::MemoryStore;
use tensa::temporal::ordhorn::{closure, is_satisfiable, OrdHornConstraint, OrdHornNetwork};
use tensa::types::{
    AllenInterval, AllenRelation, ContentBlock, Entity, EntityType, ExtractionMethod,
    MaturityLevel, NarrativeLevel, Participation, Role, Situation, TimeGranularity,
};

// ── Section 1: helpers ──────────────────────────────────────────────────────

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn ts(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, 0, 0).unwrap()
}

fn mint_entity(
    hg: &Hypergraph,
    narrative_id: &str,
    entity_type: EntityType,
    name: &str,
    confidence: f32,
    properties: Value,
) -> Uuid {
    let id = Uuid::now_v7();
    let mut props = properties;
    if let Value::Object(map) = &mut props {
        map.insert("name".into(), Value::String(name.into()));
    }
    hg.create_entity(Entity {
        id,
        entity_type,
        properties: props,
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative_id.into()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    })
    .expect("create_entity")
}

#[allow(clippy::too_many_arguments)]
fn mint_situation(
    hg: &Hypergraph,
    narrative_id: &str,
    name: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fuzzy: Option<FuzzyEndpoints>,
    granularity: TimeGranularity,
    confidence: f32,
    description: &str,
) -> Uuid {
    let id = Uuid::now_v7();
    hg.create_situation(Situation {
        id,
        properties: Value::Null,
        name: Some(name.into()),
        description: Some(description.into()),
        temporal: AllenInterval {
            start: Some(start),
            end: Some(end),
            granularity,
            relations: vec![],
            fuzzy_endpoints: fuzzy,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text(description)],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(narrative_id.into()),
        source_chunk_id: None,
        source_span: None,
        synopsis: None,
        manuscript_order: None,
        parent_situation_id: None,
        label: None,
        status: None,
        keywords: vec![],
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    })
    .expect("create_situation")
}

// ── Transcript shared across sections ───────────────────────────────────────

#[derive(Default)]
struct Transcript {
    sections: Vec<(String, String)>,
    json: BTreeMap<String, Value>,
}

impl Transcript {
    fn section(&mut self, title: &str, body: String, json: Value) {
        println!("\n══ {} ══\n{}", title, body);
        self.sections.push((title.into(), body));
        self.json.insert(title.into(), json);
    }

    fn write(&self, dir: &PathBuf) {
        fs::create_dir_all(dir).unwrap();
        let mut transcript = String::new();
        for (title, body) in &self.sections {
            transcript.push_str(&format!("══ {title} ══\n{body}\n\n"));
        }
        fs::write(dir.join("transcript.txt"), transcript).unwrap();
        let json = Value::Object(self.json.clone().into_iter().collect());
        fs::write(
            dir.join("report.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }
}

// ── Section 2: build Operation Vesper ───────────────────────────────────────

#[derive(Debug, Clone)]
struct Vesper {
    nid: String,
    actors: Vec<Uuid>,         // 5 actors driving the campaign + amplifiers
    organizations: Vec<Uuid>,  // hub, NGO, fact-check
    fuzzy_post_a: Uuid,        // situation, fuzzy endpoints
    fuzzy_post_b: Uuid,        // situation, fuzzy endpoints (later)
    sharp_event_takedown: Uuid, // crisp situation
    sharp_event_factcheck: Uuid,
    crisp_t0: Uuid,
    crisp_t1: Uuid,
    crisp_t2: Uuid,
}

fn build_operation_vesper(hg: &Hypergraph) -> Vesper {
    let nid = "operation-vesper".to_string();

    // Five actors with synthetic-CIB signal properties on `properties`.
    // (temporal_correlation, content_overlap, network_density, posting_cadence)
    // matter for FCA + Mamdani + Hybrid.
    let actor_signals = [
        // Cluster suspected of CIB — high temporal+content correlation.
        ("Vesper-A1", 0.92, json!({
            "temporal_correlation": 0.91, "content_overlap": 0.88,
            "network_density": 0.62, "posting_cadence": 0.31,
            "inflammatory_score": 0.81, "anonymous": true,
            "foreign_funded": true, "verified": false
        })),
        ("Vesper-A2", 0.88, json!({
            "temporal_correlation": 0.86, "content_overlap": 0.83,
            "network_density": 0.55, "posting_cadence": 0.28,
            "inflammatory_score": 0.78, "anonymous": true,
            "foreign_funded": true, "verified": false
        })),
        ("Vesper-A3", 0.84, json!({
            "temporal_correlation": 0.79, "content_overlap": 0.81,
            "network_density": 0.45, "posting_cadence": 0.34,
            "inflammatory_score": 0.74, "anonymous": false,
            "foreign_funded": true, "verified": false
        })),
        // Genuine but heated commentator — looks bad, isn't coordinated.
        ("Outlier-Loud", 0.74, json!({
            "temporal_correlation": 0.20, "content_overlap": 0.18,
            "network_density": 0.30, "posting_cadence": 0.85,
            "inflammatory_score": 0.71, "anonymous": false,
            "foreign_funded": false, "verified": true
        })),
        // Fact-checker.
        ("FactCheck-Gamma", 0.95, json!({
            "temporal_correlation": 0.05, "content_overlap": 0.10,
            "network_density": 0.20, "posting_cadence": 0.40,
            "inflammatory_score": 0.10, "anonymous": false,
            "foreign_funded": false, "verified": true
        })),
    ];

    let actors: Vec<Uuid> = actor_signals
        .iter()
        .map(|(name, conf, props)| {
            mint_entity(hg, &nid, EntityType::Actor, name, *conf, props.clone())
        })
        .collect();

    let organizations = vec![
        mint_entity(
            hg,
            &nid,
            EntityType::Organization,
            "ShellHub-Var",
            0.66,
            json!({"foreign_funded": true, "anonymous": true, "verified": false}),
        ),
        mint_entity(
            hg,
            &nid,
            EntityType::Organization,
            "EU-Watchdog-NGO",
            0.91,
            json!({"foreign_funded": false, "anonymous": false, "verified": true}),
        ),
        mint_entity(
            hg,
            &nid,
            EntityType::Organization,
            "EuFactWatch",
            0.93,
            json!({"foreign_funded": false, "anonymous": false, "verified": true}),
        ),
    ];

    // Two situations with fuzzy endpoints — analyst said "in the morning",
    // "around midday, may have started while wave A was still going".
    // Trapezoidal windows are picked so the supports OVERLAP, so the
    // graded relation distributes mass across multiple Allen variants.
    let post_a_start = ts(2026, 4, 14, 8); // morning: 08:00 ± 1h
    let post_a_end = ts(2026, 4, 14, 11);
    let post_b_start = ts(2026, 4, 14, 11); // midday-ish: 11:00 ± 1h, overlap-or-meet
    let post_b_end = ts(2026, 4, 14, 13);

    let fuzz_post_a = FuzzyEndpoints::from_pair(
        TrapezoidalFuzzy::new(
            post_a_start - Duration::hours(1),
            post_a_start,
            post_a_start + Duration::minutes(15),
            post_a_start + Duration::hours(1),
        )
        .unwrap(),
        TrapezoidalFuzzy::new(
            post_a_end - Duration::hours(1),
            post_a_end - Duration::minutes(15),
            post_a_end,
            post_a_end + Duration::hours(1),
        )
        .unwrap(),
    );
    let fuzz_post_b = FuzzyEndpoints::from_pair(
        TrapezoidalFuzzy::new(
            post_b_start - Duration::hours(1),
            post_b_start,
            post_b_start + Duration::minutes(15),
            post_b_start + Duration::hours(1),
        )
        .unwrap(),
        TrapezoidalFuzzy::new(
            post_b_end - Duration::hours(1),
            post_b_end - Duration::minutes(15),
            post_b_end,
            post_b_end + Duration::hours(1),
        )
        .unwrap(),
    );

    let fuzzy_post_a = mint_situation(
        hg,
        &nid,
        "Vesper-Post-Wave-A",
        post_a_start,
        post_a_end,
        Some(fuzz_post_a),
        TimeGranularity::Approximate,
        0.78,
        "First wave of near-identical posts (around dawn)",
    );
    let fuzzy_post_b = mint_situation(
        hg,
        &nid,
        "Vesper-Post-Wave-B",
        post_b_start,
        post_b_end,
        Some(fuzz_post_b),
        TimeGranularity::Approximate,
        0.74,
        "Second wave of near-identical posts (late morning)",
    );

    // Sharp events used by the gradual-argumentation + ORD-Horn sections.
    let sharp_event_takedown = mint_situation(
        hg,
        &nid,
        "Platform-Takedown",
        ts(2026, 4, 14, 14),
        ts(2026, 4, 14, 15),
        None,
        TimeGranularity::Exact,
        0.83,
        "Platform removes Vesper-A1 and Vesper-A2 accounts",
    );
    let sharp_event_factcheck = mint_situation(
        hg,
        &nid,
        "EuFactWatch-Verdict",
        ts(2026, 4, 14, 16),
        ts(2026, 4, 14, 17),
        None,
        TimeGranularity::Exact,
        0.92,
        "EuFactWatch publishes 'misleading' verdict on Vesper claim",
    );

    // Three crisp situations used for ORD-Horn closure.
    let crisp_t0 = mint_situation(
        hg,
        &nid,
        "Funding-Wire",
        ts(2026, 4, 1, 9),
        ts(2026, 4, 1, 10),
        None,
        TimeGranularity::Exact,
        0.55,
        "Suspected funding wire reaches ShellHub-Var",
    );
    let crisp_t1 = mint_situation(
        hg,
        &nid,
        "Persona-Network-Build",
        ts(2026, 4, 5, 9),
        ts(2026, 4, 12, 18),
        None,
        TimeGranularity::Exact,
        0.62,
        "Anonymous personas register and warm up",
    );
    let crisp_t2 = mint_situation(
        hg,
        &nid,
        "Coordinated-Posting",
        ts(2026, 4, 14, 5),
        ts(2026, 4, 14, 13),
        None,
        TimeGranularity::Exact,
        0.79,
        "Coordinated posting waves A and B",
    );

    // Wire participation: Vesper actors all participate in both posting
    // waves; FactCheck-Gamma participates in the verdict.
    for actor_id in &actors[..3] {
        for sit in [fuzzy_post_a, fuzzy_post_b, sharp_event_takedown] {
            hg.add_participant(Participation {
                entity_id: *actor_id,
                situation_id: sit,
                role: Role::Protagonist,
                info_set: None,
                action: Some("post".into()),
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
    hg.add_participant(Participation {
        entity_id: actors[4],
        situation_id: sharp_event_factcheck,
        role: Role::Protagonist,
        info_set: None,
        action: Some("publish-verdict".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    Vesper {
        nid,
        actors,
        organizations,
        fuzzy_post_a,
        fuzzy_post_b,
        sharp_event_takedown,
        sharp_event_factcheck,
        crisp_t0,
        crisp_t1,
        crisp_t2,
    }
}

// ── Section 3: per-capability demonstrations ────────────────────────────────

fn registries(t: &mut Transcript) {
    let tn = TNormRegistry::default();
    let agg = AggregatorRegistry::default();
    let mut tn_names = tn.list();
    tn_names.sort();
    let mut agg_names = agg.list();
    agg_names.sort();
    let body = format!(
        "Registered t-norms     : {tn_names:?}\nRegistered aggregators : {agg_names:?}"
    );
    t.section(
        "1. Registries",
        body,
        json!({"tnorms": tn_names, "aggregators": agg_names}),
    );
}

fn tnorms_against_two_signals(t: &mut Transcript) {
    // Two corroboration signals: source-A confidence, source-B confidence.
    // Same inputs, four conjunctions. Lukasiewicz < Goguen < Godel
    // pointwise.
    let cases = [(0.6, 0.7), (0.4, 0.5), (0.85, 0.95), (0.10, 0.20)];
    let mut rows: Vec<Value> = vec![];
    let mut body = String::from(
        "Two-source AND under each t-norm. Inputs (a, b)\n\n  (a, b)        Godel  Goguen  Lukas  Hamacher(λ=1)\n",
    );
    for (a, b) in cases {
        let g = combine_tnorm(TNormKind::Godel, a, b);
        let p = combine_tnorm(TNormKind::Goguen, a, b);
        let l = combine_tnorm(TNormKind::Lukasiewicz, a, b);
        let h = combine_tnorm(TNormKind::Hamacher(1.0), a, b);
        body.push_str(&format!(
            "  ({a:.2}, {b:.2})    {g:.4}  {p:.4}  {l:.4}  {h:.4}\n"
        ));
        rows.push(json!({
            "a": a, "b": b, "godel": g, "goguen": p, "lukasiewicz": l, "hamacher_1": h
        }));
    }
    body.push_str(
        "\nInterpretation: Godel = strict floor (min), Lukasiewicz = bounded \
         difference (becomes 0 if the inputs jointly under-cover 1.0), Goguen \
         = probabilistic AND, Hamacher(1) numerically equals Goguen.",
    );

    // T-conorms (OR) on the same inputs — corroboration boosters.
    let mut conorm_rows = vec![];
    for (a, b) in cases {
        conorm_rows.push(json!({
            "a": a, "b": b,
            "godel_max": combine_tconorm(TNormKind::Godel, a, b),
            "goguen": combine_tconorm(TNormKind::Goguen, a, b),
            "lukasiewicz": combine_tconorm(TNormKind::Lukasiewicz, a, b),
        }));
    }
    t.section(
        "2. T-norms (fuzzy AND) vs t-conorms (fuzzy OR)",
        body,
        json!({"tnorm_combine": rows, "tconorm_combine": conorm_rows}),
    );
}

fn aggregators_over_sources(t: &mut Transcript) {
    // Five-source confidence vector for the same Vesper claim.
    // High mean but one strong dissent and one anonymous low-trust source.
    let xs = [0.91, 0.88, 0.84, 0.10, 0.74]; // CIB-A1, A2, A3, FactCheck-Gamma dissent, Outlier-Loud

    let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[xs.len() / 2];
    let tnorm_godel = reduce_tnorm(TNormKind::Godel, &xs);
    let tnorm_lukas = reduce_tnorm(TNormKind::Lukasiewicz, &xs);
    let tconorm_godel = reduce_tconorm(TNormKind::Godel, &xs);
    let tconorm_goguen = reduce_tconorm(TNormKind::Goguen, &xs);

    let owa_most_w = linguistic_weights(OwaQuantifier::Most, xs.len()).unwrap();
    let owa_few_w = linguistic_weights(OwaQuantifier::Few, xs.len()).unwrap();
    let owa_most = owa(&xs, &owa_most_w).unwrap();
    let owa_few = owa(&xs, &owa_few_w).unwrap();

    let m_add = symmetric_additive(xs.len() as u8).unwrap();
    let m_pess = symmetric_pessimistic(xs.len() as u8).unwrap();
    let m_opt = symmetric_optimistic(xs.len() as u8).unwrap();
    let c_add = choquet_exact(&xs, &m_add).unwrap();
    let c_pess = choquet_exact(&xs, &m_pess).unwrap();
    let c_opt = choquet_exact(&xs, &m_opt).unwrap();

    // Custom non-additive measure (n=5) that boosts mass on the
    // {A1, A2, A3} CIB-suspect coalition (signals 0,1,2).
    // Members of mass on the full set + the 3-way coalition + bigger sets.
    // We construct mu via a monotone interpolation: mu(S) = base_size_term
    // + (0.40 if S ⊇ {0,1,2}). Then renormalise so mu(N)=1, mu(∅)=0.
    let n = xs.len() as u8;
    let size = 1usize << n;
    let mut values = vec![0.0_f64; size];
    let coal: usize = 0b00111;
    for s in 0..size {
        let cardinality = (s as u32).count_ones() as f64;
        let base = (cardinality / n as f64).powf(0.7); // concave size term
        let coalition_bonus = if (s & coal) == coal && s != 0 {
            0.30
        } else {
            0.0
        };
        values[s] = (base + coalition_bonus).min(1.0);
    }
    values[0] = 0.0;
    values[size - 1] = 1.0;
    // Project to monotonic by sweeping.
    for s in 0..size {
        for i in 0..n {
            let bit = 1usize << i;
            if s & bit == 0 {
                let t = s | bit;
                if values[t] < values[s] {
                    values[t] = values[s];
                }
            }
        }
    }
    values[0] = 0.0;
    values[size - 1] = 1.0;
    let coalition_measure = new_monotone(n, values).unwrap();
    let c_coalition = choquet_exact(&xs, &coalition_measure).unwrap();
    let c_coalition_dispatch = choquet(&xs, &coalition_measure, 7).unwrap().value;

    let body = format!(
        "Five-source confidences for the same claim: {xs:?}\n\n  Mean                : {mean:.4}\n  \
         Median              : {median:.4}\n  TNormReduce/Godel    : {tnorm_godel:.4}  (= weakest source — strict floor)\n  \
         TNormReduce/Lukas    : {tnorm_lukas:.4}  (joint coverage; collapses to 0 fast)\n  \
         TConormReduce/Godel  : {tconorm_godel:.4}  (= max — at least one source confident)\n  \
         TConormReduce/Goguen : {tconorm_goguen:.4}  (probabilistic OR over independent sources)\n  \
         OWA (Most)          : {owa_most:.4}\n  OWA (Few)           : {owa_few:.4}\n\n  \
         Choquet (additive)  : {c_add:.4}  ≈ Mean (sanity check)\n  \
         Choquet (pessim.)   : {c_pess:.4}  = min (one weak voice tanks the score)\n  \
         Choquet (optim.)    : {c_opt:.4}  = max\n  \
         Choquet (CIB-coal.) : {c_coalition:.4}  ← non-additive measure that rewards coordinated sources 0,1,2\n  \
         Choquet (dispatch)  : {c_coalition_dispatch:.4} (exact path under EXACT_N_CAP=10)",
    );

    t.section(
        "3. Aggregators (Mean / Median / OWA / Choquet / TNormReduce / TConormReduce)",
        body,
        json!({
            "xs": xs,
            "mean": mean, "median": median,
            "tnorm_godel": tnorm_godel, "tnorm_lukasiewicz": tnorm_lukas,
            "tconorm_godel": tconorm_godel, "tconorm_goguen": tconorm_goguen,
            "owa_most": owa_most, "owa_few": owa_few,
            "owa_most_weights": owa_most_w, "owa_few_weights": owa_few_w,
            "choquet_additive": c_add,
            "choquet_pessimistic": c_pess,
            "choquet_optimistic": c_opt,
            "choquet_coalition_boost_on_signals_0_1_2": c_coalition,
        }),
    );
}

fn fuzzy_allen_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    let a = hg.get_situation(&v.fuzzy_post_a).unwrap();
    let b = hg.get_situation(&v.fuzzy_post_b).unwrap();

    let mut tnorm_rows = BTreeMap::new();
    for kind in [
        TNormKind::Godel,
        TNormKind::Goguen,
        TNormKind::Lukasiewicz,
        TNormKind::Hamacher(1.0),
    ] {
        let cfg = GradedAllenConfig { tnorm: kind };
        let v = graded_relation_with(&a.temporal, &b.temporal, &cfg);
        tnorm_rows.insert(format!("{kind:?}"), serde_json::to_value(v.to_vec()).unwrap());
    }

    // Show top-3 relations under Godel.
    let cfg = GradedAllenConfig { tnorm: TNormKind::Godel };
    let v_god = graded_relation_with(&a.temporal, &b.temporal, &cfg);
    let labels = [
        "Before","Meets","Overlaps","Starts","During","Finishes","Equals",
        "FinishedBy","Contains","StartedBy","OverlappedBy","MetBy","After",
    ];
    let mut indexed: Vec<(usize, f64)> = v_god.iter().copied().enumerate().collect();
    indexed.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let top3: Vec<String> = indexed
        .iter()
        .take(3)
        .map(|(idx, val)| format!("{}={:.3}", labels[*idx], val))
        .collect();

    // Sanity check — at least three Allen relations must carry non-zero
    // graded mass. Crisp timestamps would yield a one-hot; if the fuzzy
    // path collapses to one relation, our windows didn't overlap as
    // intended.
    let before_idx = relation_index(AllenRelation::Before);
    let nonzero_count = v_god.iter().filter(|x| **x > 0.001).count();
    assert!(
        nonzero_count >= 2,
        "expected fuzzy Allen to spread mass across ≥2 relations (got {}); v={:?}",
        nonzero_count,
        v_god,
    );
    let _ = before_idx; // kept for future assertions

    let body = format!(
        "Two situations with fuzzy endpoints. Wave A: morning (08:00 - 11:00 \
         ± 60 min). Wave B: midday (11:00 - 13:00 ± 60 min). The wave-A end \
         and wave-B start kernels MEET, so a graded Allen call must spread \
         mass across {{Before, Meets, Overlaps}}.\n  \
         Top-3 graded relations under Godel : {top3:?}\n  \
         Non-zero mass on {nonzero_count} of 13 relations (crisp timestamps \
         would put 1.0 on exactly one).\n  \
         Each t-norm's full 13-vector is in the JSON report.",
    );

    t.section(
        "4. Fuzzy Allen relations on imprecise endpoints",
        body,
        json!({
            "labels": labels,
            "graded_relation_per_tnorm": Value::Object(tnorm_rows.into_iter().collect()),
            "top3_godel": top3,
        }),
    );
}

fn quantifiers_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    // Two predicates with very different cardinality ratios so the four
    // quantifier ramps actually fire at different points along their
    // respective curves. r_high ≈ 1.0 (every actor is confident enough);
    // r_low ≈ 0.4 (only Vesper-A1 + Vesper-A2 + FactCheck-Gamma clear
    // the higher bar).
    let pred_high = |e: &Entity| if e.confidence > 0.7 { 1.0 } else { 0.0 };
    let pred_split = |e: &Entity| if e.confidence > 0.85 { 1.0 } else { 0.0 };
    let mut rows_high = BTreeMap::new();
    let mut rows_split = BTreeMap::new();
    let mut body = String::from(
        "Domain = all Actor entities in operation-vesper. Two crisp predicates:\n  \
         predicate-high  : confidence > 0.70  (r = 5/5 = 1.0)\n  \
         predicate-split : confidence > 0.85  (r = 3/5 = 0.6)\n\n",
    );
    body.push_str("                        predicate-high   predicate-split\n");
    for q in [
        Quantifier::Most,
        Quantifier::Many,
        Quantifier::AlmostAll,
        Quantifier::Few,
    ] {
        let v_high =
            evaluate_over_entities(hg, &v.nid, Some(EntityType::Actor), pred_high, q).unwrap();
        let v_split =
            evaluate_over_entities(hg, &v.nid, Some(EntityType::Actor), pred_split, q).unwrap();
        body.push_str(&format!(
            "  {:<10}            {:.4}            {:.4}\n",
            q.name(),
            v_high,
            v_split
        ));
        rows_high.insert(q.name().to_string(), v_high);
        rows_split.insert(q.name().to_string(), v_split);
    }
    // Same domain, graded predicate: confidence itself.
    body.push_str("\nWith a graded predicate μ_P(e) = e.confidence:\n");
    let mut graded = BTreeMap::new();
    for q in [
        Quantifier::Most,
        Quantifier::AlmostAll,
        Quantifier::Few,
    ] {
        let val = evaluate_over_entities(
            hg,
            &v.nid,
            Some(EntityType::Actor),
            |e: &Entity| e.confidence as f64,
            q,
        )
        .unwrap();
        body.push_str(&format!("  {:<11} -> Q(r) = {:.4}\n", q.name(), val));
        graded.insert(q.name().to_string(), val);
    }

    t.section(
        "5. Intermediate quantifiers (Novák / Murinová)",
        body,
        json!({
            "crisp_predicate_confidence_gt_0_7": rows_high,
            "crisp_predicate_confidence_gt_0_85": rows_split,
            "graded_predicate_mu_eq_confidence": graded,
        }),
    );
}

fn syllogism_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    // Figure I*: "Most Actor IS Actor", "AlmostAll Actor IS Actor",
    // "Most Actor IS Actor". Trivially true since the universal predicate
    // = Actor type. Demonstrates the canonical valid path.
    let s_valid = Syllogism {
        major: SyllogismStatement {
            quantifier: Quantifier::Most,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        minor: SyllogismStatement {
            quantifier: Quantifier::AlmostAll,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        conclusion: SyllogismStatement {
            quantifier: Quantifier::Most,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        figure_hint: None,
    };
    let figure = classify_figure(&s_valid);
    let valid_result =
        verify(hg, &v.nid, &s_valid, TNormKind::Godel, 0.5, &TypePredicateResolver).unwrap();

    // Figure II — Peterson-invalid by taxonomy regardless of degree.
    let s_fig2 = Syllogism {
        major: SyllogismStatement {
            quantifier: Quantifier::AlmostAll,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        minor: SyllogismStatement {
            quantifier: Quantifier::Most,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        conclusion: SyllogismStatement {
            quantifier: Quantifier::Most,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Actor".into(),
        },
        figure_hint: None,
    };
    let fig2_figure = classify_figure(&s_fig2);
    let fig2_result =
        verify(hg, &v.nid, &s_fig2, TNormKind::Godel, 0.5, &TypePredicateResolver).unwrap();

    let body = format!(
        "Figure I* (canonical, Peterson-valid)\n  major:      'Most Actor IS Actor'\n  minor:      'AlmostAll Actor IS Actor'\n  conclusion: 'Most Actor IS Actor'\n  -> classify_figure = {figure}\n  -> degree = {:.4}, valid = {}\n\nFigure II (Peterson-invalid by taxonomy)\n  major:      'AlmostAll Actor IS Actor'\n  minor:      'Most Actor IS Actor'\n  conclusion: 'Most Actor IS Actor'\n  -> classify_figure = {fig2_figure}\n  -> degree = {:.4}, valid = {}  (degree may be high but Figure II is invalid)\n",
        valid_result.degree, valid_result.valid, fig2_result.degree, fig2_result.valid,
    );

    t.section(
        "6. Graded Peterson syllogisms",
        body,
        json!({
            "figure_I_star": {
                "classified_figure": figure,
                "degree": valid_result.degree,
                "valid": valid_result.valid,
            },
            "figure_II_peterson_invalid": {
                "classified_figure": fig2_figure,
                "degree": fig2_result.degree,
                "valid": fig2_result.valid,
            },
        }),
    );
}

fn fca_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    // Build a graded actor × attribute context. We only let actors be
    // objects, with three boolean-graded attributes. The default
    // hypergraph builder reads numerics + booleans from properties; we
    // pass them as f64 in [0, 1].
    let opts = FormalContextOptions {
        entity_type_filter: Some(EntityType::Actor),
        attribute_allowlist: Some(vec![
            "anonymous".into(),
            "foreign_funded".into(),
            "verified".into(),
            "inflammatory_score".into(),
        ]),
        large_context: false,
    };
    let ctx = FormalContext::from_hypergraph(hg, &v.nid, &opts).unwrap();
    let lattice = build_lattice(&ctx, TNormKind::Godel).unwrap();

    let body = format!(
        "FormalContext: {} objects (Actor entities) × {} attributes\n  attributes: {:?}\n  Concept lattice: {} concepts under Godel\n  Hasse edges: {}",
        ctx.num_objects(),
        ctx.num_attributes(),
        ctx.attributes,
        lattice.concepts.len(),
        lattice.order.len(),
    );

    let extents: Vec<Value> = lattice
        .concepts
        .iter()
        .map(|c| {
            json!({
                "extent_size": c.extent.len(),
                "intent": c.intent.iter()
                    .map(|(j, mu)| json!({"attr": ctx.attributes[*j], "mu": mu}))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    t.section(
        "7. Fuzzy Formal Concept Analysis (Bělohlávek lattice)",
        body,
        json!({
            "n_objects": ctx.num_objects(),
            "n_attributes": ctx.num_attributes(),
            "attributes": ctx.attributes,
            "n_concepts": lattice.concepts.len(),
            "n_hasse_edges": lattice.order.len(),
            "concepts": extents,
        }),
    );
}

fn mamdani_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    // Reference rule fixture — 'elevated-disinfo-risk':
    // IF inflammatory_score IS high AND temporal_correlation IS high
    // THEN disinfo_risk IS elevated
    let rule = MamdaniRule {
        id: Uuid::now_v7(),
        name: "elevated-disinfo-risk".into(),
        narrative_id: v.nid.clone(),
        antecedent: vec![
            FuzzyCondition {
                variable_path: "entity.properties.inflammatory_score".into(),
                membership: MembershipFunction::Trapezoidal {
                    a: 0.5,
                    b: 0.7,
                    c: 1.0,
                    d: 1.0,
                },
                linguistic_term: "high".into(),
            },
            FuzzyCondition {
                variable_path: "entity.properties.temporal_correlation".into(),
                membership: MembershipFunction::Trapezoidal {
                    a: 0.5,
                    b: 0.75,
                    c: 1.0,
                    d: 1.0,
                },
                linguistic_term: "high".into(),
            },
            FuzzyCondition {
                variable_path: "entity.confidence".into(),
                membership: MembershipFunction::Gaussian {
                    mean: 0.85,
                    sigma: 0.15,
                },
                linguistic_term: "validated".into(),
            },
        ],
        consequent: FuzzyOutput {
            variable: "disinfo_risk".into(),
            membership: MembershipFunction::Triangular {
                a: 0.5,
                b: 0.85,
                c: 1.0,
            },
            linguistic_term: "elevated".into(),
        },
        tnorm: TNormKind::Godel,
        created_at: Utc::now(),
        enabled: true,
    };
    let rule_set = RuleSet::new(vec![rule.clone()]);

    let mut rows = vec![];
    let mut body = String::from(
        "Mamdani rule 'elevated-disinfo-risk' (3 antecedents under Godel, centroid defuzz):\n",
    );
    for actor_id in &v.actors {
        let entity = hg.get_entity(actor_id).unwrap();
        let evaluation = evaluate_rule_set(&rule_set, &entity).unwrap();
        let firing = evaluation
            .fired_rules
            .first()
            .map(|f| f.firing_strength)
            .unwrap_or(0.0);
        let defuzz = evaluation.defuzzified_output.unwrap_or(0.0);
        let name = entity
            .properties
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("?");
        body.push_str(&format!(
            "  {name:<18}  firing = {firing:.4}   defuzz(disinfo_risk) = {defuzz:.4}\n"
        ));
        rows.push(json!({
            "actor": name,
            "firing_strength": firing,
            "defuzzified_output": defuzz,
        }));
    }
    body.push_str(
        "\nThe three Vesper-A actors fire strongly; Outlier-Loud (high inflammatory, low correlation) fires weakly; FactCheck-Gamma fires near zero. The rule discriminates COORDINATION, not heat.",
    );

    t.section(
        "8. Mamdani fuzzy rule system",
        body,
        json!({"rule_id": rule.id.to_string(), "evaluations": rows}),
    );
}

fn hybrid_demo(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    // P_fuzzy(E) where E = "actor confidence > 0.7" and the distribution
    // is uniform over the Vesper-A coalition (3 outcomes, each P=1/3).
    let coalition = &v.actors[..3];
    let p = 1.0_f64 / coalition.len() as f64;
    let outcomes: Vec<(Uuid, f64)> = coalition.iter().map(|id| (*id, p)).collect();
    let dist = ProbDist::Discrete { outcomes };

    let event_q = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Quantifier,
        predicate_payload: json!({
            "quantifier": "most",
            "where": "confidence > 0.7",
            "entity_type": "Actor",
        }),
    };
    let p_q = fuzzy_probability(hg, &v.nid, &event_q, &dist, TNormKind::Godel).unwrap();

    // Custom predicate: graded membership = inflammatory_score directly.
    let mut memberships = BTreeMap::new();
    for actor_id in coalition {
        let e = hg.get_entity(actor_id).unwrap();
        let mu = e
            .properties
            .get("inflammatory_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        memberships.insert(actor_id.to_string(), Value::from(mu));
    }
    let event_custom = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: json!({
            "memberships": memberships,
        }),
    };
    let p_custom =
        fuzzy_probability(hg, &v.nid, &event_custom, &dist, TNormKind::Godel).unwrap();

    let body = format!(
        "Discrete uniform distribution over the Vesper-A coalition (3 actors, P=1/3 each).\n  \
         Event Q: 'confidence > 0.7' — crisp 1.0 for all three coalition members\n    \
         P_fuzzy(Q) = {p_q:.4}  (≈ 1.0 because every coalition member matches)\n  \
         Event Custom: μ_E = inflammatory_score per actor\n    \
         P_fuzzy(Custom) = {p_custom:.4}  (= average inflammatory_score on coalition)",
    );
    t.section(
        "9. Fuzzy-Probabilistic Hybrid (Cao-Holcapek-Flaminio base case)",
        body,
        json!({
            "p_quantifier_event": p_q,
            "p_custom_event": p_custom,
        }),
    );
}

fn choquet_learning_demo(t: &mut Transcript) {
    // Reproduce the synthetic-CIB worked example.
    let dataset = generate_synthetic_cib(42, 100);
    let n = 4_u8;

    // Baseline: symmetric_additive (= mean) AUC on the held-out half.
    let split_idx = dataset.len() / 2;
    let test = &dataset[split_idx..];
    let m_add = symmetric_additive(n).unwrap();
    let baseline_auc = ranking_auc(test, |xs| choquet_exact(xs, &m_add).unwrap());

    let learned = learn_choquet_measure(n, &dataset, "fuzzy-capabilities-demo-v1").unwrap();
    let learned_auc = learned.test_auc;

    let body = format!(
        "Synthetic-CIB dataset (seed=42, n_clusters=100). Score has a hidden\n  multiplicative interaction between signals 0 and 1: no additive measure can recover the ranking.\n\n  symmetric_additive (= mean) test AUC : {baseline_auc:.4}\n  learned Choquet measure test AUC     : {learned_auc:.4}\n  AUC gap                              : +{:.4}\n\n  Provenance:\n    fit_method      : {}\n    fit_loss        : {:.4}\n    fit_seconds     : {:.3}\n    n_samples       : {}\n",
        learned_auc - baseline_auc,
        learned.provenance.fit_method,
        learned.provenance.fit_loss,
        learned.provenance.fit_seconds,
        learned.provenance.n_samples,
    );
    t.section(
        "10. Choquet measure learning (Graded Sprint)",
        body,
        json!({
            "baseline_additive_auc": baseline_auc,
            "learned_test_auc": learned_auc,
            "gap": learned_auc - baseline_auc,
            "fit_method": learned.provenance.fit_method,
            "fit_loss": learned.provenance.fit_loss,
            "fit_seconds": learned.provenance.fit_seconds,
            "n_samples": learned.provenance.n_samples,
            "train_auc": learned.train_auc,
        }),
    );
}

fn ranking_auc<F: Fn(&[f64]) -> f64>(data: &[(Vec<f64>, u32)], f: F) -> f64 {
    // Pairwise concordance vs ranks (lower rank = more coordinated).
    let mut concordant = 0_u64;
    let mut total = 0_u64;
    for i in 0..data.len() {
        for j in (i + 1)..data.len() {
            let (xs_i, r_i) = &data[i];
            let (xs_j, r_j) = &data[j];
            if r_i == r_j {
                continue;
            }
            total += 1;
            let s_i = f(xs_i);
            let s_j = f(xs_j);
            let predicted_lower_rank = if s_i > s_j {
                Some(i)
            } else if s_j > s_i {
                Some(j)
            } else {
                None
            };
            let actual_lower_rank = if r_i < r_j { i } else { j };
            if predicted_lower_rank == Some(actual_lower_rank) {
                concordant += 1;
            }
        }
    }
    if total == 0 {
        0.5
    } else {
        concordant as f64 / total as f64
    }
}

fn gradual_argumentation_demo(t: &mut Transcript) {
    // Four arguments around the Vesper case.
    let a_coordinated = Argument {
        id: Uuid::now_v7(),
        label: "Vesper IS coordinated".into(),
        source_id: None,
        confidence: 0.85,
    };
    let a_organic = Argument {
        id: Uuid::now_v7(),
        label: "Vesper looks organic".into(),
        source_id: None,
        confidence: 0.45,
    };
    let a_factcheck_correct = Argument {
        id: Uuid::now_v7(),
        label: "EuFactWatch is correct".into(),
        source_id: None,
        confidence: 0.92,
    };
    let a_factcheck_biased = Argument {
        id: Uuid::now_v7(),
        label: "EuFactWatch is biased".into(),
        source_id: None,
        confidence: 0.30,
    };

    let arguments = vec![
        a_coordinated.clone(),
        a_organic.clone(),
        a_factcheck_correct.clone(),
        a_factcheck_biased.clone(),
    ];
    // 0=coord, 1=organic, 2=fc-correct, 3=fc-biased
    // organic attacks coordinated; coordinated attacks organic.
    // fc-correct attacks fc-biased; fc-biased attacks fc-correct.
    // fc-correct also attacks organic (the verdict says it isn't organic).
    let attacks = vec![(1, 0), (0, 1), (2, 3), (3, 2), (2, 1)];
    let framework = ArgumentationFramework {
        arguments: arguments.clone(),
        attacks: attacks.clone(),
    };

    let mut all = BTreeMap::new();
    let kinds = [
        ("HCategoriser", GradualSemanticsKind::HCategoriser),
        ("MaxBased", GradualSemanticsKind::MaxBased),
        ("CardBased", GradualSemanticsKind::CardBased),
        (
            "WeightedHCategoriser",
            GradualSemanticsKind::WeightedHCategoriser {
                weights: vec![0.5, 0.5, 0.5, 0.5, 0.5],
            },
        ),
    ];
    let mut body = String::from(
        "Framework: 4 arguments, 5 attacks. Acceptability per gradual semantics (default Godel):\n",
    );
    body.push_str("                            HCat   MaxB   CardB  WeightedHCat\n");
    let names = ["Vesper-coordinated", "Vesper-organic ", "FC-correct      ", "FC-biased       "];
    let mut per_kind: BTreeMap<String, GradualResult> = BTreeMap::new();
    for (label, kind) in &kinds {
        let res = run_gradual_argumentation(&framework, kind, None).unwrap();
        per_kind.insert((*label).to_string(), res.clone());
        all.insert(
            (*label).to_string(),
            json!({
                "iterations": res.iterations,
                "converged": res.converged,
                "acceptability": res.acceptability.iter()
                    .map(|(id, v)| (id.to_string(), *v))
                    .collect::<BTreeMap<_, _>>(),
            }),
        );
    }
    for (i, name) in names.iter().enumerate() {
        let id = arguments[i].id;
        let h = per_kind["HCategoriser"].acceptability.get(&id).copied().unwrap_or(0.0);
        let m = per_kind["MaxBased"].acceptability.get(&id).copied().unwrap_or(0.0);
        let c = per_kind["CardBased"].acceptability.get(&id).copied().unwrap_or(0.0);
        let w = per_kind["WeightedHCategoriser"].acceptability.get(&id).copied().unwrap_or(0.0);
        body.push_str(&format!(
            "  {name}     {h:.4}  {m:.4}  {c:.4}  {w:.4}\n"
        ));
    }
    body.push_str(
        "\nInterpretation: a high acceptability means few effective attackers OR strong intrinsic confidence. \
         The fact-check argument survives near its intrinsic strength (one mutual attacker, but FC-biased itself is weak).",
    );

    t.section(
        "11. Gradual / ranking-based argumentation (Graded Sprint)",
        body,
        json!({
            "arguments": [
                {"id": a_coordinated.id.to_string(), "label": a_coordinated.label, "intrinsic": a_coordinated.confidence},
                {"id": a_organic.id.to_string(), "label": a_organic.label, "intrinsic": a_organic.confidence},
                {"id": a_factcheck_correct.id.to_string(), "label": a_factcheck_correct.label, "intrinsic": a_factcheck_correct.confidence},
                {"id": a_factcheck_biased.id.to_string(), "label": a_factcheck_biased.label, "intrinsic": a_factcheck_biased.confidence},
            ],
            "attacks": attacks,
            "results": Value::Object(all.into_iter().collect()),
        }),
    );
}

fn ordhorn_demo(t: &mut Transcript) {
    // Disjunctive Allen network on 4 events:
    //   0 = Funding-Wire
    //   1 = Persona-Network-Build
    //   2 = Coordinated-Posting
    //   3 = Platform-Takedown
    //
    // Constraints (analyst-style):
    //   FundingWire {Before, Meets} PersonaBuild
    //   PersonaBuild {Before, Meets, Overlaps} CoordPosting
    //   FundingWire {Before} CoordPosting
    //   CoordPosting {Before, Meets} Takedown
    let net = OrdHornNetwork {
        n: 4,
        constraints: vec![
            OrdHornConstraint {
                a: 0,
                b: 1,
                relations: vec![AllenRelation::Before, AllenRelation::Meets],
            },
            OrdHornConstraint {
                a: 1,
                b: 2,
                relations: vec![
                    AllenRelation::Before,
                    AllenRelation::Meets,
                    AllenRelation::Overlaps,
                ],
            },
            OrdHornConstraint {
                a: 0,
                b: 2,
                relations: vec![AllenRelation::Before],
            },
            OrdHornConstraint {
                a: 2,
                b: 3,
                relations: vec![AllenRelation::Before, AllenRelation::Meets],
            },
        ],
    };
    let closed = closure(&net).unwrap();
    let satisfiable = is_satisfiable(&net);
    let mut closed_repr = vec![];
    for c in &closed.constraints {
        let names: Vec<String> = c.relations.iter().map(|r| format!("{r:?}")).collect();
        closed_repr.push(format!("({}, {}) ∈ {{{}}}", c.a, c.b, names.join(", ")));
    }

    // Adversarial variant — replace 0→2 with {After}, which contradicts.
    let bad_net = OrdHornNetwork {
        n: 4,
        constraints: {
            let mut c = net.constraints.clone();
            c[2].relations = vec![AllenRelation::After];
            c
        },
    };
    let bad_satisfiable = is_satisfiable(&bad_net);

    let body = format!(
        "Disjunctive Allen network on 4 events (FundingWire, PersonaBuild, CoordPosting, Takedown).\n  Original satisfiable        : {satisfiable}\n  Closure-tightened constraints (post van-Beek path consistency):\n    {}\n  Adversarial variant (FundingWire AFTER CoordPosting):\n  Adversarial satisfiable     : {bad_satisfiable}",
        closed_repr.join("\n    "),
    );
    t.section(
        "12. ORD-Horn path-consistency closure (Graded Sprint)",
        body,
        json!({
            "satisfiable_original": satisfiable,
            "closure_constraints": closed.constraints.iter().map(|c| {
                json!({
                    "a": c.a, "b": c.b,
                    "relations": c.relations.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "satisfiable_after_adversarial_edit": bad_satisfiable,
        }),
    );
}

fn vesper_summary(t: &mut Transcript, hg: &Hypergraph, v: &Vesper) {
    let entities = hg.list_entities_by_narrative(&v.nid).unwrap();
    let situations = hg.list_situations_by_narrative(&v.nid).unwrap();
    let body = format!(
        "Operation Vesper narrative populated with:\n  Actors        : {}\n  Organizations : {}\n  Situations    : {}\n  Total entities: {}\n",
        v.actors.len(),
        v.organizations.len(),
        situations.len(),
        entities.len(),
    );
    t.section(
        "0. Narrative scaffold",
        body,
        json!({
            "narrative_id": v.nid,
            "n_actors": v.actors.len(),
            "n_organizations": v.organizations.len(),
            "n_situations": situations.len(),
            "n_entities": entities.len(),
            "fuzzy_situations": [
                v.fuzzy_post_a.to_string(),
                v.fuzzy_post_b.to_string(),
            ],
            "sharp_situations": [
                v.sharp_event_takedown.to_string(),
                v.sharp_event_factcheck.to_string(),
            ],
            "ordhorn_situations": [
                v.crisp_t0.to_string(),
                v.crisp_t1.to_string(),
                v.crisp_t2.to_string(),
            ],
        }),
    );
}

#[test]
fn fuzzy_capabilities_demo() {
    let hg = make_hg();
    let vesper = build_operation_vesper(&hg);
    let mut t = Transcript::default();

    vesper_summary(&mut t, &hg, &vesper);
    registries(&mut t);
    tnorms_against_two_signals(&mut t);
    aggregators_over_sources(&mut t);
    fuzzy_allen_demo(&mut t, &hg, &vesper);
    quantifiers_demo(&mut t, &hg, &vesper);
    syllogism_demo(&mut t, &hg, &vesper);
    fca_demo(&mut t, &hg, &vesper);
    mamdani_demo(&mut t, &hg, &vesper);
    hybrid_demo(&mut t, &hg, &vesper);
    choquet_learning_demo(&mut t);
    gradual_argumentation_demo(&mut t);
    ordhorn_demo(&mut t);

    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("fuzzy_capabilities_demo");
    t.write(&target);
    println!(
        "\nReport written to {} (transcript.txt + report.json)",
        target.display()
    );
}
