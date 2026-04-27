// Cites: [klement2000] [yager1988owa] [grabisch1996choquet].
use pest::Parser;
use pest_derive::Parser;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::tnorm::TNormKind;

#[derive(Parser)]
#[grammar = "query/tensa.pest"]
struct TensaParser;

// ─── AST Types ───────────────────────────────────────────────

/// Path traversal mode.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PathMode {
    Shortest,
    All,
    Longest,
}

/// GQL ISO 39075 path restrictor controlling traversal semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub enum PathRestrictor {
    /// Allow any repetition of nodes and edges (default).
    #[default]
    Walk,
    /// No repeated edges (but nodes may repeat).
    Trail,
    /// No repeated nodes (no cycles).
    Acyclic,
    /// No repeated nodes except start may equal end (simple cycle).
    Simple,
}

/// PATH clause for graph path queries.
#[derive(Debug, Clone, Serialize)]
pub struct PathClause {
    pub mode: PathMode,
    /// GQL path restrictor. Default: Walk (no restrictions).
    pub restrictor: PathRestrictor,
    pub start_binding: String,
    pub end_binding: String,
    pub rel_type: String,
    pub min_depth: usize,
    pub max_depth: usize,
    /// TOP k: return k shortest paths (Yen's algorithm).
    pub top_k: Option<usize>,
    /// WEIGHT field: use weighted shortest path (Dijkstra).
    pub weight_field: Option<String>,
}

/// Flow query type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum FlowType {
    Max,
    MinCut,
}

/// FLOW clause for max-flow / min-cut queries.
#[derive(Debug, Clone, Serialize)]
pub struct FlowClause {
    pub flow_type: FlowType,
    pub start_binding: String,
    pub end_binding: String,
    pub rel_type: String,
}

/// Fuzzy Sprint Phase 3 — aggregator as expressed in TensaQL source, before the
/// planner resolves it into a concrete [`AggregatorKind`].
///
/// The split is deliberate: `ChoquetByRef` stores only the referenced measure
/// name (opaque to the parser) because the `/fuzzy/measures` CRUD surface that
/// resolves it lands in Phase 4. Phase 3 carries the name forward unchanged so
/// the planner can surface a useful "measure '<name>' not resolvable until
/// Phase 4 ships /fuzzy/measures" error at plan time.
///
/// Cites: [klement2000] [yager1988owa] [grabisch1996choquet].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregatorSpec {
    Mean,
    Median,
    Owa(Vec<f64>),
    /// Named reference to a fuzzy measure — resolved by the planner via the
    /// (future) measure registry. Phase 3 only carries the name.
    ChoquetByRef(String),
    /// T-norm reduce with the named t-norm.
    TNormReduce(TNormKind),
    /// T-conorm reduce with the named t-norm (De Morgan dual).
    TConormReduce(TNormKind),
}

/// Fuzzy Sprint Phase 3 — per-statement t-norm / aggregator override carried by
/// the `WITH TNORM '<kind>'` and `AGGREGATE <kind>` tail clauses on MATCH,
/// INFER, and ASK statements.
///
/// Both fields default to `None`; when `None`, the engine falls through to the
/// site-specific defaults wired in Phases 1–2 (Goguen for Dempster–Shafer mass
/// combination, Gödel for WHERE AND/OR folding, etc.). The planner resolves
/// the string forms (`TNormKind` via `TNormRegistry::default`, `AggregatorKind`
/// via `AggregatorRegistry` + explicit OWA weight / Choquet measure validation)
/// so that unknown names surface as `TensaError::InvalidInput` at plan time
/// rather than crashing mid-execute.
///
/// `#[serde(default)]` on every carrier struct preserves backward-compat: plans
/// serialized before Phase 3 deserialize cleanly as `FuzzyConfig::default()`.
/// Cites: [klement2000] [yager1988owa] [grabisch1996choquet].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FuzzyConfig {
    /// Selected t-norm name (raw — from `WITH TNORM '<name>'`). Parser does
    /// not resolve this; unknown names surface as `TensaError::InvalidInput`
    /// at plan time via `TNormRegistry::default().get(name)` so parse errors
    /// vs. semantic errors are cleanly separated.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Selected aggregator — `None` = engine default.
    ///
    /// Parser stores `AggregatorSpec` (raw AST shape); the planner converts
    /// into a concrete [`AggregatorKind`] after measure-ref resolution. The
    /// plan-step slot holds the resolved `AggregatorKind` so the executor
    /// never sees `AggregatorSpec`.
    #[serde(default)]
    pub aggregator: Option<AggregatorSpec>,
    /// Graded Acceptability Phase 0 — id of a learned Choquet measure
    /// referenced by the query. Phase 2 wires the planner to resolve
    /// this through the existing `fz/tn/measures/{name}` KV slot;
    /// symmetric defaults leave it `None` so AST round-trips stay
    /// bit-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_id: Option<String>,
    /// Echoed version stamp of the referenced learned measure.
    /// Pairs with `measure_id` — `None` iff `measure_id` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_version: Option<u32>,
}

impl FuzzyConfig {
    /// `true` when no override was supplied (no t-norm, no aggregator,
    /// no learned-measure reference).
    pub fn is_empty(&self) -> bool {
        self.tnorm.is_none()
            && self.aggregator.is_none()
            && self.measure_id.is_none()
            && self.measure_version.is_none()
    }
}

/// Fuzzy Sprint Phase 3 — the planner's resolved counterpart to [`FuzzyConfig`].
/// Carries the concrete [`AggregatorKind`] (after measure-ref resolution) so
/// every plan step's fuzzy slot is immediately executable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FuzzyPlanConfig {
    #[serde(default)]
    pub tnorm: Option<TNormKind>,
    #[serde(default)]
    pub aggregator: Option<AggregatorKind>,
}

impl FuzzyPlanConfig {
    pub fn is_empty(&self) -> bool {
        self.tnorm.is_none() && self.aggregator.is_none()
    }
}

/// A parsed TensaQL query.
#[derive(Debug, Clone)]
pub struct TensaQuery {
    pub explain: bool,
    pub infer_clause: Option<InferClause>,
    pub discover_clause: Option<DiscoverClause>,
    pub ask_clause: Option<AskClause>,
    pub across_clause: Option<AcrossClause>,
    pub match_clause: Option<MatchClause>,
    pub path_clause: Option<PathClause>,
    pub flow_clause: Option<FlowClause>,
    pub where_clause: Option<WhereClause>,
    pub at_clause: Option<AtClause>,
    pub near_clause: Option<NearClause>,
    pub spatial_clause: Option<SpatialClause>,
    pub assuming_clause: Option<AssumingClause>,
    pub under_clause: Option<UnderClause>,
    pub group_by: Option<GroupByClause>,
    /// Bi-temporal AS OF clause: filter results to system state at given transaction time.
    pub as_of: Option<String>,
    pub return_clause: ReturnClause,
    /// Fuzzy Sprint Phase 3 — optional t-norm / aggregator override from
    /// trailing `WITH TNORM '<kind>' AGGREGATE <kind>` clauses. Default is
    /// `FuzzyConfig::default()` (both `None`). Present on every read-query
    /// variant so MATCH / INFER / DISCOVER / ASK / PATH / FLOW plans can all
    /// thread a single config through to the executor.
    pub fuzzy_config: FuzzyConfig,
}

/// INFER clause for async inference queries.
#[derive(Debug, Clone, Serialize)]
pub struct InferClause {
    pub infer_type: InferType,
    pub target_binding: String,
    pub target_type: String,
}

/// Type of inference to perform.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum InferType {
    Causes,
    Motivation,
    Game,
    Counterfactual,
    Missing,
    Anomalies,
    Centrality,
    Communities,
    Entropy,
    Beliefs,
    Evidence,
    Arguments,
    Contagion,
    /// Phase 7b — higher-order SIR contagion on the narrative's hyperedges.
    /// Routes to a synchronous `compute_higher_order_contagion`-style
    /// dispatch in the executor (no job submission needed; algorithm runs
    /// in the executor thread for the real narrative).
    HigherOrderContagion,
    /// Phase 14 — bistability/hysteresis significance on the narrative's
    /// hyperedges. Routes to a `SurrogateBistabilitySignificance` job that
    /// runs the bistability sweep on the source AND on K surrogate samples.
    ContagionBistability,
    Style,
    StyleCompare,
    StyleAnomalies,
    VerifyAuthorship,
    TemporalRules,
    MeanField,
    Psl,
    Trajectory,
    Simulate,
    PageRank,
    Eigenvector,
    Harmonic,
    Hits,
    Topology,
    LabelPropagation,
    KCore,
    TemporalPageRank,
    CausalInfluence,
    InfoBottleneck,
    Assortativity,
    TemporalMotifs,
    FactionEvolution,
    FastRP,
    Node2Vec,
    NetworkInference,
    // Disinfo Sprint D1: dual fingerprints
    BehavioralFingerprint,
    DisinfoFingerprint,
    // Disinfo Sprint D2: spread dynamics
    SpreadVelocity,
    SpreadIntervention,
    // Disinfo Sprint D3: CIB detection + superspreaders
    Cib,
    Superspreaders,
    // Disinfo Sprint D4: claims & fact-check pipeline
    ClaimOrigin,
    ClaimMatch,
    // Disinfo Sprint D5: archetypes + DS fusion
    ArchetypeClassification,
    DisinfoAssessment,
    // Sprint D9: Narrative architecture
    CommitmentDetection,
    FabulaExtraction,
    SjuzetExtraction,
    DramaticIrony,
    Focalization,
    CharacterArc,
    SubplotDetection,
    SceneSequel,
    SjuzetReordering,
    // Sprint D12: Adversarial narrative wargaming
    AdversaryPolicy,
    CognitiveHierarchy,
    Wargame,
    CounterNarrative,
    RewardFingerprint,
    Retrodiction,
    // Narrative-scoped inference forms for clauses also reachable via DISCOVER.
    // `INFER ARCS FOR n:Narrative WHERE n.narrative_id = "x" RETURN n` →
    // runs the Reagan arc classifier on a single narrative. `INFER PATTERNS`
    // and `INFER MISSING_EVENTS` are the corresponding per-narrative forms
    // for pattern mining and missing-event prediction.
    Arcs,
    Patterns,
    MissingEvents,
    /// Per-actor Reagan arc classification. Target semantics:
    /// - `INFER ACTOR_ARCS FOR n:Narrative WHERE n.narrative_id = "x"` →
    ///   classify every Actor in the narrative.
    /// - `INFER ARCS FOR e:Actor WHERE e.id = "..."` also dispatches here
    ///   (the dispatcher routes by target type).
    ActorArcs,
}

/// DISCOVER clause for cross-narrative pattern mining.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoverClause {
    pub discover_type: DiscoverType,
    pub target_binding: Option<String>,
    pub target_type: Option<String>,
}

/// Type of discovery to perform.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DiscoverType {
    Patterns,
    Arcs,
    Missing,
}

/// ACROSS NARRATIVES clause for cross-narrative queries.
#[derive(Debug, Clone, Serialize)]
pub struct AcrossClause {
    /// Specific narrative IDs to scope the query, or None for all narratives.
    pub narrative_ids: Option<Vec<String>>,
}

/// ASSUMING clause for counterfactual interventions.
#[derive(Debug, Clone, Serialize)]
pub struct AssumingClause {
    pub assumptions: Vec<Assumption>,
}

/// A single counterfactual assumption.
#[derive(Debug, Clone, Serialize)]
pub struct Assumption {
    pub field: String,
    pub value: QueryValue,
}

/// UNDER clause for game-theoretic constraints.
#[derive(Debug, Clone, Serialize)]
pub struct UnderClause {
    pub conditions: Vec<UnderCondition>,
}

/// A game-theoretic constraint.
#[derive(Debug, Clone, Serialize)]
pub enum UnderCondition {
    Rationality(f64),
    Information(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchClause {
    pub elements: Vec<PatternElement>,
}

#[derive(Debug, Clone, Serialize)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

#[derive(Debug, Clone, Serialize)]
pub struct NodePattern {
    pub binding: Option<String>,
    pub type_name: String,
    pub properties: Vec<PropPair>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgePattern {
    pub binding: Option<String>,
    pub rel_type: String,
    /// `true` for `-[...]->` or `<-[...]-`, `false` for undirected `-[...]-`.
    pub directed: bool,
    /// `true` only for `<-[...]-` (right-to-left directed edge).
    /// Ignored when `directed = false`.
    #[serde(default)]
    pub reversed: bool,
    pub properties: Vec<PropPair>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PropPair {
    pub key: String,
    pub value: QueryValue,
}

/// Boolean expression tree for WHERE clause conditions.
#[derive(Debug, Clone, Serialize)]
pub enum ConditionExpr {
    /// A single condition (leaf node).
    Single(Condition),
    /// Conjunction: all children must be true.
    And(Vec<ConditionExpr>),
    /// Disjunction: at least one child must be true.
    Or(Vec<ConditionExpr>),
}

#[derive(Debug, Clone, Serialize)]
pub struct WhereClause {
    pub expr: ConditionExpr,
}

impl WhereClause {
    /// Collect all leaf conditions from the expression tree (for tests and debugging).
    /// Traverses AND, OR, and Single nodes — returns all conditions regardless of boolean structure.
    pub fn all_conditions(&self) -> Vec<&Condition> {
        let mut out = Vec::new();
        Self::collect_all_conditions(&self.expr, &mut out);
        out
    }

    fn collect_all_conditions<'a>(expr: &'a ConditionExpr, out: &mut Vec<&'a Condition>) {
        match expr {
            ConditionExpr::Single(c) => out.push(c),
            ConditionExpr::And(children) | ConditionExpr::Or(children) => {
                for child in children {
                    Self::collect_all_conditions(child, out);
                }
            }
        }
    }
}

/// An inline graph function call (Level 2).
#[derive(Debug, Clone, Serialize)]
pub struct GraphFunc {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Condition {
    pub field: String,
    /// When set, the left-hand side is a graph function, not a field path.
    pub graph_func: Option<GraphFunc>,
    pub op: CompareOp,
    pub value: QueryValue,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    In,
    Contains,
}

#[derive(Debug, Clone, Serialize)]
pub struct AtClause {
    pub field: String,
    pub relation: String,
    pub value: QueryValue,
    /// Fuzzy Sprint Phase 5 — optional "AS FUZZY <rel> THRESHOLD <f>" tail.
    /// When `Some`, the executor switches from crisp [`relation_between`]
    /// filtering to graded-Allen thresholding via
    /// [`crate::fuzzy::allen::fuzzy_relation_holds`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fuzzy: Option<FuzzyAtConfig>,
}

/// Fuzzy Sprint Phase 5 — per-AT fuzzy configuration.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct FuzzyAtConfig {
    /// Relation whose graded degree is thresholded.
    pub relation: String,
    /// Minimum graded-Allen degree `[0.0, 1.0]` a candidate must reach.
    pub threshold: f64,
}

/// NEAR clause for vector similarity search.
#[derive(Debug, Clone, Serialize)]
pub struct NearClause {
    pub binding: String,
    pub query_text: String,
    pub k: usize,
}

/// SPATIAL clause for geospatial proximity filtering.
#[derive(Debug, Clone, Serialize)]
pub struct SpatialClause {
    pub field: String,
    pub radius_km: f64,
    pub center_lat: f64,
    pub center_lon: f64,
}

/// Retrieval mode for ASK queries (RAG).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RetrievalMode {
    Local,
    Global,
    Hybrid,
    Mix,
    Drift,
    /// LazyGraphRAG: no pre-computation. Vector search → local subgraph → on-demand community summary.
    Lazy,
    /// Personalized PageRank: seed from query-relevant entities, run PPR with restart.
    Ppr,
}

/// ASK clause for RAG (natural language question answering) queries.
#[derive(Debug, Clone, Serialize)]
pub struct AskClause {
    /// The natural language question.
    pub question: String,
    /// Optional narrative ID scope (from OVER clause).
    pub narrative_id: Option<String>,
    /// Optional retrieval mode (from MODE clause).
    pub mode: Option<RetrievalMode>,
    /// Optional desired response format (from RESPOND AS clause).
    pub response_type: Option<String>,
    /// Optional session ID for multi-turn conversation context.
    pub session_id: Option<String>,
    /// Whether to generate follow-up question suggestions.
    pub suggest: bool,
}

/// TUNE clause for prompt auto-tuning.
#[derive(Debug, Clone, Serialize)]
pub struct TuneClause {
    /// Narrative ID to tune prompts for.
    pub narrative_id: String,
}

/// Aggregate function type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// A return expression: either a field path or an aggregate function.
#[derive(Debug, Clone, Serialize)]
pub enum ReturnExpr {
    /// A field path (e.g., "e", "e.name", "*").
    Field(String),
    /// An aggregate function (e.g., COUNT(*), AVG(e.confidence)).
    Aggregate { func: AggregateFunc, field: String },
    /// An inline graph function (e.g., triangles(e), jaccard(a, b)).
    GraphFunction(GraphFunc),
}

impl ReturnExpr {
    /// Returns the field name if this is a Field variant.
    pub fn as_field(&self) -> Option<&str> {
        match self {
            ReturnExpr::Field(s) => Some(s),
            _ => None,
        }
    }

    /// Returns true if this is an aggregate expression.
    pub fn is_aggregate(&self) -> bool {
        matches!(self, ReturnExpr::Aggregate { .. })
    }
}

/// GROUP BY clause.
#[derive(Debug, Clone, Serialize)]
pub struct GroupByClause {
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReturnClause {
    pub expressions: Vec<ReturnExpr>,
    pub order_by: Option<OrderBy>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderBy {
    pub field: String,
    pub ascending: bool,
}

#[derive(Debug, Clone, Serialize)]
pub enum QueryValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
}

impl QueryValue {
    /// Convert to a serde_json::Value.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            QueryValue::String(s) => serde_json::json!(s),
            QueryValue::Integer(i) => serde_json::json!(i),
            QueryValue::Float(f) => serde_json::json!(f),
            QueryValue::Boolean(b) => serde_json::json!(b),
            QueryValue::Null => serde_json::Value::Null,
        }
    }
}

// ─── DML AST Types ─────────────────────────���────────────────

/// Top-level parsed statement: either a read query or a mutation.
#[derive(Debug, Clone)]
pub enum TensaStatement {
    Query(Box<TensaQuery>),
    Mutation(MutationStatement),
    Export {
        narrative_id: String,
        format: String,
    },
    /// Prompt auto-tuning for a narrative.
    Tune(TuneClause),
    /// EATH Phase 5 — calibrate a registered surrogate model against a
    /// real narrative. Model name is REQUIRED (no default — calibration
    /// is O(dataset) and downstream runs depend on its output, so
    /// ambiguity about which model was fitted is a real footgun).
    CalibrateSurrogate {
        narrative_id: String,
        model: String,
    },
    /// EATH Phase 5 — generate a synthetic narrative via a registered
    /// surrogate model. Defaults `model` to `"eath"` when the optional
    /// `USING SURROGATE '...'` clause is omitted (generation is per-call
    /// cheap and visible, so the convenience is safe — see
    /// docs/EATH_sprint.md Phase 5 spec for the asymmetry rationale).
    GenerateNarrative {
        output_id: String,
        source_id: String,
        model: String,
        params: Option<serde_json::Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<String>,
    },
    /// EATH Phase 9 — generate a synthetic narrative as a weighted mixture
    /// of `n` calibrated surrogate processes.
    /// `GENERATE NARRATIVE "<output_id>" USING HYBRID
    ///     FROM "<src_a>" WEIGHT 0.7, FROM "<src_b>" WEIGHT 0.3
    ///     [SEED <u64>] [STEPS <n>]`.
    /// Each component pairs `(narrative_id, weight)` — the model defaults
    /// to `"eath"` (Phase 9 only ships EATH-flavoured hybrid). Weights are
    /// validated to sum to 1.0 within `1e-6` at the planner.
    GenerateHybridNarrative {
        output_id: String,
        components: Vec<HybridComponentSpec>,
        seed: Option<u64>,
        num_steps: Option<usize>,
    },
    /// EATH Extension Phase 13c — dual-null-model significance.
    /// `COMPUTE DUAL_SIGNIFICANCE FOR "<narrative_id>" USING '<metric>'
    ///     [K_PER_MODEL <n>] [MODELS '<m1>','<m2>'...]`.
    /// Default `models` (when clause omitted): `["eath", "nudhy"]`.
    /// The planner validates each model against the registry; unknown names
    /// fail at plan time rather than at job execution.
    ComputeDualSignificance {
        narrative_id: String,
        metric: String,
        k_per_model: Option<u16>,
        models: Option<Vec<String>>,
    },
    /// EATH Extension Phase 15c — submit a SINDy hypergraph reconstruction job.
    /// `INFER HYPERGRAPH FROM DYNAMICS FOR "<narrative_id>"
    ///     [USING OBSERVATION '<source>'] [MAX_ORDER <n>] [LAMBDA <f>]`.
    /// Default observation source (when `USING OBSERVATION` omitted) is
    /// `participation_rate` — the only observation source fully implemented in
    /// Phase 15b. Other sources (`sentiment_mean`, `belief_mass`, `engagement`)
    /// are accepted by the parser but the engine returns
    /// `InferenceError("PrerequisiteMissing: ...")` for the first two and
    /// `InferenceError("MVP: ...")` for `engagement` until follow-up phases.
    /// Defaults: `max_order = 3`, `lambda` auto-selected via the λ_max
    /// heuristic when omitted.
    InferHypergraphReconstruction {
        narrative_id: String,
        observation: Option<String>,
        max_order: Option<usize>,
        lambda: Option<f32>,
        /// Fuzzy Sprint Phase 3 — optional trailing `WITH TNORM '<kind>'` /
        /// `AGGREGATE <kind>` clauses. Default `FuzzyConfig::default()`
        /// means no override (engine-specific defaults apply).
        fuzzy_config: FuzzyConfig,
    },
    /// EATH Extension Phase 16c — opinion-dynamics simulation (synchronous).
    /// `INFER OPINION_DYNAMICS( confidence_bound := <f>, variant := '<v>',
    ///   [mu := <f>], [initial := '<u|g|b>'] ) FOR "<narrative_id>"`.
    /// Maps to a synchronous executor step that calls
    /// [`crate::analysis::opinion_dynamics::simulate_opinion_dynamics`] and
    /// returns a descriptor row for the planner. Defaults applied at the
    /// engine layer for every omitted field.
    InferOpinionDynamics {
        narrative_id: String,
        confidence_bound: Option<f32>,
        variant: Option<String>,
        mu: Option<f32>,
        initial: Option<String>,
        /// Fuzzy Sprint Phase 3 — optional trailing fuzzy-config clauses.
        fuzzy_config: FuzzyConfig,
    },
    /// EATH Extension Phase 16c — opinion phase-transition sweep
    /// (synchronous). `INFER OPINION_PHASE_TRANSITION( c_start := <f>,
    /// c_end := <f>, c_steps := <n> ) FOR "<narrative_id>"`.
    InferOpinionPhaseTransition {
        narrative_id: String,
        c_start: f32,
        c_end: f32,
        c_steps: usize,
        /// Fuzzy Sprint Phase 3 — optional trailing fuzzy-config clauses.
        fuzzy_config: FuzzyConfig,
    },
    /// Fuzzy Sprint Phase 6 — intermediate-quantifier statement.
    /// `QUANTIFY MOST (e:Actor) WHERE e.confidence > 0.7 FOR "<nid>" AS "<label>"`
    /// — computes the graded truth value `Q(r)` where `r` is the fraction
    /// of domain members satisfying `WHERE`. Executes synchronously.
    Quantify {
        /// Canonical quantifier name (lowercased at parse time).
        quantifier: String,
        /// Node-pattern binding name (`e` in `(e:Actor)`).
        binding: Option<String>,
        /// Node-pattern type name (`Actor` / `Location` / `Situation` / ...).
        type_name: String,
        /// Optional WHERE condition expression.
        where_clause: Option<WhereClause>,
        /// Narrative id from the optional `FOR "<nid>"` clause.
        narrative_id: Option<String>,
        /// Result-row column name (from the optional `AS "<label>"`
        /// clause). Defaults to `"quantifier_result"` at the executor.
        label: Option<String>,
    },
    /// Fuzzy Sprint Phase 7 — graded syllogism verification.
    /// `VERIFY SYLLOGISM { major: '<stmt>', minor: '<stmt>',
    /// conclusion: '<stmt>' } FOR "<nid>" [THRESHOLD <f>]
    /// [WITH TNORM '<kind>']`. Executes synchronously; returns a
    /// graded-validity descriptor row. Cites: [murinovanovak2014peterson].
    VerifySyllogism {
        /// Raw tiny-DSL string for the major premise.
        major: String,
        /// Raw tiny-DSL string for the minor premise.
        minor: String,
        /// Raw tiny-DSL string for the conclusion.
        conclusion: String,
        /// Target narrative id.
        narrative_id: String,
        /// Optional validity threshold; defaults to `0.5`.
        threshold: Option<f64>,
        /// Optional `WITH TNORM '<kind>'` override; defaults to Gödel.
        fuzzy_config: FuzzyConfig,
    },
    /// Fuzzy Sprint Phase 8 — build + persist a graded concept lattice.
    /// `FCA LATTICE FOR "<nid>" [THRESHOLD <n>] [ATTRIBUTES [...]]
    /// [ENTITY_TYPE <type>] [WITH TNORM '<kind>']`. Synchronous; returns
    /// a descriptor row {lattice_id, num_concepts, num_objects,
    /// num_attributes, tnorm}.
    /// Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].
    FcaLattice {
        narrative_id: String,
        threshold: Option<usize>,
        attribute_allowlist: Option<Vec<String>>,
        entity_type: Option<String>,
        fuzzy_config: FuzzyConfig,
    },
    /// Fuzzy Sprint Phase 8 — retrieve one concept from a persisted
    /// lattice. `FCA CONCEPT <idx> FROM "<lattice_id>"`.
    FcaConcept {
        lattice_id: String,
        concept_idx: usize,
    },
    /// Fuzzy Sprint Phase 9 — evaluate a Mamdani rule set against every
    /// entity in a narrative that matches the entity-type pattern.
    /// `EVALUATE RULES FOR "<nid>" AGAINST (e:Actor)
    ///     [RULES ['rule-a', 'rule-b']] [WITH TNORM '<kind>']`. Cites:
    /// [mamdani1975mamdani].
    EvaluateRules {
        narrative_id: String,
        entity_type: String,
        rule_ids: Option<Vec<String>>,
        fuzzy_config: FuzzyConfig,
    },
    /// Fuzzy Sprint Phase 10 — hybrid fuzzy-probability query.
    /// `INFER FUZZY_PROBABILITY(event_kind := '<kind>', event_ref :=
    /// '<payload>', distribution := '<spec>') FOR "<nid>" [WITH TNORM
    /// '<k>']`. Executes synchronously; returns
    /// `{ value, event_kind, distribution_summary, _narrative_id }`.
    /// Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
    FuzzyProbability {
        narrative_id: String,
        event_kind: String,
        event_ref: String,
        distribution: String,
        /// Optional `WITH TNORM '<kind>'`. Carried through for
        /// forward-compat — Phase 10 base case does not consume it.
        fuzzy_config: FuzzyConfig,
    },
}

/// Phase 9 hybrid component as parsed from `FROM "<nid>" WEIGHT <w>`. The
/// model is intentionally NOT carried at the grammar level — Phase 9 ships
/// `"eath"` exclusively; future surrogate families add a `USING SURROGATE
/// '<name>'` per-component clause when the registry has more than one
/// non-trivial implementation.
#[derive(Debug, Clone, Serialize)]
pub struct HybridComponentSpec {
    pub narrative_id: String,
    pub weight: f32,
}

/// A DML mutation statement.
#[derive(Debug, Clone, Serialize)]
pub enum MutationStatement {
    CreateNarrative {
        id: String,
        title: Option<String>,
        genre: Option<String>,
        tags: Vec<String>,
    },
    CreateEntity {
        entity_type: String,
        properties: Vec<PropPair>,
        narrative_id: Option<String>,
        confidence: Option<f64>,
    },
    CreateSituation {
        level: String,
        content: Vec<String>,
        narrative_id: Option<String>,
        confidence: Option<f64>,
    },
    DeleteEntity {
        id: String,
    },
    DeleteSituation {
        id: String,
    },
    DeleteNarrative {
        id: String,
    },
    UpdateEntity {
        id: String,
        set_pairs: Vec<(String, QueryValue)>,
    },
    UpdateNarrative {
        id: String,
        set_pairs: Vec<(String, QueryValue)>,
    },
    AddParticipant {
        entity_id: String,
        situation_id: String,
        role: String,
        action: Option<String>,
    },
    RemoveParticipant {
        entity_id: String,
        situation_id: String,
    },
    AddCause {
        from_id: String,
        to_id: String,
        causal_type: Option<String>,
        strength: Option<f64>,
        mechanism: Option<String>,
    },
    RemoveCause {
        from_id: String,
        to_id: String,
    },
}

// ─── Parser Implementation ───────────────────────────────────

/// Parse a TensaQL statement (query or mutation) into an AST.
pub fn parse_statement(input: &str) -> Result<TensaStatement> {
    let pairs = TensaParser::parse(Rule::query, input)
        .map_err(|e| TensaError::ParseError(e.to_string()))?;

    let query_pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| TensaError::ParseError("Empty parse result".into()))?;

    for pair in query_pair.into_inner() {
        match pair.as_rule() {
            Rule::mutation_query => {
                let inner = pair
                    .into_inner()
                    .next()
                    .ok_or_else(|| TensaError::ParseError("Empty mutation".into()))?;
                return Ok(TensaStatement::Mutation(parse_mutation_stmt(inner)?));
            }
            Rule::export_query => {
                let mut inner = pair.into_inner();
                let narrative_id_pair = inner.next().ok_or_else(|| {
                    TensaError::ParseError("Missing narrative ID in EXPORT".into())
                })?;
                let narrative_id = extract_string(&narrative_id_pair);
                let format_pair = inner
                    .next()
                    .ok_or_else(|| TensaError::ParseError("Missing format in EXPORT".into()))?;
                let format = format_pair.as_str().to_lowercase();
                return Ok(TensaStatement::Export {
                    narrative_id,
                    format,
                });
            }
            Rule::tune_query => {
                let narrative_id = pair
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::string_value)
                    .and_then(|p| {
                        p.into_inner()
                            .find(|q| q.as_rule() == Rule::inner_string)
                            .map(|q| q.as_str().to_string())
                    })
                    .ok_or_else(|| {
                        TensaError::ParseError("Missing narrative ID in TUNE PROMPTS".into())
                    })?;
                return Ok(TensaStatement::Tune(TuneClause { narrative_id }));
            }
            Rule::calibrate_surrogate_query => {
                return parse_calibrate_surrogate_query(pair);
            }
            Rule::generate_narrative_query => {
                return parse_generate_narrative_query(pair);
            }
            Rule::generate_hybrid_narrative_query => {
                return parse_generate_hybrid_narrative_query(pair);
            }
            Rule::dual_significance_query => {
                return parse_dual_significance_query(pair);
            }
            Rule::infer_hypergraph_reconstruction_query => {
                return parse_infer_hypergraph_reconstruction_query(pair);
            }
            Rule::opinion_dynamics_query => {
                return parse_opinion_dynamics_query(pair);
            }
            Rule::opinion_phase_transition_query => {
                return parse_opinion_phase_transition_query(pair);
            }
            Rule::quantify_query => {
                return parse_quantify_query(pair);
            }
            Rule::verify_syllogism_query => {
                return parse_verify_syllogism_query(pair);
            }
            Rule::fca_lattice_query => {
                return parse_fca_lattice_query(pair);
            }
            Rule::fca_concept_query => {
                return parse_fca_concept_query(pair);
            }
            Rule::evaluate_rules_query => {
                return parse_evaluate_rules_query(pair);
            }
            Rule::fuzzy_probability_query => {
                return parse_fuzzy_probability_query(pair);
            }
            Rule::explain_query
            | Rule::match_query
            | Rule::infer_query
            | Rule::discover_query
            | Rule::ask_query
            | Rule::path_query => {
                // Re-parse as a read query
                let q = parse_query(input)?;
                return Ok(TensaStatement::Query(Box::new(q)));
            }
            Rule::EOI => {}
            _ => {}
        }
    }

    Err(TensaError::ParseError("Could not parse statement".into()))
}

/// Parse a TensaQL query string into an AST (backward-compatible).
pub fn parse_query(input: &str) -> Result<TensaQuery> {
    let pairs = TensaParser::parse(Rule::query, input)
        .map_err(|e| TensaError::ParseError(e.to_string()))?;

    let query_pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| TensaError::ParseError("Empty parse result".into()))?;

    let mut explain = false;
    let mut infer_clause = None;
    let mut discover_clause = None;
    let mut ask_clause = None;
    let mut across_clause = None;
    let mut match_clause = None;
    let mut path_clause = None;
    let mut flow_clause = None;
    let mut where_clause = None;
    let mut at_clause = None;
    let mut near_clause = None;
    let mut spatial_clause = None;
    let mut assuming_clause = None;
    let mut under_clause = None;
    let mut group_by = None;
    let mut as_of = None;
    let mut return_clause = None;
    let mut fuzzy_config = FuzzyConfig::default();

    #[allow(clippy::too_many_arguments)]
    fn extract_clauses(
        pair: pest::iterators::Pair<Rule>,
        infer_clause: &mut Option<InferClause>,
        discover_clause: &mut Option<DiscoverClause>,
        ask_clause: &mut Option<AskClause>,
        across_clause: &mut Option<AcrossClause>,
        match_clause: &mut Option<MatchClause>,
        path_clause: &mut Option<PathClause>,
        where_clause: &mut Option<WhereClause>,
        at_clause: &mut Option<AtClause>,
        near_clause: &mut Option<NearClause>,
        spatial_clause: &mut Option<SpatialClause>,
        assuming_clause: &mut Option<AssumingClause>,
        under_clause: &mut Option<UnderClause>,
        group_by: &mut Option<GroupByClause>,
        as_of: &mut Option<String>,
        return_clause: &mut Option<ReturnClause>,
        fuzzy_config: &mut FuzzyConfig,
    ) -> Result<()> {
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::infer_clause => *infer_clause = Some(parse_infer_clause(inner)?),
                Rule::discover_clause => *discover_clause = Some(parse_discover_clause(inner)?),
                Rule::across_clause => *across_clause = Some(parse_across_clause(inner)?),
                Rule::match_clause => *match_clause = Some(parse_match_clause(inner)?),
                Rule::path_mode => {
                    // path_mode is handled inside parse_path_query
                }
                Rule::path_pattern => {
                    // path_pattern is handled inside parse_path_query
                }
                Rule::where_clause => *where_clause = Some(parse_where_clause(inner)?),
                Rule::at_clause => *at_clause = Some(parse_at_clause(inner)?),
                Rule::near_clause => *near_clause = Some(parse_near_clause(inner)?),
                Rule::spatial_clause => *spatial_clause = Some(parse_spatial_clause(inner)?),
                Rule::assuming_clause => *assuming_clause = Some(parse_assuming_clause(inner)?),
                Rule::under_clause => *under_clause = Some(parse_under_clause(inner)?),
                Rule::group_by_clause => *group_by = Some(parse_group_by_clause(inner)?),
                Rule::as_of_clause => {
                    // AS OF "timestamp" — extract the string value
                    for p in inner.into_inner() {
                        if p.as_rule() == Rule::string_value {
                            let raw = p.as_str();
                            let trimmed = raw.trim_matches('"');
                            *as_of = Some(trimmed.to_string());
                        }
                    }
                }
                Rule::with_tnorm_clause => {
                    fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
                }
                Rule::with_aggregator_clause => {
                    fuzzy_config.aggregator = Some(parse_with_aggregator_clause(inner)?);
                }
                Rule::return_clause => *return_clause = Some(parse_return_clause(inner)?),
                Rule::over_clause => {
                    // Handled inside parse_ask_query
                }
                Rule::mode_clause => {
                    // Handled inside parse_ask_query
                }
                Rule::retrieval_mode => {
                    // Handled inside parse_ask_query
                }
                Rule::respond_clause => {
                    // Handled inside parse_ask_query
                }
                Rule::session_clause => {
                    // Handled inside parse_ask_query
                }
                Rule::suggest_clause => {
                    // Handled inside parse_ask_query
                }
                Rule::string_value => {
                    // Handled by parent rule (ask_query question)
                }
                // Recurse into nested query types (e.g., inside explain_query)
                Rule::match_query | Rule::infer_query | Rule::discover_query => {
                    extract_clauses(
                        inner,
                        infer_clause,
                        discover_clause,
                        ask_clause,
                        across_clause,
                        match_clause,
                        path_clause,
                        where_clause,
                        at_clause,
                        near_clause,
                        spatial_clause,
                        assuming_clause,
                        under_clause,
                        group_by,
                        as_of,
                        return_clause,
                        fuzzy_config,
                    )?;
                }
                Rule::ask_query => {
                    *ask_clause = Some(parse_ask_query(inner.clone())?);
                    // Also extract return_clause from within the ask_query
                    extract_clauses(
                        inner,
                        infer_clause,
                        discover_clause,
                        ask_clause,
                        across_clause,
                        match_clause,
                        path_clause,
                        where_clause,
                        at_clause,
                        near_clause,
                        spatial_clause,
                        assuming_clause,
                        under_clause,
                        group_by,
                        as_of,
                        return_clause,
                        fuzzy_config,
                    )?;
                }
                Rule::path_query => {
                    *path_clause = Some(parse_path_query(inner.clone())?);
                    // Also extract where/return from within the path_query
                    extract_clauses(
                        inner,
                        infer_clause,
                        discover_clause,
                        ask_clause,
                        across_clause,
                        match_clause,
                        path_clause,
                        where_clause,
                        at_clause,
                        near_clause,
                        spatial_clause,
                        assuming_clause,
                        under_clause,
                        group_by,
                        as_of,
                        return_clause,
                        fuzzy_config,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    for pair in query_pair.into_inner() {
        match pair.as_rule() {
            Rule::explain_query => {
                explain = true;
                extract_clauses(
                    pair,
                    &mut infer_clause,
                    &mut discover_clause,
                    &mut ask_clause,
                    &mut across_clause,
                    &mut match_clause,
                    &mut path_clause,
                    &mut where_clause,
                    &mut at_clause,
                    &mut near_clause,
                    &mut spatial_clause,
                    &mut assuming_clause,
                    &mut under_clause,
                    &mut group_by,
                    &mut as_of,
                    &mut return_clause,
                    &mut fuzzy_config,
                )?;
            }
            Rule::path_query => {
                path_clause = Some(parse_path_query(pair.clone())?);
                extract_clauses(
                    pair,
                    &mut infer_clause,
                    &mut discover_clause,
                    &mut ask_clause,
                    &mut across_clause,
                    &mut match_clause,
                    &mut path_clause,
                    &mut where_clause,
                    &mut at_clause,
                    &mut near_clause,
                    &mut spatial_clause,
                    &mut assuming_clause,
                    &mut under_clause,
                    &mut group_by,
                    &mut as_of,
                    &mut return_clause,
                    &mut fuzzy_config,
                )?;
            }
            Rule::ask_query => {
                ask_clause = Some(parse_ask_query(pair.clone())?);
                extract_clauses(
                    pair,
                    &mut infer_clause,
                    &mut discover_clause,
                    &mut ask_clause,
                    &mut across_clause,
                    &mut match_clause,
                    &mut path_clause,
                    &mut where_clause,
                    &mut at_clause,
                    &mut near_clause,
                    &mut spatial_clause,
                    &mut assuming_clause,
                    &mut under_clause,
                    &mut group_by,
                    &mut as_of,
                    &mut return_clause,
                    &mut fuzzy_config,
                )?;
            }
            Rule::flow_query => {
                flow_clause = Some(parse_flow_query(pair.clone())?);
                // Extract return_clause from within the flow_query
                for inner in pair.into_inner() {
                    if inner.as_rule() == Rule::return_clause {
                        return_clause = Some(parse_return_clause(inner)?);
                    }
                }
            }
            Rule::match_query | Rule::infer_query | Rule::discover_query => {
                extract_clauses(
                    pair,
                    &mut infer_clause,
                    &mut discover_clause,
                    &mut ask_clause,
                    &mut across_clause,
                    &mut match_clause,
                    &mut path_clause,
                    &mut where_clause,
                    &mut at_clause,
                    &mut near_clause,
                    &mut spatial_clause,
                    &mut assuming_clause,
                    &mut under_clause,
                    &mut group_by,
                    &mut as_of,
                    &mut return_clause,
                    &mut fuzzy_config,
                )?;
            }
            Rule::EOI => {}
            _ => {}
        }
    }

    // ASK queries may omit RETURN clause — provide a default
    let resolved_return = if ask_clause.is_some() && return_clause.is_none() {
        ReturnClause {
            expressions: vec![ReturnExpr::Field("*".to_string())],
            order_by: None,
            limit: None,
        }
    } else {
        return_clause.ok_or_else(|| TensaError::ParseError("Missing RETURN clause".into()))?
    };

    Ok(TensaQuery {
        explain,
        infer_clause,
        discover_clause,
        ask_clause,
        across_clause,
        match_clause,
        path_clause,
        flow_clause,
        where_clause,
        at_clause,
        near_clause,
        spatial_clause,
        assuming_clause,
        under_clause,
        group_by,
        as_of,
        return_clause: resolved_return,
        fuzzy_config,
    })
}

// ─── Fuzzy Sprint Phase 3 — WITH TNORM / AGGREGATE clause parsers ───────────
// Cites: [klement2000] [yager1988owa] [grabisch1996choquet].
//
// Each helper only parses and returns the parsed Kind; the planner performs
// the registry lookup + OWA-weight / Choquet-measure validation. We keep the
// parser pure on purpose — unknown t-norm names are strings at this layer and
// only become `TensaError::InvalidInput` when `plan_query` calls
// `TNormRegistry::default().get(...)`.

/// Extract the raw t-norm name from `WITH TNORM '<name>'` — does NOT validate.
/// Unknown names parse cleanly at this layer; the planner resolves via
/// `TNormRegistry::default().get(name)` and surfaces
/// `TensaError::InvalidInput` with the "known: [...]" list so parse errors
/// stay limited to syntactic problems.
pub(crate) fn parse_with_tnorm_clause(pair: pest::iterators::Pair<Rule>) -> Result<String> {
    let name_pair = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::model_name)
        .ok_or_else(|| TensaError::ParseError("WITH TNORM missing '<name>' value".into()))?;
    extract_model_name(&name_pair)
}

/// Resolve `AGGREGATE <kind>` to an [`AggregatorSpec`]. Parameterised variants
/// (OWA / Choquet / TNormReduce / TConormReduce) are constructed here;
/// t-norm names are resolved immediately via `TNormRegistry` (unknown → parse
/// error) but Choquet measure refs are preserved as strings — the planner
/// resolves them via the measure registry Phase 4 ships, and errors with
/// `TensaError::InvalidInput` ahead of execute time if the ref is unknown.
pub(crate) fn parse_with_aggregator_clause(
    pair: pest::iterators::Pair<Rule>,
) -> Result<AggregatorSpec> {
    let kind_pair = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::aggregator_kind)
        .ok_or_else(|| TensaError::ParseError("AGGREGATE missing aggregator kind".into()))?;
    let inner = kind_pair
        .into_inner()
        .next()
        .ok_or_else(|| TensaError::ParseError("AGGREGATE aggregator kind is empty".into()))?;
    match inner.as_rule() {
        Rule::mean_kind => Ok(AggregatorSpec::Mean),
        Rule::median_kind => Ok(AggregatorSpec::Median),
        Rule::owa_kind => {
            let weights = parse_weight_vector(inner)?;
            Ok(AggregatorSpec::Owa(weights))
        }
        Rule::choquet_kind => {
            let name_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::model_name)
                .ok_or_else(|| {
                    TensaError::ParseError("AGGREGATE CHOQUET missing measure reference".into())
                })?;
            let measure_name = extract_model_name(&name_pair)?;
            Ok(AggregatorSpec::ChoquetByRef(measure_name))
        }
        Rule::tnorm_reduce_kind => {
            let name_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::model_name)
                .ok_or_else(|| {
                    TensaError::ParseError("AGGREGATE TNORM_REDUCE missing '<tnorm>'".into())
                })?;
            let name = extract_model_name(&name_pair)?;
            let tnorm = crate::fuzzy::registry::TNormRegistry::default().get(&name)?;
            Ok(AggregatorSpec::TNormReduce(tnorm))
        }
        Rule::tconorm_reduce_kind => {
            let name_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::model_name)
                .ok_or_else(|| {
                    TensaError::ParseError("AGGREGATE TCONORM_REDUCE missing '<tnorm>'".into())
                })?;
            let name = extract_model_name(&name_pair)?;
            let tnorm = crate::fuzzy::registry::TNormRegistry::default().get(&name)?;
            Ok(AggregatorSpec::TConormReduce(tnorm))
        }
        other => Err(TensaError::ParseError(format!(
            "unexpected aggregator kind rule: {:?}",
            other
        ))),
    }
}

/// Parse `OWA [w1, w2, ...]` weight vector. Empty vectors error out of the
/// parser because no downstream layer has a legitimate use for a zero-weight
/// OWA. Non-empty lists are handed to the `OwaAggregator` without normalising
/// — callers wanting auto-normalisation go through
/// `crate::fuzzy::aggregation_owa::owa_normalize`.
fn parse_weight_vector(pair: pest::iterators::Pair<Rule>) -> Result<Vec<f64>> {
    let vec_pair = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::weight_vector)
        .ok_or_else(|| TensaError::ParseError("OWA missing weight vector".into()))?;
    let mut weights: Vec<f64> = Vec::new();
    for p in vec_pair.into_inner() {
        if p.as_rule() == Rule::weight_entry {
            let w: f64 = p.as_str().parse::<f64>().map_err(|e| {
                TensaError::ParseError(format!("OWA weight '{}' not a float: {e}", p.as_str()))
            })?;
            weights.push(w);
        }
    }
    if weights.is_empty() {
        return Err(TensaError::ParseError(
            "OWA weight vector must contain at least one entry".into(),
        ));
    }
    Ok(weights)
}

fn parse_path_query(pair: pest::iterators::Pair<Rule>) -> Result<PathClause> {
    let mut mode = PathMode::Shortest;
    let mut restrictor = PathRestrictor::Walk;
    let mut start_binding = String::new();
    let mut end_binding = String::new();
    let mut rel_type = String::new();
    let mut min_depth: usize = 1;
    let mut max_depth: usize = 10;
    let mut top_k: Option<usize> = None;
    let mut weight_field: Option<String> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::path_mode => {
                let text = inner.as_str().to_uppercase();
                mode = match text.as_str() {
                    "ALL" => PathMode::All,
                    "LONGEST" => PathMode::Longest,
                    _ => PathMode::Shortest,
                };
            }
            Rule::path_top_k => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::integer {
                        top_k = p.as_str().trim().parse().ok();
                    }
                }
            }
            Rule::path_weight => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::field_path {
                        weight_field = Some(p.as_str().to_string());
                    }
                }
            }
            Rule::path_restrictor => {
                let text = inner.as_str().to_uppercase();
                restrictor = match text.as_str() {
                    "TRAIL" => PathRestrictor::Trail,
                    "ACYCLIC" => PathRestrictor::Acyclic,
                    "SIMPLE" => PathRestrictor::Simple,
                    _ => PathRestrictor::Walk,
                };
            }
            Rule::path_pattern => {
                let mut bindings = Vec::new();
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::binding => bindings.push(p.as_str().to_string()),
                        Rule::rel_type => rel_type = p.as_str().to_string(),
                        Rule::depth_range => {
                            let nums: Vec<&str> = p.as_str().split("..").collect();
                            if nums.len() == 2 {
                                min_depth = nums[0].trim().parse().unwrap_or(1);
                                max_depth = nums[1].trim().parse().unwrap_or(10);
                            }
                        }
                        _ => {}
                    }
                }
                if bindings.len() >= 2 {
                    start_binding = bindings[0].clone();
                    end_binding = bindings[1].clone();
                }
            }
            _ => {} // where_clause, return_clause handled by extract_clauses
        }
    }

    Ok(PathClause {
        mode,
        restrictor,
        start_binding,
        end_binding,
        rel_type,
        min_depth,
        max_depth,
        top_k,
        weight_field,
    })
}

/// Parse a MATCH FLOW query.
fn parse_flow_query(pair: pest::iterators::Pair<Rule>) -> Result<FlowClause> {
    let mut flow_type = FlowType::Max;
    let mut start_binding = String::new();
    let mut end_binding = String::new();
    let mut rel_type = String::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::flow_type => {
                let text = inner.as_str().to_uppercase();
                flow_type = if text == "MIN_CUT" {
                    FlowType::MinCut
                } else {
                    FlowType::Max
                };
            }
            Rule::path_pattern => {
                let mut bindings = Vec::new();
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::binding => bindings.push(p.as_str().to_string()),
                        Rule::rel_type => rel_type = p.as_str().to_string(),
                        _ => {}
                    }
                }
                if bindings.len() >= 2 {
                    start_binding = bindings[0].clone();
                    end_binding = bindings[1].clone();
                }
            }
            _ => {} // return_clause handled separately
        }
    }

    Ok(FlowClause {
        flow_type,
        start_binding,
        end_binding,
        rel_type,
    })
}

fn parse_match_clause(pair: pest::iterators::Pair<Rule>) -> Result<MatchClause> {
    let mut elements = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::pattern {
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::node_pattern => elements.push(PatternElement::Node(parse_node(p)?)),
                    Rule::edge_pattern => elements.push(PatternElement::Edge(parse_edge(p)?)),
                    _ => {}
                }
            }
        }
    }
    Ok(MatchClause { elements })
}

fn parse_node(pair: pest::iterators::Pair<Rule>) -> Result<NodePattern> {
    let mut binding = None;
    let mut type_name = String::new();
    let mut properties = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binding => binding = Some(inner.as_str().to_string()),
            Rule::type_name => type_name = inner.as_str().to_string(),
            Rule::properties => properties = parse_properties(inner)?,
            _ => {}
        }
    }
    Ok(NodePattern {
        binding,
        type_name,
        properties,
    })
}

fn parse_edge(pair: pest::iterators::Pair<Rule>) -> Result<EdgePattern> {
    let inner = pair.into_inner().next().unwrap();
    let (directed, reversed) = match inner.as_rule() {
        Rule::directed_edge => (true, false),
        Rule::reverse_edge => (true, true),
        _ => (false, false),
    };

    let mut binding = None;
    let mut rel_type = String::new();
    let mut properties = Vec::new();

    for p in inner.into_inner() {
        match p.as_rule() {
            Rule::binding => binding = Some(p.as_str().to_string()),
            Rule::rel_type => rel_type = p.as_str().to_string(),
            Rule::properties => properties = parse_properties(p)?,
            _ => {}
        }
    }
    Ok(EdgePattern {
        binding,
        rel_type,
        directed,
        reversed,
        properties,
    })
}

fn parse_properties(pair: pest::iterators::Pair<Rule>) -> Result<Vec<PropPair>> {
    let mut props = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::prop_pair {
            let mut key = String::new();
            let mut value = QueryValue::Null;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::binding => key = p.as_str().to_string(),
                    Rule::value => value = parse_value(p)?,
                    _ => {}
                }
            }
            props.push(PropPair { key, value });
        }
    }
    Ok(props)
}

fn parse_where_clause(pair: pest::iterators::Pair<Rule>) -> Result<WhereClause> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| TensaError::ParseError("Empty WHERE clause".into()))?;
    let expr = parse_bool_expr(inner)?;
    Ok(WhereClause { expr })
}

fn parse_bool_expr(pair: pest::iterators::Pair<Rule>) -> Result<ConditionExpr> {
    let mut terms: Vec<ConditionExpr> = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::bool_term {
            terms.push(parse_bool_term(inner)?);
        }
    }
    if terms.len() == 1 {
        Ok(terms.remove(0))
    } else {
        Ok(ConditionExpr::Or(terms))
    }
}

fn parse_bool_term(pair: pest::iterators::Pair<Rule>) -> Result<ConditionExpr> {
    let mut factors: Vec<ConditionExpr> = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::bool_factor {
            factors.push(parse_bool_factor(inner)?);
        }
    }
    if factors.len() == 1 {
        Ok(factors.remove(0))
    } else {
        Ok(ConditionExpr::And(factors))
    }
}

fn parse_bool_factor(pair: pest::iterators::Pair<Rule>) -> Result<ConditionExpr> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::bool_expr => return parse_bool_expr(inner),
            Rule::condition => return Ok(ConditionExpr::Single(parse_condition(inner)?)),
            _ => {}
        }
    }
    Err(TensaError::ParseError("Invalid boolean factor".into()))
}

fn parse_condition(pair: pest::iterators::Pair<Rule>) -> Result<Condition> {
    let mut field = String::new();
    let mut graph_func = None;
    let mut op = CompareOp::Eq;
    let mut value = QueryValue::Null;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::field_path => field = inner.as_str().to_string(),
            Rule::graph_func => {
                let gf = parse_graph_func(inner)?;
                // Use the function signature as the "field" for display
                field = format!("{}({})", gf.name, gf.args.join(", "));
                graph_func = Some(gf);
            }
            Rule::comparator => op = parse_comparator(inner)?,
            Rule::value => value = parse_value(inner)?,
            _ => {}
        }
    }
    Ok(Condition {
        field,
        graph_func,
        op,
        value,
    })
}

fn parse_graph_func(pair: pest::iterators::Pair<Rule>) -> Result<GraphFunc> {
    let mut name = String::new();
    let mut args = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::graph_func_name => name = inner.as_str().to_lowercase(),
            Rule::func_args => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::binding {
                        args.push(p.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(GraphFunc { name, args })
}

fn parse_comparator(pair: pest::iterators::Pair<Rule>) -> Result<CompareOp> {
    let s = pair.as_str().to_uppercase();
    match s.as_str() {
        "=" => Ok(CompareOp::Eq),
        "!=" => Ok(CompareOp::Ne),
        ">" => Ok(CompareOp::Gt),
        "<" => Ok(CompareOp::Lt),
        ">=" => Ok(CompareOp::Gte),
        "<=" => Ok(CompareOp::Lte),
        "IN" => Ok(CompareOp::In),
        "CONTAINS" => Ok(CompareOp::Contains),
        other => Err(TensaError::ParseError(format!(
            "Unknown comparator: {}",
            other
        ))),
    }
}

fn parse_at_clause(pair: pest::iterators::Pair<Rule>) -> Result<AtClause> {
    let mut field = String::new();
    let mut relation = String::new();
    let mut value = QueryValue::Null;
    let mut fuzzy: Option<FuzzyAtConfig> = None;

    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::temporal_expr => {
                for inner in child.into_inner() {
                    match inner.as_rule() {
                        Rule::field_path => field = inner.as_str().to_string(),
                        Rule::allen_op => relation = inner.as_str().to_uppercase(),
                        Rule::value => value = parse_value(inner)?,
                        _ => {}
                    }
                }
            }
            Rule::fuzzy_at_tail => {
                let mut fuzzy_relation = String::new();
                let mut threshold = 0.0f64;
                for inner in child.into_inner() {
                    match inner.as_rule() {
                        Rule::fuzzy_allen_rel => {
                            fuzzy_relation = inner.as_str().to_uppercase();
                        }
                        Rule::float_value => {
                            threshold = inner.as_str().parse::<f64>().map_err(|e| {
                                TensaError::ParseError(format!(
                                    "Invalid THRESHOLD float '{}': {}",
                                    inner.as_str(),
                                    e
                                ))
                            })?;
                        }
                        _ => {}
                    }
                }
                if !(0.0..=1.0).contains(&threshold) {
                    return Err(TensaError::ParseError(format!(
                        "AS FUZZY THRESHOLD must be in [0, 1], got {}",
                        threshold
                    )));
                }
                fuzzy = Some(FuzzyAtConfig {
                    relation: fuzzy_relation,
                    threshold,
                });
            }
            _ => {}
        }
    }

    if field.is_empty() || relation.is_empty() {
        return Err(TensaError::ParseError("Missing temporal expression".into()));
    }

    Ok(AtClause {
        field,
        relation,
        value,
        fuzzy,
    })
}

fn parse_return_expr(pair: pest::iterators::Pair<Rule>) -> Result<ReturnExpr> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::agg_expr => {
                let mut func = AggregateFunc::Count;
                let mut field = String::new();
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::agg_func => {
                            func = match p.as_str().to_uppercase().as_str() {
                                "COUNT" => AggregateFunc::Count,
                                "SUM" => AggregateFunc::Sum,
                                "AVG" => AggregateFunc::Avg,
                                "MIN" => AggregateFunc::Min,
                                "MAX" => AggregateFunc::Max,
                                _ => AggregateFunc::Count,
                            };
                        }
                        Rule::agg_arg => {
                            field = p.as_str().to_string();
                        }
                        _ => {}
                    }
                }
                return Ok(ReturnExpr::Aggregate { func, field });
            }
            Rule::graph_func => {
                return Ok(ReturnExpr::GraphFunction(parse_graph_func(inner)?));
            }
            Rule::return_field => {
                return Ok(ReturnExpr::Field(inner.as_str().to_string()));
            }
            _ => {}
        }
    }
    Err(TensaError::ParseError("Invalid return expression".into()))
}

fn parse_group_by_clause(pair: pest::iterators::Pair<Rule>) -> Result<GroupByClause> {
    let mut fields = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::field_path {
            fields.push(inner.as_str().to_string());
        }
    }
    Ok(GroupByClause { fields })
}

fn parse_return_clause(pair: pest::iterators::Pair<Rule>) -> Result<ReturnClause> {
    let mut expressions = Vec::new();
    let mut order_by = None;
    let mut limit = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::return_expr => {
                expressions.push(parse_return_expr(inner)?);
            }
            Rule::order_by => {
                let mut field = String::new();
                let mut ascending = true;
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::field_path => field = p.as_str().to_string(),
                        Rule::sort_dir => {
                            if p.as_str().eq_ignore_ascii_case("DESC") {
                                ascending = false;
                            }
                        }
                        _ => {}
                    }
                }
                order_by = Some(OrderBy { field, ascending });
            }
            Rule::limit => {
                if let Some(int_pair) = inner.into_inner().find(|p| p.as_rule() == Rule::integer) {
                    limit = Some(
                        int_pair
                            .as_str()
                            .parse::<usize>()
                            .map_err(|e| TensaError::ParseError(e.to_string()))?,
                    );
                }
            }
            _ => {}
        }
    }
    Ok(ReturnClause {
        expressions,
        order_by,
        limit,
    })
}

fn parse_near_clause(pair: pest::iterators::Pair<Rule>) -> Result<NearClause> {
    let mut binding = String::new();
    let mut query_text = String::new();
    let mut k = 10usize;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binding => binding = inner.as_str().to_string(),
            Rule::string_value => {
                query_text = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::inner_string)
                    .map(|p| p.as_str().to_string())
                    .unwrap_or_default();
            }
            Rule::integer => {
                k = inner
                    .as_str()
                    .parse::<usize>()
                    .map_err(|e| TensaError::ParseError(e.to_string()))?;
            }
            _ => {}
        }
    }
    Ok(NearClause {
        binding,
        query_text,
        k,
    })
}

fn parse_ask_query(pair: pest::iterators::Pair<Rule>) -> Result<AskClause> {
    let mut question = String::new();
    let mut narrative_id = None;
    let mut mode = None;
    let mut response_type = None;
    let mut session_id = None;
    let mut suggest = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                // First string_value is the question (not inside over_clause or respond_clause)
                if question.is_empty() {
                    question = inner
                        .into_inner()
                        .find(|p| p.as_rule() == Rule::inner_string)
                        .map(|p| p.as_str().to_string())
                        .unwrap_or_default();
                }
            }
            Rule::over_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        narrative_id = p
                            .into_inner()
                            .find(|q| q.as_rule() == Rule::inner_string)
                            .map(|q| q.as_str().to_string());
                    }
                }
            }
            Rule::mode_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::retrieval_mode {
                        mode = Some(match p.as_str().to_lowercase().as_str() {
                            "local" => RetrievalMode::Local,
                            "global" => RetrievalMode::Global,
                            "hybrid" => RetrievalMode::Hybrid,
                            "mix" => RetrievalMode::Mix,
                            "drift" => RetrievalMode::Drift,
                            "lazy" => RetrievalMode::Lazy,
                            "ppr" => RetrievalMode::Ppr,
                            other => {
                                return Err(TensaError::ParseError(format!(
                                    "Unknown retrieval mode: {}",
                                    other
                                )))
                            }
                        });
                    }
                }
            }
            Rule::respond_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        response_type = p
                            .into_inner()
                            .find(|q| q.as_rule() == Rule::inner_string)
                            .map(|q| q.as_str().to_string());
                    }
                }
            }
            Rule::session_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        session_id = p
                            .into_inner()
                            .find(|q| q.as_rule() == Rule::inner_string)
                            .map(|q| q.as_str().to_string());
                    }
                }
            }
            Rule::suggest_clause => {
                suggest = true;
            }
            Rule::return_clause => {
                // Handled by extract_clauses
            }
            _ => {}
        }
    }

    Ok(AskClause {
        question,
        narrative_id,
        mode,
        response_type,
        session_id,
        suggest,
    })
}

fn parse_spatial_clause(pair: pest::iterators::Pair<Rule>) -> Result<SpatialClause> {
    let mut field = String::new();
    let mut floats = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::field_path => field = inner.as_str().to_string(),
            Rule::float_value => {
                let f = inner.as_str().parse::<f64>().map_err(|e| {
                    TensaError::ParseError(format!("Invalid float in SPATIAL: {}", e))
                })?;
                floats.push(f);
            }
            _ => {}
        }
    }

    if floats.len() != 3 {
        return Err(TensaError::ParseError(format!(
            "SPATIAL clause requires 3 float values (radius, lat, lon), got {}",
            floats.len()
        )));
    }

    Ok(SpatialClause {
        field,
        radius_km: floats[0],
        center_lat: floats[1],
        center_lon: floats[2],
    })
}

fn parse_discover_clause(pair: pest::iterators::Pair<Rule>) -> Result<DiscoverClause> {
    let mut discover_type = DiscoverType::Patterns;
    let mut target_binding = None;
    let mut target_type = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::discover_type => {
                discover_type = match inner.as_str().to_uppercase().as_str() {
                    "PATTERNS" => DiscoverType::Patterns,
                    "ARCS" => DiscoverType::Arcs,
                    "MISSING" => DiscoverType::Missing,
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unknown discover type: {}",
                            other
                        )))
                    }
                };
            }
            Rule::discover_target => {
                for t in inner.into_inner() {
                    match t.as_rule() {
                        Rule::binding => target_binding = Some(t.as_str().to_string()),
                        Rule::type_name => target_type = Some(t.as_str().to_string()),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(DiscoverClause {
        discover_type,
        target_binding,
        target_type,
    })
}

fn parse_across_clause(pair: pest::iterators::Pair<Rule>) -> Result<AcrossClause> {
    let mut narrative_ids = None;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::narrative_list {
            let mut ids = Vec::new();
            for item in inner.into_inner() {
                if item.as_rule() == Rule::string_value {
                    let s = item
                        .into_inner()
                        .next()
                        .map(|p| p.as_str().to_string())
                        .unwrap_or_default();
                    ids.push(s);
                }
            }
            narrative_ids = Some(ids);
        }
    }

    Ok(AcrossClause { narrative_ids })
}

fn parse_infer_clause(pair: pest::iterators::Pair<Rule>) -> Result<InferClause> {
    let mut infer_type = InferType::Causes;
    let mut target_binding = String::new();
    let mut target_type = String::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::infer_type => {
                infer_type = match inner.as_str().to_uppercase().as_str() {
                    "CAUSES" => InferType::Causes,
                    "MOTIVATION" => InferType::Motivation,
                    "GAME" => InferType::Game,
                    "COUNTERFACTUAL" => InferType::Counterfactual,
                    "MISSING" => InferType::Missing,
                    "ANOMALIES" => InferType::Anomalies,
                    "CENTRALITY" => InferType::Centrality,
                    "COMMUNITIES" => InferType::Communities,
                    "ENTROPY" => InferType::Entropy,
                    "BELIEFS" => InferType::Beliefs,
                    "EVIDENCE" => InferType::Evidence,
                    "ARGUMENTS" => InferType::Arguments,
                    "CONTAGION" => InferType::Contagion,
                    "HIGHER_ORDER_CONTAGION" => InferType::HigherOrderContagion,
                    "CONTAGION_BISTABILITY" => InferType::ContagionBistability,
                    "STYLE_COMPARE" => InferType::StyleCompare,
                    "STYLE_ANOMALIES" => InferType::StyleAnomalies,
                    "STYLE" => InferType::Style,
                    "VERIFY_AUTHORSHIP" => InferType::VerifyAuthorship,
                    "TEMPORAL_RULES" => InferType::TemporalRules,
                    "MEAN_FIELD" => InferType::MeanField,
                    "PSL" => InferType::Psl,
                    "TRAJECTORY" => InferType::Trajectory,
                    "SIMULATE" => InferType::Simulate,
                    "PAGERANK" => InferType::PageRank,
                    "EIGENVECTOR" => InferType::Eigenvector,
                    "HARMONIC" => InferType::Harmonic,
                    "HITS" => InferType::Hits,
                    "TOPOLOGY" => InferType::Topology,
                    "LABEL_PROPAGATION" => InferType::LabelPropagation,
                    "KCORE" => InferType::KCore,
                    "TEMPORAL_PAGERANK" => InferType::TemporalPageRank,
                    "CAUSAL_INFLUENCE" => InferType::CausalInfluence,
                    "INFO_BOTTLENECK" => InferType::InfoBottleneck,
                    "ASSORTATIVITY" => InferType::Assortativity,
                    "TEMPORAL_MOTIFS" => InferType::TemporalMotifs,
                    "FACTION_EVOLUTION" => InferType::FactionEvolution,
                    "FAST_RP" => InferType::FastRP,
                    "NODE2VEC" => InferType::Node2Vec,
                    "NETWORK_INFERENCE" => InferType::NetworkInference,
                    "BEHAVIORAL_FINGERPRINT" => InferType::BehavioralFingerprint,
                    "DISINFO_FINGERPRINT" => InferType::DisinfoFingerprint,
                    "SPREAD_VELOCITY" => InferType::SpreadVelocity,
                    "SPREAD_INTERVENTION" => InferType::SpreadIntervention,
                    "CIB" => InferType::Cib,
                    "SUPERSPREADERS" => InferType::Superspreaders,
                    "CLAIM_ORIGIN" => InferType::ClaimOrigin,
                    "CLAIM_MATCH" => InferType::ClaimMatch,
                    "ARCHETYPE" => InferType::ArchetypeClassification,
                    "DISINFO_ASSESSMENT" => InferType::DisinfoAssessment,
                    // Sprint D9: Narrative architecture
                    "COMMITMENTS" => InferType::CommitmentDetection,
                    "FABULA" => InferType::FabulaExtraction,
                    "SJUZET" => InferType::SjuzetExtraction,
                    "DRAMATIC_IRONY" => InferType::DramaticIrony,
                    "FOCALIZATION" => InferType::Focalization,
                    "CHARACTER_ARC" => InferType::CharacterArc,
                    "SUBPLOTS" => InferType::SubplotDetection,
                    "SCENE_SEQUEL" => InferType::SceneSequel,
                    "REORDERING" => InferType::SjuzetReordering,
                    // Sprint D12: Adversarial narrative wargaming
                    "ADVERSARY_POLICY" => InferType::AdversaryPolicy,
                    "COGNITIVE_HIERARCHY" => InferType::CognitiveHierarchy,
                    "WARGAME" => InferType::Wargame,
                    "COUNTER_NARRATIVE" => InferType::CounterNarrative,
                    "REWARD_FINGERPRINT" => InferType::RewardFingerprint,
                    "RETRODICTION" => InferType::Retrodiction,
                    "ARCS" | "ARC" => InferType::Arcs,
                    "ACTOR_ARCS" | "ACTOR_ARC" => InferType::ActorArcs,
                    "PATTERNS" => InferType::Patterns,
                    "MISSING_EVENTS" => InferType::MissingEvents,
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unknown infer type: {}",
                            other
                        )))
                    }
                };
            }
            Rule::infer_target => {
                for t in inner.into_inner() {
                    match t.as_rule() {
                        Rule::binding => target_binding = t.as_str().to_string(),
                        Rule::type_name => target_type = t.as_str().to_string(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(InferClause {
        infer_type,
        target_binding,
        target_type,
    })
}

fn parse_assuming_clause(pair: pest::iterators::Pair<Rule>) -> Result<AssumingClause> {
    let mut assumptions = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::assumption {
            let mut field = String::new();
            let mut value = QueryValue::Null;

            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::field_path => field = p.as_str().to_string(),
                    Rule::value => value = parse_value(p)?,
                    _ => {}
                }
            }

            assumptions.push(Assumption { field, value });
        }
    }

    Ok(AssumingClause { assumptions })
}

fn parse_under_clause(pair: pest::iterators::Pair<Rule>) -> Result<UnderClause> {
    let mut conditions = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::under_condition {
            let mut key = String::new();
            let mut value = QueryValue::Null;

            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::under_key => key = p.as_str().to_uppercase(),
                    Rule::value => value = parse_value(p)?,
                    _ => {}
                }
            }

            let condition = match key.as_str() {
                "RATIONALITY" => match value {
                    QueryValue::Float(f) => UnderCondition::Rationality(f),
                    QueryValue::Integer(i) => UnderCondition::Rationality(i as f64),
                    _ => {
                        return Err(TensaError::ParseError(
                            "RATIONALITY requires numeric value".into(),
                        ))
                    }
                },
                "INFORMATION" => match value {
                    QueryValue::String(s) => UnderCondition::Information(s),
                    _ => {
                        return Err(TensaError::ParseError(
                            "INFORMATION requires string value".into(),
                        ))
                    }
                },
                other => {
                    return Err(TensaError::ParseError(format!(
                        "Unknown UNDER key: {}",
                        other
                    )))
                }
            };

            conditions.push(condition);
        }
    }

    Ok(UnderClause { conditions })
}

// ─── DML Parse Functions ────────────────────────────────────

fn parse_mutation_stmt(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    match pair.as_rule() {
        Rule::create_narrative_stmt => parse_create_narrative(pair),
        Rule::create_entity_stmt => parse_create_entity(pair),
        Rule::create_situation_stmt => parse_create_situation(pair),
        Rule::delete_stmt => parse_delete(pair),
        Rule::update_entity_stmt => parse_update_entity(pair),
        Rule::update_narrative_stmt => parse_update_narrative(pair),
        Rule::add_participant_stmt => parse_add_participant(pair),
        Rule::remove_participant_stmt => parse_remove_participant(pair),
        Rule::add_cause_stmt => parse_add_cause(pair),
        Rule::remove_cause_stmt => parse_remove_cause(pair),
        _ => Err(TensaError::ParseError(format!(
            "Unknown mutation rule: {:?}",
            pair.as_rule()
        ))),
    }
}

fn parse_create_narrative(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut strings = Vec::new();
    let mut tags = Vec::new();

    // PEG consumes keywords (CREATE, NARRATIVE, TITLE, GENRE) silently.
    // String values arrive in order: id, then title (if present), then genre (if present).
    // We use the raw span text to detect which optional clauses are present.
    let raw = pair.as_str().to_uppercase();
    let has_title = raw.contains("TITLE");
    let has_genre = raw.contains("GENRE");

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => strings.push(extract_string(&inner)),
            Rule::tags_clause => {
                for t in inner.into_inner() {
                    if t.as_rule() == Rule::string_value {
                        tags.push(extract_string(&t));
                    }
                }
            }
            _ => {}
        }
    }

    let id = strings.first().cloned().unwrap_or_default();
    let mut idx = 1;
    let title = if has_title {
        let v = strings.get(idx).cloned();
        idx += 1;
        v
    } else {
        None
    };
    let genre = if has_genre {
        strings.get(idx).cloned()
    } else {
        None
    };

    Ok(MutationStatement::CreateNarrative {
        id,
        title,
        genre,
        tags,
    })
}

fn parse_create_entity(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut entity_type = String::new();
    let mut properties = Vec::new();
    let mut narrative_id = None;
    let mut confidence = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::entity_type_name => entity_type = inner.as_str().to_string(),
            Rule::binding => {} // optional binding, not used for creation
            Rule::properties => properties = parse_properties(inner)?,
            Rule::in_narrative_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        narrative_id = Some(extract_string(&p));
                    }
                }
            }
            Rule::confidence_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::float_value {
                        confidence = Some(
                            p.as_str()
                                .parse::<f64>()
                                .map_err(|e| TensaError::ParseError(e.to_string()))?,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    Ok(MutationStatement::CreateEntity {
        entity_type,
        properties,
        narrative_id,
        confidence,
    })
}

fn parse_create_situation(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut level = String::new();
    let mut content = Vec::new();
    let mut narrative_id = None;
    let mut confidence = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::narrative_level_name => level = inner.as_str().to_string(),
            Rule::string_value => content.push(extract_string(&inner)),
            Rule::in_narrative_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        narrative_id = Some(extract_string(&p));
                    }
                }
            }
            Rule::confidence_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::float_value {
                        confidence = Some(
                            p.as_str()
                                .parse::<f64>()
                                .map_err(|e| TensaError::ParseError(e.to_string()))?,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    Ok(MutationStatement::CreateSituation {
        level,
        content,
        narrative_id,
        confidence,
    })
}

fn parse_delete(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut target = String::new();
    let mut id = String::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::delete_target => target = inner.as_str().to_uppercase(),
            Rule::string_value => id = extract_string(&inner),
            _ => {}
        }
    }

    match target.as_str() {
        "ENTITY" => Ok(MutationStatement::DeleteEntity { id }),
        "SITUATION" => Ok(MutationStatement::DeleteSituation { id }),
        "NARRATIVE" => Ok(MutationStatement::DeleteNarrative { id }),
        other => Err(TensaError::ParseError(format!(
            "Unknown delete target: {}",
            other
        ))),
    }
}

fn parse_update_entity(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut id = String::new();
    let mut set_pairs = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => id = extract_string(&inner),
            Rule::set_pair => {
                let (key, val) = parse_set_pair(inner)?;
                set_pairs.push((key, val));
            }
            _ => {}
        }
    }

    Ok(MutationStatement::UpdateEntity { id, set_pairs })
}

fn parse_update_narrative(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut id = String::new();
    let mut set_pairs = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => id = extract_string(&inner),
            Rule::set_pair => {
                let (key, val) = parse_set_pair(inner)?;
                set_pairs.push((key, val));
            }
            _ => {}
        }
    }

    Ok(MutationStatement::UpdateNarrative { id, set_pairs })
}

fn parse_set_pair(pair: pest::iterators::Pair<Rule>) -> Result<(String, QueryValue)> {
    let mut key = String::new();
    let mut value = QueryValue::Null;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::field_path => key = inner.as_str().to_string(),
            Rule::value => value = parse_value(inner)?,
            _ => {}
        }
    }

    Ok((key, value))
}

fn parse_add_participant(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut strings = Vec::new();
    let mut role = String::new();

    // PEG consumes ADD, PARTICIPANT, TO, SITUATION, ROLE, ACTION keywords.
    // String values in order: entity_id, situation_id, then optionally action.
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => strings.push(extract_string(&inner)),
            Rule::role_name => {
                // role_name can be a keyword (Protagonist, etc.) or a string_value
                let text = inner.as_str();
                if text.starts_with('"') {
                    for p in inner.into_inner() {
                        if p.as_rule() == Rule::string_value {
                            role = extract_string(&p);
                        }
                    }
                } else {
                    role = text.to_string();
                }
            }
            _ => {}
        }
    }

    let entity_id = strings.first().cloned().unwrap_or_default();
    let situation_id = strings.get(1).cloned().unwrap_or_default();
    let action = strings.get(2).cloned();

    Ok(MutationStatement::AddParticipant {
        entity_id,
        situation_id,
        role,
        action,
    })
}

fn parse_remove_participant(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut strings = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::string_value {
            strings.push(extract_string(&inner));
        }
    }

    Ok(MutationStatement::RemoveParticipant {
        entity_id: strings.first().cloned().unwrap_or_default(),
        situation_id: strings.get(1).cloned().unwrap_or_default(),
    })
}

fn parse_add_cause(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut strings = Vec::new();
    let mut causal_type = None;
    let mut strength = None;

    // PEG consumes ADD, CAUSE, FROM, TO, TYPE, STRENGTH, MECHANISM keywords.
    // String values in order: from_id, to_id, then optionally mechanism.
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => strings.push(extract_string(&inner)),
            Rule::causal_type_name => causal_type = Some(inner.as_str().to_string()),
            Rule::float_value => {
                strength = Some(
                    inner
                        .as_str()
                        .parse::<f64>()
                        .map_err(|e| TensaError::ParseError(e.to_string()))?,
                );
            }
            _ => {}
        }
    }

    let mechanism = strings.get(2).cloned();

    Ok(MutationStatement::AddCause {
        from_id: strings.first().cloned().unwrap_or_default(),
        to_id: strings.get(1).cloned().unwrap_or_default(),
        causal_type,
        strength,
        mechanism,
    })
}

fn parse_remove_cause(pair: pest::iterators::Pair<Rule>) -> Result<MutationStatement> {
    let mut strings = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::string_value {
            strings.push(extract_string(&inner));
        }
    }

    Ok(MutationStatement::RemoveCause {
        from_id: strings.first().cloned().unwrap_or_default(),
        to_id: strings.get(1).cloned().unwrap_or_default(),
    })
}

/// Extract the inner string content from a string_value pair.
fn extract_string(pair: &pest::iterators::Pair<Rule>) -> String {
    pair.clone()
        .into_inner()
        .find(|p| p.as_rule() == Rule::inner_string)
        .map(|p| p.as_str().to_string())
        .unwrap_or_default()
}

/// EATH Phase 5 — strip the surrounding single quotes from a model_name
/// pair and return the inner content. Empty inner content errors so a
/// future grammar tweak that allows `''` doesn't silently slip through.
fn extract_model_name(pair: &pest::iterators::Pair<Rule>) -> Result<String> {
    let inner = pair
        .clone()
        .into_inner()
        .find(|p| p.as_rule() == Rule::model_inner)
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    if inner.is_empty() {
        return Err(TensaError::ParseError(
            "model name must not be empty (use 'eath' or another registered model)".into(),
        ));
    }
    Ok(inner)
}

/// EATH Phase 5 — parse a CALIBRATE SURROGATE statement.
fn parse_calibrate_surrogate_query(pair: pest::iterators::Pair<Rule>) -> Result<TensaStatement> {
    let mut model: Option<String> = None;
    let mut narrative_id: Option<String> = None;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::model_name => {
                model = Some(extract_model_name(&inner)?);
            }
            Rule::string_value => {
                narrative_id = Some(extract_string(&inner));
            }
            _ => {}
        }
    }
    let model = model
        .ok_or_else(|| TensaError::ParseError("CALIBRATE SURROGATE missing USING <model>".into()))?;
    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError("CALIBRATE SURROGATE missing FOR \"<narrative_id>\"".into())
    })?;
    Ok(TensaStatement::CalibrateSurrogate {
        narrative_id,
        model,
    })
}

/// EATH Phase 5 — parse a GENERATE NARRATIVE statement.
///
/// Default model when the optional `USING SURROGATE '<model>'` clause is
/// omitted: `"eath"` (asymmetric with CALIBRATE — see grammar comment).
/// PARAMS json is parsed eagerly via serde_json so a malformed body
/// errors at parse time, not at execute time.
fn parse_generate_narrative_query(pair: pest::iterators::Pair<Rule>) -> Result<TensaStatement> {
    let mut narrative_strings: Vec<String> = Vec::new();
    let mut model: Option<String> = None;
    let mut params: Option<serde_json::Value> = None;
    let mut seed: Option<u64> = None;
    let mut num_steps: Option<usize> = None;
    let mut label_prefix: Option<String> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                // First string is output_id, second is source_id (per grammar order).
                narrative_strings.push(extract_string(&inner));
            }
            Rule::model_name => {
                model = Some(extract_model_name(&inner)?);
            }
            Rule::generate_clause => {
                let body = inner.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError("Empty GENERATE clause body".into())
                })?;
                match body.as_rule() {
                    Rule::generate_params => {
                        // The params_json child carries the captured `{...}`
                        // span — parse it via serde_json eagerly so malformed
                        // JSON errors at parse time, not at execute time.
                        let json_pair = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::params_json)
                            .ok_or_else(|| {
                                TensaError::ParseError(
                                    "GENERATE PARAMS missing JSON object body".into(),
                                )
                            })?;
                        let body_str = json_pair.as_str();
                        params = Some(serde_json::from_str(body_str).map_err(|e| {
                            TensaError::ParseError(format!(
                                "GENERATE PARAMS body is not valid JSON: {e}"
                            ))
                        })?);
                    }
                    Rule::generate_seed => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        seed = Some(n_str.parse::<u64>().map_err(|e| {
                            TensaError::ParseError(format!("GENERATE SEED must be u64: {e}"))
                        })?);
                    }
                    Rule::generate_steps => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        num_steps = Some(n_str.parse::<usize>().map_err(|e| {
                            TensaError::ParseError(format!("GENERATE STEPS must be usize: {e}"))
                        })?);
                    }
                    Rule::generate_label_prefix => {
                        let name_pair = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::model_name)
                            .ok_or_else(|| {
                                TensaError::ParseError(
                                    "GENERATE LABEL_PREFIX missing single-quoted value".into(),
                                )
                            })?;
                        label_prefix = Some(extract_model_name(&name_pair)?);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unexpected GENERATE clause body: {:?}",
                            other
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    if narrative_strings.len() < 2 {
        return Err(TensaError::ParseError(
            "GENERATE NARRATIVE \"<output>\" LIKE \"<source>\" requires both narrative IDs".into(),
        ));
    }
    let output_id = narrative_strings[0].clone();
    let source_id = narrative_strings[1].clone();
    let model = model.unwrap_or_else(|| "eath".to_string());

    Ok(TensaStatement::GenerateNarrative {
        output_id,
        source_id,
        model,
        params,
        seed,
        num_steps,
        label_prefix,
    })
}

/// EATH Phase 9 — parse a `GENERATE NARRATIVE ... USING HYBRID ...` statement.
fn parse_generate_hybrid_narrative_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut output_id: Option<String> = None;
    let mut components: Vec<HybridComponentSpec> = Vec::new();
    let mut seed: Option<u64> = None;
    let mut num_steps: Option<usize> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                if output_id.is_none() {
                    output_id = Some(extract_string(&inner));
                }
            }
            Rule::hybrid_component => {
                let mut nid: Option<String> = None;
                let mut weight: Option<f32> = None;
                for c in inner.into_inner() {
                    match c.as_rule() {
                        Rule::string_value => nid = Some(extract_string(&c)),
                        Rule::hybrid_weight => {
                            let s = c.as_str();
                            weight = Some(s.parse::<f32>().map_err(|e| {
                                TensaError::ParseError(format!(
                                    "GENERATE HYBRID: weight '{s}' is not a valid float: {e}"
                                ))
                            })?);
                        }
                        _ => {}
                    }
                }
                let nid = nid.ok_or_else(|| {
                    TensaError::ParseError("hybrid component missing FROM \"<nid>\"".into())
                })?;
                let weight = weight.ok_or_else(|| {
                    TensaError::ParseError("hybrid component missing WEIGHT".into())
                })?;
                components.push(HybridComponentSpec {
                    narrative_id: nid,
                    weight,
                });
            }
            Rule::hybrid_clause => {
                let body = inner.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError("Empty hybrid clause body".into())
                })?;
                match body.as_rule() {
                    Rule::hybrid_seed => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        seed = Some(n_str.parse::<u64>().map_err(|e| {
                            TensaError::ParseError(format!("HYBRID SEED must be u64: {e}"))
                        })?);
                    }
                    Rule::hybrid_steps => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        num_steps = Some(n_str.parse::<usize>().map_err(|e| {
                            TensaError::ParseError(format!("HYBRID STEPS must be usize: {e}"))
                        })?);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unexpected hybrid clause body: {:?}",
                            other
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    let output_id = output_id.ok_or_else(|| {
        TensaError::ParseError(
            "GENERATE NARRATIVE \"<output>\" USING HYBRID requires output narrative id".into(),
        )
    })?;
    if components.len() < 2 {
        return Err(TensaError::ParseError(
            "GENERATE HYBRID requires at least 2 components".into(),
        ));
    }

    Ok(TensaStatement::GenerateHybridNarrative {
        output_id,
        components,
        seed,
        num_steps,
    })
}

/// EATH Extension Phase 13c — parse `COMPUTE DUAL_SIGNIFICANCE FOR ... USING ...
///     [K_PER_MODEL <n>] [MODELS '<m1>','<m2>',...]`.
///
/// Default model list (when `MODELS` clause omitted) is set to `None` here;
/// the planner expands `None` into `["eath", "nudhy"]` so the planner-side
/// surrogate-name validation runs unconditionally.
fn parse_dual_significance_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut metric: Option<String> = None;
    let mut k_per_model: Option<u16> = None;
    let mut models: Option<Vec<String>> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                if narrative_id.is_none() {
                    narrative_id = Some(extract_string(&inner));
                }
            }
            Rule::model_name => {
                // The first model_name slot after FOR ... USING is the metric
                // (single-quoted per the model_name lexical convention).
                if metric.is_none() {
                    metric = Some(extract_model_name(&inner)?);
                }
            }
            Rule::dual_sig_clause => {
                let body = inner.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError(
                        "Empty COMPUTE DUAL_SIGNIFICANCE clause body".into(),
                    )
                })?;
                match body.as_rule() {
                    Rule::dual_sig_k_per_model => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        k_per_model = Some(n_str.parse::<u16>().map_err(|e| {
                            TensaError::ParseError(format!(
                                "K_PER_MODEL must be u16: {e}"
                            ))
                        })?);
                    }
                    Rule::dual_sig_models => {
                        let mut list: Vec<String> = Vec::new();
                        for c in body.into_inner() {
                            if c.as_rule() == Rule::model_name {
                                list.push(extract_model_name(&c)?);
                            }
                        }
                        if list.is_empty() {
                            return Err(TensaError::ParseError(
                                "MODELS clause requires at least one model name".into(),
                            ));
                        }
                        models = Some(list);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unexpected dual-sig clause body: {other:?}"
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError(
            "COMPUTE DUAL_SIGNIFICANCE missing FOR \"<narrative_id>\"".into(),
        )
    })?;
    let metric = metric.ok_or_else(|| {
        TensaError::ParseError(
            "COMPUTE DUAL_SIGNIFICANCE missing USING '<metric>'".into(),
        )
    })?;

    Ok(TensaStatement::ComputeDualSignificance {
        narrative_id,
        metric,
        k_per_model,
        models,
    })
}

/// EATH Extension Phase 15c — parse `INFER HYPERGRAPH FROM DYNAMICS FOR
/// "<narrative_id>" [USING OBSERVATION '<source>'] [MAX_ORDER <n>]
/// [LAMBDA <f>]`.
///
/// All clauses are optional except the narrative id. Defaults applied at the
/// planner / executor layer; the parser keeps `Option<...>` so the planner
/// can distinguish "user explicitly set X to default" from "X was omitted."
fn parse_infer_hypergraph_reconstruction_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut observation: Option<String> = None;
    let mut max_order: Option<usize> = None;
    let mut lambda: Option<f32> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                if narrative_id.is_none() {
                    narrative_id = Some(extract_string(&inner));
                }
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            Rule::with_aggregator_clause => {
                fuzzy_config.aggregator = Some(parse_with_aggregator_clause(inner)?);
            }
            Rule::recon_clause => {
                let body = inner.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError(
                        "Empty INFER HYPERGRAPH FROM DYNAMICS clause body".into(),
                    )
                })?;
                match body.as_rule() {
                    Rule::recon_observation => {
                        let name_pair = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::model_name)
                            .ok_or_else(|| {
                                TensaError::ParseError(
                                    "USING OBSERVATION requires a single-quoted source name"
                                        .into(),
                                )
                            })?;
                        observation = Some(extract_model_name(&name_pair)?);
                    }
                    Rule::recon_max_order => {
                        let n_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::unsigned_int)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        max_order = Some(n_str.parse::<usize>().map_err(|e| {
                            TensaError::ParseError(format!("MAX_ORDER must be unsigned: {e}"))
                        })?);
                    }
                    Rule::recon_lambda => {
                        let f_str = body
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::float_value)
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        lambda = Some(f_str.parse::<f32>().map_err(|e| {
                            TensaError::ParseError(format!("LAMBDA must be a float: {e}"))
                        })?);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "Unexpected reconstruction clause body: {other:?}"
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError(
            "INFER HYPERGRAPH FROM DYNAMICS missing FOR \"<narrative_id>\"".into(),
        )
    })?;

    Ok(TensaStatement::InferHypergraphReconstruction {
        narrative_id,
        observation,
        max_order,
        lambda,
        fuzzy_config,
    })
}

/// EATH Extension Phase 16c — parse `INFER OPINION_DYNAMICS(... named ...)
/// FOR "<narrative_id>"`.
fn parse_opinion_dynamics_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut confidence_bound: Option<f32> = None;
    let mut variant: Option<String> = None;
    let mut mu: Option<f32> = None;
    let mut initial: Option<String> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                if narrative_id.is_none() {
                    narrative_id = Some(extract_string(&inner));
                }
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            Rule::with_aggregator_clause => {
                fuzzy_config.aggregator = Some(parse_with_aggregator_clause(inner)?);
            }
            Rule::opinion_param => {
                let mut iter = inner.into_inner();
                let key_pair = iter.next().ok_or_else(|| {
                    TensaError::ParseError("Empty opinion_param body".into())
                })?;
                let key = key_pair.as_str().to_ascii_lowercase();
                let value_pair = iter.next().ok_or_else(|| {
                    TensaError::ParseError(format!("opinion_param '{key}' missing value"))
                })?;
                let value_inner = value_pair.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError(format!("opinion_param '{key}' empty value"))
                })?;
                match key.as_str() {
                    "confidence_bound" => {
                        confidence_bound = Some(parse_f32_value(&value_inner)?);
                    }
                    "mu" => {
                        mu = Some(parse_f32_value(&value_inner)?);
                    }
                    "variant" => {
                        variant = Some(parse_model_or_string(&value_inner)?);
                    }
                    "initial" => {
                        initial = Some(parse_model_or_string(&value_inner)?);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "unknown OPINION_DYNAMICS param '{other}'; expected one of: \
                             confidence_bound, variant, mu, initial"
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError("INFER OPINION_DYNAMICS missing FOR \"<narrative_id>\"".into())
    })?;

    Ok(TensaStatement::InferOpinionDynamics {
        narrative_id,
        confidence_bound,
        variant,
        mu,
        initial,
        fuzzy_config,
    })
}

/// EATH Extension Phase 16c — parse `INFER OPINION_PHASE_TRANSITION(...)
/// FOR "<narrative_id>"`.
fn parse_opinion_phase_transition_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut c_start: Option<f32> = None;
    let mut c_end: Option<f32> = None;
    let mut c_steps: Option<usize> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value => {
                if narrative_id.is_none() {
                    narrative_id = Some(extract_string(&inner));
                }
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            Rule::with_aggregator_clause => {
                fuzzy_config.aggregator = Some(parse_with_aggregator_clause(inner)?);
            }
            Rule::opinion_pt_param => {
                let mut iter = inner.into_inner();
                let key_pair = iter.next().ok_or_else(|| {
                    TensaError::ParseError("Empty opinion_pt_param body".into())
                })?;
                let key = key_pair.as_str().to_ascii_lowercase();
                let value_pair = iter.next().ok_or_else(|| {
                    TensaError::ParseError(format!("opinion_pt_param '{key}' missing value"))
                })?;
                let value_inner = value_pair.into_inner().next().ok_or_else(|| {
                    TensaError::ParseError(format!("opinion_pt_param '{key}' empty value"))
                })?;
                match key.as_str() {
                    "c_start" => c_start = Some(parse_f32_value(&value_inner)?),
                    "c_end" => c_end = Some(parse_f32_value(&value_inner)?),
                    "c_steps" => {
                        let n_str = value_inner.as_str();
                        c_steps = Some(n_str.parse::<usize>().map_err(|e| {
                            TensaError::ParseError(format!("c_steps must be unsigned: {e}"))
                        })?);
                    }
                    other => {
                        return Err(TensaError::ParseError(format!(
                            "unknown OPINION_PHASE_TRANSITION param '{other}'"
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError(
            "INFER OPINION_PHASE_TRANSITION missing FOR \"<narrative_id>\"".into(),
        )
    })?;
    let c_start = c_start.ok_or_else(|| {
        TensaError::ParseError("INFER OPINION_PHASE_TRANSITION missing c_start".into())
    })?;
    let c_end = c_end.ok_or_else(|| {
        TensaError::ParseError("INFER OPINION_PHASE_TRANSITION missing c_end".into())
    })?;
    let c_steps = c_steps.ok_or_else(|| {
        TensaError::ParseError("INFER OPINION_PHASE_TRANSITION missing c_steps".into())
    })?;
    Ok(TensaStatement::InferOpinionPhaseTransition {
        narrative_id,
        c_start,
        c_end,
        c_steps,
        fuzzy_config,
    })
}

/// Fuzzy Sprint Phase 6 — parse a QUANTIFY statement.
/// `QUANTIFY MOST (e:Actor) WHERE e.confidence > 0.7 FOR "nid" AS "label"`.
/// Unknown quantifier names are rejected here; only the four Phase 6
/// built-ins (`MOST` / `MANY` / `ALMOST_ALL` / `FEW`) are accepted.
///
/// Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].
fn parse_quantify_query(pair: pest::iterators::Pair<Rule>) -> Result<TensaStatement> {
    let mut quantifier: Option<String> = None;
    let mut binding: Option<String> = None;
    let mut type_name: Option<String> = None;
    let mut where_clause: Option<WhereClause> = None;
    let mut narrative_id: Option<String> = None;
    let mut label: Option<String> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::quantifier_name_q6 => {
                quantifier = Some(inner.as_str().to_ascii_lowercase());
            }
            Rule::node_pattern => {
                let node = parse_node(inner)?;
                binding = node.binding;
                type_name = Some(node.type_name);
            }
            Rule::where_clause => {
                where_clause = Some(parse_where_clause(inner)?);
            }
            Rule::quantify_for_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        narrative_id = Some(extract_string(&p));
                    }
                }
            }
            Rule::quantify_as_clause => {
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::string_value {
                        label = Some(extract_string(&p));
                    }
                }
            }
            _ => {}
        }
    }

    let quantifier = quantifier
        .ok_or_else(|| TensaError::ParseError("QUANTIFY missing quantifier name".into()))?;
    let type_name = type_name
        .ok_or_else(|| TensaError::ParseError("QUANTIFY missing entity pattern".into()))?;

    Ok(TensaStatement::Quantify {
        quantifier,
        binding,
        type_name,
        where_clause,
        narrative_id,
        label,
    })
}

/// Fuzzy Sprint Phase 7 — parse a VERIFY SYLLOGISM statement.
/// `VERIFY SYLLOGISM { major: '<dsl>', minor: '<dsl>',
/// conclusion: '<dsl>' } FOR "<nid>" [THRESHOLD <f>] [WITH TNORM '<kind>']`.
/// The three statement strings use the tiny DSL in
/// [`crate::fuzzy::syllogism::parse_statement`]; parsing happens at the
/// executor layer to keep the pest grammar small.
/// Cites: [murinovanovak2014peterson].
fn parse_verify_syllogism_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut major: Option<String> = None;
    let mut minor: Option<String> = None;
    let mut conclusion: Option<String> = None;
    let mut narrative_id: Option<String> = None;
    let mut threshold: Option<f64> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::syllogism_kv => {
                let mut key: Option<String> = None;
                let mut val: Option<String> = None;
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::syllogism_key => {
                            key = Some(p.as_str().to_ascii_lowercase());
                        }
                        Rule::model_name => {
                            val = Some(extract_model_name(&p)?);
                        }
                        _ => {}
                    }
                }
                match (key.as_deref(), val) {
                    (Some("major"), Some(v)) => major = Some(v),
                    (Some("minor"), Some(v)) => minor = Some(v),
                    (Some("conclusion"), Some(v)) => conclusion = Some(v),
                    _ => {
                        return Err(TensaError::ParseError(
                            "VERIFY SYLLOGISM key/value malformed; expected \
                             major/minor/conclusion with a single-quoted DSL \
                             string value"
                                .into(),
                        ));
                    }
                }
            }
            Rule::string_value => {
                narrative_id = Some(extract_string(&inner));
            }
            Rule::syllogism_threshold_clause => {
                let fv = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::float_value)
                    .ok_or_else(|| {
                        TensaError::ParseError(
                            "THRESHOLD missing numeric value".into(),
                        )
                    })?;
                threshold = Some(fv.as_str().parse::<f64>().map_err(|e| {
                    TensaError::ParseError(format!(
                        "THRESHOLD value must be a float: {e}"
                    ))
                })?);
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            _ => {}
        }
    }

    let major = major.ok_or_else(|| {
        TensaError::ParseError("VERIFY SYLLOGISM missing major premise".into())
    })?;
    let minor = minor.ok_or_else(|| {
        TensaError::ParseError("VERIFY SYLLOGISM missing minor premise".into())
    })?;
    let conclusion = conclusion.ok_or_else(|| {
        TensaError::ParseError("VERIFY SYLLOGISM missing conclusion".into())
    })?;
    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError(
            "VERIFY SYLLOGISM missing FOR \"<narrative_id>\"".into(),
        )
    })?;

    Ok(TensaStatement::VerifySyllogism {
        major,
        minor,
        conclusion,
        narrative_id,
        threshold,
        fuzzy_config,
    })
}

/// Fuzzy Sprint Phase 8 — parse
/// `FCA LATTICE FOR "<nid>" [THRESHOLD <n>] [ATTRIBUTES [...]]
/// [ENTITY_TYPE <type>] [WITH TNORM '<kind>']`.
fn parse_fca_lattice_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut threshold: Option<usize> = None;
    let mut attribute_allowlist: Option<Vec<String>> = None;
    let mut entity_type: Option<String> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value if narrative_id.is_none() => {
                narrative_id = Some(extract_string(&inner));
            }
            Rule::fca_lattice_clause => {
                for sub in inner.into_inner() {
                    match sub.as_rule() {
                        Rule::fca_threshold_clause => {
                            let n = sub
                                .into_inner()
                                .find(|p| p.as_rule() == Rule::unsigned_int)
                                .ok_or_else(|| {
                                    TensaError::ParseError(
                                        "FCA THRESHOLD missing integer".into(),
                                    )
                                })?;
                            threshold = Some(n.as_str().parse::<usize>().map_err(|e| {
                                TensaError::ParseError(format!(
                                    "FCA THRESHOLD: {e}"
                                ))
                            })?);
                        }
                        Rule::fca_attributes_clause => {
                            let mut attrs = Vec::new();
                            for p in sub.into_inner() {
                                if p.as_rule() == Rule::model_name {
                                    attrs.push(extract_model_name(&p)?);
                                }
                            }
                            if attrs.is_empty() {
                                return Err(TensaError::ParseError(
                                    "FCA ATTRIBUTES list is empty".into(),
                                ));
                            }
                            attribute_allowlist = Some(attrs);
                        }
                        Rule::fca_entity_type_clause => {
                            let tname = sub
                                .into_inner()
                                .find(|p| p.as_rule() == Rule::entity_type_name)
                                .map(|p| p.as_str().to_string())
                                .ok_or_else(|| {
                                    TensaError::ParseError(
                                        "FCA ENTITY_TYPE missing type name".into(),
                                    )
                                })?;
                            entity_type = Some(tname);
                        }
                        _ => {}
                    }
                }
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError("FCA LATTICE missing FOR \"<narrative_id>\"".into())
    })?;

    Ok(TensaStatement::FcaLattice {
        narrative_id,
        threshold,
        attribute_allowlist,
        entity_type,
        fuzzy_config,
    })
}

/// Fuzzy Sprint Phase 8 — parse `FCA CONCEPT <idx> FROM "<lattice_id>"`.
fn parse_fca_concept_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut idx: Option<usize> = None;
    let mut lattice_id: Option<String> = None;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::unsigned_int => {
                idx = Some(inner.as_str().parse::<usize>().map_err(|e| {
                    TensaError::ParseError(format!("FCA CONCEPT index: {e}"))
                })?);
            }
            Rule::string_value => {
                lattice_id = Some(extract_string(&inner));
            }
            _ => {}
        }
    }
    let concept_idx = idx.ok_or_else(|| {
        TensaError::ParseError("FCA CONCEPT missing concept index".into())
    })?;
    let lattice_id = lattice_id.ok_or_else(|| {
        TensaError::ParseError("FCA CONCEPT missing FROM \"<lattice_id>\"".into())
    })?;
    Ok(TensaStatement::FcaConcept {
        lattice_id,
        concept_idx,
    })
}

/// Fuzzy Sprint Phase 9 — parse
/// `EVALUATE RULES FOR "<nid>" AGAINST (e:Actor)
///     [RULES ['rule-a', 'rule-b']] [WITH TNORM '<kind>']`.
fn parse_evaluate_rules_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut entity_type: Option<String> = None;
    let mut rule_ids: Option<Vec<String>> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_value if narrative_id.is_none() => {
                narrative_id = Some(extract_string(&inner));
            }
            Rule::entity_type_name => {
                entity_type = Some(inner.as_str().to_string());
            }
            Rule::binding => {
                // Bind identifier — captured by the grammar so callers can
                // reference it downstream; Phase 9 ignores the binding and
                // always returns one row per matched entity.
            }
            Rule::evaluate_rules_clause => {
                let mut ids = Vec::new();
                for sub in inner.into_inner() {
                    if sub.as_rule() == Rule::model_name {
                        ids.push(extract_model_name(&sub)?);
                    }
                }
                if ids.is_empty() {
                    return Err(TensaError::ParseError(
                        "EVALUATE RULES rule-id list is empty".into(),
                    ));
                }
                rule_ids = Some(ids);
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError("EVALUATE RULES missing FOR \"<narrative_id>\"".into())
    })?;
    let entity_type = entity_type.ok_or_else(|| {
        TensaError::ParseError("EVALUATE RULES missing AGAINST (e:<EntityType>)".into())
    })?;

    Ok(TensaStatement::EvaluateRules {
        narrative_id,
        entity_type,
        rule_ids,
        fuzzy_config,
    })
}

/// Fuzzy Sprint Phase 10 — parse
/// `INFER FUZZY_PROBABILITY(event_kind := '<kind>',
///     event_ref := '<payload>', distribution := '<spec>') FOR "<nid>"
///     [WITH TNORM '<kind>']`.
/// Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
fn parse_fuzzy_probability_query(
    pair: pest::iterators::Pair<Rule>,
) -> Result<TensaStatement> {
    let mut narrative_id: Option<String> = None;
    let mut event_kind: Option<String> = None;
    let mut event_ref: Option<String> = None;
    let mut distribution: Option<String> = None;
    let mut fuzzy_config = FuzzyConfig::default();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::fuzzy_prob_param => {
                let mut key: Option<String> = None;
                let mut val: Option<String> = None;
                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::fuzzy_prob_key => {
                            key = Some(p.as_str().to_ascii_lowercase());
                        }
                        Rule::model_name => {
                            val = Some(extract_model_name(&p)?);
                        }
                        _ => {}
                    }
                }
                match (key.as_deref(), val) {
                    (Some("event_kind"), Some(v)) => event_kind = Some(v),
                    (Some("event_ref"), Some(v)) => event_ref = Some(v),
                    (Some("distribution"), Some(v)) => distribution = Some(v),
                    _ => {
                        return Err(TensaError::ParseError(
                            "INFER FUZZY_PROBABILITY param must be \
                             event_kind / event_ref / distribution with a \
                             single-quoted string value"
                                .into(),
                        ));
                    }
                }
            }
            Rule::string_value => {
                narrative_id = Some(extract_string(&inner));
            }
            Rule::with_tnorm_clause => {
                fuzzy_config.tnorm = Some(parse_with_tnorm_clause(inner)?);
            }
            _ => {}
        }
    }

    let narrative_id = narrative_id.ok_or_else(|| {
        TensaError::ParseError(
            "INFER FUZZY_PROBABILITY missing FOR \"<narrative_id>\"".into(),
        )
    })?;
    let event_kind = event_kind.ok_or_else(|| {
        TensaError::ParseError("INFER FUZZY_PROBABILITY missing event_kind".into())
    })?;
    let event_ref = event_ref.ok_or_else(|| {
        TensaError::ParseError("INFER FUZZY_PROBABILITY missing event_ref".into())
    })?;
    let distribution = distribution.ok_or_else(|| {
        TensaError::ParseError("INFER FUZZY_PROBABILITY missing distribution".into())
    })?;

    Ok(TensaStatement::FuzzyProbability {
        narrative_id,
        event_kind,
        event_ref,
        distribution,
        fuzzy_config,
    })
}

/// Parse a Pest pair as f32 — accepts either `float_value` or `unsigned_int`
/// (the latter so callers can pass `confidence_bound := 0` style without
/// enforcing the decimal point).
fn parse_f32_value(pair: &pest::iterators::Pair<Rule>) -> Result<f32> {
    pair.as_str()
        .parse::<f32>()
        .map_err(|e| TensaError::ParseError(format!("expected numeric value: {e}")))
}

/// Extract a single-quoted model_name token's inner string. Reused for both
/// `variant` and `initial` opinion-dynamics parameters.
fn parse_model_or_string(pair: &pest::iterators::Pair<Rule>) -> Result<String> {
    if pair.as_rule() == Rule::model_name {
        extract_model_name(pair)
    } else {
        Err(TensaError::ParseError(format!(
            "expected single-quoted '<value>' token, got {:?}",
            pair.as_rule()
        )))
    }
}

fn parse_value(pair: pest::iterators::Pair<Rule>) -> Result<QueryValue> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| TensaError::ParseError("Empty value".into()))?;
    match inner.as_rule() {
        Rule::string_value => {
            let s = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::inner_string)
                .map(|p| p.as_str().to_string())
                .unwrap_or_default();
            Ok(QueryValue::String(s))
        }
        Rule::float_value => {
            let f = inner
                .as_str()
                .parse::<f64>()
                .map_err(|e| TensaError::ParseError(e.to_string()))?;
            Ok(QueryValue::Float(f))
        }
        Rule::integer => {
            let i = inner
                .as_str()
                .parse::<i64>()
                .map_err(|e| TensaError::ParseError(e.to_string()))?;
            Ok(QueryValue::Integer(i))
        }
        Rule::boolean => {
            let b = inner.as_str().eq_ignore_ascii_case("true");
            Ok(QueryValue::Boolean(b))
        }
        Rule::null_value => Ok(QueryValue::Null),
        _ => Err(TensaError::ParseError(format!(
            "Unexpected value rule: {:?}",
            inner.as_rule()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_match() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        assert_eq!(q.match_clause.as_ref().unwrap().elements.len(), 1);
        if let PatternElement::Node(n) = &q.match_clause.as_ref().unwrap().elements[0] {
            assert_eq!(n.binding.as_deref(), Some("e"));
            assert_eq!(n.type_name, "Actor");
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_match_with_properties() {
        let q = parse_query(r#"MATCH (e:Actor {name: "Raskolnikov"}) RETURN e"#).unwrap();
        if let PatternElement::Node(n) = &q.match_clause.as_ref().unwrap().elements[0] {
            assert_eq!(n.properties.len(), 1);
            assert_eq!(n.properties[0].key, "name");
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_match_with_edge() {
        let q = parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s").unwrap();
        assert_eq!(q.match_clause.as_ref().unwrap().elements.len(), 3);
        assert!(matches!(
            &q.match_clause.as_ref().unwrap().elements[1],
            PatternElement::Edge(e) if e.directed && e.rel_type == "PARTICIPATES"
        ));
    }

    #[test]
    fn test_parse_where_equals() {
        let q = parse_query(r#"MATCH (e:Actor) WHERE e.name = "Sonya" RETURN e"#).unwrap();
        let wc = q.where_clause.unwrap();
        let conds = wc.all_conditions();
        assert_eq!(conds.len(), 1);
        assert_eq!(conds[0].field, "e.name");
        assert_eq!(conds[0].op, CompareOp::Eq);
    }

    #[test]
    fn test_parse_where_greater_than() {
        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.8 RETURN e").unwrap();
        let wc = q.where_clause.unwrap();
        let conds = wc.all_conditions();
        assert_eq!(conds[0].op, CompareOp::Gt);
        assert!(matches!(conds[0].value, QueryValue::Float(f) if f == 0.8));
    }

    #[test]
    fn test_parse_where_multiple_conditions() {
        let q =
            parse_query(r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "Test" RETURN e"#)
                .unwrap();
        assert_eq!(q.where_clause.unwrap().all_conditions().len(), 2);
    }

    #[test]
    fn test_parse_at_before() {
        let q = parse_query(r#"MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01" RETURN s"#)
            .unwrap();
        let at = q.at_clause.unwrap();
        assert_eq!(at.field, "s.temporal");
        assert_eq!(at.relation, "BEFORE");
    }

    #[test]
    fn test_parse_at_within() {
        let q = parse_query(r#"MATCH (s:Situation) AT s.temporal WITHIN "2025-01-01" RETURN s"#)
            .unwrap();
        let at = q.at_clause.unwrap();
        assert_eq!(at.relation, "WITHIN");
    }

    // Fuzzy Sprint Phase 5 — AT clause with `AS FUZZY <rel> THRESHOLD <f>` tail.
    #[test]
    fn test_grammar_at_as_fuzzy_parses() {
        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01" AS FUZZY BEFORE THRESHOLD 0.6 RETURN s"#,
        )
        .unwrap();
        let at = q.at_clause.unwrap();
        assert_eq!(at.relation, "BEFORE");
        let fuzzy = at.fuzzy.expect("fuzzy tail must be present");
        assert_eq!(fuzzy.relation, "BEFORE");
        assert!((fuzzy.threshold - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_grammar_at_without_fuzzy_tail_is_none() {
        let q = parse_query(r#"MATCH (s:Situation) AT s.temporal AFTER "2024-01-01" RETURN s"#)
            .unwrap();
        let at = q.at_clause.unwrap();
        assert!(at.fuzzy.is_none());
    }

    #[test]
    fn test_parse_return_single_field() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        assert_eq!(q.return_clause.expressions.len(), 1);
        assert!(matches!(&q.return_clause.expressions[0], ReturnExpr::Field(s) if s == "e"));
    }

    #[test]
    fn test_parse_return_multiple_fields() {
        let q = parse_query("MATCH (e:Actor) RETURN e, e.name").unwrap();
        assert_eq!(q.return_clause.expressions.len(), 2);
    }

    #[test]
    fn test_parse_return_with_order_by() {
        let q = parse_query("MATCH (e:Actor) RETURN e ORDER BY e.confidence DESC").unwrap();
        let ob = q.return_clause.order_by.unwrap();
        assert_eq!(ob.field, "e.confidence");
        assert!(!ob.ascending);
    }

    #[test]
    fn test_parse_return_with_limit() {
        let q = parse_query("MATCH (e:Actor) RETURN e LIMIT 10").unwrap();
        assert_eq!(q.return_clause.limit, Some(10));
    }

    #[test]
    fn test_parse_full_instant_query() {
        let q = parse_query(
            r#"MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) WHERE e.confidence > 0.5 AT s.temporal BEFORE "2025-06-01" RETURN e.name, s ORDER BY e.confidence DESC LIMIT 5"#,
        )
        .unwrap();
        assert_eq!(q.match_clause.as_ref().unwrap().elements.len(), 3);
        assert!(q.where_clause.is_some());
        assert!(q.at_clause.is_some());
        assert_eq!(q.return_clause.expressions.len(), 2);
        assert_eq!(q.return_clause.limit, Some(5));
    }

    #[test]
    fn test_parse_invalid_syntax_error() {
        let result = parse_query("INVALID QUERY");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_query_error() {
        let result = parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_return_error() {
        let result = parse_query("MATCH (e:Actor)");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_node_without_binding() {
        let q = parse_query("MATCH (:Actor) RETURN *").unwrap();
        if let PatternElement::Node(n) = &q.match_clause.as_ref().unwrap().elements[0] {
            assert!(n.binding.is_none());
            assert_eq!(n.type_name, "Actor");
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_integer_value() {
        let q = parse_query("MATCH (e:Actor) WHERE e.age = 23 RETURN e").unwrap();
        let conds = q
            .where_clause
            .unwrap()
            .all_conditions()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        assert!(matches!(conds[0].value, QueryValue::Integer(23)));
    }

    #[test]
    fn test_parse_boolean_value() {
        let q = parse_query("MATCH (e:Actor) WHERE e.active = true RETURN e").unwrap();
        let conds = q
            .where_clause
            .unwrap()
            .all_conditions()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        assert!(matches!(conds[0].value, QueryValue::Boolean(true)));
    }

    #[test]
    fn test_parse_case_insensitive_keywords() {
        let q = parse_query("match (e:Actor) where e.x = 1 return e").unwrap();
        assert_eq!(q.match_clause.as_ref().unwrap().elements.len(), 1);
        assert!(q.where_clause.is_some());
    }

    #[test]
    fn test_parse_near_clause() {
        let q = parse_query(r#"MATCH (e:Actor) NEAR(e, "confession scene", 5) RETURN e"#).unwrap();
        let near = q.near_clause.unwrap();
        assert_eq!(near.binding, "e");
        assert_eq!(near.query_text, "confession scene");
        assert_eq!(near.k, 5);
    }

    #[test]
    fn test_parse_near_with_where() {
        let q = parse_query(
            r#"MATCH (e:Actor) WHERE e.confidence > 0.5 NEAR(e, "murder", 10) RETURN e"#,
        )
        .unwrap();
        assert!(q.where_clause.is_some());
        assert!(q.near_clause.is_some());
    }

    // ─── INFER Query Tests ───────────────────────────────────

    #[test]
    fn test_parse_infer_causes() {
        let q = parse_query("INFER CAUSES FOR s:Situation RETURN s").unwrap();
        let infer = q.infer_clause.unwrap();
        assert_eq!(infer.infer_type, InferType::Causes);
        assert_eq!(infer.target_binding, "s");
        assert_eq!(infer.target_type, "Situation");
        assert!(q.match_clause.is_none());
    }

    #[test]
    fn test_parse_infer_arcs_per_narrative() {
        let q = parse_query(
            r#"INFER ARCS FOR n:Narrative WHERE n.narrative_id = "oliver_twist" RETURN n"#,
        )
        .unwrap();
        let infer = q.infer_clause.unwrap();
        assert_eq!(infer.infer_type, InferType::Arcs);
        assert_eq!(infer.target_type, "Narrative");
        assert!(q.where_clause.is_some());
    }

    #[test]
    fn test_parse_infer_patterns_per_narrative() {
        let q =
            parse_query(r#"INFER PATTERNS FOR n:Narrative WHERE n.narrative_id = "x" RETURN n"#)
                .unwrap();
        assert_eq!(q.infer_clause.unwrap().infer_type, InferType::Patterns);
    }

    #[test]
    fn test_parse_infer_motivation() {
        let q = parse_query("INFER MOTIVATION FOR e:Actor RETURN e").unwrap();
        let infer = q.infer_clause.unwrap();
        assert_eq!(infer.infer_type, InferType::Motivation);
        assert_eq!(infer.target_type, "Actor");
    }

    #[test]
    fn test_parse_infer_game() {
        let q = parse_query("INFER GAME FOR s:Scene RETURN s").unwrap();
        let infer = q.infer_clause.unwrap();
        assert_eq!(infer.infer_type, InferType::Game);
        assert_eq!(infer.target_type, "Scene");
    }

    #[test]
    fn test_parse_infer_counterfactual() {
        let q = parse_query("INFER COUNTERFACTUAL FOR s:Situation RETURN s").unwrap();
        let infer = q.infer_clause.unwrap();
        assert_eq!(infer.infer_type, InferType::Counterfactual);
    }

    #[test]
    fn test_parse_infer_with_match() {
        let q = parse_query(
            "INFER MOTIVATION FOR e:Actor MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e",
        )
        .unwrap();
        assert!(q.infer_clause.is_some());
        assert!(q.match_clause.is_some());
    }

    #[test]
    fn test_parse_infer_with_assuming() {
        let q = parse_query(
            r#"INFER COUNTERFACTUAL FOR s:Situation ASSUMING s.action = "cooperate" RETURN s"#,
        )
        .unwrap();
        let assuming = q.assuming_clause.unwrap();
        assert_eq!(assuming.assumptions.len(), 1);
        assert_eq!(assuming.assumptions[0].field, "s.action");
        assert!(matches!(
            assuming.assumptions[0].value,
            QueryValue::String(ref s) if s == "cooperate"
        ));
    }

    #[test]
    fn test_parse_infer_with_under() {
        let q = parse_query("INFER GAME FOR s:Scene UNDER RATIONALITY = 0.5 RETURN s").unwrap();
        let under = q.under_clause.unwrap();
        assert_eq!(under.conditions.len(), 1);
        assert!(matches!(under.conditions[0], UnderCondition::Rationality(r) if r == 0.5));
    }

    #[test]
    fn test_parse_infer_with_under_information() {
        let q = parse_query(r#"INFER GAME FOR s:Scene UNDER INFORMATION = "complete" RETURN s"#)
            .unwrap();
        let under = q.under_clause.unwrap();
        assert!(matches!(
            &under.conditions[0],
            UnderCondition::Information(s) if s == "complete"
        ));
    }

    #[test]
    fn test_parse_infer_multiple_assuming() {
        let q = parse_query(
            r#"INFER COUNTERFACTUAL FOR s:Situation ASSUMING s.action = "cooperate" AND s.payoff = 5 RETURN s"#,
        )
        .unwrap();
        let assuming = q.assuming_clause.unwrap();
        assert_eq!(assuming.assumptions.len(), 2);
    }

    #[test]
    fn test_parse_infer_case_insensitive() {
        let q = parse_query("infer causes for s:Situation return s").unwrap();
        assert!(q.infer_clause.is_some());
        assert_eq!(q.infer_clause.unwrap().infer_type, InferType::Causes);
    }

    #[test]
    fn test_parse_backward_compatibility_match() {
        // Ensure existing MATCH queries still parse correctly
        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e LIMIT 10").unwrap();
        assert!(q.infer_clause.is_none());
        assert!(q.discover_clause.is_none());
        assert!(q.match_clause.is_some());
        assert!(q.where_clause.is_some());
        assert_eq!(q.return_clause.limit, Some(10));
    }

    // ─── DISCOVER Query Tests ────────────────────────────────

    #[test]
    fn test_parse_discover_patterns() {
        let q = parse_query("DISCOVER PATTERNS RETURN *").unwrap();
        let d = q.discover_clause.unwrap();
        assert_eq!(d.discover_type, DiscoverType::Patterns);
        assert!(d.target_binding.is_none());
    }

    #[test]
    fn test_parse_discover_arcs() {
        let q = parse_query("DISCOVER ARCS RETURN *").unwrap();
        let d = q.discover_clause.unwrap();
        assert_eq!(d.discover_type, DiscoverType::Arcs);
    }

    #[test]
    fn test_parse_discover_missing() {
        let q = parse_query("DISCOVER MISSING IN s:Situation RETURN s").unwrap();
        let d = q.discover_clause.unwrap();
        assert_eq!(d.discover_type, DiscoverType::Missing);
        assert_eq!(d.target_binding, Some("s".to_string()));
        assert_eq!(d.target_type, Some("Situation".to_string()));
    }

    #[test]
    fn test_parse_discover_across_narratives_all() {
        let q = parse_query("DISCOVER PATTERNS ACROSS NARRATIVES RETURN *").unwrap();
        assert!(q.discover_clause.is_some());
        let across = q.across_clause.unwrap();
        assert!(across.narrative_ids.is_none()); // all narratives
    }

    #[test]
    fn test_parse_discover_across_narratives_specific() {
        let q =
            parse_query(r#"DISCOVER PATTERNS ACROSS NARRATIVES ("hamlet", "macbeth") RETURN *"#)
                .unwrap();
        let across = q.across_clause.unwrap();
        let ids = across.narrative_ids.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], "hamlet");
        assert_eq!(ids[1], "macbeth");
    }

    #[test]
    fn test_parse_match_across_narratives() {
        let q = parse_query(
            r#"MATCH (e:Actor) ACROSS NARRATIVES ("hamlet") WHERE e.confidence > 0.5 RETURN e"#,
        )
        .unwrap();
        assert!(q.match_clause.is_some());
        let across = q.across_clause.unwrap();
        let ids = across.narrative_ids.unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "hamlet");
    }

    #[test]
    fn test_parse_discover_with_where() {
        let q =
            parse_query(r#"DISCOVER PATTERNS ACROSS NARRATIVES WHERE confidence > 0.5 RETURN *"#)
                .unwrap();
        assert!(q.discover_clause.is_some());
        assert!(q.across_clause.is_some());
        assert!(q.where_clause.is_some());
    }

    #[test]
    fn test_parse_discover_backward_compat() {
        // Old queries still parse
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        assert!(q.discover_clause.is_none());
        assert!(q.across_clause.is_none());
        assert!(q.match_clause.is_some());
    }

    // ─── DML Mutation Tests ─────────────────────────────────

    #[test]
    fn test_parse_create_narrative() {
        let stmt = parse_statement(
            r#"CREATE NARRATIVE "hamlet" TITLE "Hamlet" GENRE "tragedy" TAGS ("shakespeare", "revenge")"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateNarrative {
            id,
            title,
            genre,
            tags,
        }) = stmt
        {
            assert_eq!(id, "hamlet");
            assert_eq!(title, Some("Hamlet".to_string()));
            assert_eq!(genre, Some("tragedy".to_string()));
            assert_eq!(tags, vec!["shakespeare", "revenge"]);
        } else {
            panic!("Expected CreateNarrative mutation");
        }
    }

    #[test]
    fn test_parse_create_narrative_minimal() {
        let stmt = parse_statement(r#"CREATE NARRATIVE "test-narrative""#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateNarrative {
            id,
            title,
            genre,
            tags,
        }) = stmt
        {
            assert_eq!(id, "test-narrative");
            assert!(title.is_none());
            assert!(genre.is_none());
            assert!(tags.is_empty());
        } else {
            panic!("Expected CreateNarrative mutation");
        }
    }

    #[test]
    fn test_parse_create_entity() {
        let stmt = parse_statement(
            r#"CREATE (e:Actor {name: "Raskolnikov", age: 23}) IN NARRATIVE "crime" CONFIDENCE 0.9"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateEntity {
            entity_type,
            properties,
            narrative_id,
            confidence,
        }) = stmt
        {
            assert_eq!(entity_type, "Actor");
            assert_eq!(properties.len(), 2);
            assert_eq!(properties[0].key, "name");
            assert_eq!(narrative_id, Some("crime".to_string()));
            assert_eq!(confidence, Some(0.9));
        } else {
            panic!("Expected CreateEntity mutation");
        }
    }

    #[test]
    fn test_parse_create_entity_minimal() {
        let stmt = parse_statement(r#"CREATE (:Location {name: "Moscow"})"#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateEntity {
            entity_type,
            properties,
            narrative_id,
            confidence,
        }) = stmt
        {
            assert_eq!(entity_type, "Location");
            assert_eq!(properties.len(), 1);
            assert!(narrative_id.is_none());
            assert!(confidence.is_none());
        } else {
            panic!("Expected CreateEntity mutation");
        }
    }

    #[test]
    fn test_parse_create_situation() {
        let stmt = parse_statement(
            r#"CREATE SITUATION AT Scene CONTENT "He entered the room" IN NARRATIVE "crime" CONFIDENCE 0.75"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateSituation {
            level,
            content,
            narrative_id,
            confidence,
        }) = stmt
        {
            assert_eq!(level, "Scene");
            assert_eq!(content, vec!["He entered the room"]);
            assert_eq!(narrative_id, Some("crime".to_string()));
            assert_eq!(confidence, Some(0.75));
        } else {
            panic!("Expected CreateSituation mutation");
        }
    }

    #[test]
    fn test_parse_create_situation_multi_content() {
        let stmt =
            parse_statement(r#"CREATE SITUATION AT Beat CONTENT "Line one" CONTENT "Line two""#)
                .unwrap();
        if let TensaStatement::Mutation(MutationStatement::CreateSituation { content, .. }) = stmt {
            assert_eq!(content.len(), 2);
        } else {
            panic!("Expected CreateSituation mutation");
        }
    }

    #[test]
    fn test_parse_delete_entity() {
        let stmt =
            parse_statement(r#"DELETE ENTITY "550e8400-e29b-41d4-a716-446655440000""#).unwrap();
        assert!(matches!(
            stmt,
            TensaStatement::Mutation(MutationStatement::DeleteEntity { .. })
        ));
    }

    #[test]
    fn test_parse_delete_narrative() {
        let stmt = parse_statement(r#"DELETE NARRATIVE "hamlet""#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::DeleteNarrative { id }) = stmt {
            assert_eq!(id, "hamlet");
        } else {
            panic!("Expected DeleteNarrative mutation");
        }
    }

    #[test]
    fn test_parse_update_entity() {
        let stmt = parse_statement(
            r#"UPDATE ENTITY "550e8400-e29b-41d4-a716-446655440000" SET name = "Rodion", confidence = 0.95"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::UpdateEntity { id, set_pairs, .. }) =
            stmt
        {
            assert_eq!(id, "550e8400-e29b-41d4-a716-446655440000");
            assert_eq!(set_pairs.len(), 2);
            assert_eq!(set_pairs[0].0, "name");
            assert_eq!(set_pairs[1].0, "confidence");
        } else {
            panic!("Expected UpdateEntity mutation");
        }
    }

    #[test]
    fn test_parse_update_narrative() {
        let stmt =
            parse_statement(r#"UPDATE NARRATIVE "hamlet" SET title = "The Tragedy of Hamlet""#)
                .unwrap();
        if let TensaStatement::Mutation(MutationStatement::UpdateNarrative { id, set_pairs }) = stmt
        {
            assert_eq!(id, "hamlet");
            assert_eq!(set_pairs.len(), 1);
            assert_eq!(set_pairs[0].0, "title");
        } else {
            panic!("Expected UpdateNarrative mutation");
        }
    }

    #[test]
    fn test_parse_add_participant() {
        let stmt = parse_statement(
            r#"ADD PARTICIPANT "eid" TO SITUATION "sid" ROLE Protagonist ACTION "confesses""#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::AddParticipant {
            entity_id,
            situation_id,
            role,
            action,
        }) = stmt
        {
            assert_eq!(entity_id, "eid");
            assert_eq!(situation_id, "sid");
            assert_eq!(role, "Protagonist");
            assert_eq!(action, Some("confesses".to_string()));
        } else {
            panic!("Expected AddParticipant mutation");
        }
    }

    #[test]
    fn test_parse_add_participant_no_action() {
        let stmt =
            parse_statement(r#"ADD PARTICIPANT "eid" TO SITUATION "sid" ROLE Witness"#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::AddParticipant { action, .. }) = stmt {
            assert!(action.is_none());
        } else {
            panic!("Expected AddParticipant mutation");
        }
    }

    #[test]
    fn test_parse_remove_participant() {
        let stmt = parse_statement(r#"REMOVE PARTICIPANT "eid" FROM SITUATION "sid""#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::RemoveParticipant {
            entity_id,
            situation_id,
        }) = stmt
        {
            assert_eq!(entity_id, "eid");
            assert_eq!(situation_id, "sid");
        } else {
            panic!("Expected RemoveParticipant mutation");
        }
    }

    #[test]
    fn test_parse_add_cause() {
        let stmt = parse_statement(
            r#"ADD CAUSE FROM "sid1" TO "sid2" TYPE Contributing STRENGTH 0.8 MECHANISM "guilt""#,
        )
        .unwrap();
        if let TensaStatement::Mutation(MutationStatement::AddCause {
            from_id,
            to_id,
            causal_type,
            strength,
            mechanism,
        }) = stmt
        {
            assert_eq!(from_id, "sid1");
            assert_eq!(to_id, "sid2");
            assert_eq!(causal_type, Some("Contributing".to_string()));
            assert_eq!(strength, Some(0.8));
            assert_eq!(mechanism, Some("guilt".to_string()));
        } else {
            panic!("Expected AddCause mutation");
        }
    }

    #[test]
    fn test_parse_add_cause_minimal() {
        let stmt = parse_statement(r#"ADD CAUSE FROM "sid1" TO "sid2""#).unwrap();
        if let TensaStatement::Mutation(MutationStatement::AddCause {
            causal_type,
            strength,
            mechanism,
            ..
        }) = stmt
        {
            assert!(causal_type.is_none());
            assert!(strength.is_none());
            assert!(mechanism.is_none());
        } else {
            panic!("Expected AddCause mutation");
        }
    }

    #[test]
    fn test_parse_remove_cause() {
        let stmt = parse_statement(r#"REMOVE CAUSE FROM "sid1" TO "sid2""#).unwrap();
        assert!(matches!(
            stmt,
            TensaStatement::Mutation(MutationStatement::RemoveCause { .. })
        ));
    }

    #[test]
    fn test_parse_statement_match_backward_compat() {
        // parse_statement should work for read queries too
        let stmt = parse_statement("MATCH (e:Actor) RETURN e").unwrap();
        assert!(matches!(stmt, TensaStatement::Query(_)));
    }

    #[test]
    fn test_parse_dml_case_insensitive() {
        let stmt = parse_statement(r#"create narrative "test""#).unwrap();
        assert!(matches!(
            stmt,
            TensaStatement::Mutation(MutationStatement::CreateNarrative { .. })
        ));
    }

    // ─── EXPLAIN Tests ──────────────────────────────────────────

    #[test]
    fn test_parse_explain_match() {
        let q = parse_query("EXPLAIN MATCH (e:Actor) RETURN e").unwrap();
        assert!(q.explain);
        assert!(q.match_clause.is_some());
    }

    #[test]
    fn test_parse_explain_with_where() {
        let q = parse_query("EXPLAIN MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
        assert!(q.explain);
        assert!(q.where_clause.is_some());
    }

    #[test]
    fn test_parse_explain_infer() {
        let q = parse_query("EXPLAIN INFER CAUSES FOR s:Situation RETURN s").unwrap();
        assert!(q.explain);
        assert!(q.infer_clause.is_some());
    }

    #[test]
    fn test_parse_non_explain_has_false_flag() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        assert!(!q.explain);
    }

    // ─── OR Condition Tests ─────────────────────────────────────

    #[test]
    fn test_parse_or_simple() {
        let q =
            parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 OR e.confidence < 0.1 RETURN e")
                .unwrap();
        let wc = q.where_clause.unwrap();
        assert!(matches!(wc.expr, ConditionExpr::Or(_)));
        if let ConditionExpr::Or(children) = &wc.expr {
            assert_eq!(children.len(), 2);
        }
    }

    #[test]
    fn test_parse_and_or_precedence() {
        // AND has higher precedence than OR: a AND b OR c → Or(And(a,b), c)
        let q = parse_query(
            r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "A" OR e.confidence < 0.1 RETURN e"#,
        ).unwrap();
        let wc = q.where_clause.unwrap();
        if let ConditionExpr::Or(children) = &wc.expr {
            assert_eq!(children.len(), 2);
            assert!(matches!(&children[0], ConditionExpr::And(inner) if inner.len() == 2));
            assert!(matches!(&children[1], ConditionExpr::Single(_)));
        } else {
            panic!("Expected Or at top level");
        }
    }

    #[test]
    fn test_parse_parenthesized_or() {
        // (a OR b) AND c → And(Or(a,b), c)
        let q = parse_query(
            r#"MATCH (e:Actor) WHERE (e.confidence > 0.5 OR e.confidence < 0.1) AND e.name = "A" RETURN e"#,
        ).unwrap();
        let wc = q.where_clause.unwrap();
        if let ConditionExpr::And(children) = &wc.expr {
            assert_eq!(children.len(), 2);
            assert!(matches!(&children[0], ConditionExpr::Or(_)));
            assert!(matches!(&children[1], ConditionExpr::Single(_)));
        } else {
            panic!("Expected And at top level, got {:?}", wc.expr);
        }
    }

    #[test]
    fn test_parse_nested_parens() {
        let q = parse_query("MATCH (e:Actor) WHERE ((e.confidence > 0.5)) RETURN e").unwrap();
        let wc = q.where_clause.unwrap();
        // Should flatten to a single condition
        assert!(matches!(wc.expr, ConditionExpr::Single(_)));
    }

    #[test]
    fn test_parse_and_only_backward_compat() {
        let q =
            parse_query(r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "A" RETURN e"#)
                .unwrap();
        let wc = q.where_clause.unwrap();
        assert_eq!(wc.all_conditions().len(), 2);
    }

    // ─── Aggregation Tests ──────────────────────────────────────

    #[test]
    fn test_parse_count_star() {
        let q = parse_query("MATCH (e:Actor) RETURN COUNT(*)").unwrap();
        assert_eq!(q.return_clause.expressions.len(), 1);
        assert!(matches!(
            &q.return_clause.expressions[0],
            ReturnExpr::Aggregate { func: AggregateFunc::Count, field } if field == "*"
        ));
    }

    #[test]
    fn test_parse_count_field() {
        let q = parse_query("MATCH (e:Actor) RETURN COUNT(e.id)").unwrap();
        assert!(matches!(
            &q.return_clause.expressions[0],
            ReturnExpr::Aggregate { func: AggregateFunc::Count, field } if field == "e.id"
        ));
    }

    #[test]
    fn test_parse_avg_field() {
        let q = parse_query("MATCH (e:Actor) RETURN AVG(e.confidence)").unwrap();
        assert!(matches!(
            &q.return_clause.expressions[0],
            ReturnExpr::Aggregate { func: AggregateFunc::Avg, field } if field == "e.confidence"
        ));
    }

    #[test]
    fn test_parse_multiple_aggregates() {
        let q =
            parse_query("MATCH (e:Actor) RETURN COUNT(*), AVG(e.confidence), MAX(e.confidence)")
                .unwrap();
        assert_eq!(q.return_clause.expressions.len(), 3);
        assert!(q.return_clause.expressions[0].is_aggregate());
        assert!(q.return_clause.expressions[1].is_aggregate());
        assert!(q.return_clause.expressions[2].is_aggregate());
    }

    #[test]
    fn test_parse_mixed_fields_and_aggs() {
        let q =
            parse_query("MATCH (e:Actor) GROUP BY e.entity_type RETURN e.entity_type, COUNT(*)")
                .unwrap();
        assert_eq!(q.return_clause.expressions.len(), 2);
        assert!(
            matches!(&q.return_clause.expressions[0], ReturnExpr::Field(s) if s == "e.entity_type")
        );
        assert!(matches!(
            &q.return_clause.expressions[1],
            ReturnExpr::Aggregate {
                func: AggregateFunc::Count,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_group_by_single() {
        let q =
            parse_query("MATCH (e:Actor) GROUP BY e.entity_type RETURN e.entity_type, COUNT(*)")
                .unwrap();
        let gb = q.group_by.unwrap();
        assert_eq!(gb.fields, vec!["e.entity_type"]);
    }

    #[test]
    fn test_parse_group_by_multiple() {
        let q = parse_query(
            "MATCH (e:Actor) GROUP BY e.entity_type, e.narrative_id RETURN e.entity_type, COUNT(*)",
        )
        .unwrap();
        let gb = q.group_by.unwrap();
        assert_eq!(gb.fields.len(), 2);
    }

    #[test]
    fn test_parse_full_aggregate_query() {
        let q = parse_query(
            "MATCH (e:Actor) WHERE e.confidence > 0.3 GROUP BY e.entity_type RETURN e.entity_type, COUNT(*), AVG(e.confidence) LIMIT 10",
        ).unwrap();
        assert!(q.where_clause.is_some());
        assert!(q.group_by.is_some());
        assert_eq!(q.return_clause.expressions.len(), 3);
        assert_eq!(q.return_clause.limit, Some(10));
    }

    // ─── SPATIAL Clause Parser Tests (Sprint P3.7) ──────────────

    #[test]
    fn test_parse_spatial_clause() {
        let q = parse_query(
            "MATCH (s:Situation) SPATIAL s.spatial WITHIN 10.0 KM OF (40.7, -74.0) RETURN s",
        )
        .unwrap();
        let spatial = q.spatial_clause.unwrap();
        assert_eq!(spatial.field, "s.spatial");
        assert!((spatial.radius_km - 10.0).abs() < 0.01);
        assert!((spatial.center_lat - 40.7).abs() < 0.01);
        assert!((spatial.center_lon - (-74.0)).abs() < 0.01);
    }

    #[test]
    fn test_parse_spatial_with_where() {
        let q = parse_query(
            r#"MATCH (s:Situation) WHERE s.confidence > 0.5 SPATIAL s.spatial WITHIN 50.0 KM OF (51.5, -0.1) RETURN s"#,
        )
        .unwrap();
        assert!(q.where_clause.is_some());
        assert!(q.spatial_clause.is_some());
        let spatial = q.spatial_clause.unwrap();
        assert!((spatial.radius_km - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_no_spatial_has_none() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        assert!(q.spatial_clause.is_none());
    }

    // ─── PATH Query Tests ──────────────────────────────────────

    #[test]
    fn test_parse_path_shortest() {
        let q =
            parse_query(r#"MATCH PATH SHORTEST (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.mode, PathMode::Shortest);
        assert_eq!(path.start_binding, "s1");
        assert_eq!(path.end_binding, "s2");
        assert_eq!(path.rel_type, "CAUSES");
        assert_eq!(path.min_depth, 1);
        assert_eq!(path.max_depth, 5);
    }

    #[test]
    fn test_parse_path_all() {
        let q = parse_query(r#"MATCH PATH ALL (a) -[:CAUSES*2..10]-> (b) RETURN a"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.mode, PathMode::All);
        assert_eq!(path.start_binding, "a");
        assert_eq!(path.end_binding, "b");
        assert_eq!(path.min_depth, 2);
        assert_eq!(path.max_depth, 10);
    }

    #[test]
    fn test_parse_path_no_depth_range() {
        let q = parse_query(r#"MATCH PATH SHORTEST (s1) -[:CAUSES*]-> (s2) RETURN s1"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.min_depth, 1);
        assert_eq!(path.max_depth, 10);
    }

    #[test]
    fn test_parse_path_with_where() {
        let q = parse_query(
            r#"MATCH PATH SHORTEST (s1) -[:CAUSES*1..3]-> (s2) WHERE s1.confidence > 0.5 RETURN s1"#,
        )
        .unwrap();
        assert!(q.path_clause.is_some());
        assert!(q.where_clause.is_some());
    }

    #[test]
    fn test_explain_path_query() {
        let q = parse_query(r#"EXPLAIN MATCH PATH SHORTEST (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#)
            .unwrap();
        assert!(q.explain);
        assert!(q.path_clause.is_some());
    }

    // ─── Path restrictor tests ────────────────────────────

    #[test]
    fn test_parse_path_with_walk_restrictor() {
        let q = parse_query(r#"MATCH PATH SHORTEST WALK (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#)
            .unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Walk);
    }

    #[test]
    fn test_parse_path_with_trail_restrictor() {
        let q =
            parse_query(r#"MATCH PATH ALL TRAIL (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Trail);
        assert_eq!(path.mode, PathMode::All);
    }

    #[test]
    fn test_parse_path_with_acyclic_restrictor() {
        let q =
            parse_query(r#"MATCH PATH SHORTEST ACYCLIC (a) -[:CAUSES*]-> (b) RETURN a"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Acyclic);
    }

    #[test]
    fn test_parse_path_with_simple_restrictor() {
        let q =
            parse_query(r#"MATCH PATH ALL SIMPLE (x) -[:CAUSES*1..10]-> (y) RETURN x"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Simple);
    }

    #[test]
    fn test_parse_path_no_restrictor_defaults_to_walk() {
        let q =
            parse_query(r#"MATCH PATH SHORTEST (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Walk);
    }

    #[test]
    fn test_parse_path_restrictor_case_insensitive() {
        let q = parse_query(r#"MATCH PATH ALL trail (s1) -[:CAUSES*]-> (s2) RETURN s1"#).unwrap();
        let path = q.path_clause.unwrap();
        assert_eq!(path.restrictor, PathRestrictor::Trail);
    }

    #[test]
    fn test_export_csv() {
        let stmt = parse_statement(r#"EXPORT NARRATIVE "hamlet" AS csv"#).unwrap();
        match stmt {
            TensaStatement::Export {
                narrative_id,
                format,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(format, "csv");
            }
            _ => panic!("Expected Export statement"),
        }
    }

    #[test]
    fn test_export_manuscript() {
        let stmt =
            parse_statement(r#"EXPORT NARRATIVE "crime-and-punishment" AS manuscript"#).unwrap();
        match stmt {
            TensaStatement::Export {
                narrative_id,
                format,
            } => {
                assert_eq!(narrative_id, "crime-and-punishment");
                assert_eq!(format, "manuscript");
            }
            _ => panic!("Expected Export statement"),
        }
    }

    #[test]
    fn test_export_all_formats() {
        for fmt in &["csv", "graphml", "json", "manuscript", "report", "archive"] {
            let q = format!(r#"EXPORT NARRATIVE "test" AS {}"#, fmt);
            let stmt = parse_statement(&q).unwrap();
            assert!(matches!(stmt, TensaStatement::Export { .. }));
        }
    }

    #[test]
    fn test_export_case_insensitive() {
        let stmt = parse_statement(r#"EXPORT NARRATIVE "test" AS JSON"#).unwrap();
        match stmt {
            TensaStatement::Export { format, .. } => {
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Export statement"),
        }
    }

    // ─── ASK Query Tests (Sprint RAG-2) ──────────────────────

    #[test]
    fn test_parse_ask_simple() {
        let q = parse_query(r#"ASK "What happened?""#).unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.question, "What happened?");
        assert!(ask.narrative_id.is_none());
        assert!(ask.mode.is_none());
    }

    #[test]
    fn test_parse_ask_with_over() {
        let q = parse_query(r#"ASK "Who is the villain?" OVER "my-narrative""#).unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.question, "Who is the villain?");
        assert_eq!(ask.narrative_id, Some("my-narrative".to_string()));
    }

    #[test]
    fn test_parse_ask_with_mode() {
        let q = parse_query(r#"ASK "What happened?" MODE local"#).unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.mode, Some(RetrievalMode::Local));
    }

    #[test]
    fn test_parse_ask_full() {
        let q = parse_query(r#"ASK "What happened to the hero?" OVER "nar" MODE hybrid RETURN *"#)
            .unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.question, "What happened to the hero?");
        assert_eq!(ask.narrative_id, Some("nar".to_string()));
        assert_eq!(ask.mode, Some(RetrievalMode::Hybrid));
        assert_eq!(q.return_clause.expressions.len(), 1);
    }

    #[test]
    fn test_explain_ask() {
        let q = parse_query(r#"EXPLAIN ASK "Why did he leave?""#).unwrap();
        assert!(q.explain);
        assert!(q.ask_clause.is_some());
    }

    #[test]
    fn test_parse_ask_modes() {
        for (mode_str, expected) in &[
            ("local", RetrievalMode::Local),
            ("global", RetrievalMode::Global),
            ("hybrid", RetrievalMode::Hybrid),
            ("mix", RetrievalMode::Mix),
            ("lazy", RetrievalMode::Lazy),
            ("ppr", RetrievalMode::Ppr),
        ] {
            let q = parse_query(&format!(r#"ASK "q" MODE {}"#, mode_str)).unwrap();
            assert_eq!(q.ask_clause.unwrap().mode, Some(*expected));
        }
    }

    #[test]
    fn test_parse_ask_lazy_mode() {
        let q = parse_query(r#"ASK "What happened?" OVER "story" MODE lazy"#).unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.mode, Some(RetrievalMode::Lazy));
        assert_eq!(ask.narrative_id.as_deref(), Some("story"));
    }

    #[test]
    fn test_parse_ask_ppr_mode() {
        let q = parse_query(r#"ASK "Who is central?" MODE ppr"#).unwrap();
        let ask = q.ask_clause.unwrap();
        assert_eq!(ask.mode, Some(RetrievalMode::Ppr));
    }

    // ─── EATH Phase 5 — Surrogate verbs (CALIBRATE / GENERATE) ───
    //
    // Grammar shape recap:
    //   CALIBRATE SURROGATE USING '<model>' FOR "<narrative_id>"
    //   GENERATE NARRATIVE "<output>" LIKE "<source>"
    //       [USING SURROGATE '<model>']
    //       [PARAMS { json }] [SEED <u64>] [STEPS <usize>]
    //       [LABEL_PREFIX '<str>']
    //
    // Asymmetry: CALIBRATE has NO default model; GENERATE defaults to 'eath'.

    #[test]
    fn test_parse_calibrate_with_explicit_model() {
        let stmt =
            parse_statement(r#"CALIBRATE SURROGATE USING 'eath' FOR "hamlet""#).unwrap();
        match stmt {
            TensaStatement::CalibrateSurrogate {
                narrative_id,
                model,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(model, "eath");
            }
            _ => panic!("Expected CalibrateSurrogate"),
        }
    }

    #[test]
    fn test_parse_calibrate_requires_model_name() {
        // USING clause omitted — must be a parse error (no default model
        // for calibration, see parser/grammar comments).
        let result = parse_statement(r#"CALIBRATE SURROGATE FOR "hamlet""#);
        assert!(
            result.is_err(),
            "CALIBRATE without USING clause must error at parse time"
        );
    }

    #[test]
    fn test_parse_generate_with_explicit_model() {
        let stmt = parse_statement(
            r#"GENERATE NARRATIVE "synthetic-hamlet" LIKE "hamlet" USING SURROGATE 'eath'"#,
        )
        .unwrap();
        match stmt {
            TensaStatement::GenerateNarrative {
                output_id,
                source_id,
                model,
                params,
                seed,
                num_steps,
                label_prefix,
            } => {
                assert_eq!(output_id, "synthetic-hamlet");
                assert_eq!(source_id, "hamlet");
                assert_eq!(model, "eath");
                assert!(params.is_none());
                assert!(seed.is_none());
                assert!(num_steps.is_none());
                assert!(label_prefix.is_none());
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_with_default_model_when_using_omitted() {
        let stmt =
            parse_statement(r#"GENERATE NARRATIVE "synthetic-hamlet" LIKE "hamlet""#).unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { model, .. } => {
                assert_eq!(model, "eath", "default model must be 'eath'");
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_with_params_clause() {
        let stmt = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" PARAMS { "num_entities": 50, "rho_low": 0.1 }"#,
        )
        .unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { params, .. } => {
                let p = params.expect("PARAMS body must parse to Some");
                assert_eq!(p["num_entities"].as_u64(), Some(50));
                assert!((p["rho_low"].as_f64().unwrap() - 0.1).abs() < 1e-9);
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_with_seed_clause() {
        let stmt =
            parse_statement(r#"GENERATE NARRATIVE "out" LIKE "src" SEED 42"#).unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { seed, .. } => {
                assert_eq!(seed, Some(42));
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_with_params_and_seed() {
        let stmt = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" PARAMS { "rho_low": 0.2 } SEED 7"#,
        )
        .unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { params, seed, .. } => {
                assert!(params.is_some());
                assert_eq!(seed, Some(7));
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_errors_on_malformed_params_json() {
        // Invalid JSON in the PARAMS body — must error at PARSE time
        // (not deferred to execute).
        let result = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" PARAMS { not valid json }"#,
        );
        assert!(
            result.is_err(),
            "malformed PARAMS body must error at parse time"
        );
        if let Err(e) = result {
            let msg = format!("{e}");
            assert!(
                msg.contains("PARAMS") || msg.contains("JSON"),
                "error message should mention PARAMS / JSON; got: {msg}"
            );
        }
    }

    #[test]
    fn test_parse_generate_with_steps_clause() {
        let stmt = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" STEPS 250"#,
        )
        .unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { num_steps, .. } => {
                assert_eq!(num_steps, Some(250));
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    #[test]
    fn test_parse_generate_with_label_prefix_clause() {
        let stmt = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" LABEL_PREFIX 'experiment-3'"#,
        )
        .unwrap();
        match stmt {
            TensaStatement::GenerateNarrative { label_prefix, .. } => {
                assert_eq!(label_prefix.as_deref(), Some("experiment-3"));
            }
            _ => panic!("Expected GenerateNarrative"),
        }
    }

    // ── Phase 13c — COMPUTE DUAL_SIGNIFICANCE ────────────────────────────

    /// T5 — `COMPUTE DUAL_SIGNIFICANCE FOR ... USING ...` parses with the
    /// MODELS clause omitted; the AST carries `models: None` and the planner
    /// later expands that to the canonical `["eath", "nudhy"]` pair.
    #[test]
    fn test_dual_sig_tensaql_parses_default_models() {
        let stmt = parse_statement(
            r#"COMPUTE DUAL_SIGNIFICANCE FOR "hamlet" USING 'patterns'"#,
        )
        .expect("dual-significance form must parse");
        match stmt {
            TensaStatement::ComputeDualSignificance {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(metric, "patterns");
                assert!(k_per_model.is_none(), "K_PER_MODEL omitted → None");
                assert!(
                    models.is_none(),
                    "MODELS omitted → None at AST layer (planner fills in default)"
                );
            }
            other => panic!("Expected ComputeDualSignificance, got: {other:?}"),
        }
    }

    /// Exercise the K_PER_MODEL + MODELS clauses end-to-end so the parser
    /// branches stay live regression-test material.
    #[test]
    fn test_parse_dual_significance_with_explicit_clauses() {
        let stmt = parse_statement(
            r#"COMPUTE DUAL_SIGNIFICANCE FOR "hamlet" USING 'communities' K_PER_MODEL 25 MODELS 'eath','nudhy'"#,
        )
        .expect("dual-significance with explicit clauses must parse");
        match stmt {
            TensaStatement::ComputeDualSignificance {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(metric, "communities");
                assert_eq!(k_per_model, Some(25));
                assert_eq!(models.as_deref(), Some(&["eath".to_string(), "nudhy".to_string()][..]));
            }
            other => panic!("Expected ComputeDualSignificance, got: {other:?}"),
        }
    }

    // ── Phase 15c — INFER HYPERGRAPH FROM DYNAMICS ────────────────────────

    /// T2 — `INFER HYPERGRAPH FROM DYNAMICS FOR "..."` parses with no
    /// optional clauses; the AST carries `observation = None`,
    /// `max_order = None`, `lambda = None` so defaults get applied at the
    /// planner / engine layer (observation = participation_rate, max_order = 3,
    /// lambda auto-selected).
    #[test]
    fn test_reconstruction_tensaql_parses_with_defaults() {
        let stmt = parse_statement(
            r#"INFER HYPERGRAPH FROM DYNAMICS FOR "disinfo-1""#,
        )
        .expect("reconstruction form with no clauses must parse");
        match stmt {
            TensaStatement::InferHypergraphReconstruction {
                narrative_id,
                observation,
                max_order,
                lambda,
                ..
            } => {
                assert_eq!(narrative_id, "disinfo-1");
                assert!(
                    observation.is_none(),
                    "USING OBSERVATION omitted → None; planner fills participation_rate"
                );
                assert!(max_order.is_none(), "MAX_ORDER omitted → None");
                assert!(lambda.is_none(), "LAMBDA omitted → None; engine auto-selects");
            }
            other => panic!("Expected InferHypergraphReconstruction, got: {other:?}"),
        }
    }

    /// Exercise every optional clause so the parser branches stay live.
    #[test]
    fn test_parse_reconstruction_with_explicit_clauses() {
        let stmt = parse_statement(
            r#"INFER HYPERGRAPH FROM DYNAMICS FOR "tg-corpus" USING OBSERVATION 'participation_rate' MAX_ORDER 3 LAMBDA 0.05"#,
        )
        .expect("reconstruction with explicit clauses must parse");
        match stmt {
            TensaStatement::InferHypergraphReconstruction {
                narrative_id,
                observation,
                max_order,
                lambda,
                ..
            } => {
                assert_eq!(narrative_id, "tg-corpus");
                assert_eq!(observation.as_deref(), Some("participation_rate"));
                assert_eq!(max_order, Some(3));
                assert_eq!(lambda, Some(0.05));
            }
            other => panic!("Expected InferHypergraphReconstruction, got: {other:?}"),
        }
    }

    /// T3 — `INFER OPINION_DYNAMICS(...)` parses with named params.
    #[test]
    fn test_opinion_dynamics_tensaql_parses_named_params() {
        let stmt = parse_statement(
            r#"INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise', mu := 0.5, initial := 'uniform' ) FOR "n1""#,
        )
        .expect("INFER OPINION_DYNAMICS must parse");
        match stmt {
            TensaStatement::InferOpinionDynamics {
                narrative_id,
                confidence_bound,
                variant,
                mu,
                initial,
                ..
            } => {
                assert_eq!(narrative_id, "n1");
                assert_eq!(confidence_bound, Some(0.3));
                assert_eq!(variant.as_deref(), Some("pairwise"));
                assert_eq!(mu, Some(0.5));
                assert_eq!(initial.as_deref(), Some("uniform"));
            }
            other => panic!("Expected InferOpinionDynamics, got: {other:?}"),
        }

        // Minimal form — only confidence_bound + variant.
        let stmt = parse_statement(
            r#"INFER OPINION_DYNAMICS( confidence_bound := 0.4, variant := 'group_mean' ) FOR "narr-2""#,
        )
        .expect("minimal form must parse");
        match stmt {
            TensaStatement::InferOpinionDynamics {
                narrative_id,
                confidence_bound,
                variant,
                mu,
                initial,
                ..
            } => {
                assert_eq!(narrative_id, "narr-2");
                assert_eq!(confidence_bound, Some(0.4));
                assert_eq!(variant.as_deref(), Some("group_mean"));
                assert!(mu.is_none());
                assert!(initial.is_none());
            }
            other => panic!("Expected InferOpinionDynamics, got: {other:?}"),
        }

        // Phase-transition sweep form.
        let stmt = parse_statement(
            r#"INFER OPINION_PHASE_TRANSITION( c_start := 0.05, c_end := 0.5, c_steps := 10 ) FOR "n1""#,
        )
        .expect("INFER OPINION_PHASE_TRANSITION must parse");
        match stmt {
            TensaStatement::InferOpinionPhaseTransition {
                narrative_id,
                c_start,
                c_end,
                c_steps,
                ..
            } => {
                assert_eq!(narrative_id, "n1");
                assert_eq!(c_start, 0.05);
                assert_eq!(c_end, 0.5);
                assert_eq!(c_steps, 10);
            }
            other => panic!("Expected InferOpinionPhaseTransition, got: {other:?}"),
        }
    }
}
