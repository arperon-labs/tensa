# tensa-bench — Outstanding Work

Status of the academic benchmark suite (`tests/tensa_bench/`). Updated 2026-04-16.

## Done

- Reporting: `TensaBenchReport`, `DomainReport`, `DatasetReport` + JSON/Markdown/LaTeX outputs
- Metrics: MRR, Hits@1/3/10, BLEU-1/4, ROUGE-L, METEOR, Exact Match, token-F1, multi-class P/R/F1
- Dataset loaders: ICEWS14, ICEWS18, GDELT, HotpotQA, 2WikiMultiHopQA, ROCStories, NarrativeQA, MAVEN-ERE
- Adapters: `tkg_adapter` (fully wired), `multihop_adapter`, `narrative_adapter`
- Baselines: published numbers from TComplEx, RE-NET, TANGO, TimeTraveler, GraphRAG, LightRAG, HippoRAG, GPT-4, EEQA, DEGREE, TagPrime
- Datasets downloaded to `data/benchmarks/` (gitignored) — everything except ROCStories
- 34 unit tests covering metric correctness and loader format parsing

## Remaining

### 1. Wire up stubbed `#[ignore] bench_*_full` tests

Each currently prints baselines then returns — the evaluation loop is the last mile.

#### `bench_hotpotqa_full` — [tests/tensa_bench_multihop.rs:242](../tensa_bench_multihop.rs#L242)
For each item (up to `TENSA_BENCH_SAMPLE`):
1. Fresh `MemoryStore` → `Hypergraph`
2. Ingest all 10 context paragraphs via `IngestionPipeline`
3. Call `execute_ask(question, narrative_id, mode)` for each mode in [Local, Hybrid, Drift, PPR]
4. Score EM + F1 via `multihop_adapter::score_answer`
5. Accumulate `MultihopModeResult` per mode

**Requires:** `server` feature flag, LLM provider (OPENROUTER_API_KEY / ANTHROPIC_API_KEY / LOCAL_LLM_URL)
**Adapter functions ready:** `build_ingestion_text`, `build_ask_query`, `score_answer`, `MultihopModeResult::from_items`

#### `bench_maven_ere_full` — [tests/tensa_bench_narrative.rs:228](../tensa_bench_narrative.rs#L228)
For each document:
1. Ingest `doc.text` via `IngestionPipeline`
2. Read back extracted causal links + temporal relations from the hypergraph
3. Convert to `ExtractedRelation` entries (map TENSA → MAVEN via `tensa_causal_to_maven`, `tensa_temporal_to_maven`)
4. Call `narrative_adapter::evaluate_maven_document(gold, predicted)` — already wired to return `ConfusionMatrix`
5. Aggregate macro/micro F1 across documents

**Scoring harness is ready.** Just needs the ingestion call + extraction readback.

#### `bench_rocstories_full` — [tests/tensa_bench_narrative.rs:266](../tensa_bench_narrative.rs#L266)
**Blocked:** requires academic agreement at https://cs.rochester.edu/nlp/rocstories/

Logic when data arrives:
1. Ingest 4-sentence prefix via `IngestionPipeline`
2. `execute_ask` with `build_story_cloze_prompt(prefix, ending_a, ending_b)`
3. `parse_ending_choice(response)` → compare to `correct_ending`
4. Accuracy over all items

#### `bench_narrativeqa_full` — [tests/tensa_bench_narrative.rs:313](../tensa_bench_narrative.rs#L313)
For each item:
1. Ingest `item.summary` (or `document_text`) via `IngestionPipeline`
2. `execute_ask(item.question)` → generated answer
3. Score against `item.answers` (2 references) with BLEU-1/4, ROUGE-L, METEOR via `aggregate_nlg_metrics`

**Requires:** LLM provider.

### 2. 2WikiMultiHopQA benchmark runner

Loader exists; no `bench_2wikimultihop_full` function yet — only a smoke test that verifies loading.
Easy win: the logic mirrors `bench_hotpotqa_full` with a different loader and baselines.

### 3. GDELT benchmark runner

Loader exists (uses `TimestampConfig::gdelt()`); the generic `run_tkg_link_prediction::<Gdelt>` helper in [tests/tensa_bench_tkg.rs](../tensa_bench_tkg.rs) is ready.
Just needs:

```rust
#[test]
#[ignore]
fn bench_gdelt_link_prediction() {
    run_tkg_link_prediction::<Gdelt>("GDELT", "gdelt", gdelt_baselines());
}
```

Add `gdelt_baselines()` to `tests/tensa_bench/baselines/tkg_baselines.rs`.

### 4. Compile verification

None of the new code has been compiled yet (written during an ongoing test run). Expected issues:
- Possible trait bound / lifetime tweaks on the generic `run_tkg_link_prediction<L: DatasetLoader<Item = TemporalTriple>>` helper
- `serde_json::Value` import in any `metrics.to_json_value()` callsite
- `Participation` / `Entity` construction may need field adjustments if the shared types changed

First step: `cargo build --tests --no-default-features`

### 5. First real benchmark run

TKG benches are the cheapest first target — no LLM, deterministic, purely structural.
ICEWS14 at 12,498 entities × 341,409 test triples will produce the first publishable number.

```bash
export TENSA_BENCHMARK_DATA=d:/BeepDeep/tensa/data/benchmarks
cargo test --no-default-features --ignored bench_icews14_link_prediction -- --nocapture
```

Expected: TENSA's general-purpose link prediction (Adamic-Adar + common-neighbors + resource-allocation composite) will score below specialized TKG models (TComplEx MRR=0.56, TANGO MRR=0.58) on raw MRR. Honest baseline; qualitative advantages (interpretability, temporal interval algebra) covered in report narrative.

### 6. Paper integration

Once numbers exist, plug LaTeX tables from `report::to_latex_table(&report)` into:
- `PAPER/paper/sections/08-evaluation.md`
- `PAPER/paper_2/` if separate empirical paper

## Suggested order

1. `cargo build --tests --no-default-features` — surface compile errors
2. `cargo test --no-default-features tensa_bench` — unit tests only (no data, no LLM)
3. `bench_icews14_link_prediction` — first real number, no LLM cost
4. `bench_icews18_link_prediction`, `bench_gdelt_link_prediction` — cheap follow-ups
5. `bench_maven_ere_full` — needs only TENSA ingestion, no LLM for scoring
6. `bench_hotpotqa_full` — biggest credibility payoff, LLM-expensive
7. `bench_narrativeqa_full`, `bench_2wikimultihop_full`
8. ROCStories — once academic agreement is signed
9. LaTeX table export → paper evaluation section

## Cost estimates (rough)

| Benchmark | Deterministic? | LLM cost | Wall time |
|-----------|---------------|----------|-----------|
| ICEWS14/18, GDELT | Yes | $0 | minutes |
| MAVEN-ERE (valid, 710 docs) | Depends on ingestion LLM | $5-20 | hours |
| HotpotQA (500-item sample) | LLM-dependent | $15-30 | ~2 hours |
| 2WikiMultiHopQA (500-item sample) | LLM-dependent | $15-30 | ~2 hours |
| NarrativeQA (100-item sample) | LLM-dependent | $10-20 | ~1 hour |
| ROCStories (100-item sample) | LLM-dependent | $2-5 | ~30 min |

Set `TENSA_BENCH_SAMPLE=<N>` to cap item counts on any LLM-backed benchmark.

## Dataset inventory

All stored at `$TENSA_BENCHMARK_DATA/` (default `data/benchmarks/`):

| Dataset | Path | Size | Items |
|---------|------|------|-------|
| ICEWS14 | `icews14/` | ~20MB | 665K triples, 12,498 entities, 260 relations |
| ICEWS18 | `icews18/` | ~30MB | 468K triples, 23,033 entities, 256 relations |
| GDELT | `gdelt/` | ~200MB | 2.3M triples, 7,691 entities, 240 relations |
| HotpotQA | `hotpotqa/hotpot_dev_distractor_v1.json` | 45MB | 7,405 items |
| 2WikiMultiHop | `2wikimultihop/{train,dev,test}.json` | 756MB | dev+test+train |
| MAVEN-ERE | `maven_ere/{train,valid,test}.jsonl` | 131MB | 4,480 docs |
| NarrativeQA | `narrativeqa/{qaps,summaries,documents}.csv` | 22MB | 46,766 QA pairs |
| ROCStories | `rocstories/` | — | **Not downloaded** (academic agreement required) |
