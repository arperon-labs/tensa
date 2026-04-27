//! TKG (Temporal Knowledge Graph) benchmark suite.
//!
//! Evaluates TENSA's link prediction on ICEWS14/18 and GDELT against
//! TComplEx, RE-NET, TANGO, and TimeTraveler.
//!
//! Run: TENSA_BENCHMARK_DATA=/path cargo test --no-default-features --ignored bench_tkg

mod tensa_bench;

use tensa_bench::baselines::tkg_baselines;
use tensa_bench::datasets::icews::{Icews14, Icews18};
use tensa_bench::datasets::{self, DatasetLoader, Split};
use tensa_bench::metrics::ranking::RankingMetrics;
use tensa_bench::{benchmark_data_dir, timed, DatasetReport, DomainReport};

// ─── Unit Tests (no data required) ───

#[test]
fn test_ranking_metrics_unit() {
    // Perfect ranks
    let m = RankingMetrics::from_ranks(&[1, 1, 1]);
    assert!((m.mrr - 1.0).abs() < 1e-9);
    assert!((m.hits_at_1 - 1.0).abs() < 1e-9);

    // Mixed ranks
    let m = RankingMetrics::from_ranks(&[1, 5, 10, 50]);
    assert!(m.mrr > 0.0 && m.mrr < 1.0);
    assert!((m.hits_at_1 - 0.25).abs() < 1e-9); // only rank 1
    assert!((m.hits_at_10 - 0.75).abs() < 1e-9); // ranks 1, 5, 10

    // Empty
    let m = RankingMetrics::from_ranks(&[]);
    assert!((m.mrr).abs() < 1e-9);
}

#[test]
fn test_icews_triple_format() {
    // Verify the parser handles the expected format
    let sample = "42\t7\t100\t2014-01-15";
    let parts: Vec<&str> = sample.split('\t').collect();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0].parse::<u32>().unwrap(), 42);
    assert_eq!(parts[1].parse::<u32>().unwrap(), 7);
    assert_eq!(parts[2].parse::<u32>().unwrap(), 100);
    assert!(chrono::NaiveDate::parse_from_str(parts[3], "%Y-%m-%d").is_ok());
}

#[test]
fn test_baseline_constants() {
    let baselines = tkg_baselines::icews14_baselines();
    assert!(baselines.len() >= 4, "Expected at least 4 baselines");
    for (name, metrics) in &baselines {
        assert!(
            metrics.mrr > 0.0 && metrics.mrr < 1.0,
            "{}: invalid MRR",
            name
        );
        assert!(
            metrics.hits_at_1 <= metrics.hits_at_3,
            "{}: H@1 should <= H@3",
            name
        );
        assert!(
            metrics.hits_at_3 <= metrics.hits_at_10,
            "{}: H@3 should <= H@10",
            name
        );
    }
}

// ─── Full Benchmarks (require TENSA_BENCHMARK_DATA) ───

/// Run link prediction for any ICEWS-style dataset. `L` is a DatasetLoader whose
/// items are TemporalTriples (ICEWS14, ICEWS18, or GDELT).
fn run_tkg_link_prediction<L>(
    display_name: &str,
    download_key: &str,
    baselines: Vec<(&'static str, tensa_bench::metrics::ranking::RankingMetrics)>,
) where
    L: tensa_bench::datasets::DatasetLoader<Item = tensa_bench::datasets::icews::TemporalTriple>,
{
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions(download_key));
            return;
        }
    };
    if !L::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions(download_key));
        return;
    }

    eprintln!("=== {} Link Prediction ===", display_name);

    let (train, load_ms) = timed(|| L::load(&data_dir, Split::Train));
    let train = train.unwrap_or_else(|e| panic!("Failed to load {} train: {}", display_name, e));
    eprintln!("Loaded {} train triples in {}ms", train.len(), load_ms);

    let test = L::load(&data_dir, Split::Test)
        .unwrap_or_else(|e| panic!("Failed to load {} test: {}", display_name, e));
    eprintln!("Loaded {} test triples", test.len());

    let dataset_dir = data_dir.join(L::dir_name());
    let dataset = tensa_bench::datasets::icews::load_icews_full(&dataset_dir)
        .unwrap_or_else(|e| panic!("Failed to load {} full: {}", display_name, e));
    eprintln!(
        "Dataset: {} entities, {} relations",
        dataset.num_entities(),
        dataset.num_relations()
    );

    let ((hg, mapping), ingest_ms) = timed(|| {
        tensa_bench::adapters::tkg_adapter::ingest_tkg_triples(
            &dataset,
            &train,
            &display_name.to_lowercase(),
        )
        .expect("Failed to ingest")
    });
    eprintln!("Ingested {} triples in {}ms", train.len(), ingest_ms);

    let (acc, eval_ms) = timed(|| {
        tensa_bench::adapters::tkg_adapter::evaluate_link_prediction(&hg, &mapping, &test)
            .expect("Failed to evaluate")
    });
    let metrics = acc.finalize();
    eprintln!("Evaluated {} test triples in {}ms", test.len(), eval_ms);

    let comparisons = tkg_baselines::build_tkg_comparisons(&metrics, &baselines);
    let report = DatasetReport {
        name: display_name.to_string(),
        task: "temporal_link_prediction".to_string(),
        metrics: metrics.to_json_value(),
        baselines: comparisons,
        num_items: test.len(),
        duration_ms: ingest_ms + eval_ms,
    };
    report.print_markdown();

    let domain = DomainReport {
        domain: "tkg".to_string(),
        datasets: vec![report],
    };
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&domain).unwrap_or_default()
    );
}

#[test]
#[ignore]
fn bench_icews14_link_prediction() {
    run_tkg_link_prediction::<Icews14>("ICEWS14", "icews14", tkg_baselines::icews14_baselines());
}

#[test]
#[ignore]
fn bench_icews18_link_prediction() {
    run_tkg_link_prediction::<Icews18>("ICEWS18", "icews18", tkg_baselines::icews18_baselines());
}
