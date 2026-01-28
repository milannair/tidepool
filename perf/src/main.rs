use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use hdrhistogram::Histogram;
use rand::distributions::Uniform;
use rand::{rngs::StdRng, Rng, SeedableRng};
use reqwest::Client;
use serde::Serialize;
use std::collections::BTreeMap;
use tidepool_common::attributes::AttrValue;
use tidepool_common::document::{Document, QueryRequest, UpsertRequest};

const MAX_LATENCY_US: u64 = 120_000_000;

#[derive(Parser, Debug, Clone)]
#[command(name = "tidepool-perf", about = "Performance harness for Tidepool ingest and query services.")]
struct Args {
    #[arg(long, value_enum, default_value_t = Mode::Both)]
    mode: Mode,

    #[arg(long, default_value = "http://localhost:8081")]
    ingest_url: String,

    #[arg(long, default_value = "http://localhost:8080")]
    query_url: String,

    #[arg(long, default_value = "default")]
    namespace: String,

    #[arg(long, default_value_t = 1536)]
    dimensions: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "doc-")]
    id_prefix: String,

    #[arg(long, default_value_t = 100_000)]
    num_vectors: usize,

    #[arg(long, default_value_t = 200)]
    batch_size: usize,

    #[arg(long, default_value_t = 4)]
    ingest_concurrency: usize,

    #[arg(long, default_value_t = 0)]
    tag_cardinality: usize,

    #[arg(long, default_value_t = 10)]
    top_k: usize,

    #[arg(long, default_value_t = 100)]
    ef_search: usize,

    #[arg(long, default_value_t = 32)]
    query_concurrency: usize,

    #[arg(long, default_value_t = 30)]
    duration_secs: u64,

    #[arg(long, default_value_t = 0)]
    warmup_secs: u64,

    #[arg(long)]
    include_vectors: bool,

    #[arg(long)]
    query_use_tags: bool,

    #[arg(long)]
    filters: Option<String>,

    #[arg(long)]
    distance_metric: Option<String>,

    #[arg(long, default_value_t = 30)]
    http_timeout_secs: u64,

    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum Mode {
    Ingest,
    Query,
    Both,
}

#[derive(Clone)]
enum FilterMode {
    None,
    Fixed(AttrValue),
    Tags { cardinality: usize },
}

impl FilterMode {
    fn from_args(args: &Args) -> Result<Self> {
        if args.query_use_tags && args.filters.is_some() {
            bail!("--query-use-tags and --filters are mutually exclusive");
        }
        if args.query_use_tags {
            if args.tag_cardinality == 0 {
                bail!("--query-use-tags requires --tag-cardinality > 0");
            }
            return Ok(FilterMode::Tags {
                cardinality: args.tag_cardinality,
            });
        }
        if let Some(ref raw) = args.filters {
            let value: AttrValue = serde_json::from_str(raw)
                .with_context(|| format!("failed to parse --filters JSON: {raw}"))?;
            return Ok(FilterMode::Fixed(value));
        }
        Ok(FilterMode::None)
    }

    fn make_filter(&self, rng: &mut StdRng) -> Option<AttrValue> {
        match self {
            FilterMode::None => None,
            FilterMode::Fixed(value) => Some(value.clone()),
            FilterMode::Tags { cardinality } => {
                let idx = rng.gen_range(0..*cardinality);
                Some(tag_attr(&format!("tag-{idx}")))
            }
        }
    }
}

fn tag_attr(value: &str) -> AttrValue {
    let mut map = BTreeMap::new();
    map.insert("tag".to_string(), AttrValue::String(value.to_string()));
    AttrValue::Object(map)
}

#[derive(Serialize)]
struct Summary {
    ingest: Option<IngestSummary>,
    query: Option<QuerySummary>,
}

#[derive(Serialize)]
struct IngestSummary {
    total_vectors: u64,
    total_batches: u64,
    errors: u64,
    elapsed_secs: f64,
    docs_per_sec: f64,
    reqs_per_sec: f64,
    latency_us: Option<LatencySummary>,
}

#[derive(Serialize)]
struct QuerySummary {
    total_requests: u64,
    errors: u64,
    elapsed_secs: f64,
    reqs_per_sec: f64,
    latency_us: Option<LatencySummary>,
}

#[derive(Serialize)]
struct LatencySummary {
    p50: u64,
    p90: u64,
    p95: u64,
    p99: u64,
    max: u64,
}

struct WorkerStats {
    total: u64,
    errors: u64,
    histogram: Histogram<u64>,
}

impl WorkerStats {
    fn new() -> Result<Self> {
        Ok(Self {
            total: 0,
            errors: 0,
            histogram: Histogram::new_with_bounds(1, MAX_LATENCY_US, 3)?,
        })
    }

    fn record(&mut self, latency: Duration, ok: bool, record_stats: bool) {
        if !record_stats {
            return;
        }
        self.total += 1;
        if !ok {
            self.errors += 1;
        }
        let micros = latency.as_micros() as u64;
        let value = micros.clamp(1, MAX_LATENCY_US);
        let _ = self.histogram.record(value);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let client = Client::builder()
        .timeout(Duration::from_secs(args.http_timeout_secs))
        .build()
        .context("failed to build http client")?;

    let mut summary = Summary {
        ingest: None,
        query: None,
    };

    match args.mode {
        Mode::Ingest => {
            summary.ingest = Some(run_ingest(&args, &client).await?);
        }
        Mode::Query => {
            summary.query = Some(run_query(&args, &client).await?);
        }
        Mode::Both => {
            summary.ingest = Some(run_ingest(&args, &client).await?);
            summary.query = Some(run_query(&args, &client).await?);
        }
    }

    if let Some(path) = &args.output_json {
        let contents = serde_json::to_string_pretty(&summary)?;
        std::fs::write(path, contents)
            .with_context(|| format!("failed to write output json to {}", path.display()))?;
    }

    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.batch_size == 0 {
        bail!("--batch-size must be > 0");
    }
    if args.dimensions == 0 {
        bail!("--dimensions must be > 0");
    }
    if args.num_vectors == 0 && matches!(args.mode, Mode::Ingest | Mode::Both) {
        bail!("--num-vectors must be > 0 for ingest");
    }
    if args.ingest_concurrency == 0 {
        bail!("--ingest-concurrency must be > 0");
    }
    if args.query_concurrency == 0 {
        bail!("--query-concurrency must be > 0");
    }
    if args.duration_secs == 0 && matches!(args.mode, Mode::Query | Mode::Both) {
        bail!("--duration-secs must be > 0 for query");
    }
    if args.query_use_tags && args.tag_cardinality == 0 {
        bail!("--query-use-tags requires --tag-cardinality > 0");
    }
    Ok(())
}

async fn run_ingest(args: &Args, client: &Client) -> Result<IngestSummary> {
    let endpoint = format!(
        "{}/v1/vectors/{}",
        args.ingest_url.trim_end_matches('/'),
        args.namespace
    );

    let total_batches = (args.num_vectors + args.batch_size - 1) / args.batch_size;
    let next = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(args.ingest_concurrency);
    for worker_id in 0..args.ingest_concurrency {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let next = Arc::clone(&next);
        let args = args.clone();
        let handle = tokio::spawn(async move {
            let mut stats = WorkerStats::new()?;
            let mut rng = StdRng::seed_from_u64(args.seed ^ (worker_id as u64).wrapping_mul(31));
            let dist = Uniform::new(0.0f32, 1.0f32);

            loop {
                let start_idx = next.fetch_add(args.batch_size, Ordering::SeqCst);
                if start_idx >= args.num_vectors {
                    break;
                }
                let end_idx = (start_idx + args.batch_size).min(args.num_vectors);
                let mut vectors = Vec::with_capacity(end_idx - start_idx);

                for i in start_idx..end_idx {
                    let mut vector = Vec::with_capacity(args.dimensions);
                    for _ in 0..args.dimensions {
                        vector.push(rng.sample(dist));
                    }

                    let attributes = if args.tag_cardinality > 0 {
                        Some(tag_attr(&format!("tag-{}", i % args.tag_cardinality)))
                    } else {
                        None
                    };

                    vectors.push(Document {
                        id: format!("{}{}", args.id_prefix, i),
                        vector,
                        attributes,
                    });
                }

                let req = UpsertRequest {
                    vectors,
                    distance_metric: args.distance_metric.clone(),
                };

                let start = Instant::now();
                let resp = client.post(&endpoint).json(&req).send().await;
                let latency = start.elapsed();

                match resp {
                    Ok(resp) => {
                        let ok = resp.status().is_success();
                        let _ = resp.bytes().await;
                        stats.record(latency, ok, true);
                    }
                    Err(_) => {
                        stats.record(latency, false, true);
                    }
                }
            }

            Ok::<WorkerStats, anyhow::Error>(stats)
        });
        handles.push(handle);
    }

    let mut worker_stats = Vec::with_capacity(handles.len());
    for handle in handles {
        worker_stats.push(handle.await??);
    }
    let elapsed = start.elapsed();

    let (histogram, total_reqs, errors) = merge_stats(worker_stats)?;
    let latency_summary = summarize_latency(&histogram);
    let elapsed_secs = elapsed.as_secs_f64();

    let docs_per_sec = if elapsed_secs > 0.0 {
        args.num_vectors as f64 / elapsed_secs
    } else {
        0.0
    };
    let reqs_per_sec = if elapsed_secs > 0.0 {
        total_reqs as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("Ingest complete");
    println!("  vectors: {}", args.num_vectors);
    println!("  batches: {}", total_batches);
    println!("  errors: {}", errors);
    println!("  elapsed: {:.2}s", elapsed_secs);
    println!("  throughput: {:.2} docs/s, {:.2} req/s", docs_per_sec, reqs_per_sec);
    print_latency("  batch latency", &latency_summary);

    Ok(IngestSummary {
        total_vectors: args.num_vectors as u64,
        total_batches: total_batches as u64,
        errors,
        elapsed_secs,
        docs_per_sec,
        reqs_per_sec,
        latency_us: latency_summary,
    })
}

async fn run_query(args: &Args, client: &Client) -> Result<QuerySummary> {
    let endpoint = format!(
        "{}/v1/vectors/{}",
        args.query_url.trim_end_matches('/'),
        args.namespace
    );

    let filter_mode = FilterMode::from_args(args)?;
    let warmup = Duration::from_secs(args.warmup_secs);
    let duration = Duration::from_secs(args.duration_secs);

    let start = Instant::now();
    let warmup_end = start + warmup;
    let end = warmup_end + duration;

    let mut handles = Vec::with_capacity(args.query_concurrency);
    for worker_id in 0..args.query_concurrency {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let args = args.clone();
        let filter_mode = filter_mode.clone();

        let handle = tokio::spawn(async move {
            let mut stats = WorkerStats::new()?;
            let mut rng = StdRng::seed_from_u64(args.seed ^ (worker_id as u64).wrapping_mul(97));
            let dist = Uniform::new(0.0f32, 1.0f32);

            loop {
                let now = Instant::now();
                if now >= end {
                    break;
                }

                let mut vector = Vec::with_capacity(args.dimensions);
                for _ in 0..args.dimensions {
                    vector.push(rng.sample(dist));
                }

                let filters = filter_mode.make_filter(&mut rng);
                let req = QueryRequest {
                    vector,
                    top_k: args.top_k,
                    ef_search: args.ef_search,
                    nprobe: 0,
                    distance_metric: args.distance_metric.clone(),
                    include_vectors: args.include_vectors,
                    filters,
                };

                let start_req = Instant::now();
                let resp = client.post(&endpoint).json(&req).send().await;
                let latency = start_req.elapsed();

                let record_stats = Instant::now() >= warmup_end;
                match resp {
                    Ok(resp) => {
                        let ok = resp.status().is_success();
                        let _ = resp.bytes().await;
                        stats.record(latency, ok, record_stats);
                    }
                    Err(_) => {
                        stats.record(latency, false, record_stats);
                    }
                }
            }

            Ok::<WorkerStats, anyhow::Error>(stats)
        });
        handles.push(handle);
    }

    let mut worker_stats = Vec::with_capacity(handles.len());
    for handle in handles {
        worker_stats.push(handle.await??);
    }

    let elapsed = duration.as_secs_f64();
    let (histogram, total_reqs, errors) = merge_stats(worker_stats)?;
    let latency_summary = summarize_latency(&histogram);
    let reqs_per_sec = if elapsed > 0.0 {
        total_reqs as f64 / elapsed
    } else {
        0.0
    };

    println!("Query complete");
    println!("  requests: {}", total_reqs);
    println!("  errors: {}", errors);
    println!("  elapsed: {:.2}s", elapsed);
    println!("  throughput: {:.2} req/s", reqs_per_sec);
    print_latency("  query latency", &latency_summary);

    Ok(QuerySummary {
        total_requests: total_reqs,
        errors,
        elapsed_secs: elapsed,
        reqs_per_sec,
        latency_us: latency_summary,
    })
}

fn merge_stats(stats: Vec<WorkerStats>) -> Result<(Histogram<u64>, u64, u64)> {
    let mut merged = Histogram::new_with_bounds(1, MAX_LATENCY_US, 3)?;
    let mut total = 0;
    let mut errors = 0;
    for stat in stats {
        total += stat.total;
        errors += stat.errors;
        merged.add(&stat.histogram)?;
    }
    Ok((merged, total, errors))
}

fn summarize_latency(histogram: &Histogram<u64>) -> Option<LatencySummary> {
    if histogram.len() == 0 {
        return None;
    }
    Some(LatencySummary {
        p50: histogram.value_at_quantile(0.50),
        p90: histogram.value_at_quantile(0.90),
        p95: histogram.value_at_quantile(0.95),
        p99: histogram.value_at_quantile(0.99),
        max: histogram.max(),
    })
}

fn print_latency(label: &str, summary: &Option<LatencySummary>) {
    match summary {
        Some(summary) => {
            println!(
                "{}: p50={}us p90={}us p95={}us p99={}us max={}us",
                label, summary.p50, summary.p90, summary.p95, summary.p99, summary.max
            );
        }
        None => {
            println!("{}: no samples", label);
        }
    }
}
