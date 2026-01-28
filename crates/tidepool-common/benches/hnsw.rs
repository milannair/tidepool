use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tidepool_common::index::hnsw::HnswIndex;
use tidepool_common::index::recall::{exact_search, measure_recall};
use tidepool_common::vector::DistanceMetric;

fn make_vectors(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_hnsw_search(c: &mut Criterion) {
    let dims = 128;
    let vectors = make_vectors(10000, dims, 42);
    let queries = make_vectors(200, dims, 7);

    let mut index = HnswIndex::new(16, 200, 100, DistanceMetric::Cosine);
    for (i, vec) in vectors.iter().cloned().enumerate() {
        index.insert(i, vec);
    }

    c.bench_function("hnsw_search", |b| {
        b.iter_batched(
            || queries[0].clone(),
            |query| {
                let _ = index.search(&query, 10, 0);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_exact_vs_hnsw(c: &mut Criterion) {
    let dims = 128;
    let vectors = make_vectors(10000, dims, 123);
    let query = make_vectors(1, dims, 999).pop().unwrap();

    let mut index = HnswIndex::new(16, 200, 100, DistanceMetric::Cosine);
    for (i, vec) in vectors.iter().cloned().enumerate() {
        index.insert(i, vec);
    }

    c.bench_function("exact_search", |b| {
        b.iter(|| {
            let _ = exact_search(&query, &vectors, 10, DistanceMetric::Cosine);
        })
    });

    c.bench_function("hnsw_recall_measurement", |b| {
        b.iter(|| {
            let _ = measure_recall(&index, &vectors, &[query.clone()], 10);
        })
    });
}

criterion_group!(benches, bench_hnsw_search, bench_exact_vs_hnsw);
criterion_main!(benches);
