use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tidepool_common::index::ivf::IVFIndex;
use tidepool_common::vector::DistanceMetric;

fn bench_ivf_build(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let dims = 128;
    let count = 10_000;
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let k = IVFIndex::compute_k(count, 1.0, 16, 4096);

    c.bench_function("ivf_build", |b| {
        b.iter(|| {
            let _ = IVFIndex::build(&refs, DistanceMetric::Cosine, k, 10, 20, 42).unwrap();
        })
    });
}

criterion_group!(benches, bench_ivf_build);
criterion_main!(benches);
