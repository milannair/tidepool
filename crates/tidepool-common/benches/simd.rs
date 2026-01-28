//! Benchmark suite comparing scalar vs SIMD distance kernels
//!
//! Run with: cargo bench --bench simd
//!
//! Expected results (AVX2+FMA):
//! - 4-8x speedup for dot product and L2 distance
//! - >1B ops/sec on 128-dim vectors
//! - Linear scan 10k vectors in <5ms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tidepool_common::simd::{
    self, dot_f32, dot_f32_scalar, l2_norm_f32, l2_squared_f32, l2_squared_f32_scalar,
};
use tidepool_common::vector::{cosine_distance, euclidean_squared, magnitude};

/// Generate random vectors for benchmarking
fn make_vectors(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

/// Generate a single random vector
fn make_vector(dims: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Benchmark dot product: scalar vs SIMD
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dims in [64, 128, 256, 384, 512, 768, 1024, 1536].iter() {
        let a = make_vector(*dims, 42);
        let b = make_vector(*dims, 123);

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| dot_f32_scalar(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| dot_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark L2 squared distance: scalar vs SIMD
fn bench_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");

    for dims in [64, 128, 256, 384, 512, 768, 1024, 1536].iter() {
        let a = make_vector(*dims, 42);
        let b = make_vector(*dims, 123);

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("scalar", dims), dims, |bench, _| {
            bench.iter(|| l2_squared_f32_scalar(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| l2_squared_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark L2 norm computation
fn bench_l2_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_norm");

    for dims in [128, 256, 512, 1024, 1536].iter() {
        let v = make_vector(*dims, 42);

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("simd", dims), dims, |bench, _| {
            bench.iter(|| l2_norm_f32(black_box(&v)))
        });

        // Compare with old magnitude implementation
        group.bench_with_input(BenchmarkId::new("magnitude", dims), dims, |bench, _| {
            bench.iter(|| magnitude(black_box(&v)))
        });
    }

    group.finish();
}

/// Benchmark cosine distance (full computation)
fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dims in [128, 256, 512, 768, 1024, 1536].iter() {
        let a = make_vector(*dims, 42);
        let b = make_vector(*dims, 123);

        group.throughput(Throughput::Elements(*dims as u64));

        group.bench_with_input(BenchmarkId::new("full", dims), dims, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });

        // Pre-normalized vectors for fast path
        let a_norm = simd::normalized_f32(&a);
        let b_norm = simd::normalized_f32(&b);

        group.bench_with_input(BenchmarkId::new("prenorm", dims), dims, |bench, _| {
            bench.iter(|| simd::cosine_distance_prenorm(black_box(&a_norm), black_box(&b_norm)))
        });
    }

    group.finish();
}

/// Benchmark distance computation throughput over N vectors
fn bench_distance_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_throughput");
    group.sample_size(50); // Reduce samples for slower benchmarks

    let dims = 128;
    let query = make_vector(dims, 999);
    let query_norm = simd::l2_norm_f32(&query);

    for count in [1000, 5000, 10000, 20000].iter() {
        let vectors = make_vectors(*count, dims, 42);

        // Precompute norms
        let norms: Vec<f32> = vectors.iter().map(|v| simd::l2_norm_f32(v)).collect();

        group.throughput(Throughput::Elements(*count as u64));

        // Cosine distance with precomputed norms (typical query path)
        group.bench_with_input(
            BenchmarkId::new("cosine_prenorm", count),
            count,
            |bench, _| {
                bench.iter(|| {
                    let mut min_dist = f32::MAX;
                    for (i, vec) in vectors.iter().enumerate() {
                        let dist = simd::cosine_distance_with_norms(
                            black_box(&query),
                            black_box(vec),
                            query_norm,
                            norms[i],
                        );
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                    min_dist
                })
            },
        );

        // Euclidean distance (simpler, faster)
        group.bench_with_input(BenchmarkId::new("euclidean", count), count, |bench, _| {
            bench.iter(|| {
                let mut min_dist = f32::MAX;
                for vec in vectors.iter() {
                    let dist = euclidean_squared(black_box(&query), black_box(vec));
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                min_dist
            })
        });

        // Dot product
        group.bench_with_input(BenchmarkId::new("dot_product", count), count, |bench, _| {
            bench.iter(|| {
                let mut max_dot = f32::MIN;
                for vec in vectors.iter() {
                    let dot = dot_f32(black_box(&query), black_box(vec));
                    if dot > max_dot {
                        max_dot = dot;
                    }
                }
                max_dot
            })
        });
    }

    group.finish();
}

/// Benchmark top-k selection with distance computation
fn bench_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk");
    group.sample_size(30);

    let dims = 128;
    let count = 10000;
    let query = make_vector(dims, 999);
    let vectors = make_vectors(count, dims, 42);
    let norms: Vec<f32> = vectors.iter().map(|v| simd::l2_norm_f32(v)).collect();
    let query_norm = simd::l2_norm_f32(&query);

    for k in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("heap_cosine", k), k, |bench, &k| {
            bench.iter(|| {
                let mut heap: std::collections::BinaryHeap<ordered_float::OrderedFloat<f32>> =
                    std::collections::BinaryHeap::with_capacity(k);

                for (i, vec) in vectors.iter().enumerate() {
                    let dist = simd::cosine_distance_with_norms(&query, vec, query_norm, norms[i]);

                    if heap.len() < k {
                        heap.push(ordered_float::OrderedFloat(dist));
                    } else if let Some(&top) = heap.peek() {
                        if dist < top.0 {
                            heap.pop();
                            heap.push(ordered_float::OrderedFloat(dist));
                        }
                    }
                }

                heap.into_sorted_vec()
            })
        });
    }

    group.finish();
}

/// Benchmark throughput in ops/sec for 128-dim vectors
fn bench_throughput_128d(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_128d");

    let dims = 128;
    let iterations = 10000;

    // Create many vector pairs
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..iterations)
        .map(|i| {
            (
                make_vector(dims, i as u64),
                make_vector(dims, (i + 1000) as u64),
            )
        })
        .collect();

    group.throughput(Throughput::Elements(iterations as u64));

    group.bench_function("dot_product", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for (a, b) in pairs.iter() {
                sum += dot_f32(black_box(a), black_box(b));
            }
            sum
        })
    });

    group.bench_function("l2_squared", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for (a, b) in pairs.iter() {
                sum += l2_squared_f32(black_box(a), black_box(b));
            }
            sum
        })
    });

    group.finish();
}

/// Report SIMD availability
fn bench_simd_info(c: &mut Criterion) {
    let has_simd = simd::has_avx2_fma();
    println!("\n=== SIMD Info ===");
    println!("AVX2+FMA available: {}", has_simd);

    // Quick sanity check
    let a = vec![1.0; 128];
    let b = vec![1.0; 128];
    let dot = dot_f32(&a, &b);
    println!("Sanity check dot([1;128], [1;128]) = {} (expected 128)", dot);
    println!("==================\n");

    // Minimal benchmark to keep criterion happy
    c.bench_function("simd_available", |bench| {
        bench.iter(|| simd::has_avx2_fma())
    });
}

criterion_group!(
    benches,
    bench_simd_info,
    bench_dot_product,
    bench_l2_squared,
    bench_l2_norm,
    bench_cosine_distance,
    bench_distance_throughput,
    bench_topk,
    bench_throughput_128d,
);
criterion_main!(benches);
