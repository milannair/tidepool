use std::collections::HashSet;

use crate::index::hnsw::HnswIndex;
use crate::vector::{distance, DistanceMetric};

pub fn measure_recall(hnsw: &HnswIndex, vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f32 {
    if queries.is_empty() || k == 0 {
        return 0.0;
    }

    let mut correct = 0usize;
    let mut total = 0usize;

    for query in queries {
        let hnsw_results = hnsw.search(query, k, 0);
        let brute_results = brute_force(query, vectors, k, hnsw.metric);

        let hnsw_set: HashSet<usize> = hnsw_results.iter().map(|r| r.id).collect();
        for r in brute_results {
            if hnsw_set.contains(&r.0) {
                correct += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

pub fn brute_force(
    query: &[f32],
    vectors: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, vec)| (i, distance(query, vec, metric)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(k);
    results
}
