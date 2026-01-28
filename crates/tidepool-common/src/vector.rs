//! Vector distance functions with SIMD acceleration
//!
//! This module provides distance metrics for vector similarity search.
//! All functions automatically use SIMD (AVX2+FMA) when available,
//! with transparent fallback to scalar implementations.
//!
//! # Metrics
//! - **Cosine**: `1 - (a·b)/(||a||*||b||)` — measures angle between vectors
//! - **Euclidean**: `||a - b||²` — squared L2 distance
//! - **DotProduct**: `-a·b` — negated for use as distance (higher dot = lower distance)
//!
//! # Performance
//! SIMD provides 4-8x speedup for distance computations on AVX2-capable hardware.

use crate::simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    pub fn parse(metric: Option<&str>) -> Self {
        match metric.unwrap_or("") {
            "cosine" | "cosine_distance" => Self::Cosine,
            "euclidean" | "euclidean_squared" => Self::Euclidean,
            "dot" | "dot_product" => Self::DotProduct,
            _ => Self::Cosine,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine_distance",
            Self::Euclidean => "euclidean_squared",
            Self::DotProduct => "dot_product",
        }
    }
}

/// Compute distance between two vectors using the specified metric
///
/// Uses SIMD-accelerated implementations when available.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_squared(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b),
    }
}

/// Compute distance using precomputed norms (optimized for cosine)
///
/// For cosine distance, avoids recomputing norms.
/// For other metrics, norms are ignored.
#[inline]
pub fn distance_with_norms(
    a: &[f32],
    b: &[f32],
    norm_a: f32,
    norm_b: f32,
    metric: DistanceMetric,
) -> f32 {
    match metric {
        DistanceMetric::Cosine => simd::cosine_distance_with_norms(a, b, norm_a, norm_b),
        DistanceMetric::Euclidean => euclidean_squared(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b),
    }
}

/// Cosine distance: 1 - cos(θ) where θ is angle between vectors
///
/// Returns a value in [0, 2]:
/// - 0: identical direction
/// - 1: orthogonal
/// - 2: opposite direction
///
/// Uses SIMD-accelerated dot product and norm computations.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 2.0;
    }

    let norm_a = simd::l2_norm_f32(a);
    let norm_b = simd::l2_norm_f32(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0;
    }

    let dot = simd::dot_f32(a, b);
    let similarity = dot / (norm_a * norm_b);
    
    // Clamp to handle floating point errors
    1.0 - similarity.clamp(-1.0, 1.0)
}

/// Cosine distance for pre-normalized vectors
///
/// When ||a|| = ||b|| = 1, simplifies to: 1 - a·b
/// Much faster than full cosine distance.
#[inline]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    simd::cosine_distance_prenorm(a, b)
}

/// Squared Euclidean distance: ||a - b||²
///
/// Note: Returns squared distance to avoid sqrt computation.
/// For ranking purposes, squared distance preserves ordering.
#[inline]
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    simd::l2_squared_f32(a, b)
}

/// Dot product: a·b = Σ(aᵢ × bᵢ)
///
/// Higher values indicate more similar vectors.
/// When used as distance, negate: distance = -dot(a, b)
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MIN;
    }
    simd::dot_f32(a, b)
}

/// Normalize a vector to unit length
///
/// Returns a new vector with ||v|| = 1
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    simd::normalized_f32(v)
}

/// Compute magnitude (L2 norm) of a vector
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    simd::l2_norm_f32(v)
}

/// Check if SIMD acceleration is available
#[inline]
pub fn simd_available() -> bool {
    simd::has_avx2_fma()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_cosine_distance() {
        // Orthogonal vectors -> distance = 1
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!(approx_eq(dist, 1.0), "orthogonal: got {}", dist);

        // Identical vectors -> distance = 0
        let c = vec![1.0, 1.0];
        let dist2 = cosine_distance(&c, &c);
        assert!(approx_eq(dist2, 0.0), "identical: got {}", dist2);

        // Opposite vectors -> distance = 2
        let d = vec![1.0, 0.0];
        let e = vec![-1.0, 0.0];
        let dist3 = cosine_distance(&d, &e);
        assert!(approx_eq(dist3, 2.0), "opposite: got {}", dist3);
    }

    #[test]
    fn test_euclidean_squared() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_squared(&a, &b);
        assert!(approx_eq(dist, 25.0), "got {}", dist);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!(approx_eq(dot, 32.0), "got {}", dot);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!(approx_eq(n[0], 0.6), "got {}", n[0]);
        assert!(approx_eq(n[1], 0.8), "got {}", n[1]);
    }

    #[test]
    fn test_magnitude() {
        let v = vec![3.0, 4.0];
        let m = magnitude(&v);
        assert!(approx_eq(m, 5.0), "got {}", m);
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        
        let cos_dist = distance(&a, &b, DistanceMetric::Cosine);
        assert!(approx_eq(cos_dist, 1.0), "cosine: got {}", cos_dist);
        
        let euc_dist = distance(&a, &b, DistanceMetric::Euclidean);
        assert!(approx_eq(euc_dist, 2.0), "euclidean: got {}", euc_dist);
    }

    #[test]
    fn test_cosine_normalized() {
        let a = normalize(&[1.0, 1.0]);
        let b = normalize(&[1.0, 0.0]);
        
        let dist = cosine_distance_normalized(&a, &b);
        let expected = cosine_distance(&a, &b);
        
        assert!((dist - expected).abs() < 0.01, "got {} expected {}", dist, expected);
    }
}
