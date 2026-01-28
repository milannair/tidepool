//! SIMD-accelerated distance kernels
//!
//! Provides AVX2/FMA (x86_64) and NEON (ARM64) implementations for dot product
//! and L2 squared distance with automatic fallback to scalar implementations.
//!
//! # Performance
//! - AVX2 processes 8 f32 values per instruction (256-bit registers)
//! - NEON processes 4 f32 values per instruction (128-bit registers)
//! - FMA (Fused Multiply-Add) reduces latency for multiply-accumulate operations
//! - Loop unrolling processes 32 floats per iteration for better instruction pipelining
//!
//! # Usage
//! ```ignore
//! use tidepool_common::simd::{dot_f32, l2_squared_f32, cosine_distance_prenorm};
//!
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![4.0, 3.0, 2.0, 1.0];
//!
//! let dot = dot_f32(&a, &b);
//! let l2 = l2_squared_f32(&a, &b);
//! ```

use std::sync::atomic::{AtomicU8, Ordering};

/// Feature detection state
/// 0 = unknown, 1 = AVX2+FMA available (x86_64), 2 = NEON available (ARM64), 3 = scalar only
static SIMD_LEVEL: AtomicU8 = AtomicU8::new(0);

const SIMD_UNKNOWN: u8 = 0;
const SIMD_AVX2_FMA: u8 = 1;
const SIMD_NEON: u8 = 2;
#[allow(dead_code)]
const SIMD_SCALAR: u8 = 3;

/// Detect SIMD capabilities at runtime
#[inline]
fn detect_simd() -> u8 {
    let cached = SIMD_LEVEL.load(Ordering::Relaxed);
    if cached != SIMD_UNKNOWN {
        return cached;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        // Compiled with AVX2+FMA, check runtime support
        let level = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SIMD_AVX2_FMA
        } else {
            SIMD_SCALAR
        };
        SIMD_LEVEL.store(level, Ordering::Relaxed);
        return level;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let level = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SIMD_AVX2_FMA
        } else {
            SIMD_SCALAR
        };
        SIMD_LEVEL.store(level, Ordering::Relaxed);
        return level;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        SIMD_LEVEL.store(SIMD_NEON, Ordering::Relaxed);
        return SIMD_NEON;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SIMD_LEVEL.store(SIMD_SCALAR, Ordering::Relaxed);
        SIMD_SCALAR
    }
}

/// Check if AVX2+FMA is available (x86_64)
#[inline]
pub fn has_avx2_fma() -> bool {
    detect_simd() == SIMD_AVX2_FMA
}

/// Check if NEON is available (ARM64)
#[inline]
pub fn has_neon() -> bool {
    detect_simd() == SIMD_NEON
}

/// Check if any SIMD acceleration is available
#[inline]
pub fn has_simd() -> bool {
    let level = detect_simd();
    level == SIMD_AVX2_FMA || level == SIMD_NEON
}

// ============================================================================
// Public API - dispatches to SIMD or scalar based on runtime detection
// ============================================================================

/// Compute dot product of two f32 slices
///
/// Automatically uses AVX2+FMA or NEON when available, falls back to scalar otherwise.
/// Returns 0.0 if slices have different lengths or are empty.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let level = detect_simd();

    #[cfg(target_arch = "x86_64")]
    {
        if level == SIMD_AVX2_FMA {
            // Safety: we've verified AVX2+FMA support at runtime
            return unsafe { dot_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if level == SIMD_NEON {
            // Safety: NEON is always available on aarch64
            return unsafe { neon::dot_f32_neon_impl(a, b) };
        }
    }

    dot_f32_scalar(a, b)
}

/// Compute squared L2 (Euclidean) distance between two f32 slices
///
/// Automatically uses AVX2+FMA or NEON when available, falls back to scalar otherwise.
/// Returns f32::MAX if slices have different lengths.
#[inline]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    if a.is_empty() {
        return 0.0;
    }

    let level = detect_simd();

    #[cfg(target_arch = "x86_64")]
    {
        if level == SIMD_AVX2_FMA {
            // Safety: we've verified AVX2+FMA support at runtime
            return unsafe { l2_squared_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if level == SIMD_NEON {
            // Safety: NEON is always available on aarch64
            return unsafe { neon::l2_squared_f32_neon_impl(a, b) };
        }
    }

    l2_squared_f32_scalar(a, b)
}

/// Compute cosine distance for pre-normalized vectors
///
/// When vectors are already normalized (||a|| = ||b|| = 1), cosine distance
/// simplifies to: `1 - dot(a, b)`
///
/// This is significantly faster than computing norms on the fly.
#[inline]
pub fn cosine_distance_prenorm(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_f32(a, b)
}

/// Compute cosine distance with precomputed norms
///
/// Uses the formula: `1 - dot(a, b) / (norm_a * norm_b)`
/// This avoids recomputing norms when they're already available.
#[inline]
pub fn cosine_distance_with_norms(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0; // Maximum distance for zero vectors
    }
    let dot = dot_f32(a, b);
    1.0 - dot / (norm_a * norm_b)
}

/// Compute L2 norm of a vector
#[inline]
pub fn l2_norm_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }

    let level = detect_simd();

    #[cfg(target_arch = "x86_64")]
    {
        if level == SIMD_AVX2_FMA {
            // Safety: we've verified AVX2+FMA support at runtime
            return unsafe { l2_norm_f32_avx2(v) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if level == SIMD_NEON {
            // Safety: NEON is always available on aarch64
            return unsafe { neon::l2_norm_f32_neon_impl(v) };
        }
    }

    l2_norm_f32_scalar(v)
}

/// Normalize a vector in-place
#[inline]
pub fn normalize_f32(v: &mut [f32]) {
    let norm = l2_norm_f32(v);
    if norm == 0.0 {
        return;
    }
    let inv_norm = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

/// Normalize a vector, returning a new vector
#[inline]
pub fn normalized_f32(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm_f32(v);
    if norm == 0.0 {
        return v.to_vec();
    }
    let inv_norm = 1.0 / norm;
    v.iter().map(|x| x * inv_norm).collect()
}

// ============================================================================
// Scalar implementations (fallback)
// ============================================================================

/// Scalar dot product implementation
#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Use f64 accumulator for better numerical precision
    let mut sum = 0.0f64;
    
    // Unroll by 4 for better instruction-level parallelism
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let base = i * 4;
        sum += (a[base] as f64) * (b[base] as f64);
        sum += (a[base + 1] as f64) * (b[base + 1] as f64);
        sum += (a[base + 2] as f64) * (b[base + 2] as f64);
        sum += (a[base + 3] as f64) * (b[base + 3] as f64);
    }
    
    let base = chunks * 4;
    for i in 0..remainder {
        sum += (a[base + i] as f64) * (b[base + i] as f64);
    }
    
    sum as f32
}

/// Scalar L2 squared distance implementation
#[inline]
pub fn l2_squared_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    
    // Unroll by 4
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let base = i * 4;
        let d0 = (a[base] as f64) - (b[base] as f64);
        let d1 = (a[base + 1] as f64) - (b[base + 1] as f64);
        let d2 = (a[base + 2] as f64) - (b[base + 2] as f64);
        let d3 = (a[base + 3] as f64) - (b[base + 3] as f64);
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    
    let base = chunks * 4;
    for i in 0..remainder {
        let d = (a[base + i] as f64) - (b[base + i] as f64);
        sum += d * d;
    }
    
    sum as f32
}

/// Scalar L2 norm implementation
#[inline]
pub fn l2_norm_f32_scalar(v: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &x in v {
        let xf = x as f64;
        sum += xf * xf;
    }
    sum.sqrt() as f32
}

// ============================================================================
// AVX2 + FMA implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2+FMA dot product
    ///
    /// Processes 32 floats per iteration (4 × 8-wide AVX2 registers)
    /// Uses FMA for fused multiply-add operations
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_f32_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Initialize 4 accumulators for better pipelining
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 32 elements per iteration (4 × 8)
        let chunks = n / 32;
        for i in 0..chunks {
            let base = i * 32;

            // Load 32 elements from each array
            let a0 = _mm256_loadu_ps(a_ptr.add(base));
            let a1 = _mm256_loadu_ps(a_ptr.add(base + 8));
            let a2 = _mm256_loadu_ps(a_ptr.add(base + 16));
            let a3 = _mm256_loadu_ps(a_ptr.add(base + 24));

            let b0 = _mm256_loadu_ps(b_ptr.add(base));
            let b1 = _mm256_loadu_ps(b_ptr.add(base + 8));
            let b2 = _mm256_loadu_ps(b_ptr.add(base + 16));
            let b3 = _mm256_loadu_ps(b_ptr.add(base + 24));

            // FMA: acc = acc + a * b
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        }

        // Process remaining 8-element chunks
        let remainder_base = chunks * 32;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 8;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 8;
            let av = _mm256_loadu_ps(a_ptr.add(base));
            let bv = _mm256_loadu_ps(b_ptr.add(base));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
        }

        // Combine accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // Horizontal sum of 8 f32 values
        let sum = horizontal_sum_ps(acc0);

        // Handle final scalar elements
        let scalar_base = remainder_base + remaining_chunks * 8;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            scalar_sum += *a_ptr.add(i) * *b_ptr.add(i);
        }

        scalar_sum
    }

    /// AVX2+FMA squared L2 distance
    ///
    /// Computes sum((a[i] - b[i])^2)
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn l2_squared_f32_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 32 elements per iteration
        let chunks = n / 32;
        for i in 0..chunks {
            let base = i * 32;

            let a0 = _mm256_loadu_ps(a_ptr.add(base));
            let a1 = _mm256_loadu_ps(a_ptr.add(base + 8));
            let a2 = _mm256_loadu_ps(a_ptr.add(base + 16));
            let a3 = _mm256_loadu_ps(a_ptr.add(base + 24));

            let b0 = _mm256_loadu_ps(b_ptr.add(base));
            let b1 = _mm256_loadu_ps(b_ptr.add(base + 8));
            let b2 = _mm256_loadu_ps(b_ptr.add(base + 16));
            let b3 = _mm256_loadu_ps(b_ptr.add(base + 24));

            // Compute differences
            let d0 = _mm256_sub_ps(a0, b0);
            let d1 = _mm256_sub_ps(a1, b1);
            let d2 = _mm256_sub_ps(a2, b2);
            let d3 = _mm256_sub_ps(a3, b3);

            // FMA: acc = acc + d * d
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        }

        // Process remaining 8-element chunks
        let remainder_base = chunks * 32;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 8;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 8;
            let av = _mm256_loadu_ps(a_ptr.add(base));
            let bv = _mm256_loadu_ps(b_ptr.add(base));
            let d = _mm256_sub_ps(av, bv);
            acc0 = _mm256_fmadd_ps(d, d, acc0);
        }

        // Combine accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let sum = horizontal_sum_ps(acc0);

        // Handle final scalar elements
        let scalar_base = remainder_base + remaining_chunks * 8;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            let d = *a_ptr.add(i) - *b_ptr.add(i);
            scalar_sum += d * d;
        }

        scalar_sum
    }

    /// AVX2 L2 norm (magnitude)
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn l2_norm_f32_avx2_impl(v: &[f32]) -> f32 {
        let n = v.len();
        let v_ptr = v.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let chunks = n / 32;
        for i in 0..chunks {
            let base = i * 32;

            let v0 = _mm256_loadu_ps(v_ptr.add(base));
            let v1 = _mm256_loadu_ps(v_ptr.add(base + 8));
            let v2 = _mm256_loadu_ps(v_ptr.add(base + 16));
            let v3 = _mm256_loadu_ps(v_ptr.add(base + 24));

            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            acc2 = _mm256_fmadd_ps(v2, v2, acc2);
            acc3 = _mm256_fmadd_ps(v3, v3, acc3);
        }

        let remainder_base = chunks * 32;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 8;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 8;
            let vv = _mm256_loadu_ps(v_ptr.add(base));
            acc0 = _mm256_fmadd_ps(vv, vv, acc0);
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let sum = horizontal_sum_ps(acc0);

        let scalar_base = remainder_base + remaining_chunks * 8;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            let x = *v_ptr.add(i);
            scalar_sum += x * x;
        }

        scalar_sum.sqrt()
    }

    /// Horizontal sum of 8 f32 values in AVX2 register
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn horizontal_sum_ps(v: __m256) -> f32 {
        // Add high 128 bits to low 128 bits
        let high = _mm256_extractf128_ps(v, 1);
        let low = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(high, low);

        // Horizontal add within 128-bit register
        let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
        let sums = _mm_add_ps(sum128, shuf); // [0+1,1+1,2+3,3+3]
        let shuf2 = _mm_movehl_ps(sums, sums); // [2+3,3+3,2+3,3+3]
        let result = _mm_add_ss(sums, shuf2); // [0+1+2+3,...]

        _mm_cvtss_f32(result)
    }
}

// Wrapper functions that call into the avx2 module
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    avx2::dot_f32_avx2_impl(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn l2_squared_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    avx2::l2_squared_f32_avx2_impl(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn l2_norm_f32_avx2(v: &[f32]) -> f32 {
    avx2::l2_norm_f32_avx2_impl(v)
}

// ============================================================================
// NEON (ARM64) implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON dot product
    ///
    /// Processes 16 floats per iteration (4 × 4-wide NEON registers)
    ///
    /// # Safety
    /// Caller must ensure aarch64 target with NEON support
    #[inline]
    pub unsafe fn dot_f32_neon_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Initialize 4 accumulators for better pipelining
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        // Process 16 elements per iteration (4 × 4)
        let chunks = n / 16;
        for i in 0..chunks {
            let base = i * 16;

            // Load 16 elements from each array
            let a0 = vld1q_f32(a_ptr.add(base));
            let a1 = vld1q_f32(a_ptr.add(base + 4));
            let a2 = vld1q_f32(a_ptr.add(base + 8));
            let a3 = vld1q_f32(a_ptr.add(base + 12));

            let b0 = vld1q_f32(b_ptr.add(base));
            let b1 = vld1q_f32(b_ptr.add(base + 4));
            let b2 = vld1q_f32(b_ptr.add(base + 8));
            let b3 = vld1q_f32(b_ptr.add(base + 12));

            // FMA: acc = acc + a * b
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            acc2 = vfmaq_f32(acc2, a2, b2);
            acc3 = vfmaq_f32(acc3, a3, b3);
        }

        // Process remaining 4-element chunks
        let remainder_base = chunks * 16;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 4;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 4;
            let av = vld1q_f32(a_ptr.add(base));
            let bv = vld1q_f32(b_ptr.add(base));
            acc0 = vfmaq_f32(acc0, av, bv);
        }

        // Combine accumulators
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);

        // Horizontal sum of 4 f32 values
        let sum = vaddvq_f32(acc0);

        // Handle final scalar elements
        let scalar_base = remainder_base + remaining_chunks * 4;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            scalar_sum += *a_ptr.add(i) * *b_ptr.add(i);
        }

        scalar_sum
    }

    /// NEON squared L2 distance
    ///
    /// Computes sum((a[i] - b[i])^2)
    ///
    /// # Safety
    /// Caller must ensure aarch64 target with NEON support
    #[inline]
    pub unsafe fn l2_squared_f32_neon_impl(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        // Process 16 elements per iteration
        let chunks = n / 16;
        for i in 0..chunks {
            let base = i * 16;

            let a0 = vld1q_f32(a_ptr.add(base));
            let a1 = vld1q_f32(a_ptr.add(base + 4));
            let a2 = vld1q_f32(a_ptr.add(base + 8));
            let a3 = vld1q_f32(a_ptr.add(base + 12));

            let b0 = vld1q_f32(b_ptr.add(base));
            let b1 = vld1q_f32(b_ptr.add(base + 4));
            let b2 = vld1q_f32(b_ptr.add(base + 8));
            let b3 = vld1q_f32(b_ptr.add(base + 12));

            // Compute differences
            let d0 = vsubq_f32(a0, b0);
            let d1 = vsubq_f32(a1, b1);
            let d2 = vsubq_f32(a2, b2);
            let d3 = vsubq_f32(a3, b3);

            // FMA: acc = acc + d * d
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }

        // Process remaining 4-element chunks
        let remainder_base = chunks * 16;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 4;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 4;
            let av = vld1q_f32(a_ptr.add(base));
            let bv = vld1q_f32(b_ptr.add(base));
            let d = vsubq_f32(av, bv);
            acc0 = vfmaq_f32(acc0, d, d);
        }

        // Combine accumulators
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);

        let sum = vaddvq_f32(acc0);

        // Handle final scalar elements
        let scalar_base = remainder_base + remaining_chunks * 4;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            let d = *a_ptr.add(i) - *b_ptr.add(i);
            scalar_sum += d * d;
        }

        scalar_sum
    }

    /// NEON L2 norm (magnitude)
    ///
    /// # Safety
    /// Caller must ensure aarch64 target with NEON support
    #[inline]
    pub unsafe fn l2_norm_f32_neon_impl(v: &[f32]) -> f32 {
        let n = v.len();
        let v_ptr = v.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let chunks = n / 16;
        for i in 0..chunks {
            let base = i * 16;

            let v0 = vld1q_f32(v_ptr.add(base));
            let v1 = vld1q_f32(v_ptr.add(base + 4));
            let v2 = vld1q_f32(v_ptr.add(base + 8));
            let v3 = vld1q_f32(v_ptr.add(base + 12));

            acc0 = vfmaq_f32(acc0, v0, v0);
            acc1 = vfmaq_f32(acc1, v1, v1);
            acc2 = vfmaq_f32(acc2, v2, v2);
            acc3 = vfmaq_f32(acc3, v3, v3);
        }

        let remainder_base = chunks * 16;
        let remaining = n - remainder_base;
        let remaining_chunks = remaining / 4;

        for i in 0..remaining_chunks {
            let base = remainder_base + i * 4;
            let vv = vld1q_f32(v_ptr.add(base));
            acc0 = vfmaq_f32(acc0, vv, vv);
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);

        let sum = vaddvq_f32(acc0);

        let scalar_base = remainder_base + remaining_chunks * 4;
        let mut scalar_sum = sum;
        for i in scalar_base..n {
            let x = *v_ptr.add(i);
            scalar_sum += x * x;
        }

        scalar_sum.sqrt()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()) < EPSILON
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let expected = 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0;
        let result = dot_f32(&a, &b);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_dot_product_large() {
        // Test with 128 dimensions (common for embeddings)
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (127 - i) as f32 * 0.01).collect();
        
        let expected = dot_f32_scalar(&a, &b);
        let result = dot_f32(&a, &b);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_dot_product_1536() {
        // Test with 1536 dimensions (OpenAI embeddings)
        let a: Vec<f32> = (0..1536).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
        let b: Vec<f32> = (0..1536).map(|i| ((i % 100) as f32) * 0.01).collect();
        
        let expected = dot_f32_scalar(&a, &b);
        let result = dot_f32(&a, &b);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_l2_squared_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let expected = 3.0 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0; // 27.0
        let result = l2_squared_f32(&a, &b);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_l2_squared_large() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (127 - i) as f32 * 0.01).collect();
        
        let expected = l2_squared_f32_scalar(&a, &b);
        let result = l2_squared_f32(&a, &b);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let expected = 5.0;
        let result = l2_norm_f32(&v);
        assert!(approx_eq(result, expected), "got {} expected {}", result, expected);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize_f32(&mut v);
        assert!(approx_eq(v[0], 0.6), "got {} expected 0.6", v[0]);
        assert!(approx_eq(v[1], 0.8), "got {} expected 0.8", v[1]);
        
        // Verify norm is 1
        let norm = l2_norm_f32(&v);
        assert!(approx_eq(norm, 1.0), "normalized vector has norm {}", norm);
    }

    #[test]
    fn test_cosine_prenorm() {
        // Orthogonal normalized vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance_prenorm(&a, &b);
        assert!(approx_eq(dist, 1.0), "orthogonal vectors should have distance 1.0, got {}", dist);

        // Identical normalized vectors
        let c = normalized_f32(&[1.0, 1.0]);
        let dist2 = cosine_distance_prenorm(&c, &c);
        assert!(approx_eq(dist2, 0.0), "identical vectors should have distance 0.0, got {}", dist2);
    }

    #[test]
    fn test_cosine_with_norms() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let norm_a = l2_norm_f32(&a);
        let norm_b = l2_norm_f32(&b);
        
        let dist = cosine_distance_with_norms(&a, &b, norm_a, norm_b);
        
        // Compare with standard cosine distance
        let dot = dot_f32(&a, &b);
        let expected = 1.0 - dot / (norm_a * norm_b);
        assert!(approx_eq(dist, expected), "got {} expected {}", dist, expected);
    }

    #[test]
    fn test_empty_vectors() {
        let empty: Vec<f32> = vec![];
        assert_eq!(dot_f32(&empty, &empty), 0.0);
        assert_eq!(l2_squared_f32(&empty, &empty), 0.0);
        assert_eq!(l2_norm_f32(&empty), 0.0);
    }

    #[test]
    fn test_mismatched_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert_eq!(dot_f32(&a, &b), 0.0);
        assert_eq!(l2_squared_f32(&a, &b), f32::MAX);
    }

    #[test]
    fn test_odd_lengths() {
        // Test non-power-of-2 lengths to exercise remainder handling
        for len in [1, 3, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.1).collect();
            
            let dot_expected = dot_f32_scalar(&a, &b);
            let dot_result = dot_f32(&a, &b);
            assert!(approx_eq(dot_result, dot_expected), 
                "dot mismatch at len={}: got {} expected {}", len, dot_result, dot_expected);
            
            let l2_expected = l2_squared_f32_scalar(&a, &b);
            let l2_result = l2_squared_f32(&a, &b);
            assert!(approx_eq(l2_result, l2_expected),
                "l2 mismatch at len={}: got {} expected {}", len, l2_result, l2_expected);
        }
    }

    #[test]
    fn test_simd_detection() {
        // Just verify detection doesn't panic
        let has_simd = has_avx2_fma();
        println!("AVX2+FMA available: {}", has_simd);
    }
}
