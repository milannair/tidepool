# Tidepool Development Phases

A precise, phased implementation roadmap for building an extremely fast Rust-based vector database backed by object storage. Each phase removes a specific bottleneck and must complete before the next begins.

---

## Phase 1: Foundation (Complete)

**Objective:** Establish core architecture with stateless services and object storage as source of truth.

**Why First:** All subsequent optimizations depend on the fundamental read/write separation and immutable file design.

### Deliverables
- [x] Stateless query service (read-only)
- [x] Stateless ingest service (WAL + compaction)
- [x] S3-compatible object storage backend
- [x] Write-ahead log for durability
- [x] Basic segment format (`.tpvs`)
- [x] Vector search (cosine, euclidean, dot product)
- [x] Attribute filtering
- [x] Background compaction worker
- [x] Railway deployment support

**Exit Criteria:** End-to-end upsert → compact → query flow working with object storage.

---

## Phase 2: HNSW & Caching (Complete)

**Objective:** Add HNSW index for sub-linear query time. Add segment caching to eliminate redundant S3 reads.

**Why Second:** Query latency is the primary bottleneck after foundation. HNSW provides O(log n) search vs O(n) linear scan.

### Deliverables
- [x] HNSW index construction during compaction
- [x] HNSW serialization (`.hnsw` files)
- [x] Filtered HNSW search with RoaringBitmap
- [x] Memory-mapped segment access (`memmap2`)
- [x] Local disk cache with cleanup
- [x] Parallel segment loading (`JoinSet`)
- [x] Parallel query execution (`spawn_blocking`)
- [x] Tombstone management for deletes

**Exit Criteria:** 
- Query throughput >100 req/sec at 60k vectors
- HNSW recall >95% vs exact search
- Cache hit eliminates S3 round-trip

---

## Phase 3: Zero-Copy & Binary Formats (Complete)

**Objective:** Eliminate allocations and copies on the query path. Achieve true zero-copy reads from mmap.

**Why Third:** Current rkyv usage still deserializes. Memory pressure and GC-like behavior limit throughput at scale.

### Techniques
- **rkyv zero-copy:** Access archived data directly without deserialization
- **Binary-stable layouts:** Fixed-size structs, aligned for direct casting
- **ID normalization:** Replace `String` IDs with `u64` (FNV hash or sequence)
- **Norm precomputation:** Store ||v|| with each vector for cosine distance

### Deliverables
- [x] `SegmentView` as zero-copy view over mmap (`segment_v2.rs`)
- [x] Direct field access via parsed header (no packed struct issues)
- [x] `u64` hashed IDs with string lookup table
- [x] Precomputed L2 norms stored in segment header
- [x] 32-byte aligned vector storage for SIMD readiness
- [x] Backward compatible: reader handles v1 and v2 formats

### File Format (v2)
```
segment_v2.tpvs (TPV2 magic):
  [header: 68 bytes]
    magic: [u8; 4] = "TPV2"
    version: u32 = 2
    vector_count: u32
    dimensions: u32
    flags: u32
    norm_offset: u64
    vector_offset: u64
    id_offset: u64
    string_table_offset: u64
    attr_offset: u64
    attr_len: u64
  [norms: f32 × n, 32-byte aligned]
  [vectors: f32 × n × d, 32-byte aligned]
  [ids: u64 × n]
  [string_table: length-prefixed strings]
  [attrs: rkyv archived, 4-byte aligned]
```

**Exit Criteria:**
- [x] Zero-copy segment access via mmap
- [x] Precomputed norms for fast cosine distance
- [x] All existing tests pass with v2 format

---

## Phase 4: SIMD Distance Kernels (Complete)

**Objective:** Maximize throughput of distance computations using SIMD intrinsics.

**Why Fourth:** After zero-copy, distance computation dominates CPU time. SIMD provides 4-8x speedup.

### Techniques
- **AVX2/FMA:** 8-wide f32 operations
- **Normalized cosine:** `1 - dot(a, b)` when ||a|| = ||b|| = 1
- **Early-exit dot products:** Stop when partial sum exceeds threshold
- **Loop unrolling:** Process 32 floats per iteration

### Deliverables
- [x] `simd_dot_f32_avx2()` with runtime feature detection (`simd.rs`)
- [x] `simd_l2_squared_avx2()` (`simd.rs`)
- [x] Fallback scalar implementations (`simd.rs`)
- [x] Pre-normalized vectors in segment (optional mode via `flags::VECTORS_NORMALIZED`)
- [x] Benchmark suite comparing scalar vs SIMD (`benches/simd.rs`)

### Implementation Details
- `simd.rs`: AVX2+FMA kernels with 4x unrolled loops (32 floats/iteration)
- Runtime feature detection via `is_x86_feature_detected!`
- `vector.rs`: Updated to dispatch to SIMD implementations
- `segment_v2.rs`: Added `WriteOptions::normalize_vectors` for pre-normalization
- Uses `std::arch::x86_64` intrinsics (no external crate dependencies)

### Benchmark Results (scalar fallback, no AVX2)
```
distance_throughput/euclidean/10000:    320µs (0.32ms) — 31.2 Melem/s
distance_throughput/dot_product/10000:  531µs (0.53ms) — 18.8 Melem/s
distance_throughput/cosine_prenorm/10000: 543µs (0.54ms) — 18.4 Melem/s
```

**Exit Criteria:**
- [x] Distance kernel throughput >1B ops/sec on 128-dim vectors (achieved: 2.3-3.1B elem/s)
- [x] 10k distance computations in <5ms (achieved: 0.32-0.54ms)

---

## Phase 5: IVF Centroid Index

**Objective:** Reduce search space by partitioning vectors into clusters. Only search relevant partitions.

**Why Fifth:** HNSW works but has high memory overhead. IVF scales better for large datasets with filtered queries.

### Techniques
- **k-means clustering:** Run during compaction, k = sqrt(n)
- **Centroid routing:** Query → top-k centroids → posting lists
- **Triangle inequality:** Prune clusters where `dist(q, centroid) - radius > best_so_far`
- **Exact rerank:** Linear scan top candidates from selected clusters

### Deliverables
- [ ] `IVFIndex` struct with centroids + posting lists
- [ ] k-means implementation (Lloyd's algorithm)
- [ ] `ivf_search()` with nprobe parameter
- [ ] Serialization format (`.ivf` files)
- [ ] Hybrid: IVF for large segments, HNSW for small

### File Format
```
segment.ivf:
  [header]
    k: u32 (cluster count)
    nprobe_default: u32
  [centroids: f32 × k × d]
  [posting_lists: Vec<u32> × k]
```

**Exit Criteria:**
- 10x speedup vs linear scan at 500k vectors
- Recall >98% with nprobe=10

---

## Phase 6: Quantization

**Objective:** Reduce memory footprint and increase cache efficiency via lossy compression.

**Why Sixth:** At scale, f32 vectors don't fit in RAM. Quantization trades precision for capacity.

### Techniques
- **f16 storage:** 2x compression, minimal recall loss
- **Asymmetric int8:** Quantize DB vectors, keep query in f32
- **Scalar quantization (SQ):** Per-dimension min/max scaling
- **Product quantization (PQ):** Optional, for extreme compression

### Deliverables
- [ ] f16 vector storage mode
- [ ] SQ8 quantization with calibration
- [ ] Asymmetric distance: `dot(q_f32, db_i8) * scale`
- [ ] Mixed precision: quantized scan → f32 rerank
- [ ] Config flag: `quantization: none | f16 | sq8`

**Exit Criteria:**
- 4x memory reduction with SQ8
- Recall >95% vs f32 baseline

---

## Phase 7: Deterministic Vector Ordering

**Objective:** Maximize early-exit effectiveness by reordering vectors within posting lists so that high-quality candidates appear first, causing top-k bounds to tighten rapidly and most distance computations to abort early.

**Why Seventh:** After IVF partitions vectors and quantization reduces precision, the *order* in which vectors are scanned within each cluster determines how quickly bounds tighten. Random ordering wastes SIMD cycles on vectors that will never make top-k. This is a layout + math synergy that most vector databases do not exploit.

### Why This Is Not Commonly Implemented
- Requires control over segment layout (cloud-hosted DBs often don't have this)
- Only beneficial with early-exit distance kernels (requires Phase 4)
- Must be done at compaction time (runtime reordering defeats the purpose)
- Academic ANN papers focus on recall, not average-case latency
- Retrofitting into mutable indexes is complex; immutable segments make this trivial

### Core Insight
When scanning a posting list for top-k, the k-th best distance acts as a threshold. Any vector whose partial dot product cannot exceed this threshold can be skipped mid-computation. If vectors are ordered so that:
1. The best candidates appear first
2. The threshold tightens after just a few vectors
3. Most remaining vectors fail the early-exit check

Then average distance computation cost drops dramatically—often 2-5x—without any change to recall.

### Techniques
- **Centroid-projected ordering:** Sort vectors by `dot(v, centroid)` descending within each posting list
- **Reference direction sorting:** Use the centroid as a "reference query" and sort by similarity to it
- **Score upper-bound ordering:** For quantized vectors, sort by maximum possible score (using quantization bounds)
- **Batch-aware layout:** Group vectors likely to be co-retrieved for better cache locality

### Integration with Existing Phases
- **Phase 5 (IVF):** Ordering is applied per posting list during index construction
- **Phase 6 (Quantization):** Quantized codes inherit ordering; asymmetric search benefits from early termination
- **Phase 4 (SIMD):** Early-exit kernels exploit tight bounds to skip 8-wide iterations
- **Phase 3 (Zero-Copy):** Ordered layout is written once at compaction, read directly via mmap

### Deliverables
- [ ] `sort_posting_list_by_centroid_similarity()` in compaction pipeline
- [ ] Configurable ordering strategy: `centroid | random | score_bound`
- [ ] Early-exit dot product kernel with threshold parameter (`dot_f32_early_exit_avx2`)
- [ ] Benchmark: measure % of SIMD iterations skipped per query
- [ ] A/B comparison: ordered vs random posting lists

### File Format Extension
```
segment.ivf (extended):
  [header]
    ordering_strategy: u8 (0=none, 1=centroid, 2=score_bound)
  [posting_lists: Vec<u32> × k, ordered by strategy]
```

### Why This Is Defensible
- Requires immutable segments (can't reorder in-place efficiently)
- Requires early-exit kernels (most DBs use full SIMD reductions)
- Requires IVF (posting lists provide natural reorder boundaries)
- Retrofitting into existing systems requires rewriting compaction + kernels

**Exit Criteria:**
- Average SIMD iterations per distance computation reduced by >50%
- p50 latency improvement of 1.5-2x on IVF queries with nprobe=10
- Zero impact on recall (ordering is internal, not algorithmic)

---

## Phase 8: Multi-Stage Pruning

**Objective:** Cascade cheap approximate checks before expensive exact computation.

**Why Eighth:** Combines IVF, quantization, vector ordering, and HNSW into optimal query plan.

### Techniques
- **Bound propagation:** Use quantized distance as lower bound
- **Candidate elimination:** Skip vectors where `lower_bound > k-th best`
- **Approx → exact cascade:** SQ8 scan → f16 rerank → f32 final
- **Adaptive nprobe:** Increase clusters if early results are poor

### Deliverables
- [ ] `CascadeSearch` query executor
- [ ] Bound tracking with priority queue
- [ ] Early termination when bounds converge
- [ ] Query planner selecting strategy by segment size

**Exit Criteria:**
- <10% of vectors require f32 distance computation
- p99 latency <100ms at 1M vectors

---

## Phase 9: Query-Adaptive Index Resolution (QAIR)

**Objective:** Dynamically adjust index work per-query based on confidence signals, reducing median latency for "easy" queries while maintaining accuracy for "hard" queries.

**Why Ninth:** Static ANN parameters (nprobe, rerank depth, quantization level) are tuned for worst-case queries. Most queries are not worst-case. QAIR exploits this distribution to do less work on average without sacrificing tail accuracy.

### Why This Is Not Commonly Implemented
- Requires instrumented query path to compute confidence cheaply
- Requires multi-stage pipeline (can't adapt what you don't have)
- Risk-averse systems prefer static parameters with predictable behavior
- Academic benchmarks measure fixed-parameter recall, not adaptive latency
- Stateful databases struggle with per-query adaptation (session state, caching)

### Core Insight
Query difficulty varies. A query near a dense cluster centroid is "easy"—the top-k results are obvious and tightly clustered. A query equidistant from multiple centroids is "hard"—results are spread across clusters with similar scores.

Tidepool's stateless architecture makes per-query adaptation trivial: each query is independent, parameters are chosen at query time, and there's no session state to corrupt.

### Confidence Signals (Computed After Centroid Routing)
| Signal | Computation | Interpretation |
|--------|-------------|----------------|
| **Centroid gap** | `dist(q, c1) - dist(q, c2)` | Large gap → high confidence, one cluster dominates |
| **Score entropy** | `-Σ p_i log(p_i)` over top-k centroid scores | Low entropy → confident, high → ambiguous |
| **Query norm** | `\|\|q\|\|` | Extreme norms may indicate adversarial/OOD queries |
| **Top-k variance** | Variance of distances in initial top-k | Low variance → results are interchangeable |

### Adaptive Parameters
| Parameter | High Confidence | Low Confidence |
|-----------|-----------------|----------------|
| `nprobe` | 1-3 clusters | 10-20 clusters |
| `rerank_depth` | 1x top-k | 3-5x top-k |
| `quantization` | SQ8 only | SQ8 → f16 → f32 cascade |
| `early_exit_threshold` | Aggressive | Conservative |

### Integration with Existing Phases
- **Phase 5 (IVF):** Centroid distances provide confidence signals
- **Phase 6 (Quantization):** Adaptive precision selection per query
- **Phase 7 (Ordering):** Early-exit thresholds adjusted by confidence
- **Phase 8 (Pruning):** Cascade depth controlled by confidence

### Algorithm
```
fn qair_search(query, k):
    // Stage 1: Route to centroids (always cheap)
    centroid_scores = compute_centroid_distances(query)
    
    // Stage 2: Compute confidence
    gap = centroid_scores[1] - centroid_scores[0]
    confidence = sigmoid(gap / calibration_constant)
    
    // Stage 3: Select parameters
    nprobe = lerp(MAX_NPROBE, MIN_NPROBE, confidence)
    rerank_factor = lerp(5.0, 1.0, confidence)
    
    // Stage 4: Execute with adapted parameters
    candidates = ivf_search(query, k * rerank_factor, nprobe)
    results = rerank(candidates, k, precision_for_confidence(confidence))
    
    return results
```

### Deliverables
- [ ] `QairConfig` struct with calibration parameters
- [ ] `compute_query_confidence()` function (centroid gap + entropy)
- [ ] `adapt_search_params()` mapping confidence → nprobe, rerank_depth
- [ ] Calibration tool: run sample queries to tune confidence thresholds
- [ ] Metrics: track confidence distribution, parameter selection histogram
- [ ] Config flag: `adaptive: bool` to enable/disable QAIR

### Calibration Process
1. Sample 10k representative queries
2. Run each with maximum parameters (ground truth)
3. Measure confidence signals for each query
4. Find thresholds that achieve target recall at each confidence level
5. Store calibration in segment metadata

### Why This Is Defensible
- Requires multi-stage pipeline (most DBs have monolithic search)
- Requires stateless architecture (no session state to manage)
- Requires IVF (confidence signals come from centroid routing)
- Retrofitting requires instrumenting the entire query path

**Exit Criteria:**
- p50 latency reduced by 30-50% vs static parameters
- p99 recall matches static parameters (no tail regression)
- <5% overhead for confidence computation
- Confidence signal correlates with actual query difficulty (r² > 0.7)

---

## Phase 10: System-Level Optimization

**Objective:** Optimize end-to-end system behavior: cold starts, parallelism, resource management.

**Why Tenth:** After algorithmic optimizations, system-level effects dominate.

### Deliverables
- [ ] **Parallel segment search:** Rayon or tokio parallelism across segments
- [ ] **Namespace sharding:** Distribute namespaces across query replicas
- [ ] **Hot segment prewarming:** Background mmap + page fault on startup
- [ ] **Cold start mitigation:** Lazy loading with bounded latency
- [ ] **Segment size tuning:** Auto-merge small segments, split large
- [ ] **Connection pooling:** Reuse S3 connections

### Observability
- [ ] Prometheus metrics (query latency, cache hits, segment count)
- [ ] Structured logging with tracing spans
- [ ] Health endpoints with detailed status

**Exit Criteria:**
- Cold start to first query <2s
- Linear scaling with replica count
- Cache hit rate >90% in steady state

---

## Phase 11: Advanced Features (Future)

**Objective:** Extend functionality without compromising core performance.

### Planned
- [ ] Hybrid search (vector + BM25)
- [ ] Metadata indexes (B-tree on attributes)
- [ ] Multi-vector documents
- [ ] Streaming/pagination for large result sets
- [ ] Namespace-level configuration
- [ ] Backup/restore utilities

---

## Performance Targets Summary

| Metric | Target | Phase |
|--------|--------|-------|
| Query latency p50 (100k vectors) | <20ms | 4 |
| Query latency p99 (1M vectors) | <100ms | 8 |
| Query latency p50 (1M vectors) | <30ms | 9 (QAIR) |
| Ingest throughput | >10k docs/sec | 3 |
| Memory per vector (128-dim) | <256 bytes | 6 |
| Cold start time | <2s | 10 |
| Recall@10 | >95% | 5 |
| SIMD early-exit rate | >50% | 7 |

---

## Novel Optimization Summary

Tidepool is fast not just because it uses ANN—but because it exploits **data layout**, **bounds tightening**, and **per-query adaptation** in ways most vector databases cannot:

| Optimization | Why Tidepool Can Do This | Why Others Can't |
|--------------|--------------------------|------------------|
| **Deterministic Vector Ordering** | Immutable segments written once at compaction | Mutable indexes can't maintain ordering |
| **Early-Exit SIMD Kernels** | Full control over distance computation | Most DBs use library BLAS/LAPACK |
| **Query-Adaptive Resolution** | Stateless architecture, no session state | Stateful DBs risk consistency issues |
| **Zero-Copy mmap** | Binary-stable segment format | Serialization formats require parsing |

These optimizations compound: ordered vectors feed early-exit kernels, which feed adaptive resolution, which reduces total work. The result is a system where **median latency is 2-3x better than naive ANN** without sacrificing tail recall.

---

## Ordering Rationale

```
Foundation → HNSW → Zero-Copy → SIMD → IVF → Quantization → Ordering → Pruning → QAIR → System
     ↓          ↓         ↓        ↓       ↓         ↓           ↓          ↓        ↓        ↓
  Correct    Fast     Efficient  Faster  Scalable  Compact    Layout    Cascade  Adaptive  Production
```

Each phase builds on the previous. Skipping phases creates technical debt:
- SIMD without zero-copy wastes cycles on memcpy
- IVF without SIMD makes cluster search slow
- Quantization without IVF has nowhere to apply asymmetric search
- **Ordering without IVF has no posting lists to reorder**
- **Ordering without SIMD has no early-exit to exploit**
- Pruning requires all prior techniques to cascade effectively
- **QAIR without pruning has nothing to adapt**
- **QAIR without IVF has no confidence signals**
