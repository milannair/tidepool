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
- [x] Brute-force vector search (cosine, euclidean, dot product)
- [x] Attribute filtering
- [x] Background compaction worker
- [x] Railway deployment support

**Exit Criteria:** End-to-end upsert → compact → query flow working with object storage.

---

## Phase 2: HNSW & Caching (Complete)

**Objective:** Replace brute-force with HNSW for sub-linear query time. Add segment caching to eliminate redundant S3 reads.

**Why Second:** Query latency is the primary bottleneck after foundation. HNSW provides O(log n) search vs O(n) brute-force.

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
- HNSW recall >95% vs brute-force
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

## Phase 4: SIMD Distance Kernels

**Objective:** Maximize throughput of distance computations using SIMD intrinsics.

**Why Fourth:** After zero-copy, distance computation dominates CPU time. SIMD provides 4-8x speedup.

### Techniques
- **AVX2/FMA:** 8-wide f32 operations
- **Normalized cosine:** `1 - dot(a, b)` when ||a|| = ||b|| = 1
- **Early-exit dot products:** Stop when partial sum exceeds threshold
- **Loop unrolling:** Process 32 floats per iteration

### Deliverables
- [ ] `simd_dot_f32_avx2()` with runtime feature detection
- [ ] `simd_l2_squared_avx2()`
- [ ] Fallback scalar implementations
- [ ] Pre-normalized vectors in segment (optional mode)
- [ ] Benchmark suite comparing scalar vs SIMD

### Dependencies
- `std::arch` intrinsics or `pulp` / `simdeez` crate

**Exit Criteria:**
- Distance kernel throughput >1B ops/sec on 128-dim vectors
- Brute-force 10k vectors in <5ms

---

## Phase 5: IVF Centroid Index

**Objective:** Reduce search space by partitioning vectors into clusters. Only search relevant partitions.

**Why Fifth:** HNSW works but has high memory overhead. IVF scales better for large datasets with filtered queries.

### Techniques
- **k-means clustering:** Run during compaction, k = sqrt(n)
- **Centroid routing:** Query → top-k centroids → posting lists
- **Triangle inequality:** Prune clusters where `dist(q, centroid) - radius > best_so_far`
- **Exact rerank:** Brute-force top candidates from selected clusters

### Deliverables
- [ ] `IVFIndex` struct with centroids + posting lists
- [ ] k-means implementation (Lloyd's algorithm)
- [ ] `ivf_search()` with nprobe parameter
- [ ] Serialization format (`.ivf` files)
- [ ] Hybrid: IVF for large segments, brute-force for small

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
- 10x speedup vs brute-force at 500k vectors
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

## Phase 7: Multi-Stage Pruning

**Objective:** Cascade cheap approximate checks before expensive exact computation.

**Why Seventh:** Combines IVF, quantization, and HNSW into optimal query plan.

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

## Phase 8: System-Level Optimization

**Objective:** Optimize end-to-end system behavior: cold starts, parallelism, resource management.

**Why Eighth:** After algorithmic optimizations, system-level effects dominate.

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

## Phase 9: Advanced Features (Future)

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
| Query latency p99 (1M vectors) | <100ms | 7 |
| Ingest throughput | >10k docs/sec | 3 |
| Memory per vector (128-dim) | <256 bytes | 6 |
| Cold start time | <2s | 8 |
| Recall@10 | >95% | 5 |

---

## Ordering Rationale

```
Foundation → HNSW → Zero-Copy → SIMD → IVF → Quantization → Pruning → System
     ↓          ↓         ↓        ↓       ↓         ↓           ↓        ↓
  Correct    Fast     Efficient  Faster  Scalable  Compact   Optimal   Production
```

Each phase builds on the previous. Skipping phases creates technical debt:
- SIMD without zero-copy wastes cycles on memcpy
- IVF without SIMD makes cluster search slow
- Quantization without IVF has nowhere to apply asymmetric search
- Pruning requires all prior techniques to cascade effectively
