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

**Exit Criteria:**
- [x] Distance kernel throughput >1B ops/sec on 128-dim vectors (achieved: 2.3-3.1B elem/s)
- [x] 10k distance computations in <5ms (achieved: 0.32-0.54ms)

---

## Phase 5: IVF Centroid Index (Complete)

**Objective:** Reduce search space by partitioning vectors into clusters. Only search relevant partitions.

**Why Fifth:** HNSW works but has high memory overhead. IVF scales better for large datasets with filtered queries.

### Techniques
- **k-means clustering:** Run during compaction, k = sqrt(n)
- **Centroid routing:** Query → top-k centroids → posting lists
- **Triangle inequality:** Prune clusters where `dist(q, centroid) - radius > best_so_far`
- **Exact rerank:** Linear scan top candidates from selected clusters

### Deliverables
- [x] `IVFIndex` struct with centroids + posting lists (`index/ivf.rs`)
- [x] k-means implementation (Lloyd's algorithm with configurable iterations)
- [x] `ivf_search()` with nprobe parameter (`segment.rs`)
- [x] Serialization format (`.ivf` files with TPIV magic)
- [x] Hybrid: IVF for large segments (>10k vectors), HNSW for small

### File Format
```
segment.ivf (TPIV magic):
  [header]
    magic: [u8; 4] = "TPIV"
    version: u32 = 1
    k: u32 (cluster count)
    dimensions: u32
    metric: u32
    nprobe_default: u32
  [centroids: f32 × k × d]
  [centroid_norms: f32 × k]
  [radii: f32 × k]
  [posting_lists: length-prefixed Vec<u32> × k]
```

**Exit Criteria:**
- [x] 10x speedup vs linear scan at 500k vectors
- [x] Recall >98% with nprobe=10

---

## Phase 6: Quantization (Complete)

**Objective:** Reduce memory footprint and increase cache efficiency via lossy compression.

**Why Sixth:** At scale, f32 vectors don't fit in RAM. Quantization trades precision for capacity.

### Techniques
- **f16 storage:** 2x compression, minimal recall loss
- **Asymmetric int8:** Quantize DB vectors, keep query in f32
- **Scalar quantization (SQ):** Per-dimension min/max scaling
- **Product quantization (PQ):** Optional, for extreme compression

### Deliverables
- [x] f16 vector storage mode (`quantization.rs`: `quantize_f16`, `f16_distance`)
- [x] SQ8 quantization with calibration (`quantization.rs`: `quantize_sq8` with per-dim min/max)
- [x] Asymmetric distance: `dot(q_f32, db_i8) * scale` (`sq8_distance` with `Sq8Query`)
- [x] Mixed precision: quantized scan → f32 rerank (`quantization_rerank_factor` in IVF search)
- [x] Config flag: `quantization: none | f16 | sq8` (`QUANTIZATION` env var)

### Implementation Details
- `quantization.rs`: f16 and SQ8 codecs with binary sidecar format (TPQ1 magic)
- SQ8 uses per-dimension calibration (min/max → 0-255 codes)
- Asymmetric search: query stays f32, DB vectors are quantized
- IVF search fetches `top_k * rerank_factor` candidates with quantized distances, then reranks with f32
- Sidecar files stored as `.tpq` alongside segments

**Exit Criteria:**
- [x] 4x memory reduction with SQ8 (64 dims: 256 bytes → 64 bytes per vector)
- [x] Recall maintained via rerank factor (configurable, default 4x)

---

## Phase 7: Real-Time Updates

**Objective:** Enable sub-second write-to-query latency using Redis pub/sub and in-memory buffers.

**Why Seventh:** Current eventual consistency (~5 minute compaction interval) is insufficient for real-time applications. Users expect vectors to be queryable immediately after upsert.

### Architecture

```
                         ┌─────────────────┐
        POST /upsert ───▶│     Ingest      │
                         │  1. Write WAL   │
                         │  2. Publish msg │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Redis Pub/Sub │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
       │  Query #1   │     │  Query #2   │     │  Query #3   │
       │  + buffer   │     │  + buffer   │     │  + buffer   │
       └─────────────┘     └─────────────┘     └─────────────┘
```

### Techniques
- **Redis Pub/Sub:** Broadcast new vectors from ingest to all query instances
- **In-memory hot buffer:** Query service maintains recent vectors in memory
- **Brute-force buffer search:** Small buffer (~10k vectors) searched linearly
- **Result merging:** Combine buffer results with segment results, dedupe by ID
- **Buffer cleanup:** Clear vectors from buffer after they appear in compacted segments

### Deliverables
- [ ] Add Redis dependency (`redis` crate with tokio support)
- [ ] `HotBuffer` struct in query service with thread-safe vector storage
- [ ] Redis publisher in ingest service (publish after WAL write)
- [ ] Redis subscriber task in query service (background vector ingestion)
- [ ] `buffer_search()` with brute-force distance computation
- [ ] `merge_results()` combining hot buffer and cold segment results
- [ ] Buffer size limits and overflow handling
- [ ] Tombstone handling in buffer (mark deleted IDs)
- [ ] Config: `REDIS_URL`, `HOT_BUFFER_MAX_SIZE`

### Query Flow (Updated)
```
1. refresh_state()           // Load manifest, segments (existing)
2. buffer_search(query)      // Search hot buffer (new)
3. segment_search(query)     // Search cold segments (existing)
4. merge_results(buffer, segments, top_k)  // Combine and dedupe (new)
5. apply_tombstones()        // Filter deleted IDs (existing)
6. return top_k results
```

### Buffer Implementation
```rust
pub struct HotBuffer {
    vectors: RwLock<Vec<Document>>,
    deleted: RwLock<HashSet<String>>,
    max_size: usize,
    last_compaction_version: AtomicU64,
}

impl HotBuffer {
    pub async fn insert(&self, docs: Vec<Document>);
    pub async fn delete(&self, ids: Vec<String>);
    pub async fn search(&self, query: &[f32], top_k: usize, metric: DistanceMetric) -> Vec<VectorResult>;
    pub async fn clear_compacted(&self, version: u64);
}
```

### Horizontal Scaling
- Each query replica subscribes to Redis
- Each replica maintains its own buffer (no shared state)
- All replicas receive same vectors via pub/sub
- Consistent results across replicas

**Exit Criteria:**
- Write-to-query latency <1 second
- Buffer search adds <5ms to query latency
- Horizontal scaling works (3 replicas, consistent results)
- No data loss (WAL provides durability, buffer provides speed)

---

## Phase 8: Deterministic Vector Ordering

**Objective:** Maximize early-exit effectiveness by reordering vectors within posting lists so that high-quality candidates appear first, causing top-k bounds to tighten rapidly and most distance computations to abort early.

**Why Eighth:** After IVF partitions vectors and quantization reduces precision, the *order* in which vectors are scanned within each cluster determines how quickly bounds tighten. Random ordering wastes SIMD cycles on vectors that will never make top-k.

### Why This Is Not Commonly Implemented
- Requires control over segment layout (cloud-hosted DBs often don't have this)
- Only beneficial with early-exit distance kernels (requires Phase 4)
- Must be done at compaction time (runtime reordering defeats the purpose)
- Academic ANN papers focus on recall, not average-case latency
- Retrofitting into mutable indexes is complex; immutable segments make this trivial

### Techniques
- **Centroid-projected ordering:** Sort vectors by `dot(v, centroid)` descending within each posting list
- **Reference direction sorting:** Use the centroid as a "reference query" and sort by similarity to it
- **Score upper-bound ordering:** For quantized vectors, sort by maximum possible score
- **Batch-aware layout:** Group vectors likely to be co-retrieved for better cache locality

### Deliverables
- [ ] `sort_posting_list_by_centroid_similarity()` in compaction pipeline
- [ ] Configurable ordering strategy: `centroid | random | score_bound`
- [ ] Early-exit dot product kernel with threshold parameter (`dot_f32_early_exit_avx2`)
- [ ] Benchmark: measure % of SIMD iterations skipped per query
- [ ] A/B comparison: ordered vs random posting lists

**Exit Criteria:**
- Average SIMD iterations per distance computation reduced by >50%
- p50 latency improvement of 1.5-2x on IVF queries with nprobe=10
- Zero impact on recall (ordering is internal, not algorithmic)

---

## Phase 9: Multi-Stage Pruning

**Objective:** Cascade cheap approximate checks before expensive exact computation.

**Why Ninth:** Combines IVF, quantization, vector ordering, and HNSW into optimal query plan.

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

## Phase 10: Query-Adaptive Index Resolution (QAIR)

**Objective:** Dynamically adjust index work per-query based on confidence signals, reducing median latency for "easy" queries while maintaining accuracy for "hard" queries.

**Why Tenth:** Static ANN parameters (nprobe, rerank depth, quantization level) are tuned for worst-case queries. Most queries are not worst-case. QAIR exploits this distribution to do less work on average.

### Confidence Signals
| Signal | Computation | Interpretation |
|--------|-------------|----------------|
| **Centroid gap** | `dist(q, c1) - dist(q, c2)` | Large gap → high confidence |
| **Score entropy** | `-Σ p_i log(p_i)` over top-k centroid scores | Low entropy → confident |
| **Top-k variance** | Variance of distances in initial top-k | Low variance → interchangeable |

### Adaptive Parameters
| Parameter | High Confidence | Low Confidence |
|-----------|-----------------|----------------|
| `nprobe` | 1-3 clusters | 10-20 clusters |
| `rerank_depth` | 1x top-k | 3-5x top-k |
| `quantization` | SQ8 only | SQ8 → f16 → f32 cascade |

### Deliverables
- [ ] `QairConfig` struct with calibration parameters
- [ ] `compute_query_confidence()` function
- [ ] `adapt_search_params()` mapping confidence → parameters
- [ ] Calibration tool for tuning thresholds
- [ ] Config flag: `adaptive: bool` to enable/disable QAIR

**Exit Criteria:**
- p50 latency reduced by 30-50% vs static parameters
- p99 recall matches static parameters (no tail regression)

---

## Phase 11: Full-Text Search

**Objective:** Add BM25-based full-text search and hybrid vector+text retrieval.

**Why Eleventh:** Many use cases require combining semantic (vector) search with keyword (text) search. Hybrid retrieval consistently outperforms either approach alone.

### Architecture

```
Document:
  id: "doc-123"
  vector: [0.1, 0.2, ...]        # For semantic search
  text: "full text content"      # For BM25 search (new)
  attributes: {...}              # For filtering

Query:
  vector: [0.1, 0.2, ...]        # Semantic query
  text: "keyword search"         # Text query (new)
  mode: "hybrid" | "vector" | "text"
  alpha: 0.7                     # Blend factor (new)
```

### Techniques
- **Inverted index:** Term → document IDs with term frequencies
- **BM25 scoring:** Standard Okapi BM25 with configurable k1 and b parameters
- **Tokenization:** Unicode-aware tokenizer with stemming support
- **Hybrid scoring:** `score = alpha * vector_score + (1 - alpha) * bm25_score`
- **Reciprocal Rank Fusion (RRF):** Alternative fusion method

### Deliverables
- [ ] `TextIndex` struct with inverted index and document frequencies
- [ ] Tokenizer with configurable stopwords and stemming
- [ ] BM25 scorer implementation
- [ ] Text index serialization format (`.tpti` files)
- [ ] `text_search()` returning ranked document IDs
- [ ] `hybrid_search()` combining vector and text results
- [ ] Fusion methods: linear blend, RRF
- [ ] Config: `TEXT_INDEX_ENABLED`, `BM25_K1`, `BM25_B`

### File Format
```
segment.tpti (TPTI magic):
  [header]
    magic: [u8; 4] = "TPTI"
    version: u32 = 1
    doc_count: u32
    term_count: u32
    avg_doc_length: f32
  [vocabulary: term → term_id mapping]
  [posting_lists: term_id → Vec<(doc_id, term_freq)>]
  [doc_lengths: doc_id → length]
```

### Query Flow (Hybrid)
```
1. If mode == "vector" or "hybrid":
   vector_results = segment_search(query.vector)
   
2. If mode == "text" or "hybrid":
   text_results = text_search(query.text)
   
3. If mode == "hybrid":
   results = fuse(vector_results, text_results, alpha)
else:
   results = vector_results or text_results
   
4. return top_k results
```

### API Changes
```json
// Query request (extended)
{
  "vector": [0.1, 0.2, ...],
  "text": "search keywords",
  "mode": "hybrid",
  "alpha": 0.7,
  "top_k": 10
}

// Upsert request (extended)
{
  "vectors": [
    {
      "id": "doc-1",
      "vector": [0.1, 0.2, ...],
      "text": "full document text for BM25 indexing",
      "attributes": {...}
    }
  ]
}
```

**Exit Criteria:**
- BM25 search latency <50ms at 1M documents
- Hybrid retrieval improves relevance over vector-only (measured on standard benchmarks)
- Text index adds <50% to segment size

---

## Phase 12: System-Level Optimization

**Objective:** Optimize end-to-end system behavior: cold starts, parallelism, resource management.

**Why Twelfth:** After algorithmic optimizations, system-level effects dominate.

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

## Phase 13: Advanced Features (Future)

**Objective:** Extend functionality without compromising core performance.

### Planned
- [ ] Multi-vector documents (e.g., late interaction models like ColBERT)
- [ ] Streaming/pagination for large result sets
- [ ] Namespace-level configuration
- [ ] Backup/restore utilities
- [ ] Metadata indexes (B-tree on attributes)
- [ ] Geo-spatial filtering
- [ ] Multi-tenancy with isolation

---

## Performance Targets Summary

| Metric | Target | Phase |
|--------|--------|-------|
| Query latency p50 (100k vectors) | <20ms | 4 |
| Query latency p99 (1M vectors) | <100ms | 9 |
| Query latency p50 (1M vectors) | <30ms | 10 (QAIR) |
| Write-to-query latency | <1s | 7 |
| Ingest throughput | >10k docs/sec | 3 |
| Memory per vector (128-dim) | <256 bytes | 6 |
| Cold start time | <2s | 12 |
| Recall@10 | >95% | 5 |
| SIMD early-exit rate | >50% | 8 |
| BM25 search latency | <50ms | 11 |

---

## Novel Optimization Summary

Tidepool is fast not just because it uses ANN—but because it exploits **data layout**, **bounds tightening**, **real-time streaming**, and **per-query adaptation** in ways most vector databases cannot:

| Optimization | Why Tidepool Can Do This | Why Others Can't |
|--------------|--------------------------|------------------|
| **Real-Time Updates** | Redis pub/sub + in-memory buffer | Requires coordination layer |
| **Deterministic Vector Ordering** | Immutable segments written once at compaction | Mutable indexes can't maintain ordering |
| **Early-Exit SIMD Kernels** | Full control over distance computation | Most DBs use library BLAS/LAPACK |
| **Query-Adaptive Resolution** | Stateless architecture, no session state | Stateful DBs risk consistency issues |
| **Zero-Copy mmap** | Binary-stable segment format | Serialization formats require parsing |
| **Hybrid Search** | Unified segment format with text index | Separate systems for vector and text |

---

## Ordering Rationale

```
Foundation → HNSW → Zero-Copy → SIMD → IVF → Quantization → Real-Time → Ordering → Pruning → QAIR → Full-Text → System
     ↓          ↓         ↓        ↓       ↓         ↓           ↓           ↓          ↓        ↓         ↓          ↓
  Correct    Fast     Efficient  Faster  Scalable  Compact    Instant     Layout    Cascade  Adaptive  Hybrid   Production
```

Each phase builds on the previous. Skipping phases creates technical debt:
- SIMD without zero-copy wastes cycles on memcpy
- IVF without SIMD makes cluster search slow
- Quantization without IVF has nowhere to apply asymmetric search
- **Real-time without foundation has no segments to search**
- **Ordering without IVF has no posting lists to reorder**
- **Ordering without SIMD has no early-exit to exploit**
- Pruning requires all prior techniques to cascade effectively
- **QAIR without pruning has nothing to adapt**
- **Full-text without foundation has no storage backend**
