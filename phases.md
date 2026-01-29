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

## Phase 7: Real-Time Updates (Complete)

**Objective:** Enable sub-second write-to-query latency using WAL-aware queries and in-memory buffers.

**Why Seventh:** Current eventual consistency (~5 minute compaction interval) is insufficient for real-time applications. Users expect vectors to be queryable immediately after upsert.

### Architecture

```
                         ┌─────────────────┐
        POST /upsert ───▶│     Ingest      │
                         │  1. Write WAL   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   S3 (WAL)      │◄─────────────────┐
                         └─────────────────┘                  │
                                                              │
              ┌───────────────────┬───────────────────┐       │
              ▼                   ▼                   ▼       │
       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
       │  Query #1   │     │  Query #2   │     │  Query #3   ││
       │  scan WAL   │─────│  scan WAL   │─────│  scan WAL   │┘
       │  + buffer   │     │  + buffer   │     │  + buffer   │
       └─────────────┘     └─────────────┘     └─────────────┘
```

### Techniques
- **WAL-Aware Queries:** Query nodes scan WAL files directly from S3 on each query
- **In-memory hot buffer:** Query service maintains recent vectors in HNSW-indexed buffer
- **HNSW buffer search:** O(log n) search via mini-HNSW index in hot buffer
- **Result merging:** Combine buffer results with segment results, dedupe by ID
- **WAL watermark:** Track manifest.created_at to filter already-compacted WAL entries

### Deliverables
- [x] `HotBuffer` struct with HNSW index for O(log n) search (`buffer.rs`)
- [x] WAL scanner in query engine (`scan_wal()` in `engine.rs`)
- [x] WAL reader integration (`WalReader` from `tidepool-common`)
- [x] `merge_results()` combining hot buffer and cold segment results
- [x] Buffer size limits with FIFO eviction
- [x] Tombstone handling in buffer (RoaringBitmap for deleted IDs)
- [x] Config: `HOT_BUFFER_MAX_SIZE`

### Implementation Details
- **No Redis required:** Object storage (S3) is the only stateful dependency
- **WAL scanning:** On each query, list WAL files → filter by manifest watermark → read new entries
- **HNSW-indexed buffer:** Uses mini-HNSW (M=12, ef_construction=100) for fast buffer search
- **Scanned file tracking:** Avoid re-reading WAL files already in buffer
- **Automatic cleanup:** Buffer cleared when manifest changes (data now in segments)

### Query Flow (Updated)
```
1. refresh_state()           // Load manifest, segments
2. scan_wal()                // Scan new WAL files → populate buffer
3. buffer_search(query)      // Search hot buffer (HNSW)
4. segment_search(query)     // Search cold segments
5. merge_results(buffer, segments, top_k)  // Combine and dedupe
6. apply_tombstones()        // Filter deleted IDs
7. return top_k results
```

### Buffer Implementation
```rust
pub struct HotBuffer {
    index: RwLock<HnswIndex>,           // Mini-HNSW for O(log n) search
    id_to_index: RwLock<HashMap<String, usize>>,
    index_to_id: RwLock<Vec<String>>,
    attributes: RwLock<Vec<Option<AttrValue>>>,
    active: RwLock<RoaringBitmap>,      // Active (non-deleted) vectors
    insertion_order: RwLock<Vec<String>>, // For FIFO eviction
    max_size: usize,
}
```

### Horizontal Scaling
- Each query replica scans WAL from S3 independently
- Each replica maintains its own buffer (no shared state)
- All replicas read same WAL files → consistent results
- Object storage is the single source of truth

**Exit Criteria:**
- [x] Write-to-query latency <1 second (achieved: ~10-50ms)
- [x] Buffer search adds <5ms to query latency (HNSW: ~0.2ms)
- [x] Horizontal scaling works (all replicas read same WAL)
- [x] No data loss (WAL provides durability, buffer provides speed)
- [x] No Redis dependency (S3-only architecture)

---

## Phase 8: Dynamic Namespace Support

**Objective:** Enable a single deployment to handle multiple namespaces dynamically, without requiring separate services per namespace.

**Why Eighth:** Currently, each Tidepool deployment handles exactly one namespace (configured via `NAMESPACE` env var). This requires deploying separate ingest/query service pairs for each namespace, increasing operational complexity and resource usage.

### Current Limitation

```yaml
# Current: One deployment per namespace
tidepool-default:
  NAMESPACE: default
  
tidepool-products:  # Separate deployment!
  NAMESPACE: products
```

### Target Architecture

```yaml
# Target: Single deployment, multiple namespaces
tidepool:
  # No NAMESPACE env var - handles all dynamically
```

```
                    ┌─────────────────────────────────────┐
    /v1/vectors/ns1 │           Tidepool Service          │
    /v1/vectors/ns2 │  ┌─────────┐  ┌─────────┐           │
    /v1/vectors/ns3 │  │ ns1 buf │  │ ns1 eng │           │
         ...        │  ├─────────┤  ├─────────┤           │
                    │  │ ns2 buf │  │ ns2 eng │           │
                    │  ├─────────┤  ├─────────┤           │
                    │  │ ns3 buf │  │ ns3 eng │           │
                    │  └─────────┘  └─────────┘           │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │              S3 Bucket              │
                    │  namespaces/ns1/...                 │
                    │  namespaces/ns2/...                 │
                    │  namespaces/ns3/...                 │
                    └─────────────────────────────────────┘
```

### Techniques
- **Lazy initialization:** Create namespace state on first request
- **Per-namespace engines:** Separate `Engine` instance per namespace
- **Per-namespace buffers:** Separate `HotBuffer` per namespace
- **Shared storage client:** Single S3 client shared across namespaces
- **LRU namespace eviction:** Evict least-recently-used namespace state under memory pressure

### Deliverables

#### Query Service
- [ ] `NamespaceManager` struct managing multiple `Engine` instances
- [ ] `get_or_create_engine(namespace)` with lazy initialization
- [ ] Per-namespace hot buffers and WAL scanners
- [ ] Remove `NAMESPACE` config validation (accept any namespace)
- [ ] Namespace listing endpoint: `GET /v1/namespaces`
- [ ] Optional: LRU eviction of inactive namespace state

#### Ingest Service
- [ ] `NamespaceManager` for WAL writers and compactors
- [ ] Per-namespace compaction cycles
- [ ] Dynamic namespace discovery from requests
- [ ] Optional: Parallel compaction across namespaces

#### Configuration
- [ ] `MAX_NAMESPACES` - Maximum concurrent namespaces (default: unlimited)
- [ ] `NAMESPACE_IDLE_TIMEOUT` - Evict namespace state after inactivity
- [ ] `ALLOWED_NAMESPACES` - Optional allowlist for multi-tenant security

### Implementation Details

```rust
// Query service
pub struct NamespaceManager<S: Store + Clone> {
    storage: S,
    engines: RwLock<HashMap<String, Arc<Engine<S>>>>,
    buffers: RwLock<HashMap<String, Arc<HotBuffer>>>,
    cache_dir: Option<String>,
    options: EngineOptions,
    max_namespaces: Option<usize>,
}

impl<S: Store + Clone + 'static> NamespaceManager<S> {
    pub async fn get_engine(&self, namespace: &str) -> Arc<Engine<S>> {
        // Return existing or create new
    }
    
    pub async fn list_namespaces(&self) -> Vec<String> {
        // List from S3: namespaces/*/manifests/latest.rkyv
    }
}
```

```rust
// Ingest service  
pub struct IngestNamespaceManager<S: Store + Clone> {
    storage: S,
    wal_writers: RwLock<HashMap<String, Arc<BufferedWalWriter<S>>>>,
    compactors: RwLock<HashMap<String, Arc<Compactor<S>>>>,
}
```

### API (Unchanged)
The API already supports namespace in the URL path:

```bash
# These already work - just need backend to handle dynamically
POST /v1/vectors/products    # Creates "products" namespace if needed
POST /v1/vectors/users       # Creates "users" namespace if needed
GET  /v1/namespaces          # Lists all namespaces in bucket
```

### Horizontal Scaling Strategy: Shared-Nothing Replication

Each replica independently handles all namespaces with no coordination required:

```
                    Load Balancer
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Query 1 │     │ Query 2 │     │ Query 3 │
    │ ─────── │     │ ─────── │     │ ─────── │
    │ ns1 buf │     │ ns1 buf │     │ ns1 buf │  ← Each replica
    │ ns2 buf │     │ ns2 buf │     │ ns2 buf │    has its own
    │ ns3 buf │     │ ns3 buf │     │ ns3 buf │    namespace state
    └────┬────┘     └────┬────┘     └────┬────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │     S3 Bucket       │  ← Single source of truth
              │ namespaces/ns1/...  │
              │ namespaces/ns2/...  │
              │ namespaces/ns3/...  │
              └─────────────────────┘
```

**How it works:**
- Any replica can handle any namespace request
- Each replica lazily initializes namespace state on first request
- All replicas scan the same WAL files from S3 → consistent results
- No coordination or shared state between replicas
- LRU eviction bounds memory: only active namespaces consume resources

**Why Shared-Nothing:**
| Benefit | Description |
|---------|-------------|
| **Simple** | No coordination layer, no distributed locks |
| **Fault tolerant** | Any replica can fail without affecting others |
| **Flexible routing** | Works with any load balancer (round-robin, random) |
| **Consistent** | S3 is single source of truth for all replicas |

**Memory Management with LRU:**
```rust
pub struct NamespaceManager<S: Store + Clone> {
    engines: RwLock<LruCache<String, Arc<Engine<S>>>>,  // LRU eviction
    max_namespaces: usize,  // e.g., 50 active at once per replica
}
```

| Scenario | Memory per Replica |
|----------|-------------------|
| 10 namespaces, all active | 10 × buffer_size |
| 100 namespaces, 20 active (LRU=20) | 20 × buffer_size |
| 1000 namespaces, 50 active (LRU=50) | 50 × buffer_size |

### Security Considerations
- **Namespace isolation:** Each namespace has separate data paths in S3
- **Optional allowlist:** `ALLOWED_NAMESPACES=ns1,ns2,ns3` restricts access
- **No cross-namespace queries:** Each query targets exactly one namespace

**Exit Criteria:**
- Single deployment handles 10+ namespaces concurrently
- First request to new namespace completes in <500ms (lazy init)
- Memory usage scales with active namespaces, not total namespaces
- No regression in single-namespace performance
- Horizontal scaling works with shared-nothing replication

---

## Phase 9: Full-Text Search

**Objective:** Add BM25-based full-text search and hybrid vector+text retrieval.

**Why Ninth:** Many use cases require combining semantic (vector) search with keyword (text) search. Hybrid retrieval consistently outperforms either approach alone. Adding this early enables more complete applications.

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

## Phase 10: Deterministic Vector Ordering

**Objective:** Maximize early-exit effectiveness by reordering vectors within posting lists so that high-quality candidates appear first, causing top-k bounds to tighten rapidly and most distance computations to abort early.

**Why Tenth:** After IVF partitions vectors and quantization reduces precision, the *order* in which vectors are scanned within each cluster determines how quickly bounds tighten. Random ordering wastes SIMD cycles on vectors that will never make top-k.

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

## Phase 11: Multi-Stage Pruning

**Objective:** Cascade cheap approximate checks before expensive exact computation.

**Why Eleventh:** Combines IVF, quantization, vector ordering, and HNSW into optimal query plan.

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

## Phase 12: Query-Adaptive Index Resolution (QAIR)

**Objective:** Dynamically adjust index work per-query based on confidence signals, reducing median latency for "easy" queries while maintaining accuracy for "hard" queries.

**Why Twelfth:** Static ANN parameters (nprobe, rerank depth, quantization level) are tuned for worst-case queries. Most queries are not worst-case. QAIR exploits this distribution to do less work on average.

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

## Phase 13: System-Level Optimization

**Objective:** Optimize end-to-end system behavior: cold starts, parallelism, resource management.

**Why Thirteenth:** After algorithmic optimizations, system-level effects dominate.

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

## Phase 14: Advanced Features (Future)

**Objective:** Extend functionality without compromising core performance.

### Planned
- [ ] Multi-vector documents (e.g., late interaction models like ColBERT)
- [ ] Streaming/pagination for large result sets
- [ ] Namespace-level configuration
- [ ] Backup/restore utilities
- [ ] Metadata indexes (B-tree on attributes)
- [ ] Geo-spatial filtering

---

## Performance Targets Summary

| Metric | Target | Phase |
|--------|--------|-------|
| Query latency p50 (100k vectors) | <20ms | 4 |
| Query latency p99 (1M vectors) | <100ms | 11 |
| Query latency p50 (1M vectors) | <30ms | 12 (QAIR) |
| Write-to-query latency | <1s | 7 |
| Ingest throughput | >10k docs/sec | 3 |
| Memory per vector (128-dim) | <256 bytes | 6 |
| Cold start time | <2s | 13 |
| Recall@10 | >95% | 5 |
| Multi-namespace init | <500ms | 8 |
| BM25 search latency | <50ms | 9 |
| SIMD early-exit rate | >50% | 10 |

---

## Novel Optimization Summary

Tidepool is fast not just because it uses ANN—but because it exploits **data layout**, **bounds tightening**, **real-time streaming**, and **per-query adaptation** in ways most vector databases cannot:

| Optimization | Why Tidepool Can Do This | Why Others Can't |
|--------------|--------------------------|------------------|
| **Real-Time Updates** | WAL-aware queries + HNSW-indexed buffer | Requires separate caching layer |
| **S3-Only Architecture** | Object storage is sole dependency | Need Redis/Kafka for real-time |
| **Dynamic Namespaces** | Lazy initialization, shared S3 bucket | Require separate deployments |
| **Deterministic Vector Ordering** | Immutable segments written once at compaction | Mutable indexes can't maintain ordering |
| **Early-Exit SIMD Kernels** | Full control over distance computation | Most DBs use library BLAS/LAPACK |
| **Query-Adaptive Resolution** | Stateless architecture, no session state | Stateful DBs risk consistency issues |
| **Zero-Copy mmap** | Binary-stable segment format | Serialization formats require parsing |
| **Hybrid Search** | Unified segment format with text index | Separate systems for vector and text |

---

## Ordering Rationale

```
Foundation → HNSW → Zero-Copy → SIMD → IVF → Quantization → Real-Time → Namespaces → Full-Text → Ordering → Pruning → QAIR → System
     ↓          ↓         ↓        ↓       ↓         ↓           ↓            ↓            ↓            ↓          ↓        ↓         ↓
  Correct    Fast     Efficient  Faster  Scalable  Compact    Instant     Multi-Tenant  Hybrid      Layout    Cascade  Adaptive  Production
```

Each phase builds on the previous. Skipping phases creates technical debt:
- SIMD without zero-copy wastes cycles on memcpy
- IVF without SIMD makes cluster search slow
- Quantization without IVF has nowhere to apply asymmetric search
- **Real-time without foundation has no segments to search**
- **Dynamic namespaces without real-time creates per-namespace overhead**
- **Full-text without foundation has no storage backend**
- **Ordering without IVF has no posting lists to reorder**
- **Ordering without SIMD has no early-exit to exploit**
- Pruning requires all prior techniques to cascade effectively
- **QAIR without pruning has nothing to adapt**
