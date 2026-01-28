# Tidepool

A high-performance vector database designed for cost-effective deployment on cloud infrastructure. Tidepool uses object storage (S3, R2, MinIO) as the primary data store, enabling horizontal scaling without the complexity of distributed consensus.

## Features

- **Hybrid Indexing** - HNSW for small segments, IVF with k-means clustering for large segments
- **Vector Quantization** - SQ8 (4× compression) and f16 (2× compression) with asymmetric search
- **SIMD Acceleration** - AVX2/FMA on x86_64, NEON on ARM64, with automatic runtime detection
- **Zero-Copy Access** - Memory-mapped segments eliminate deserialization overhead
- **Attribute Filtering** - Filter search results by metadata attributes
- **S3-Compatible** - Works with AWS S3, Cloudflare R2, MinIO, and other S3-compatible storage
- **Stateless Architecture** - Query and ingest services scale independently

## Quick Start

```bash
# Start services with Docker Compose (includes MinIO for local S3)
docker-compose up -d

# Insert vectors (ingest service on port 8081)
curl -X POST http://localhost:8081/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "1", "vector": [0.1, 0.2, 0.3], "attributes": {"title": "Example"}}]}'

# Trigger compaction (or wait for automatic compaction)
curl -X POST http://localhost:8081/compact

# Query vectors (query service on port 8080)
curl -X POST http://localhost:8080/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "top_k": 10}'
```

## Architecture

Tidepool consists of two stateless services:

- **tidepool-query**: HTTP API for vector similarity search
- **tidepool-ingest**: Background worker for vector upserts and compaction

### Design Principles

- Object storage (S3-compatible) is the source of truth
- All data files are immutable
- Query service never writes to object storage
- Ingest service is the only writer
- Local disk is disposable cache (mmap for zero-copy access)
- No coordination between services

## Storage Layout

```
namespaces/{namespace}/
  manifests/
    latest.json          # Current state
    {version}.json       # Versioned snapshots
  wal/
    {date}/{uuid}.wal    # Write-ahead log
  segments/
    {segment_id}.tpvs    # Vector segments (TPV2 binary format)
    {segment_id}.hnsw    # HNSW index graph
    {segment_id}.ivf     # IVF centroid index (for large segments)
    {segment_id}.tpq     # Quantized vectors sidecar (SQ8/f16)
  tombstones/
    latest.rkyv          # Deleted IDs
```

## Performance

### Indexing Strategy

| Segment Size | Index Type | Search Complexity |
|--------------|------------|-------------------|
| < 10,000 vectors | HNSW | O(log n) |
| ≥ 10,000 vectors | IVF + Quantization | O(√n × nprobe) |

- **HNSW**: Hierarchical Navigable Small World graph for sub-linear approximate nearest neighbor search
- **IVF**: Inverted file index with k-means clustering; searches only relevant partitions
- **Quantization**: SQ8 provides 4× compression using asymmetric distance (f32 query, int8 database)

### Memory Efficiency

Approximate resource usage with SQ8 quantization (default):

| Vector Count | Dimensions | Disk Usage | Query RAM |
|--------------|------------|------------|-----------|
| 100,000 | 768 | ~100 MB | ~50 MB |
| 1,000,000 | 768 | ~1 GB | ~500 MB |
| 3,000,000 | 768 | ~3 GB | ~1.5 GB |

### SIMD Optimization

Distance computations use platform-specific SIMD instructions when available:

| Platform | Instructions | Operations per Cycle |
|----------|--------------|---------------------|
| x86_64 | AVX2 + FMA | 8-wide f32 |
| ARM64 | NEON | 4-wide f32 |
| Other | Scalar | 1-wide f32 |

Runtime feature detection selects the optimal implementation automatically.

## API Reference

### Query Service (port 8080)

**POST /v1/vectors/{namespace}** or **POST /v1/namespaces/{namespace}/query**

Query vectors by similarity.

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10,
  "ef_search": 100,
  "nprobe": 10,
  "distance_metric": "cosine_distance",
  "include_vectors": false,
  "filters": {
    "category": "article"
  }
}
```

Response:

```json
{
  "results": [
    {
      "id": "doc-123",
      "attributes": {"category": "article", "title": "..."},
      "dist": 0.123
    }
  ],
  "namespace": "default"
}
```

**GET /v1/namespaces/{namespace}**

Get namespace info.

```json
{
  "namespace": "default",
  "approx_count": 10000,
  "dimensions": 1536
}
```

**GET /health**

Health check.

### Ingest Service (port 8080)

**POST /v1/vectors/{namespace}** or **POST /v1/namespaces/{namespace}**

Upsert vectors.

```json
{
  "vectors": [
    {
      "id": "doc-123",
      "vector": [0.1, 0.2, 0.3, ...],
      "attributes": {
        "category": "article",
        "title": "My Document"
      }
    }
  ]
}
```

Response:

```json
{
  "status": "OK"
}
```

**DELETE /v1/vectors/{namespace}** or **DELETE /v1/namespaces/{namespace}**

Delete vectors by ID.

```json
{
  "ids": ["doc-123", "doc-456"]
}
```

**POST /compact**

Trigger manual compaction.

**GET /status**

Get ingest/compaction status.

**GET /health**

Health check.

## Distance Metrics

- `cosine_distance` (default) - 1 - cosine similarity, range [0, 2]
- `euclidean_squared` - squared L2 distance
- `dot_product` - negative dot product (for ranking)

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | - | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | S3 secret key |
| `AWS_ENDPOINT_URL` | Yes | - | S3 endpoint URL |
| `AWS_REGION` | Yes | - | S3 region |
| `BUCKET_NAME` | Yes | - | S3 bucket name |
| `CACHE_DIR` | No | `/data` | Local cache directory |
| `NAMESPACE` | No | `default` | Data namespace |
| `COMPACTION_INTERVAL` | No | `5m` | Compaction interval |
| `PORT` | No | `8080` | HTTP port |
| `READ_TIMEOUT` | No | `30s` | HTTP read timeout |
| `WRITE_TIMEOUT` | No | `60s` | HTTP write timeout |
| `IDLE_TIMEOUT` | No | `60s` | HTTP idle timeout |
| `MAX_BODY_BYTES` | No | `26214400` | Max request body size in bytes |
| `MAX_TOP_K` | No | `1000` | Max top_k for queries |
| `CORS_ALLOW_ORIGIN` | No | `*` | CORS allow origin |
| `HNSW_M` | No | `16` | HNSW max connections per node |
| `HNSW_EF_CONSTRUCTION` | No | `200` | HNSW build-time beam width |
| `HNSW_EF_SEARCH` | No | `100` | HNSW query-time beam width |
| `IVF_ENABLED` | No | `true` | Enable IVF index build for large segments |
| `IVF_MIN_SEGMENT_SIZE` | No | `10000` | Minimum vectors per segment to build IVF |
| `IVF_NPROBE_DEFAULT` | No | `10` | Default nprobe used by IVF search |
| `IVF_K_FACTOR` | No | `1.0` | IVF k scaling factor (k ≈ sqrt(n) * factor) |
| `IVF_MIN_K` | No | `16` | IVF minimum cluster count |
| `IVF_MAX_K` | No | `65535` | IVF maximum cluster count |
| `QUANTIZATION` | No | `sq8` | Vector compression: `none`, `f16`, `sq8` |
| `QUANTIZATION_RERANK_FACTOR` | No | `4` | Fetch N× candidates, rerank with full precision |
| `WAL_BATCH_MAX_ENTRIES` | No | `1` | Max WAL entries per batch write (ingest) |
| `WAL_BATCH_FLUSH_INTERVAL` | No | `0ms` | Max time to wait before flushing WAL batch |

### Quantization Modes

| Mode | Compression | Memory (1M × 768-dim) | Recall | Description |
|------|-------------|----------------------|--------|-------------|
| `none` | 1× | 3 GB | 100% | Full precision (f32) |
| `f16` | 2× | 1.5 GB | ~99% | Half precision |
| `sq8` | 4× | 768 MB | ~97% | 8-bit scalar quantization (default) |

SQ8 is the default configuration. The `QUANTIZATION_RERANK_FACTOR` parameter controls the number of candidates fetched before reranking with full-precision vectors.

**Migration note:** Existing segments without IVF indexes continue to use HNSW for queries. Trigger compaction to build IVF indexes for large segments.

## Development

### Prerequisites

- Rust 1.93+
- Docker (for local development)

### Local Development with Docker Compose

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop and clean up
docker-compose down -v
```

### Local Development without Docker

1. Start MinIO:

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

2. Create bucket:

```bash
aws --endpoint-url http://localhost:9000 s3 mb s3://tidepool
```

3. Set environment:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_REGION=us-east-1
export BUCKET_NAME=tidepool
```

4. Run services:

```bash
# Terminal 1 - Query (port 8080)
cargo run -p tidepool-query

# Terminal 2 - Ingest (port 8081)
PORT=8081 cargo run -p tidepool-ingest
```

### Benchmarks

```bash
# HNSW search benchmarks + recall measurement
cargo bench -p tidepool-common --bench hnsw

# SIMD distance kernel benchmarks
cargo bench -p tidepool-common --bench simd

# IVF index construction benchmarks
cargo bench -p tidepool-common --bench ivf
```

### Example Usage

```bash
# Upsert vectors (to ingest service)
curl -X POST http://localhost:8081/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": "1", "vector": [0.1, 0.2, 0.3], "attributes": {"title": "Doc 1"}},
      {"id": "2", "vector": [0.4, 0.5, 0.6], "attributes": {"title": "Doc 2"}}
    ]
  }'

# Trigger compaction
curl -X POST http://localhost:8081/compact

# Query vectors (to query service)
curl -X POST http://localhost:8080/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10
  }'
```

## Deployment

### Railway

This repository includes Railway configuration for a two-service deployment:

- **tidepool-query**: Read-only vector search API
- **tidepool-ingest**: Write API with background compaction

#### Setup

1. Create a new Railway project from this repository
2. Add two services using the provided configuration files:
   - `tidepool-query` → `query/railway.toml`
   - `tidepool-ingest` → `ingest/railway.toml`
3. Set **Root Directory** to `/` (repository root) for both services
4. Add Railway Object Storage and configure the S3 environment variables
5. (Optional) Attach a volume at `/data` to the query service for caching

#### Troubleshooting

If the build fails with `no Rust files in /app`, ensure the service is configured to use the Dockerfile builder. The `RAILWAY_DOCKERFILE_PATH` environment variable is pre-configured in each `railway.toml`.

#### Environment Variables

**Required** (configure for both services using Railway Object Storage credentials):

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_ENDPOINT_URL` | S3 endpoint URL |
| `AWS_REGION` | S3 region |
| `BUCKET_NAME` | S3 bucket name |

**Optional tuning:**

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `QUANTIZATION` | ingest | `sq8` | Vector compression (`none`, `f16`, `sq8`) |
| `IVF_NPROBE_DEFAULT` | query | `10` | Number of IVF partitions to search |
| `HNSW_EF_SEARCH` | query | `100` | HNSW search beam width |
| `COMPACTION_INTERVAL` | ingest | `5m` | Background compaction frequency |

**Recommended:** Attach a persistent volume at `/data` on the query service to enable segment caching.

## How It Works

### Write Path (Ingest Service)

1. **Upsert**: Vectors written to WAL files in S3 (immediate durability)
2. **Compaction** (background, every 5min by default):
   - Reads WAL entries
   - Builds segment file (`.tpvs`) with 32-byte aligned vectors
   - Builds HNSW index (`.hnsw`) for small segments
   - Builds IVF index (`.ivf`) + quantized vectors (`.tpq`) for large segments
   - Updates manifest
   - Deletes processed WAL files
3. **Delete**: Tombstones written to WAL, filtered out during queries

### Read Path (Query Service)

1. **Load manifest** from S3 (lists active segments)
2. **Fetch segments** from S3 → local cache (mmap for zero-copy)
3. **Search** each segment in parallel:
   - Small segment: HNSW graph traversal
   - Large segment: IVF centroid routing → quantized scan → f32 rerank
4. **Merge** results across segments, apply tombstones
5. **Return** top-k results

### Architecture Benefits

- **Cost Efficiency**: Object storage pricing (~$0.02/GB/month) vs block storage
- **Durability**: S3-class durability (11 nines) without manual backup configuration
- **Horizontal Scaling**: Add query instances that share the same S3 data
- **Operational Simplicity**: No distributed consensus or leader election

## Current Limitations

- Single namespace per service deployment
- Eventual consistency model (vectors become queryable after compaction)
- No cursor-based pagination for result sets exceeding `top_k`

## License

MIT
