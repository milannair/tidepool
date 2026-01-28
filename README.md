# Tidepool

A lightweight vector database backed by object storage. Inspired by [Turbopuffer](https://turbopuffer.com/).

Deploy on Railway with Railway Buckets as the storage backend.

## Architecture

Tidepool consists of two stateless services:

- **tidepool-query**: HTTP API for vector similarity search
- **tidepool-ingest**: Background worker for vector upserts and compaction

### Design Principles

- Object storage (S3-compatible) is the source of truth
- All data files are immutable
- Query service never writes to object storage
- Ingest service is the only writer
- Local disk is disposable cache
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
    {segment_id}.tpvs    # Vector segments
    {segment_id}.hnsw    # HNSW index graph
  tombstones/
    latest.rkyv          # Deleted IDs
```

## API Reference

### Query Service (port 8080)

**POST /v1/vectors/{namespace}** or **POST /v1/namespaces/{namespace}/query**

Query vectors by similarity.

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10,
  "ef_search": 100,
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
| `WAL_BATCH_MAX_ENTRIES` | No | `1` | Max WAL entries per batch write (ingest) |
| `WAL_BATCH_FLUSH_INTERVAL` | No | `0ms` | Max time to wait before flushing WAL batch |

## Development

### Prerequisites

- Rust 1.93+
- Docker (optional)
- S3-compatible storage (MinIO for local dev)

### Local Development

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
# Terminal 1 - Query
cargo run -p tidepool-query

# Terminal 2 - Ingest
cargo run -p tidepool-ingest
```

### Benchmarks & Recall

```bash
# HNSW search benchmarks + recall sanity checks
cargo bench -p tidepool-common --bench hnsw
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

## Railway Template (Two Services)

This repo is set up for a Railway template with **two services**:

- **tidepool-query** (read-only API)
- **tidepool-ingest** (writer + compactor)

### Template Setup

1. Create a new Railway project from this repo.
2. Add two services and point each to its config file (absolute paths).
   - **tidepool-query** → `/query/railway.toml`
   - **tidepool-ingest** → `/ingest/railway.toml`
3. Set **Root Directory** for both services to `/` (repo root).
4. Add a Railway Object Storage (S3-compatible bucket).
5. Add the bucket env vars to both services (see below).
6. Optional: add a volume at `/data` for query caching.

Note: Railway config-as-code (`railway.toml`/`railway.json`) applies to a single
service deployment, while multi-service templates are created from a Railway
project in the UI. See `TEMPLATE.md` for the exact steps.

If a build uses Railpack and fails with `no Rust files in /app`, the service is not
using the Dockerfile builder. In Railway, set the service **Builder** to
Dockerfile or add the env var `RAILWAY_DOCKERFILE_PATH` (already set in the
service `railway.toml`).

### Railway Environment Variables

Set these for both services:
- `AWS_ACCESS_KEY_ID` - from Railway bucket
- `AWS_SECRET_ACCESS_KEY` - from Railway bucket  
- `AWS_ENDPOINT_URL` - from Railway bucket
- `AWS_REGION` - from Railway bucket
- `BUCKET_NAME` - your bucket name

Recommended for ingest:
- `COMPACTION_INTERVAL` - e.g. `5m`

## How It Works

1. **Upsert**: Vectors are written to WAL files in S3
2. **Compaction**: Background worker merges WAL into segments
3. **Query**: Loads segments, performs brute-force similarity search
4. **Delete**: Tombstones written to WAL, applied during compaction

## Limitations (v0)

- Brute-force search (no ANN index yet)
- Single namespace per deployment
- Incremental compaction only (no segment merging yet)
- No streaming/pagination for large result sets

## License

MIT
