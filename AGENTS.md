# Agent Instructions for Tidepool

This document provides instructions for AI agents working on the Tidepool codebase.

## Project Structure

```
tidepool/
├── Cargo.toml              # Workspace root
├── Dockerfile              # Single multi-service Dockerfile (SERVICE arg)
├── railway.toml            # Railway deployment config
├── railway.json            # Railway template schema
├── crates/
│   └── tidepool-common/    # Shared library (indexes, storage, vectors)
├── ingest/                 # Ingest service (writes, compaction)
├── query/                  # Query service (search API)
├── perf/                   # Performance testing tool
└── docker-compose.yml      # Local development stack
```

## Prerequisites

- **Rust 1.70+** (toolchain will be installed automatically if using rustup)
- **Docker** and **Docker Compose** (for local testing)

## Compilation

### Build All Packages

```bash
cargo build --release
```

### Build Specific Services

```bash
# Query service only
cargo build --release -p tidepool-query

# Ingest service only
cargo build --release -p tidepool-ingest

# Common library only
cargo build --release -p tidepool-common
```

### Check for Compilation Errors (Fast)

```bash
cargo check --all
```

## Running Tests

### Run All Tests

```bash
cargo test --all
```

### Run Tests for Specific Crate

```bash
# Common library tests (HNSW, segments, quantization)
cargo test -p tidepool-common

# Query engine tests
cargo test -p tidepool-query

# Ingest service tests
cargo test -p tidepool-ingest
```

### Run a Specific Test

```bash
cargo test -p tidepool-common test_hnsw_basic
```

### Run Tests with Output

```bash
cargo test --all -- --nocapture
```

## Running Locally with Docker

### Build Docker Image

The repository uses a single Dockerfile that builds ALL services into one image:

```bash
# Build the image (contains all services)
docker build -t tidepool .

# Run specific services
docker run -p 8080:8080 tidepool /tidepool-query
docker run -p 8080:8080 tidepool /tidepool-ingest
```

### Start the Full Stack

```bash
docker-compose up -d
```

This starts:
- **MinIO** (S3-compatible storage) on ports 9000 (API) and 9001 (console)
- **tidepool-query** on port 8080
- **tidepool-ingest** on port 8081

### Rebuild After Code Changes

```bash
docker-compose down
docker-compose up --build -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f tidepool-query
docker-compose logs -f tidepool-ingest
```

### Stop and Clean Up

```bash
# Stop containers (preserves volumes)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Testing the API

### Insert Vectors

```bash
curl -X POST http://localhost:8081/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": "vec1", "vector": [1.0, 0.0, 0.0, 0.0]},
      {"id": "vec2", "vector": [0.9, 0.1, 0.0, 0.0]},
      {"id": "vec3", "vector": [0.0, 1.0, 0.0, 0.0]}
    ]
  }'
```

Expected response: `{"status": "ok"}`

### Query Vectors

```bash
curl -X POST http://localhost:8080/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 3}'
```

Expected response:
```json
{
  "results": [
    {"id": "vec1", "score": 1.0},
    {"id": "vec2", "score": 0.9...}
  ]
}
```

### Trigger Compaction

```bash
curl -X POST http://localhost:8081/v1/namespaces/default/compact
```

### Delete Vectors

```bash
curl -X DELETE http://localhost:8081/v1/vectors/default \
  -H "Content-Type: application/json" \
  -d '{"ids": ["vec1", "vec2"]}'
```

### Health Check

```bash
# Query service
curl http://localhost:8080/health

# Ingest service
curl http://localhost:8081/health
```

## Environment Variables

### Query Service (tidepool-query)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | S3 access key | Required |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key | Required |
| `AWS_ENDPOINT_URL` | S3 endpoint (for MinIO/R2) | AWS default |
| `AWS_REGION` | S3 region | `us-east-1` |
| `BUCKET_NAME` | S3 bucket name | Required |
| `NAMESPACE` | Default namespace | `default` |
| `CACHE_DIR` | Local cache directory | `/data` |
| `HOT_BUFFER_MAX_SIZE` | Max vectors in hot buffer | `10000` |
| `RUST_LOG` | Log level | `info` |

### Ingest Service (tidepool-ingest)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | S3 access key | Required |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key | Required |
| `AWS_ENDPOINT_URL` | S3 endpoint (for MinIO/R2) | AWS default |
| `AWS_REGION` | S3 region | `us-east-1` |
| `BUCKET_NAME` | S3 bucket name | Required |
| `NAMESPACE` | Default namespace | `default` |
| `COMPACTION_INTERVAL` | Auto-compaction interval | `1m` |
| `RUST_LOG` | Log level | `info` |

## Common Issues

### "Connection refused" on curl

Wait a few seconds after `docker-compose up` for services to initialize. Check logs:
```bash
docker-compose logs tidepool-query
```

### Empty query results after insert

Vectors are immediately available via WAL-aware queries (hot buffer). If you don't see results:
1. Check the vector dimensions match between insert and query
2. Ensure you're using the correct namespace
3. Check the `vector` field name (not `values` or `embedding`)

### Compilation errors with SIMD

The project auto-detects SIMD capabilities. If you see SIMD-related errors:
```bash
# Build without SIMD optimizations
RUSTFLAGS="-C target-feature=-avx2" cargo build --release
```

### Docker build fails

Ensure you have sufficient disk space and memory:
```bash
docker system prune -f  # Clean up unused images
```

## Running Benchmarks

```bash
# HTTP API benchmarks (local Docker)
make bench-local

# HTTP API benchmarks (production)
make bench INGEST_URL=https://... QUERY_URL=https://...

# Rust-level benchmarks (HNSW, SIMD)
make bench-rust

# Or directly:
cargo bench -p tidepool-common --bench hnsw
cargo bench -p tidepool-common --bench simd
```

See `docs/PERFORMANCE.md` for detailed benchmark results.

## Code Organization

### Key Files

| File | Purpose |
|------|---------|
| `crates/tidepool-common/src/index/hnsw.rs` | HNSW index implementation |
| `crates/tidepool-common/src/simd.rs` | SIMD distance functions |
| `crates/tidepool-common/src/segment.rs` | Segment storage format |
| `crates/tidepool-common/src/wal.rs` | Write-ahead log |
| `crates/tidepool-common/src/manifest.rs` | Manifest (segment registry) |
| `query/src/engine.rs` | Query engine core |
| `query/src/main.rs` | Query service HTTP API |
| `ingest/src/main.rs` | Ingest service HTTP API |
| `ingest/src/compactor.rs` | Compaction logic |

### Data Flow

1. **Ingest**: Client → Ingest Service → WAL (S3)
2. **Compaction**: WAL → Segments + Index (S3)
3. **Query**: Client → Query Service → Hot Buffer + Segments → Results

## Linting and Formatting

```bash
# Format code
cargo fmt --all

# Run clippy
cargo clippy --all -- -D warnings
```

## Railway Deployment

The repository is configured as a Railway-compatible template with a single Dockerfile that builds all services into one image.

### How It Works

- The root `Dockerfile` builds ALL service binaries into a single image
- Each Railway service uses a different **Start Command** to run its binary
- `tidepool-query` → Start Command: `/tidepool-query`
- `tidepool-ingest` → Start Command: `/tidepool-ingest`

### Railway Template Settings

For each service in the Railway template, configure:

| Setting | tidepool-query | tidepool-ingest |
|---------|----------------|-----------------|
| Start Command | `/tidepool-query` | `/tidepool-ingest` |
| Healthcheck Path | `/health` | `/health` |

### Adding New Services

To add a new service (e.g., `compactor`):

1. Create the service crate (e.g., `compactor/` with `Cargo.toml` and `src/main.rs`)
2. Add it to the workspace in `Cargo.toml`
3. Update the root `Dockerfile`:
   - Add the `Cargo.toml` copy for dependency caching
   - Add `-p tidepool-compactor` to the cargo build command
   - Add `COPY --from=builder` for the new binary
4. In Railway template, add new service with Start Command: `/tidepool-compactor`

### Template Structure

| File | Purpose |
|------|---------|
| `Dockerfile` | Single image containing all service binaries |
| `railway.toml` | Railway build/deploy configuration |
| `railway.json` | Railway template schema |

## Making Changes

1. Make your code changes
2. Run `cargo check --all` to verify compilation
3. Run `cargo test --all` to verify tests pass
4. Run `cargo fmt --all` to format code
5. Test locally with Docker if changing API behavior
