# Tidepool

A stateless search engine backed by object storage.

## Architecture

Tidepool consists of two services:

- **tidepool-query**: HTTP API for search operations
- **tidepool-ingest**: Background worker for document ingestion and compaction

### Design Principles

- Object storage (S3-compatible) is the source of truth
- All data files are immutable
- Query service never writes to object storage
- Ingest service is the only writer
- Local disk is disposable cache
- No shared memory or coordination between services

## Storage Model

```
namespaces/{namespace}/
  manifests/
    latest.json          # Current state pointer
    {version}.json       # Versioned snapshots
  wal/
    {date}/{uuid}.jsonl  # Write-ahead log files
  segments/
    {segment_id}.parquet # Document data
  indexes/
    {segment_id}.idx     # Search indexes
```

## API Reference

### Query Service (tidepool-query)

**POST /search**

Search for documents.

```json
{
  "query": "search terms",
  "filters": {
    "tags": ["tag1", "tag2"]
  },
  "limit": 10,
  "offset": 0
}
```

Response:

```json
{
  "results": [
    {
      "document": {
        "id": "...",
        "content": "...",
        "title": "...",
        "tags": ["..."]
      },
      "score": 0.95
    }
  ],
  "total_hits": 100,
  "took_ms": 15,
  "query": "search terms"
}
```

**GET /health**

Health check endpoint.

**GET /stats**

Returns search engine statistics.

### Ingest Service (tidepool-ingest)

**POST /ingest**

Ingest documents.

```json
{
  "documents": [
    {
      "id": "optional-id",
      "content": "Document content to index",
      "title": "Document Title",
      "url": "https://example.com",
      "tags": ["tag1", "tag2"],
      "metadata": {
        "custom": "data"
      }
    }
  ]
}
```

Response:

```json
{
  "ingested": 1,
  "wal_file": "namespaces/default/wal/2024-01-15/abc123.jsonl"
}
```

**POST /compact**

Trigger manual compaction.

**GET /status**

Returns ingest/compaction status.

**GET /health**

Health check endpoint.

## Configuration

All configuration is via environment variables:

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

## Development

### Prerequisites

- Go 1.23+
- Docker
- S3-compatible object storage (MinIO for local development)

### Local Development

1. Start MinIO:

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

2. Create a bucket:

```bash
# Using MinIO client or AWS CLI
aws --endpoint-url http://localhost:9000 s3 mb s3://tidepool
```

3. Set environment variables:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_REGION=us-east-1
export BUCKET_NAME=tidepool
```

4. Run services:

```bash
# Terminal 1 - Query service
go run ./cmd/tidepool-query

# Terminal 2 - Ingest service
go run ./cmd/tidepool-ingest
```

### Building Docker Images

```bash
# Build query service
docker build -f Dockerfile.query -t tidepool-query .

# Build ingest service
docker build -f Dockerfile.ingest -t tidepool-ingest .
```

### Running with Docker Compose

```bash
docker-compose up
```

## Deployment

### Railway

1. Fork this repository
2. Create a new Railway project
3. Add two services from the repository:
   - Service 1: Use `Dockerfile.query`
   - Service 2: Use `Dockerfile.ingest`
4. Add a Railway Object Storage bucket
5. Configure environment variables for both services
6. Add a volume mount to `/data` for the query service

### Environment Variables on Railway

Set these for both services via Railway's variable management:

- `AWS_ACCESS_KEY_ID`: From Railway Object Storage
- `AWS_SECRET_ACCESS_KEY`: From Railway Object Storage
- `AWS_ENDPOINT_URL`: From Railway Object Storage
- `AWS_REGION`: From Railway Object Storage
- `BUCKET_NAME`: Your bucket name

## Data Flow

1. **Ingestion**: Documents are POSTed to the ingest service
2. **WAL Write**: Documents are appended to WAL files in S3
3. **Compaction**: Periodically, WAL files are compacted into segments
4. **Manifest Update**: New manifest version is written
5. **Query**: Query service loads manifest, downloads segments, executes search

## License

MIT
