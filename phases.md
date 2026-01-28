# Tidepool Development Phases

## Phase 1: Foundation (Complete)

- [x] Core architecture with stateless query and ingest services
- [x] Object storage (S3-compatible) as source of truth
- [x] Write-ahead log (WAL) for durable writes
- [x] Basic segment storage format
- [x] Brute-force vector similarity search
- [x] Support for cosine, euclidean, and dot product distances
- [x] Attribute filtering on queries
- [x] Background compaction worker
- [x] Railway deployment support

## Phase 2: Performance & Indexing

- [x] HNSW approximate nearest neighbor index
- [x] Incremental compaction (avoid full rebuilds)
- [x] Segment caching optimizations
- [x] Parallel query execution
- [x] Batch upsert optimizations
- [x] Memory-mapped segment access

## Phase 3: Scalability

- [ ] Multi-namespace support per deployment
- [ ] Streaming/pagination for large result sets
- [ ] Sharded segments for horizontal scaling
- [ ] Query result caching layer
- [ ] Async segment prefetching

## Phase 4: Advanced Features

- [ ] Hybrid search (vector + full-text)
- [ ] Metadata indexing for faster filtering
- [ ] Vector quantization (PQ, SQ) for compression
- [ ] Multi-vector documents
- [ ] Namespace-level configuration

## Phase 5: Operations & Observability

- [ ] Metrics export (Prometheus)
- [ ] Distributed tracing
- [ ] Admin API for namespace management
- [ ] Backup/restore utilities
- [ ] Cost estimation endpoints
