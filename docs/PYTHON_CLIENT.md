# Tidepool Python Client Design Document

This document specifies the API contract for building a Python client library for Tidepool.

## Overview

Tidepool exposes two HTTP services:
- **Query Service** (default port 8080): Read-only vector search
- **Ingest Service** (default port 8081): Write operations and compaction

Both services use JSON over HTTP with standard REST conventions.

## Base Configuration

```python
class TidepoolClient:
    def __init__(
        self,
        query_url: str = "http://localhost:8080",
        ingest_url: str = "http://localhost:8081",
        timeout: float = 30.0,
        namespace: str = "default",
    ):
        """
        Initialize Tidepool client.
        
        Args:
            query_url: Base URL for query service
            ingest_url: Base URL for ingest service
            timeout: Request timeout in seconds
            namespace: Default namespace for operations
        """
```

## Data Types

### Vector

A vector is a list of 32-bit floating point numbers. All vectors in a namespace must have the same dimensionality.

```python
Vector = List[float]
```

### Attributes

Attributes are arbitrary JSON-compatible metadata attached to vectors. Supported types:

```python
AttrValue = Union[
    None,
    bool,
    int,
    float,
    str,
    List["AttrValue"],
    Dict[str, "AttrValue"],
]
```

### Document

A document represents a single vector with its ID and optional attributes.

```python
@dataclass
class Document:
    id: str                           # Unique identifier (required)
    vector: Vector                    # Vector data (required for upsert)
    attributes: Optional[Dict[str, AttrValue]] = None  # Metadata (optional)
```

### VectorResult

A query result includes the document data plus distance score.

```python
@dataclass
class VectorResult:
    id: str                           # Document ID
    dist: float                       # Distance from query vector
    vector: Optional[Vector] = None   # Only if include_vectors=True
    attributes: Optional[Dict[str, AttrValue]] = None
```

### DistanceMetric

```python
class DistanceMetric(Enum):
    COSINE = "cosine_distance"      # 1 - cosine_similarity, range [0, 2]
    EUCLIDEAN = "euclidean_squared" # Squared L2 distance
    DOT_PRODUCT = "dot_product"     # Negative dot product
```

---

## API Endpoints

### Health Check

Both services expose a health endpoint.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "service": "tidepool-query",
  "status": "healthy"
}
```

**Python Method:**
```python
def health(self, service: str = "query") -> dict:
    """
    Check service health.
    
    Args:
        service: "query" or "ingest"
    
    Returns:
        Health status dict
    
    Raises:
        TidepoolError: If service is unhealthy
    """
```

---

### Upsert Vectors

Insert or update vectors. If a vector with the same ID exists, it is replaced.

**Service:** Ingest  
**Endpoint:** `POST /v1/vectors/{namespace}` or `POST /v1/namespaces/{namespace}`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "vectors": [
    {
      "id": "doc-123",
      "vector": [0.1, 0.2, 0.3, ...],
      "attributes": {
        "title": "Example Document",
        "category": "article",
        "score": 0.95
      }
    }
  ],
  "distance_metric": "cosine_distance"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `vectors` | array | Yes | - | List of documents to upsert |
| `vectors[].id` | string | Yes | - | Unique identifier |
| `vectors[].vector` | float[] | Yes | - | Vector data |
| `vectors[].attributes` | object | No | null | Metadata key-value pairs |
| `distance_metric` | string | No | "cosine_distance" | Distance metric for this batch |

**Response:**
```json
{
  "status": "ok"
}
```

**Python Method:**
```python
def upsert(
    self,
    vectors: List[Document],
    namespace: Optional[str] = None,
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
) -> None:
    """
    Insert or update vectors.
    
    Args:
        vectors: List of documents to upsert
        namespace: Target namespace (uses default if not specified)
        distance_metric: Distance metric for similarity calculations
    
    Raises:
        TidepoolError: On server error
        ValidationError: On invalid input
    """
```

**Batch Considerations:**
- Maximum request body size: 25 MB (configurable via `MAX_BODY_BYTES`)
- Recommended batch size: 100-1000 vectors per request
- Vectors are written to WAL immediately (durable)
- Vectors become queryable after compaction (default: 5 minutes)

---

### Query Vectors

Search for similar vectors using approximate nearest neighbor search.

**Service:** Query  
**Endpoint:** `POST /v1/vectors/{namespace}` or `POST /v1/namespaces/{namespace}/query`  
**Content-Type:** `application/json`

**Request Body:**
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

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `vector` | float[] | Yes | - | Query vector |
| `top_k` | int | No | 10 | Number of results to return |
| `ef_search` | int | No | 100 | HNSW search beam width (higher = better recall) |
| `nprobe` | int | No | 10 | IVF partitions to search (higher = better recall) |
| `distance_metric` | string | No | "cosine_distance" | Distance metric |
| `include_vectors` | bool | No | false | Include vectors in results |
| `filters` | object | No | null | Attribute filters (exact match) |

**Response:**
```json
{
  "results": [
    {
      "id": "doc-456",
      "dist": 0.123,
      "attributes": {
        "title": "Similar Document",
        "category": "article"
      }
    }
  ],
  "namespace": "default"
}
```

**Python Method:**
```python
def query(
    self,
    vector: Vector,
    top_k: int = 10,
    namespace: Optional[str] = None,
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
    include_vectors: bool = False,
    filters: Optional[Dict[str, AttrValue]] = None,
    ef_search: Optional[int] = None,
    nprobe: Optional[int] = None,
) -> List[VectorResult]:
    """
    Search for similar vectors.
    
    Args:
        vector: Query vector
        top_k: Number of results (max: 1000)
        namespace: Target namespace
        distance_metric: Distance metric
        include_vectors: Include vector data in results
        filters: Attribute filters for exact match
        ef_search: HNSW beam width (higher = better recall, slower)
        nprobe: IVF partitions to search (higher = better recall, slower)
    
    Returns:
        List of matching vectors sorted by distance (ascending)
    
    Raises:
        TidepoolError: On server error
        ValidationError: On invalid input
    """
```

**Filter Semantics:**
- Filters use exact match on attribute values
- Multiple filters use AND logic
- Nested objects are matched recursively
- Arrays match if the filter value is contained in the array

---

### Delete Vectors

Delete vectors by ID. Deletions are recorded as tombstones and applied during queries.

**Service:** Ingest  
**Endpoint:** `DELETE /v1/vectors/{namespace}` or `DELETE /v1/namespaces/{namespace}`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "ids": ["doc-123", "doc-456"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ids` | string[] | Yes | List of document IDs to delete |

**Response:**
```json
{
  "status": "ok"
}
```

**Python Method:**
```python
def delete(
    self,
    ids: List[str],
    namespace: Optional[str] = None,
) -> None:
    """
    Delete vectors by ID.
    
    Args:
        ids: List of document IDs to delete
        namespace: Target namespace
    
    Raises:
        TidepoolError: On server error
    """
```

---

### Get Namespace Info

Get information about a namespace including approximate vector count.

**Service:** Query  
**Endpoint:** `GET /v1/namespaces/{namespace}`

**Response:**
```json
{
  "namespace": "default",
  "approx_count": 10000,
  "dimensions": 768
}
```

**Python Method:**
```python
def get_namespace(
    self,
    namespace: Optional[str] = None,
) -> NamespaceInfo:
    """
    Get namespace information.
    
    Args:
        namespace: Namespace name
    
    Returns:
        NamespaceInfo with count and dimensions
    
    Raises:
        TidepoolError: If namespace doesn't exist
    """

@dataclass
class NamespaceInfo:
    namespace: str
    approx_count: int
    dimensions: int
```

---

### List Namespaces

List all available namespaces.

**Service:** Query  
**Endpoint:** `GET /v1/namespaces`

**Response:**
```json
{
  "namespaces": ["default", "embeddings", "images"]
}
```

**Python Method:**
```python
def list_namespaces(self) -> List[str]:
    """
    List all namespaces.
    
    Returns:
        List of namespace names
    """
```

---

### Get Ingest Status

Get compaction and WAL status from the ingest service.

**Service:** Ingest  
**Endpoint:** `GET /v1/namespaces/{namespace}/status`

**Response:**
```json
{
  "last_run": "2024-01-15T10:30:00Z",
  "wal_files": 3,
  "wal_entries": 1500,
  "segments": 12,
  "total_vecs": 150000,
  "dimensions": 768
}
```

**Python Method:**
```python
def status(self, namespace: Optional[str] = None) -> IngestStatus:
    """
    Get ingest service status for a namespace.
    
    Returns:
        IngestStatus with compaction info
    """

@dataclass
class IngestStatus:
    last_run: Optional[datetime]
    wal_files: int
    wal_entries: int
    segments: int
    total_vecs: int
    dimensions: int
```

---

### Trigger Compaction

Manually trigger compaction. Useful for testing or after large batch uploads.

**Service:** Ingest  
**Endpoint:** `POST /v1/namespaces/{namespace}/compact`

**Response:**
```json
{
  "status": "compaction completed"
}
```

**Python Method:**
```python
def compact(self, namespace: Optional[str] = None) -> None:
    """
    Trigger manual compaction.
    
    Blocks until compaction completes. Use for:
    - Testing (make vectors immediately queryable)
    - After large batch uploads
    
    Raises:
        TidepoolError: On compaction failure
    """
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid JSON, missing fields) |
| 404 | Namespace not found |
| 413 | Request body too large |
| 500 | Internal server error |
| 503 | Service unavailable |

### Error Response Format

```json
{
  "error": "description of the error"
}
```

### Python Exceptions

```python
class TidepoolError(Exception):
    """Base exception for Tidepool errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code

class ValidationError(TidepoolError):
    """Invalid input data."""

class NotFoundError(TidepoolError):
    """Resource not found."""

class ServiceUnavailableError(TidepoolError):
    """Service is not available."""
```

---

## Usage Examples

### Basic Usage

```python
from tidepool import TidepoolClient, Document

# Initialize client
client = TidepoolClient(
    query_url="https://query.example.com",
    ingest_url="https://ingest.example.com",
)

# Upsert vectors
documents = [
    Document(
        id="doc-1",
        vector=[0.1, 0.2, 0.3, 0.4],
        attributes={"title": "First Document", "category": "news"},
    ),
    Document(
        id="doc-2",
        vector=[0.5, 0.6, 0.7, 0.8],
        attributes={"title": "Second Document", "category": "blog"},
    ),
]
client.upsert(documents)

# Wait for compaction or trigger manually
client.compact(namespace="default")

# Query
results = client.query(
    vector=[0.1, 0.2, 0.3, 0.4],
    top_k=5,
)

for result in results:
    print(f"{result.id}: {result.dist:.4f}")
```

### Filtering

```python
# Query with filters
results = client.query(
    vector=[0.1, 0.2, 0.3, 0.4],
    top_k=10,
    filters={"category": "news"},
)
```

### Batch Upload

```python
# Efficient batch upload
BATCH_SIZE = 500

for i in range(0, len(all_documents), BATCH_SIZE):
    batch = all_documents[i:i + BATCH_SIZE]
    client.upsert(batch)

# Trigger compaction after large upload
client.compact(namespace="default")
```

### Async Client (Optional)

```python
import asyncio
from tidepool import AsyncTidepoolClient

async def main():
    client = AsyncTidepoolClient(
        query_url="http://localhost:8080",
        ingest_url="http://localhost:8081",
    )
    
    # Async operations
    await client.upsert(documents)
    results = await client.query(vector=[0.1, 0.2, 0.3, 0.4])
    
    await client.close()

asyncio.run(main())
```

---

## Implementation Notes

### Recommended Dependencies

```
httpx>=0.24.0      # HTTP client with async support
pydantic>=2.0      # Data validation (optional)
```

### Connection Pooling

Use connection pooling for better performance:

```python
import httpx

class TidepoolClient:
    def __init__(self, ...):
        self._query_client = httpx.Client(
            base_url=query_url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=10),
        )
        self._ingest_client = httpx.Client(
            base_url=ingest_url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=10),
        )
```

### Retry Logic

Implement exponential backoff for transient failures:

```python
import time
from typing import Callable

def with_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except ServiceUnavailableError:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)
```

### Vector Validation

```python
def validate_vector(vector: List[float], expected_dims: Optional[int] = None) -> None:
    if not vector:
        raise ValidationError("Vector cannot be empty")
    if not all(isinstance(v, (int, float)) for v in vector):
        raise ValidationError("Vector must contain only numbers")
    if expected_dims and len(vector) != expected_dims:
        raise ValidationError(f"Expected {expected_dims} dimensions, got {len(vector)}")
```

---

## Version Compatibility

This document describes the API for Tidepool v1.x. Future versions may add new endpoints or fields but will maintain backward compatibility within the same major version.
