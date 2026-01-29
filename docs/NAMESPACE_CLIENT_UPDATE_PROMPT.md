# Agent Task: Update Client SDKs for Dynamic Namespace Support

## Overview

Tidepool now supports dynamic namespaces (Phase 8). Update the Python, TypeScript, and Go client SDKs to support the new namespace features. The clients should allow users to work with multiple namespaces from a single client instance.

## Current Client Docs Location

- Python: `docs/PYTHON_CLIENT.md`
- TypeScript: `docs/TYPESCRIPT_CLIENT.md`
- Go: `docs/GO_CLIENT.md`

## API Changes Summary

### Namespace is Now a Per-Request Parameter

Previously, namespace was configured at client initialization. Now it should be specifiable per-request, with an optional default.

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/namespaces/:namespace/status` | Get namespace status |
| POST | `/v1/namespaces/:namespace/compact` | Trigger compaction for namespace |

### Existing Endpoints (unchanged paths, namespace is dynamic)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/vectors/:namespace` | Upsert vectors |
| DELETE | `/v1/vectors/:namespace` | Delete vectors |
| POST | `/v1/vectors/:namespace` (query service) | Query vectors |

## Required Client Changes

### 1. Constructor Changes

**Before:**
```python
client = TidepoolClient(
    query_url="http://localhost:8080",
    ingest_url="http://localhost:8081",
    namespace="default"  # Fixed namespace
)
```

**After:**
```python
client = TidepoolClient(
    query_url="http://localhost:8080",
    ingest_url="http://localhost:8081",
    default_namespace="default"  # Optional default, can be overridden per-request
)
```

### 2. Method Signature Changes

All methods should accept an optional `namespace` parameter that overrides the default:

```python
# Upsert - namespace optional, uses default if not provided
client.upsert(vectors=[...], namespace="products")
client.upsert(vectors=[...])  # Uses default_namespace

# Query - namespace optional
client.query(vector=[...], top_k=10, namespace="products")
client.query(vector=[...], top_k=10)  # Uses default_namespace

# Delete - namespace optional
client.delete(ids=["id1", "id2"], namespace="products")
client.delete(ids=["id1", "id2"])  # Uses default_namespace
```

### 3. New Methods to Add

```python
# Get namespace status
def get_namespace_status(self, namespace: str = None) -> NamespaceStatus:
    """
    Get status information for a namespace.
    
    Returns:
        NamespaceStatus with fields:
        - last_run: Optional[datetime] - Last compaction time
        - wal_files: int - Number of WAL files pending
        - wal_entries: int - Number of WAL entries pending
        - segments: int - Number of compacted segments
        - total_vecs: int - Total vector count
        - dimensions: int - Vector dimensions (0 if unknown)
    """

# Trigger compaction
def compact(self, namespace: str = None) -> None:
    """
    Trigger compaction for a namespace.
    Compaction converts WAL entries into indexed segments.
    """
```

### 4. Response Model Updates

Add `NamespaceStatus` model:

```python
@dataclass
class NamespaceStatus:
    last_run: Optional[datetime]
    wal_files: int
    wal_entries: int
    segments: int
    total_vecs: int
    dimensions: int
```

Query response now includes namespace:

```python
@dataclass
class QueryResponse:
    results: List[VectorResult]
    namespace: str  # NEW: Returns the namespace that was queried
```

## Implementation Examples

### Python

```python
class TidepoolClient:
    def __init__(
        self,
        query_url: str = "http://localhost:8080",
        ingest_url: str = "http://localhost:8081",
        default_namespace: str = "default",
        timeout: float = 30.0
    ):
        self.query_url = query_url.rstrip("/")
        self.ingest_url = ingest_url.rstrip("/")
        self.default_namespace = default_namespace
        self.timeout = timeout

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        return namespace if namespace is not None else self.default_namespace

    def upsert(
        self,
        vectors: List[Dict],
        namespace: Optional[str] = None
    ) -> None:
        ns = self._resolve_namespace(namespace)
        response = requests.post(
            f"{self.ingest_url}/v1/vectors/{ns}",
            json={"vectors": vectors},
            timeout=self.timeout
        )
        response.raise_for_status()

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None
    ) -> QueryResponse:
        ns = self._resolve_namespace(namespace)
        payload = {"vector": vector, "top_k": top_k}
        if filter:
            payload["filter"] = filter
        response = requests.post(
            f"{self.query_url}/v1/vectors/{ns}",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return QueryResponse(**response.json())

    def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> None:
        ns = self._resolve_namespace(namespace)
        response = requests.delete(
            f"{self.ingest_url}/v1/vectors/{ns}",
            json={"ids": ids},
            timeout=self.timeout
        )
        response.raise_for_status()

    def get_namespace_status(
        self,
        namespace: Optional[str] = None
    ) -> NamespaceStatus:
        ns = self._resolve_namespace(namespace)
        response = requests.get(
            f"{self.ingest_url}/v1/namespaces/{ns}/status",
            timeout=self.timeout
        )
        response.raise_for_status()
        return NamespaceStatus(**response.json())

    def compact(
        self,
        namespace: Optional[str] = None
    ) -> None:
        ns = self._resolve_namespace(namespace)
        response = requests.post(
            f"{self.ingest_url}/v1/namespaces/{ns}/compact",
            timeout=self.timeout
        )
        response.raise_for_status()
```

### TypeScript

```typescript
interface TidepoolClientOptions {
  queryUrl?: string;
  ingestUrl?: string;
  defaultNamespace?: string;
  timeout?: number;
}

class TidepoolClient {
  private queryUrl: string;
  private ingestUrl: string;
  private defaultNamespace: string;
  private timeout: number;

  constructor(options: TidepoolClientOptions = {}) {
    this.queryUrl = (options.queryUrl ?? "http://localhost:8080").replace(/\/$/, "");
    this.ingestUrl = (options.ingestUrl ?? "http://localhost:8081").replace(/\/$/, "");
    this.defaultNamespace = options.defaultNamespace ?? "default";
    this.timeout = options.timeout ?? 30000;
  }

  private resolveNamespace(namespace?: string): string {
    return namespace ?? this.defaultNamespace;
  }

  async upsert(vectors: Vector[], namespace?: string): Promise<void> {
    const ns = this.resolveNamespace(namespace);
    await fetch(`${this.ingestUrl}/v1/vectors/${ns}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vectors }),
    });
  }

  async query(
    vector: number[],
    topK: number = 10,
    options?: { namespace?: string; filter?: Filter }
  ): Promise<QueryResponse> {
    const ns = this.resolveNamespace(options?.namespace);
    const response = await fetch(`${this.queryUrl}/v1/vectors/${ns}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vector, top_k: topK, filter: options?.filter }),
    });
    return response.json();
  }

  async delete(ids: string[], namespace?: string): Promise<void> {
    const ns = this.resolveNamespace(namespace);
    await fetch(`${this.ingestUrl}/v1/vectors/${ns}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
  }

  async getNamespaceStatus(namespace?: string): Promise<NamespaceStatus> {
    const ns = this.resolveNamespace(namespace);
    const response = await fetch(`${this.ingestUrl}/v1/namespaces/${ns}/status`);
    return response.json();
  }

  async compact(namespace?: string): Promise<void> {
    const ns = this.resolveNamespace(namespace);
    await fetch(`${this.ingestUrl}/v1/namespaces/${ns}/compact`, {
      method: "POST",
    });
  }
}
```

### Go

```go
type ClientOptions struct {
    QueryURL         string
    IngestURL        string
    DefaultNamespace string
    Timeout          time.Duration
}

type Client struct {
    queryURL         string
    ingestURL        string
    defaultNamespace string
    httpClient       *http.Client
}

func NewClient(opts ClientOptions) *Client {
    if opts.DefaultNamespace == "" {
        opts.DefaultNamespace = "default"
    }
    if opts.Timeout == 0 {
        opts.Timeout = 30 * time.Second
    }
    return &Client{
        queryURL:         strings.TrimSuffix(opts.QueryURL, "/"),
        ingestURL:        strings.TrimSuffix(opts.IngestURL, "/"),
        defaultNamespace: opts.DefaultNamespace,
        httpClient:       &http.Client{Timeout: opts.Timeout},
    }
}

func (c *Client) resolveNamespace(namespace string) string {
    if namespace == "" {
        return c.defaultNamespace
    }
    return namespace
}

func (c *Client) Upsert(ctx context.Context, vectors []Vector, namespace string) error {
    ns := c.resolveNamespace(namespace)
    url := fmt.Sprintf("%s/v1/vectors/%s", c.ingestURL, ns)
    // ... implementation
}

func (c *Client) Query(ctx context.Context, vector []float32, topK int, namespace string) (*QueryResponse, error) {
    ns := c.resolveNamespace(namespace)
    url := fmt.Sprintf("%s/v1/vectors/%s", c.queryURL, ns)
    // ... implementation
}

func (c *Client) Delete(ctx context.Context, ids []string, namespace string) error {
    ns := c.resolveNamespace(namespace)
    url := fmt.Sprintf("%s/v1/vectors/%s", c.ingestURL, ns)
    // ... implementation
}

func (c *Client) GetNamespaceStatus(ctx context.Context, namespace string) (*NamespaceStatus, error) {
    ns := c.resolveNamespace(namespace)
    url := fmt.Sprintf("%s/v1/namespaces/%s/status", c.ingestURL, ns)
    // ... implementation
}

func (c *Client) Compact(ctx context.Context, namespace string) error {
    ns := c.resolveNamespace(namespace)
    url := fmt.Sprintf("%s/v1/namespaces/%s/compact", c.ingestURL, ns)
    // ... implementation
}
```

## Usage Examples to Include in Docs

### Multi-Tenant Application

```python
client = TidepoolClient(ingest_url="...", query_url="...")

# Each tenant gets their own namespace
def index_tenant_data(tenant_id: str, documents: List[Dict]):
    vectors = [embed(doc) for doc in documents]
    client.upsert(vectors, namespace=f"tenant_{tenant_id}")

def search_tenant(tenant_id: str, query: str, top_k: int = 10):
    query_vec = embed(query)
    return client.query(query_vec, top_k=top_k, namespace=f"tenant_{tenant_id}")
```

### Different Data Types

```python
client = TidepoolClient(default_namespace="products")

# Index different types of data in separate namespaces
client.upsert(product_vectors, namespace="products")
client.upsert(user_vectors, namespace="users")
client.upsert(doc_vectors, namespace="documents")

# Query specific namespace
results = client.query(query_vec, namespace="products")

# Check namespace status
status = client.get_namespace_status("products")
print(f"Products: {status.total_vecs} vectors in {status.segments} segments")
```

### Namespace Management

```python
# Check if namespace needs compaction
status = client.get_namespace_status("products")
if status.wal_entries > 1000:
    client.compact("products")
    print("Compaction triggered")
```

## Testing Requirements

Update client tests to verify:

1. **Namespace parameter works on all methods**
   - Upsert with explicit namespace
   - Query with explicit namespace
   - Delete with explicit namespace

2. **Default namespace fallback**
   - Operations without namespace use default

3. **New endpoints**
   - `get_namespace_status()` returns valid data
   - `compact()` succeeds

4. **Cross-namespace isolation**
   - Vectors in namespace A not visible in namespace B

## Documentation Updates

1. Update constructor docs to show `default_namespace` parameter
2. Add namespace parameter to all method signatures
3. Add new methods: `get_namespace_status()`, `compact()`
4. Add usage examples for multi-tenant scenarios
5. Update response models to include namespace field

## Error Handling

Handle new error case:
- `404 Not Found` with `{"error": "namespace not found"}` when namespace is restricted by `ALLOWED_NAMESPACES` config
