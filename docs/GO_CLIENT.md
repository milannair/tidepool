# Tidepool Go Client Design Document

This document specifies the API contract for building a Go client library for Tidepool.

## Overview

Tidepool exposes two HTTP services:
- **Query Service** (default port 8080): Read-only vector search
- **Ingest Service** (default port 8081): Write operations and compaction

Both services use JSON over HTTP with standard REST conventions.

## Installation

```bash
go get github.com/your-org/tidepool-go
```

## Package Structure

```
tidepool/
├── client.go       // Main client implementation
├── types.go        // Data types and enums
├── errors.go       // Error types
└── options.go      // Functional options
```

## Data Types

### Vector

A vector is a slice of 32-bit floating point numbers.

```go
type Vector []float32
```

### AttrValue

Attributes use Go's `any` type for JSON compatibility.

```go
// AttrValue represents any JSON-compatible value
type AttrValue = any

// Attributes is a map of string keys to JSON values
type Attributes map[string]AttrValue
```

### Document

```go
// Document represents a vector with metadata
type Document struct {
    ID         string     `json:"id"`
    Vector     Vector     `json:"vector,omitempty"`
    Attributes Attributes `json:"attributes,omitempty"`
}
```

### VectorResult

```go
// VectorResult is a single query result
type VectorResult struct {
    ID         string     `json:"id"`
    Dist       float32    `json:"dist"`
    Vector     Vector     `json:"vector,omitempty"`
    Attributes Attributes `json:"attributes,omitempty"`
}
```

### DistanceMetric

```go
type DistanceMetric string

const (
    DistanceCosine    DistanceMetric = "cosine_distance"
    DistanceEuclidean DistanceMetric = "euclidean_squared"
    DistanceDotProduct DistanceMetric = "dot_product"
)
```

### NamespaceInfo

```go
type NamespaceInfo struct {
    Namespace   string `json:"namespace"`
    ApproxCount int64  `json:"approx_count"`
    Dimensions  int    `json:"dimensions"`
}
```

### IngestStatus

```go
type IngestStatus struct {
    LastRun    *time.Time `json:"last_run,omitempty"`
    WALFiles   int        `json:"wal_files"`
    WALEntries int        `json:"wal_entries"`
    Segments   int        `json:"segments"`
    TotalVecs  int        `json:"total_vecs"`
    Dimensions int        `json:"dimensions"`
}
```

---

## Client Configuration

```go
// Config holds client configuration
type Config struct {
    QueryURL   string        // Default: "http://localhost:8080"
    IngestURL  string        // Default: "http://localhost:8081"
    Timeout    time.Duration // Default: 30s
    Namespace  string        // Default: "default"
    HTTPClient *http.Client  // Optional custom HTTP client
}

// Client is the Tidepool API client
type Client struct {
    config Config
    http   *http.Client
}

// New creates a new Tidepool client
func New(opts ...Option) *Client

// Option is a functional option for configuring the client
type Option func(*Config)

func WithQueryURL(url string) Option
func WithIngestURL(url string) Option
func WithTimeout(d time.Duration) Option
func WithNamespace(ns string) Option
func WithHTTPClient(c *http.Client) Option
```

**Example:**
```go
client := tidepool.New(
    tidepool.WithQueryURL("https://query.example.com"),
    tidepool.WithIngestURL("https://ingest.example.com"),
    tidepool.WithTimeout(60 * time.Second),
)
```

---

## API Methods

### Health Check

```go
// HealthResponse contains service health information
type HealthResponse struct {
    Service string `json:"service"`
    Status  string `json:"status"`
}

// Health checks service health
// service should be "query" or "ingest"
func (c *Client) Health(ctx context.Context, service string) (*HealthResponse, error)
```

**HTTP:** `GET /health`

**Example:**
```go
health, err := client.Health(ctx, "query")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Service: %s, Status: %s\n", health.Service, health.Status)
```

---

### Upsert Vectors

```go
// UpsertOptions configures upsert behavior
type UpsertOptions struct {
    Namespace      string
    DistanceMetric DistanceMetric
}

// Upsert inserts or updates vectors
func (c *Client) Upsert(ctx context.Context, docs []Document, opts *UpsertOptions) error
```

**HTTP:** `POST /v1/vectors/{namespace}`

**Request Body:**
```json
{
  "vectors": [
    {
      "id": "doc-123",
      "vector": [0.1, 0.2, 0.3],
      "attributes": { "title": "Example" }
    }
  ],
  "distance_metric": "cosine_distance"
}
```

**Example:**
```go
docs := []tidepool.Document{
    {
        ID:     "doc-1",
        Vector: []float32{0.1, 0.2, 0.3, 0.4},
        Attributes: map[string]any{
            "title":    "First Document",
            "category": "news",
        },
    },
    {
        ID:     "doc-2",
        Vector: []float32{0.5, 0.6, 0.7, 0.8},
        Attributes: map[string]any{
            "title":    "Second Document",
            "category": "blog",
        },
    },
}

err := client.Upsert(ctx, docs, nil)
if err != nil {
    log.Fatal(err)
}
```

**Batch Considerations:**
- Maximum request body size: 25 MB (configurable)
- Recommended batch size: 100-1000 vectors per request
- Vectors become queryable after compaction (default: 5 minutes)

---

### Query Vectors

```go
// QueryOptions configures query behavior
type QueryOptions struct {
    TopK           int
    Namespace      string
    DistanceMetric DistanceMetric
    IncludeVectors bool
    Filters        Attributes
    EfSearch       int // HNSW beam width
    NProbe         int // IVF partitions to search
}

// Query searches for similar vectors
func (c *Client) Query(ctx context.Context, vector Vector, opts *QueryOptions) ([]VectorResult, error)
```

**HTTP:** `POST /v1/vectors/{namespace}`

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 10,
  "ef_search": 100,
  "nprobe": 10,
  "distance_metric": "cosine_distance",
  "include_vectors": false,
  "filters": { "category": "news" }
}
```

**Example:**
```go
results, err := client.Query(ctx, []float32{0.1, 0.2, 0.3, 0.4}, &tidepool.QueryOptions{
    TopK: 5,
    Filters: map[string]any{
        "category": "news",
    },
})
if err != nil {
    log.Fatal(err)
}

for _, r := range results {
    fmt.Printf("%s: %.4f\n", r.ID, r.Dist)
}
```

---

### Delete Vectors

```go
// DeleteOptions configures delete behavior
type DeleteOptions struct {
    Namespace string
}

// Delete removes vectors by ID
func (c *Client) Delete(ctx context.Context, ids []string, opts *DeleteOptions) error
```

**HTTP:** `DELETE /v1/vectors/{namespace}`

**Request Body:**
```json
{
  "ids": ["doc-123", "doc-456"]
}
```

**Example:**
```go
err := client.Delete(ctx, []string{"doc-1", "doc-2"}, nil)
if err != nil {
    log.Fatal(err)
}
```

---

### Get Namespace Info

```go
// GetNamespace returns namespace information
func (c *Client) GetNamespace(ctx context.Context, namespace string) (*NamespaceInfo, error)
```

**HTTP:** `GET /v1/namespaces/{namespace}`

**Example:**
```go
info, err := client.GetNamespace(ctx, "default")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Vectors: %d, Dimensions: %d\n", info.ApproxCount, info.Dimensions)
```

---

### List Namespaces

```go
// ListNamespaces returns all namespace names
func (c *Client) ListNamespaces(ctx context.Context) ([]string, error)
```

**HTTP:** `GET /v1/namespaces`

**Example:**
```go
namespaces, err := client.ListNamespaces(ctx)
if err != nil {
    log.Fatal(err)
}
fmt.Println(namespaces) // ["default", "embeddings"]
```

---

### Get Ingest Status

```go
// Status returns ingest service status
func (c *Client) Status(ctx context.Context) (*IngestStatus, error)
```

**HTTP:** `GET /status`

**Example:**
```go
status, err := client.Status(ctx)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("WAL entries: %d, Segments: %d\n", status.WALEntries, status.Segments)
```

---

### Trigger Compaction

```go
// Compact triggers manual compaction
func (c *Client) Compact(ctx context.Context) error
```

**HTTP:** `POST /compact`

**Example:**
```go
err := client.Compact(ctx)
if err != nil {
    log.Fatal(err)
}
```

---

## Error Handling

### Error Types

```go
// TidepoolError is the base error type
type TidepoolError struct {
    Message    string
    StatusCode int
    Response   []byte
}

func (e *TidepoolError) Error() string {
    return e.Message
}

// Sentinel errors for type checking
var (
    ErrValidation       = errors.New("validation error")
    ErrNotFound         = errors.New("not found")
    ErrServiceUnavailable = errors.New("service unavailable")
)

// IsValidationError checks if err is a validation error
func IsValidationError(err error) bool

// IsNotFoundError checks if err is a not found error
func IsNotFoundError(err error) bool

// IsServiceUnavailable checks if err is a service unavailable error
func IsServiceUnavailable(err error) bool
```

### HTTP Status Codes

| Code | Error | Description |
|------|-------|-------------|
| 200 | nil | Success |
| 400 | ErrValidation | Invalid request |
| 404 | ErrNotFound | Namespace not found |
| 413 | ErrValidation | Request body too large |
| 500 | TidepoolError | Internal server error |
| 503 | ErrServiceUnavailable | Service unavailable |

**Example:**
```go
results, err := client.Query(ctx, vector, nil)
if err != nil {
    if tidepool.IsNotFoundError(err) {
        log.Println("Namespace not found")
        return
    }
    if tidepool.IsServiceUnavailable(err) {
        log.Println("Service temporarily unavailable")
        return
    }
    log.Fatalf("Query failed: %v", err)
}
```

---

## Usage Examples

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/your-org/tidepool-go"
)

func main() {
    ctx := context.Background()

    client := tidepool.New(
        tidepool.WithQueryURL("https://query.example.com"),
        tidepool.WithIngestURL("https://ingest.example.com"),
    )

    // Upsert vectors
    docs := []tidepool.Document{
        {
            ID:     "doc-1",
            Vector: []float32{0.1, 0.2, 0.3, 0.4},
            Attributes: map[string]any{
                "title": "First Document",
            },
        },
    }
    if err := client.Upsert(ctx, docs, nil); err != nil {
        log.Fatal(err)
    }

    // Trigger compaction
    if err := client.Compact(ctx); err != nil {
        log.Fatal(err)
    }

    // Query
    results, err := client.Query(ctx, []float32{0.1, 0.2, 0.3, 0.4}, &tidepool.QueryOptions{
        TopK: 5,
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, r := range results {
        fmt.Printf("%s: %.4f\n", r.ID, r.Dist)
    }
}
```

### Batch Upload with Concurrency

```go
func uploadBatch(ctx context.Context, client *tidepool.Client, docs []tidepool.Document) error {
    const batchSize = 500
    const maxConcurrency = 4

    sem := make(chan struct{}, maxConcurrency)
    errCh := make(chan error, 1)
    var wg sync.WaitGroup

    for i := 0; i < len(docs); i += batchSize {
        end := i + batchSize
        if end > len(docs) {
            end = len(docs)
        }
        batch := docs[i:end]

        wg.Add(1)
        sem <- struct{}{}

        go func(batch []tidepool.Document) {
            defer wg.Done()
            defer func() { <-sem }()

            if err := client.Upsert(ctx, batch, nil); err != nil {
                select {
                case errCh <- err:
                default:
                }
            }
        }(batch)
    }

    wg.Wait()
    close(errCh)

    if err := <-errCh; err != nil {
        return err
    }

    return client.Compact(ctx)
}
```

### Retry with Exponential Backoff

```go
func withRetry[T any](ctx context.Context, maxRetries int, fn func() (T, error)) (T, error) {
    var result T
    var err error

    for attempt := 0; attempt <= maxRetries; attempt++ {
        result, err = fn()
        if err == nil {
            return result, nil
        }

        if !tidepool.IsServiceUnavailable(err) {
            return result, err
        }

        if attempt < maxRetries {
            delay := time.Duration(1<<attempt) * 500 * time.Millisecond
            if delay > 10*time.Second {
                delay = 10 * time.Second
            }

            select {
            case <-ctx.Done():
                return result, ctx.Err()
            case <-time.After(delay):
            }
        }
    }

    return result, err
}

// Usage
results, err := withRetry(ctx, 3, func() ([]tidepool.VectorResult, error) {
    return client.Query(ctx, vector, nil)
})
```

### Custom HTTP Client

```go
// With custom transport for connection pooling
transport := &http.Transport{
    MaxIdleConns:        100,
    MaxIdleConnsPerHost: 10,
    IdleConnTimeout:     90 * time.Second,
}

httpClient := &http.Client{
    Transport: transport,
    Timeout:   30 * time.Second,
}

client := tidepool.New(
    tidepool.WithHTTPClient(httpClient),
)
```

---

## Implementation Notes

### Request Helper

```go
func (c *Client) doRequest(ctx context.Context, method, url string, body any) ([]byte, error) {
    var reqBody io.Reader
    if body != nil {
        data, err := json.Marshal(body)
        if err != nil {
            return nil, fmt.Errorf("marshal request: %w", err)
        }
        reqBody = bytes.NewReader(data)
    }

    req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
    if err != nil {
        return nil, fmt.Errorf("create request: %w", err)
    }

    if body != nil {
        req.Header.Set("Content-Type", "application/json")
    }

    resp, err := c.http.Do(req)
    if err != nil {
        return nil, fmt.Errorf("do request: %w", err)
    }
    defer resp.Body.Close()

    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("read response: %w", err)
    }

    if resp.StatusCode >= 400 {
        return nil, c.handleError(resp.StatusCode, respBody)
    }

    return respBody, nil
}

func (c *Client) handleError(statusCode int, body []byte) error {
    var errResp struct {
        Error string `json:"error"`
    }
    json.Unmarshal(body, &errResp)

    msg := errResp.Error
    if msg == "" {
        msg = http.StatusText(statusCode)
    }

    err := &TidepoolError{
        Message:    msg,
        StatusCode: statusCode,
        Response:   body,
    }

    switch statusCode {
    case 400, 413:
        return fmt.Errorf("%w: %s", ErrValidation, msg)
    case 404:
        return fmt.Errorf("%w: %s", ErrNotFound, msg)
    case 503:
        return fmt.Errorf("%w: %s", ErrServiceUnavailable, msg)
    default:
        return err
    }
}
```

### Vector Validation

```go
func ValidateVector(v Vector, expectedDims int) error {
    if len(v) == 0 {
        return fmt.Errorf("%w: vector cannot be empty", ErrValidation)
    }
    if expectedDims > 0 && len(v) != expectedDims {
        return fmt.Errorf("%w: expected %d dimensions, got %d", ErrValidation, expectedDims, len(v))
    }
    for i, val := range v {
        if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
            return fmt.Errorf("%w: invalid value at index %d", ErrValidation, i)
        }
    }
    return nil
}
```

---

## Version Compatibility

This document describes the API for Tidepool v1.x. Future versions may add new endpoints or fields but will maintain backward compatibility within the same major version.
