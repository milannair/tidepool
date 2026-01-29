# Tidepool TypeScript Client Design Document

This document specifies the API contract for building a TypeScript client library for Tidepool.

## Overview

Tidepool exposes two HTTP services:
- **Query Service** (default port 8080): Read-only vector search
- **Ingest Service** (default port 8081): Write operations and compaction

Both services use JSON over HTTP with standard REST conventions.

## Installation

```bash
npm install tidepool-client
# or
yarn add tidepool-client
```

## Base Configuration

```typescript
interface TidepoolConfig {
  queryUrl?: string;    // Default: "http://localhost:8080"
  ingestUrl?: string;   // Default: "http://localhost:8081"
  timeout?: number;     // Default: 30000 (ms)
  namespace?: string;   // Default: "default"
}

class TidepoolClient {
  constructor(config?: TidepoolConfig);
}
```

## Data Types

### Vector

A vector is an array of 32-bit floating point numbers. All vectors in a namespace must have the same dimensionality.

```typescript
type Vector = number[];
```

### AttrValue

Attributes are arbitrary JSON-compatible metadata attached to vectors.

```typescript
type AttrValue =
  | null
  | boolean
  | number
  | string
  | AttrValue[]
  | { [key: string]: AttrValue };
```

### Document

A document represents a single vector with its ID and optional attributes.

```typescript
interface Document {
  id: string;                              // Unique identifier (required)
  vector: Vector;                          // Vector data (required for upsert)
  attributes?: Record<string, AttrValue>;  // Metadata (optional)
}
```

### VectorResult

A query result includes the document data plus distance score.

```typescript
interface VectorResult {
  id: string;                              // Document ID
  dist: number;                            // Distance from query vector
  vector?: Vector;                         // Only if includeVectors=true
  attributes?: Record<string, AttrValue>;  // Document attributes
}
```

### DistanceMetric

```typescript
enum DistanceMetric {
  Cosine = "cosine_distance",       // 1 - cosine_similarity, range [0, 2]
  Euclidean = "euclidean_squared",  // Squared L2 distance
  DotProduct = "dot_product",       // Negative dot product
}
```

### QueryResponse

```typescript
interface QueryResponse {
  results: VectorResult[];
  namespace: string;
}
```

### NamespaceInfo

```typescript
interface NamespaceInfo {
  namespace: string;
  approxCount: number;
  dimensions: number;
}
```

### IngestStatus

```typescript
interface IngestStatus {
  lastRun: Date | null;
  walFiles: number;
  walEntries: number;
  segments: number;
  totalVecs: number;
  dimensions: number;
}
```

---

## API Methods

### Health Check

```typescript
async health(service?: "query" | "ingest"): Promise<{ service: string; status: string }>;
```

**HTTP:** `GET /health`

**Example:**
```typescript
const status = await client.health("query");
console.log(status); // { service: "tidepool-query", status: "healthy" }
```

---

### Upsert Vectors

Insert or update vectors. If a vector with the same ID exists, it is replaced.

```typescript
interface UpsertOptions {
  namespace?: string;
  distanceMetric?: DistanceMetric;
}

async upsert(vectors: Document[], options?: UpsertOptions): Promise<void>;
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
```typescript
await client.upsert([
  {
    id: "doc-1",
    vector: [0.1, 0.2, 0.3, 0.4],
    attributes: { title: "First Document", category: "news" },
  },
  {
    id: "doc-2",
    vector: [0.5, 0.6, 0.7, 0.8],
    attributes: { title: "Second Document", category: "blog" },
  },
]);
```

**Batch Considerations:**
- Maximum request body size: 25 MB (configurable)
- Recommended batch size: 100-1000 vectors per request
- Vectors are written to WAL immediately (durable)
- Vectors become queryable after compaction (default: 5 minutes)

---

### Query Vectors

Search for similar vectors using approximate nearest neighbor search.

```typescript
interface QueryOptions {
  topK?: number;              // Default: 10
  namespace?: string;
  distanceMetric?: DistanceMetric;
  includeVectors?: boolean;   // Default: false
  filters?: Record<string, AttrValue>;
  efSearch?: number;          // HNSW beam width
  nprobe?: number;            // IVF partitions to search
}

async query(vector: Vector, options?: QueryOptions): Promise<VectorResult[]>;
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
```typescript
const results = await client.query([0.1, 0.2, 0.3, 0.4], {
  topK: 5,
  filters: { category: "news" },
});

for (const result of results) {
  console.log(`${result.id}: ${result.dist.toFixed(4)}`);
}
```

---

### Delete Vectors

Delete vectors by ID.

```typescript
interface DeleteOptions {
  namespace?: string;
}

async delete(ids: string[], options?: DeleteOptions): Promise<void>;
```

**HTTP:** `DELETE /v1/vectors/{namespace}`

**Request Body:**
```json
{
  "ids": ["doc-123", "doc-456"]
}
```

**Example:**
```typescript
await client.delete(["doc-1", "doc-2"]);
```

---

### Get Namespace Info

```typescript
async getNamespace(namespace?: string): Promise<NamespaceInfo>;
```

**HTTP:** `GET /v1/namespaces/{namespace}`

**Example:**
```typescript
const info = await client.getNamespace();
console.log(`Vectors: ${info.approxCount}, Dimensions: ${info.dimensions}`);
```

---

### List Namespaces

```typescript
async listNamespaces(): Promise<string[]>;
```

**HTTP:** `GET /v1/namespaces`

**Example:**
```typescript
const namespaces = await client.listNamespaces();
console.log(namespaces); // ["default", "embeddings"]
```

---

### Get Ingest Status

```typescript
async status(namespace?: string): Promise<IngestStatus>;
```

**HTTP:** `GET /v1/namespaces/{namespace}/status`

**Example:**
```typescript
const status = await client.status("default");
console.log(`WAL entries: ${status.walEntries}, Segments: ${status.segments}`);
```

---

### Trigger Compaction

```typescript
async compact(namespace?: string): Promise<void>;
```

**HTTP:** `POST /v1/namespaces/{namespace}/compact`

**Example:**
```typescript
// After large batch upload
await client.compact("default");
```

---

## Error Handling

### Error Classes

```typescript
class TidepoolError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: unknown
  ) {
    super(message);
    this.name = "TidepoolError";
  }
}

class ValidationError extends TidepoolError {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

class NotFoundError extends TidepoolError {
  constructor(message: string) {
    super(message, 404);
    this.name = "NotFoundError";
  }
}

class ServiceUnavailableError extends TidepoolError {
  constructor(message: string) {
    super(message, 503);
    this.name = "ServiceUnavailableError";
  }
}
```

### HTTP Status Codes

| Code | Error Class | Description |
|------|-------------|-------------|
| 200 | - | Success |
| 400 | ValidationError | Invalid request |
| 404 | NotFoundError | Namespace not found |
| 413 | ValidationError | Request body too large |
| 500 | TidepoolError | Internal server error |
| 503 | ServiceUnavailableError | Service unavailable |

---

## Usage Examples

### Basic Usage

```typescript
import { TidepoolClient, Document, DistanceMetric } from "tidepool-client";

const client = new TidepoolClient({
  queryUrl: "https://query.example.com",
  ingestUrl: "https://ingest.example.com",
});

// Upsert vectors
const documents: Document[] = [
  {
    id: "doc-1",
    vector: [0.1, 0.2, 0.3, 0.4],
    attributes: { title: "First Document" },
  },
];
await client.upsert(documents);

// Trigger compaction
await client.compact("default");

// Query
const results = await client.query([0.1, 0.2, 0.3, 0.4], { topK: 5 });
console.log(results);
```

### With Filtering

```typescript
const results = await client.query([0.1, 0.2, 0.3, 0.4], {
  topK: 10,
  filters: {
    category: "news",
    published: true,
  },
});
```

### Batch Upload with Progress

```typescript
const BATCH_SIZE = 500;
const allDocuments: Document[] = [...]; // Large array

for (let i = 0; i < allDocuments.length; i += BATCH_SIZE) {
  const batch = allDocuments.slice(i, i + BATCH_SIZE);
  await client.upsert(batch);
  console.log(`Uploaded ${Math.min(i + BATCH_SIZE, allDocuments.length)}/${allDocuments.length}`);
}

await client.compact("default");
```

### Error Handling

```typescript
import { TidepoolError, NotFoundError, ServiceUnavailableError } from "tidepool-client";

try {
  const results = await client.query([0.1, 0.2, 0.3]);
} catch (error) {
  if (error instanceof NotFoundError) {
    console.error("Namespace not found");
  } else if (error instanceof ServiceUnavailableError) {
    console.error("Service temporarily unavailable, retrying...");
  } else if (error instanceof TidepoolError) {
    console.error(`Tidepool error: ${error.message}`);
  } else {
    throw error;
  }
}
```

### React Hook Example

```typescript
import { useState, useEffect } from "react";
import { TidepoolClient, VectorResult } from "tidepool-client";

const client = new TidepoolClient();

function useVectorSearch(queryVector: number[] | null, topK = 10) {
  const [results, setResults] = useState<VectorResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!queryVector) return;

    setLoading(true);
    setError(null);

    client
      .query(queryVector, { topK })
      .then(setResults)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [queryVector, topK]);

  return { results, loading, error };
}
```

---

## Implementation Notes

### Recommended Setup

```typescript
// Using fetch (Node.js 18+ or browser)
class TidepoolClient {
  private queryUrl: string;
  private ingestUrl: string;
  private timeout: number;
  private namespace: string;

  constructor(config: TidepoolConfig = {}) {
    this.queryUrl = config.queryUrl ?? "http://localhost:8080";
    this.ingestUrl = config.ingestUrl ?? "http://localhost:8081";
    this.timeout = config.timeout ?? 30000;
    this.namespace = config.namespace ?? "default";
  }

  private async request<T>(
    baseUrl: string,
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${baseUrl}${path}`, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new TidepoolError(
          body.error ?? response.statusText,
          response.status,
          body
        );
      }

      return response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
```

### Retry with Exponential Backoff

```typescript
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 500
): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (
        attempt === maxRetries ||
        !(error instanceof ServiceUnavailableError)
      ) {
        throw error;
      }
      const delay = Math.min(baseDelay * Math.pow(2, attempt), 10000);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error("Unreachable");
}
```

### Vector Validation

```typescript
function validateVector(vector: unknown, expectedDims?: number): asserts vector is Vector {
  if (!Array.isArray(vector)) {
    throw new ValidationError("Vector must be an array");
  }
  if (vector.length === 0) {
    throw new ValidationError("Vector cannot be empty");
  }
  if (!vector.every((v) => typeof v === "number" && Number.isFinite(v))) {
    throw new ValidationError("Vector must contain only finite numbers");
  }
  if (expectedDims !== undefined && vector.length !== expectedDims) {
    throw new ValidationError(
      `Expected ${expectedDims} dimensions, got ${vector.length}`
    );
  }
}
```

---

## Version Compatibility

This document describes the API for Tidepool v1.x. Future versions may add new endpoints or fields but will maintain backward compatibility within the same major version.
