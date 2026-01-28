// Package document defines the core document structure for Tidepool vector database.
package document

// Vector represents a dense vector embedding.
type Vector []float32

// Document represents a vector with associated metadata in Tidepool.
type Document struct {
	ID         string                 `json:"id"`
	Vector     Vector                 `json:"vector,omitempty"`
	Attributes map[string]interface{} `json:"attributes,omitempty"`
}

// VectorResult represents a document with its similarity score.
type VectorResult struct {
	ID         string                 `json:"id"`
	Vector     Vector                 `json:"vector,omitempty"`
	Attributes map[string]interface{} `json:"attributes,omitempty"`
	Dist       float32                `json:"dist"`
}

// QueryRequest represents a vector query.
type QueryRequest struct {
	Vector         Vector                 `json:"vector"`
	TopK           int                    `json:"top_k,omitempty"`
	DistanceMetric string                 `json:"distance_metric,omitempty"` // cosine, euclidean, dot_product
	IncludeVectors bool                   `json:"include_vectors,omitempty"`
	Filters        map[string]interface{} `json:"filters,omitempty"`
}

// QueryResponse represents the response from a vector query.
type QueryResponse struct {
	Results   []VectorResult `json:"results"`
	Namespace string         `json:"namespace"`
}

// UpsertRequest represents a request to upsert vectors.
type UpsertRequest struct {
	Vectors        []Document `json:"vectors"`
	DistanceMetric string     `json:"distance_metric,omitempty"`
}

// UpsertResponse represents the response from an upsert request.
type UpsertResponse struct {
	Status string `json:"status"`
}

// DeleteRequest represents a request to delete vectors.
type DeleteRequest struct {
	IDs []string `json:"ids"`
}

// DeleteResponse represents the response from a delete request.
type DeleteResponse struct {
	Status string `json:"status"`
}

// NamespaceInfo contains information about a namespace.
type NamespaceInfo struct {
	Namespace   string `json:"namespace"`
	ApproxCount int64  `json:"approx_count"`
	Dimensions  int    `json:"dimensions"`
}
