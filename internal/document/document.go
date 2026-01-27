// Package document defines the core document structure for Tidepool.
package document

import "time"

// Document represents a searchable document in Tidepool.
type Document struct {
	ID        string                 `json:"id" parquet:"id"`
	Content   string                 `json:"content" parquet:"content"`
	Title     string                 `json:"title,omitempty" parquet:"title,optional"`
	URL       string                 `json:"url,omitempty" parquet:"url,optional"`
	Tags      []string               `json:"tags,omitempty" parquet:"tags,list"`
	Metadata  map[string]interface{} `json:"metadata,omitempty" parquet:"-"`
	CreatedAt time.Time              `json:"created_at" parquet:"created_at"`
	UpdatedAt time.Time              `json:"updated_at" parquet:"updated_at"`
}

// SearchResult represents a document with its search score.
type SearchResult struct {
	Document *Document `json:"document"`
	Score    float64   `json:"score"`
}

// SearchResponse represents the response from a search query.
type SearchResponse struct {
	Results    []SearchResult `json:"results"`
	TotalHits  int            `json:"total_hits"`
	TookMs     int64          `json:"took_ms"`
	Query      string         `json:"query"`
}

// IngestRequest represents a request to ingest documents.
type IngestRequest struct {
	Documents []Document `json:"documents"`
}

// IngestResponse represents the response from an ingest request.
type IngestResponse struct {
	Ingested int    `json:"ingested"`
	WALFile  string `json:"wal_file"`
}

// SearchRequest represents a search query.
type SearchRequest struct {
	Query   string   `json:"query"`
	Filters Filters  `json:"filters,omitempty"`
	Limit   int      `json:"limit,omitempty"`
	Offset  int      `json:"offset,omitempty"`
}

// Filters represents search filters.
type Filters struct {
	Tags      []string `json:"tags,omitempty"`
	DateFrom  string   `json:"date_from,omitempty"`
	DateTo    string   `json:"date_to,omitempty"`
}
