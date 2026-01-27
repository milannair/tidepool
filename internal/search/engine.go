// Package search implements the search engine for Tidepool.
package search

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
)

// Engine performs search operations across all segments.
type Engine struct {
	storage         *storage.Client
	namespace       string
	cacheDir        string
	manifestManager *manifest.Manager
	segmentReader   *segment.Reader

	mu              sync.RWMutex
	currentManifest *manifest.Manifest
	loadedSegments  map[string][]*document.Document
	loadedIndexes   map[string]*segment.Index
}

// NewEngine creates a new search engine.
func NewEngine(storage *storage.Client, namespace, cacheDir string) *Engine {
	return &Engine{
		storage:         storage,
		namespace:       namespace,
		cacheDir:        cacheDir,
		manifestManager: manifest.NewManager(storage, namespace),
		segmentReader:   segment.NewReader(storage, namespace, cacheDir),
		loadedSegments:  make(map[string][]*document.Document),
		loadedIndexes:   make(map[string]*segment.Index),
	}
}

// LoadManifest loads or refreshes the manifest from storage.
func (e *Engine) LoadManifest(ctx context.Context) error {
	m, err := e.manifestManager.Load(ctx)
	if err != nil {
		return fmt.Errorf("failed to load manifest: %w", err)
	}

	e.mu.Lock()
	e.currentManifest = m
	e.mu.Unlock()

	log.Printf("Loaded manifest version %s with %d segments, %d total documents",
		m.Version, len(m.Segments), m.TotalDocCount())

	return nil
}

// EnsureSegmentsLoaded ensures all segments in the current manifest are loaded.
func (e *Engine) EnsureSegmentsLoaded(ctx context.Context) error {
	e.mu.RLock()
	m := e.currentManifest
	e.mu.RUnlock()

	if m == nil {
		return fmt.Errorf("no manifest loaded")
	}

	for _, seg := range m.Segments {
		if err := e.loadSegmentIfNeeded(ctx, seg); err != nil {
			return err
		}
	}

	return nil
}

func (e *Engine) loadSegmentIfNeeded(ctx context.Context, seg manifest.Segment) error {
	e.mu.RLock()
	_, docsLoaded := e.loadedSegments[seg.ID]
	_, indexLoaded := e.loadedIndexes[seg.ID]
	e.mu.RUnlock()

	if docsLoaded && indexLoaded {
		return nil
	}

	log.Printf("Loading segment %s", seg.ID)

	// Load segment documents
	if !docsLoaded {
		docs, err := e.segmentReader.ReadSegment(ctx, seg.SegmentKey)
		if err != nil {
			return fmt.Errorf("failed to load segment %s: %w", seg.ID, err)
		}
		e.mu.Lock()
		e.loadedSegments[seg.ID] = docs
		e.mu.Unlock()
	}

	// Load index
	if !indexLoaded {
		idx, err := e.segmentReader.ReadIndex(ctx, seg.IndexKey)
		if err != nil {
			return fmt.Errorf("failed to load index for segment %s: %w", seg.ID, err)
		}
		e.mu.Lock()
		e.loadedIndexes[seg.ID] = idx
		e.mu.Unlock()
	}

	return nil
}

// Search performs a search across all segments.
func (e *Engine) Search(ctx context.Context, req *document.SearchRequest) (*document.SearchResponse, error) {
	start := time.Now()

	// Refresh manifest
	if err := e.LoadManifest(ctx); err != nil {
		// Continue with existing manifest if refresh fails
		log.Printf("Warning: failed to refresh manifest: %v", err)
	}

	// Ensure segments are loaded
	if err := e.EnsureSegmentsLoaded(ctx); err != nil {
		return nil, err
	}

	e.mu.RLock()
	m := e.currentManifest
	segments := e.loadedSegments
	indexes := e.loadedIndexes
	e.mu.RUnlock()

	if m == nil {
		return &document.SearchResponse{
			Results:   []document.SearchResult{},
			TotalHits: 0,
			TookMs:    time.Since(start).Milliseconds(),
			Query:     req.Query,
		}, nil
	}

	// Search across all segments
	type scoredResult struct {
		doc   *document.Document
		score float64
	}
	var allResults []scoredResult

	for _, seg := range m.Segments {
		idx, ok := indexes[seg.ID]
		if !ok {
			continue
		}
		docs, ok := segments[seg.ID]
		if !ok {
			continue
		}

		// Apply tag filter if present
		var candidateDocIDs []string
		if len(req.Filters.Tags) > 0 {
			candidateDocIDs = idx.FilterByTags(req.Filters.Tags)
		}

		// Search within this segment
		var matchedDocIDs []string
		if req.Query != "" {
			matchedDocIDs = idx.Search(req.Query, 0) // No limit yet
		}

		// Build document map for quick lookup
		docMap := make(map[string]*document.Document)
		for _, doc := range docs {
			docMap[doc.ID] = doc
		}

		// Combine results
		if req.Query != "" && len(req.Filters.Tags) > 0 {
			// Intersection of query results and tag filter
			tagSet := make(map[string]struct{})
			for _, id := range candidateDocIDs {
				tagSet[id] = struct{}{}
			}

			for _, docID := range matchedDocIDs {
				if _, ok := tagSet[docID]; ok {
					if doc, exists := docMap[docID]; exists {
						score := calculateScore(doc, req.Query, idx)
						allResults = append(allResults, scoredResult{doc: doc, score: score})
					}
				}
			}
		} else if req.Query != "" {
			// Query only
			for _, docID := range matchedDocIDs {
				if doc, exists := docMap[docID]; exists {
					score := calculateScore(doc, req.Query, idx)
					allResults = append(allResults, scoredResult{doc: doc, score: score})
				}
			}
		} else if len(req.Filters.Tags) > 0 {
			// Tags only
			for _, docID := range candidateDocIDs {
				if doc, exists := docMap[docID]; exists {
					allResults = append(allResults, scoredResult{doc: doc, score: 1.0})
				}
			}
		} else {
			// No query or tags - return all documents
			for _, doc := range docs {
				allResults = append(allResults, scoredResult{doc: doc, score: 1.0})
			}
		}
	}

	// Sort by score descending
	for i := 0; i < len(allResults); i++ {
		for j := i + 1; j < len(allResults); j++ {
			if allResults[j].score > allResults[i].score {
				allResults[i], allResults[j] = allResults[j], allResults[i]
			}
		}
	}

	totalHits := len(allResults)

	// Apply offset
	if req.Offset > 0 && req.Offset < len(allResults) {
		allResults = allResults[req.Offset:]
	} else if req.Offset >= len(allResults) {
		allResults = nil
	}

	// Apply limit
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	if len(allResults) > limit {
		allResults = allResults[:limit]
	}

	// Convert to response format
	results := make([]document.SearchResult, len(allResults))
	for i, sr := range allResults {
		results[i] = document.SearchResult{
			Document: sr.doc,
			Score:    sr.score,
		}
	}

	return &document.SearchResponse{
		Results:   results,
		TotalHits: totalHits,
		TookMs:    time.Since(start).Milliseconds(),
		Query:     req.Query,
	}, nil
}

// calculateScore computes a simple relevance score.
func calculateScore(doc *document.Document, query string, idx *segment.Index) float64 {
	queryTerms := segment.Tokenize(query)
	if len(queryTerms) == 0 {
		return 0
	}

	docTerms, ok := idx.DocIDToTerms[doc.ID]
	if !ok {
		return 0
	}

	docTermSet := make(map[string]struct{})
	for _, term := range docTerms {
		docTermSet[term] = struct{}{}
	}

	// Count matching terms
	var matchCount int
	for _, qTerm := range queryTerms {
		if _, ok := docTermSet[qTerm]; ok {
			matchCount++
		}
	}

	// Boost for title matches
	titleBoost := 1.0
	if doc.Title != "" {
		titleLower := strings.ToLower(doc.Title)
		queryLower := strings.ToLower(query)
		if strings.Contains(titleLower, queryLower) {
			titleBoost = 2.0
		}
	}

	// Simple TF-based score
	score := float64(matchCount) / float64(len(queryTerms)) * titleBoost

	return score
}

// GetManifest returns the current manifest.
func (e *Engine) GetManifest() *manifest.Manifest {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.currentManifest
}

// Stats returns search engine statistics.
type Stats struct {
	ManifestVersion string `json:"manifest_version"`
	SegmentCount    int    `json:"segment_count"`
	TotalDocuments  int64  `json:"total_documents"`
	LoadedSegments  int    `json:"loaded_segments"`
	LoadedIndexes   int    `json:"loaded_indexes"`
}

// GetStats returns current engine statistics.
func (e *Engine) GetStats() Stats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	stats := Stats{}
	if e.currentManifest != nil {
		stats.ManifestVersion = e.currentManifest.Version
		stats.SegmentCount = len(e.currentManifest.Segments)
		stats.TotalDocuments = e.currentManifest.TotalDocCount()
	}
	stats.LoadedSegments = len(e.loadedSegments)
	stats.LoadedIndexes = len(e.loadedIndexes)

	return stats
}
