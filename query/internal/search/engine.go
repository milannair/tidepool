// Package search implements the vector search engine for Tidepool.
package search

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/internal/vector"
)

// Engine performs vector search operations across all segments.
type Engine struct {
	storage         storage.Store
	namespace       string
	cacheDir        string
	manifestManager *manifest.Manager
	segmentReader   *segment.Reader

	mu              sync.RWMutex
	currentManifest *manifest.Manifest
	loadedSegments  map[string]*segment.SegmentData
}

// NewEngine creates a new vector search engine.
func NewEngine(storage storage.Store, namespace, cacheDir string) *Engine {
	return &Engine{
		storage:         storage,
		namespace:       namespace,
		cacheDir:        cacheDir,
		manifestManager: manifest.NewManager(storage, namespace),
		segmentReader:   segment.NewReader(storage, namespace, cacheDir),
		loadedSegments:  make(map[string]*segment.SegmentData),
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

	log.Printf("Loaded manifest version %s with %d segments, %d total vectors",
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
	_, loaded := e.loadedSegments[seg.ID]
	e.mu.RUnlock()

	if loaded {
		return nil
	}

	log.Printf("Loading segment %s", seg.ID)

	segData, err := e.segmentReader.ReadSegment(ctx, seg.SegmentKey)
	if err != nil {
		return fmt.Errorf("failed to load segment %s: %w", seg.ID, err)
	}

	e.mu.Lock()
	e.loadedSegments[seg.ID] = segData
	e.mu.Unlock()

	return nil
}

// Query performs a vector similarity search.
func (e *Engine) Query(ctx context.Context, req *document.QueryRequest) (*document.QueryResponse, error) {
	// Refresh manifest
	if err := e.LoadManifest(ctx); err != nil {
		log.Printf("Warning: failed to refresh manifest: %v", err)
	}

	// Ensure segments are loaded
	if err := e.EnsureSegmentsLoaded(ctx); err != nil {
		// Return empty if no data
		if e.currentManifest == nil {
			return &document.QueryResponse{
				Results:   []document.VectorResult{},
				Namespace: e.namespace,
			}, nil
		}
		return nil, err
	}

	e.mu.RLock()
	m := e.currentManifest
	segments := e.loadedSegments
	e.mu.RUnlock()

	if m == nil || len(m.Segments) == 0 {
		return &document.QueryResponse{
			Results:   []document.VectorResult{},
			Namespace: e.namespace,
		}, nil
	}

	// Set defaults
	topK := req.TopK
	if topK <= 0 {
		topK = 10
	}

	metric := vector.ParseMetric(req.DistanceMetric)

	// Build filter function
	var filterFunc func(attrs map[string]interface{}) bool
	if len(req.Filters) > 0 {
		filterFunc = buildFilterFunc(req.Filters)
	}

	// Search across all segments
	type scoredResult struct {
		id    string
		vec   []float32
		attrs map[string]interface{}
		dist  float32
	}
	var allResults []scoredResult

	for _, seg := range m.Segments {
		segData, ok := segments[seg.ID]
		if !ok {
			continue
		}

		results := segData.Search(req.Vector, 0, metric, filterFunc) // Get all, sort globally
		for _, r := range results {
			sr := scoredResult{
				id:    segData.IDs[r.Index],
				attrs: segData.Attributes[r.Index],
				dist:  r.Dist,
			}
			if req.IncludeVectors {
				sr.vec = segData.Vectors[r.Index]
			}
			allResults = append(allResults, sr)
		}
	}

	// Sort by distance globally
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].dist < allResults[j].dist
	})

	// Take top K
	if len(allResults) > topK {
		allResults = allResults[:topK]
	}

	// Convert to response format
	results := make([]document.VectorResult, len(allResults))
	for i, sr := range allResults {
		results[i] = document.VectorResult{
			ID:         sr.id,
			Attributes: sr.attrs,
			Dist:       sr.dist,
		}
		if req.IncludeVectors {
			results[i].Vector = sr.vec
		}
	}

	return &document.QueryResponse{
		Results:   results,
		Namespace: e.namespace,
	}, nil
}

// buildFilterFunc creates a filter function from the filters map.
// Supports simple equality checks on attributes.
func buildFilterFunc(filters map[string]interface{}) func(attrs map[string]interface{}) bool {
	return func(attrs map[string]interface{}) bool {
		if attrs == nil {
			return false
		}

		for key, expectedValue := range filters {
			actualValue, exists := attrs[key]
			if !exists {
				return false
			}

			// Handle array filters (e.g., {"tag": ["a", "b"]} means tag must be in ["a", "b"])
			if expectedSlice, ok := expectedValue.([]interface{}); ok {
				found := false
				for _, v := range expectedSlice {
					if actualValue == v {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			} else {
				// Simple equality
				if actualValue != expectedValue {
					return false
				}
			}
		}
		return true
	}
}

// GetManifest returns the current manifest.
func (e *Engine) GetManifest() *manifest.Manifest {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.currentManifest
}

// Stats returns search engine statistics.
type Stats struct {
	Namespace       string `json:"namespace"`
	ManifestVersion string `json:"manifest_version,omitempty"`
	SegmentCount    int    `json:"segment_count"`
	TotalVectors    int64  `json:"total_vectors"`
	Dimensions      int    `json:"dimensions"`
	LoadedSegments  int    `json:"loaded_segments"`
}

// GetStats returns current engine statistics.
func (e *Engine) GetStats() Stats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	stats := Stats{Namespace: e.namespace}
	if e.currentManifest != nil {
		stats.ManifestVersion = e.currentManifest.Version
		stats.SegmentCount = len(e.currentManifest.Segments)
		stats.TotalVectors = e.currentManifest.TotalDocCount()
		stats.Dimensions = e.currentManifest.Dimensions
	}
	stats.LoadedSegments = len(e.loadedSegments)

	return stats
}

// InvalidateCache clears the loaded segments cache.
func (e *Engine) InvalidateCache() {
	e.mu.Lock()
	e.loadedSegments = make(map[string]*segment.SegmentData)
	e.currentManifest = nil
	e.mu.Unlock()
}
