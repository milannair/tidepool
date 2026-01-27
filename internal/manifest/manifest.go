// Package manifest handles manifest file operations for Tidepool.
package manifest

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/tidepool/tidepool/internal/storage"
)

// Manifest represents the current state of a namespace's data.
type Manifest struct {
	Version   string    `json:"version"`
	CreatedAt time.Time `json:"created_at"`
	Segments  []Segment `json:"segments"`
}

// Segment represents a data segment in the manifest.
type Segment struct {
	ID         string `json:"id"`
	SegmentKey string `json:"segment_key"`
	IndexKey   string `json:"index_key"`
	DocCount   int64  `json:"doc_count"`
}

// Manager handles manifest operations.
type Manager struct {
	storage   *storage.Client
	namespace string
}

// NewManager creates a new manifest manager.
func NewManager(storage *storage.Client, namespace string) *Manager {
	return &Manager{
		storage:   storage,
		namespace: namespace,
	}
}

// Load retrieves the latest manifest from storage.
func (m *Manager) Load(ctx context.Context) (*Manifest, error) {
	data, err := m.storage.Get(ctx, storage.LatestManifestPath(m.namespace))
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("failed to parse manifest: %w", err)
	}
	return &manifest, nil
}

// LoadVersion retrieves a specific manifest version from storage.
func (m *Manager) LoadVersion(ctx context.Context, version string) (*Manifest, error) {
	data, err := m.storage.Get(ctx, storage.ManifestPath(m.namespace, version))
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest version %s: %w", version, err)
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("failed to parse manifest: %w", err)
	}
	return &manifest, nil
}

// Save writes a new manifest version and updates latest.
func (m *Manager) Save(ctx context.Context, manifest *Manifest) error {
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize manifest: %w", err)
	}

	// Save versioned manifest
	versionPath := storage.ManifestPath(m.namespace, manifest.Version)
	if err := m.storage.Put(ctx, versionPath, data); err != nil {
		return fmt.Errorf("failed to save versioned manifest: %w", err)
	}

	// Update latest pointer
	latestPath := storage.LatestManifestPath(m.namespace)
	if err := m.storage.Put(ctx, latestPath, data); err != nil {
		return fmt.Errorf("failed to update latest manifest: %w", err)
	}

	return nil
}

// NewManifest creates a new manifest with the given segments.
func NewManifest(segments []Segment) *Manifest {
	return &Manifest{
		Version:   fmt.Sprintf("%d", time.Now().UnixNano()),
		CreatedAt: time.Now().UTC(),
		Segments:  segments,
	}
}

// TotalDocCount returns the total number of documents across all segments.
func (m *Manifest) TotalDocCount() int64 {
	var total int64
	for _, seg := range m.Segments {
		total += seg.DocCount
	}
	return total
}
