// Package compaction handles WAL compaction into segments.
package compaction

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/internal/wal"
)

// Compactor compacts WAL files into segments.
type Compactor struct {
	storage         *storage.Client
	namespace       string
	walReader       *wal.Reader
	segmentWriter   *segment.Writer
	manifestManager *manifest.Manager
}

// NewCompactor creates a new compactor.
func NewCompactor(storage *storage.Client, namespace string) *Compactor {
	return &Compactor{
		storage:         storage,
		namespace:       namespace,
		walReader:       wal.NewReader(storage, namespace),
		segmentWriter:   segment.NewWriter(storage, namespace),
		manifestManager: manifest.NewManager(storage, namespace),
	}
}

// Run performs a compaction cycle.
func (c *Compactor) Run(ctx context.Context) error {
	log.Println("Starting compaction cycle...")

	// Read all WAL files
	entries, walFiles, err := c.walReader.ReadAllWALFiles(ctx)
	if err != nil {
		return fmt.Errorf("failed to read WAL files: %w", err)
	}

	if len(entries) == 0 {
		log.Println("No WAL entries to compact")
		return nil
	}

	log.Printf("Found %d WAL entries across %d files", len(entries), len(walFiles))

	// Deduplicate documents by ID (keep latest)
	docMap := make(map[string]*document.Document)
	for _, entry := range entries {
		if entry.Document != nil {
			// If document already exists, keep the newer one
			existing, ok := docMap[entry.Document.ID]
			if !ok || entry.Document.UpdatedAt.After(existing.UpdatedAt) {
				docMap[entry.Document.ID] = entry.Document
			}
		}
	}

	// Convert to slice
	docs := make([]*document.Document, 0, len(docMap))
	for _, doc := range docMap {
		docs = append(docs, doc)
	}

	log.Printf("Compacting %d unique documents", len(docs))

	// Write new segment
	newSegment, err := c.segmentWriter.WriteSegment(ctx, docs)
	if err != nil {
		return fmt.Errorf("failed to write segment: %w", err)
	}

	if newSegment == nil {
		log.Println("No segment created (no documents)")
		return nil
	}

	log.Printf("Created segment %s with %d documents", newSegment.ID, newSegment.DocCount)

	// Load current manifest (or create empty one)
	var currentManifest *manifest.Manifest
	currentManifest, err = c.manifestManager.Load(ctx)
	if err != nil {
		// No existing manifest, start fresh
		log.Println("No existing manifest, creating new one")
		currentManifest = manifest.NewManifest(nil)
	}

	// Add new segment to manifest
	newSegments := append(currentManifest.Segments, *newSegment)
	newManifest := manifest.NewManifest(newSegments)

	// Save new manifest
	if err := c.manifestManager.Save(ctx, newManifest); err != nil {
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	log.Printf("Saved manifest version %s", newManifest.Version)

	// Delete compacted WAL files
	for _, walFile := range walFiles {
		if err := c.walReader.DeleteWALFile(ctx, walFile); err != nil {
			log.Printf("Warning: failed to delete WAL file %s: %v", walFile, err)
		} else {
			log.Printf("Deleted WAL file %s", walFile)
		}
	}

	log.Printf("Compaction complete: %d documents in %d segments",
		newManifest.TotalDocCount(), len(newManifest.Segments))

	return nil
}

// RunPeriodically runs compaction at regular intervals.
func (c *Compactor) RunPeriodically(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Run once immediately
	if err := c.Run(ctx); err != nil {
		log.Printf("Compaction error: %v", err)
	}

	for {
		select {
		case <-ctx.Done():
			log.Println("Compaction stopped")
			return
		case <-ticker.C:
			if err := c.Run(ctx); err != nil {
				log.Printf("Compaction error: %v", err)
			}
		}
	}
}

// Status represents the compaction status.
type Status struct {
	LastRun    time.Time `json:"last_run"`
	WALFiles   int       `json:"wal_files"`
	WALEntries int       `json:"wal_entries"`
	Segments   int       `json:"segments"`
	TotalDocs  int64     `json:"total_docs"`
}

// GetStatus returns the current compaction status.
func (c *Compactor) GetStatus(ctx context.Context) (*Status, error) {
	walFiles, err := c.walReader.ListWALFiles(ctx)
	if err != nil {
		return nil, err
	}

	var walEntries int
	for _, walFile := range walFiles {
		entries, err := c.walReader.ReadWALFile(ctx, walFile)
		if err != nil {
			continue
		}
		walEntries += len(entries)
	}

	var segments int
	var totalDocs int64
	manifest, err := c.manifestManager.Load(ctx)
	if err == nil {
		segments = len(manifest.Segments)
		totalDocs = manifest.TotalDocCount()
	}

	return &Status{
		LastRun:    time.Now(),
		WALFiles:   len(walFiles),
		WALEntries: walEntries,
		Segments:   segments,
		TotalDocs:  totalDocs,
	}, nil
}
