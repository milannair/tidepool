// Package compaction handles WAL compaction into segments.
package compaction

import (
	"context"
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/tidepool/tidepool/ingest/internal/wal"
	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
)

// Compactor compacts WAL files into segments.
type Compactor struct {
	storage         storage.Store
	namespace       string
	walReader       *wal.Reader
	segmentWriter   *segment.Writer
	manifestManager *manifest.Manager
}

// NewCompactor creates a new compactor.
func NewCompactor(storage storage.Store, namespace string) *Compactor {
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

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Timestamp.Before(entries[j].Timestamp)
	})

	// Process entries: apply upserts and deletes
	// Keep track of deleted IDs
	deletedIDs := make(map[string]bool)
	docMap := make(map[string]*document.Document)

	for _, entry := range entries {
		switch entry.Op {
		case "upsert":
			if entry.Document != nil {
				docMap[entry.Document.ID] = entry.Document
				delete(deletedIDs, entry.Document.ID) // Upsert overrides delete
			}
		case "delete":
			for _, id := range entry.DeleteIDs {
				deletedIDs[id] = true
				delete(docMap, id)
			}
		}
	}

	// Load existing manifest to get existing segments
	var existingSegments []manifest.Segment
	currentManifest, err := c.manifestManager.Load(ctx)
	if err == nil {
		existingSegments = currentManifest.Segments
	} else {
		log.Printf("No existing manifest, creating new one: %v", err)
	}

	// If we have existing segments, we need to merge with new data
	// For simplicity in v0, we rebuild the entire dataset
	// (A more sophisticated approach would append-only and mark tombstones)

	// Load all existing vectors from segments
	segReader := segment.NewReader(c.storage, c.namespace, "")
	for _, seg := range existingSegments {
		segData, err := segReader.ReadSegment(ctx, seg.SegmentKey)
		if err != nil {
			log.Printf("Warning: failed to read existing segment %s: %v", seg.ID, err)
			continue
		}

		for i := range segData.IDs {
			id := segData.IDs[i]
			// Only include if not deleted and not being overwritten
			if !deletedIDs[id] {
				if _, exists := docMap[id]; !exists {
					docMap[id] = &document.Document{
						ID:         id,
						Vector:     segData.Vectors[i],
						Attributes: segData.Attributes[i],
					}
				}
			}
		}
	}

	// Convert to slice
	docs := make([]*document.Document, 0, len(docMap))
	for _, doc := range docMap {
		if len(doc.Vector) > 0 {
			docs = append(docs, doc)
		}
	}
	var dims int
	for i, doc := range docs {
		if len(doc.Vector) == 0 {
			return fmt.Errorf("document %d has empty vector during compaction", i)
		}
		if dims == 0 {
			dims = len(doc.Vector)
		} else if len(doc.Vector) != dims {
			return fmt.Errorf("dimension mismatch during compaction: got %d want %d", len(doc.Vector), dims)
		}
	}

	if len(docs) == 0 {
		log.Println("No vectors to compact")
		// Still need to delete WAL files and update manifest
		emptyManifest := manifest.NewManifest(nil)
		if err := c.manifestManager.Save(ctx, emptyManifest); err != nil {
			return fmt.Errorf("failed to save empty manifest: %w", err)
		}
		// Delete WAL files
		for _, walFile := range walFiles {
			if err := c.walReader.DeleteWALFile(ctx, walFile); err != nil {
				log.Printf("Warning: failed to delete WAL file %s: %v", walFile, err)
			}
		}
		return nil
	}

	log.Printf("Compacting %d vectors", len(docs))

	// Write new segment
	newSegment, err := c.segmentWriter.WriteSegment(ctx, docs)
	if err != nil {
		return fmt.Errorf("failed to write segment: %w", err)
	}

	log.Printf("Created segment %s with %d vectors (%d dimensions)",
		newSegment.ID, newSegment.DocCount, newSegment.Dimensions)

	// Create new manifest with single segment (full compaction)
	newManifest := manifest.NewManifest([]manifest.Segment{*newSegment})

	// Save new manifest
	if err := c.manifestManager.Save(ctx, newManifest); err != nil {
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	log.Printf("Saved manifest version %s", newManifest.Version)

	// Delete old segments (they're replaced by the new compacted segment)
	for _, seg := range existingSegments {
		if err := c.storage.Delete(ctx, seg.SegmentKey); err != nil {
			log.Printf("Warning: failed to delete old segment %s: %v", seg.ID, err)
		}
	}

	// Delete compacted WAL files
	for _, walFile := range walFiles {
		if err := c.walReader.DeleteWALFile(ctx, walFile); err != nil {
			log.Printf("Warning: failed to delete WAL file %s: %v", walFile, err)
		} else {
			log.Printf("Deleted WAL file %s", walFile)
		}
	}

	log.Printf("Compaction complete: %d vectors in %d segments",
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
	TotalVecs  int64     `json:"total_vectors"`
	Dimensions int       `json:"dimensions"`
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
	var totalVecs int64
	var dimensions int
	manifest, err := c.manifestManager.Load(ctx)
	if err == nil {
		segments = len(manifest.Segments)
		totalVecs = manifest.TotalDocCount()
		dimensions = manifest.Dimensions
	}

	return &Status{
		LastRun:    time.Now(),
		WALFiles:   len(walFiles),
		WALEntries: walEntries,
		Segments:   segments,
		TotalVecs:  totalVecs,
		Dimensions: dimensions,
	}, nil
}
