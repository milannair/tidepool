package compaction

import (
	"context"
	"testing"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/internal/wal"
)

func TestCompactionCycle(t *testing.T) {
	store := storage.NewInMemoryStore()
	namespace := "test"

	writer := wal.NewWriter(store, namespace)
	if _, err := writer.WriteUpsert(context.Background(), []document.Document{
		{ID: "a", Vector: []float32{1, 0}},
		{ID: "b", Vector: []float32{0, 1}},
	}); err != nil {
		t.Fatalf("write upsert: %v", err)
	}
	if _, err := writer.WriteDelete(context.Background(), []string{"a"}); err != nil {
		t.Fatalf("write delete: %v", err)
	}

	compactor := NewCompactor(store, namespace)
	if err := compactor.Run(context.Background()); err != nil {
		t.Fatalf("compaction run: %v", err)
	}

	manager := manifest.NewManager(store, namespace)
	loaded, err := manager.Load(context.Background())
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if len(loaded.Segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(loaded.Segments))
	}

	reader := segment.NewReader(store, namespace, "")
	seg, err := reader.ReadSegment(context.Background(), loaded.Segments[0].SegmentKey)
	if err != nil {
		t.Fatalf("read segment: %v", err)
	}
	if len(seg.IDs) != 1 || seg.IDs[0] != "b" {
		t.Fatalf("expected only id b, got %v", seg.IDs)
	}
}
