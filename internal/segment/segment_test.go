package segment

import (
	"context"
	"testing"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
)

func TestSegmentRoundTrip(t *testing.T) {
	store := storage.NewInMemoryStore()
	writer := NewWriter(store, "test")
	reader := NewReader(store, "test", "")

	docs := []*document.Document{
		{ID: "a", Vector: []float32{1, 2}, Attributes: map[string]interface{}{"tag": "x"}},
		{ID: "b", Vector: []float32{3, 4}, Attributes: map[string]interface{}{"tag": "y"}},
	}

	seg, err := writer.WriteSegment(context.Background(), docs)
	if err != nil {
		t.Fatalf("write segment: %v", err)
	}

	loaded, err := reader.ReadSegment(context.Background(), seg.SegmentKey)
	if err != nil {
		t.Fatalf("read segment: %v", err)
	}

	if loaded.Dimensions != 2 {
		t.Fatalf("expected dimensions 2, got %d", loaded.Dimensions)
	}
	if len(loaded.IDs) != 2 {
		t.Fatalf("expected 2 ids, got %d", len(loaded.IDs))
	}
	if loaded.IDs[0] != "a" || loaded.IDs[1] != "b" {
		t.Fatalf("unexpected ids: %v", loaded.IDs)
	}
	if loaded.Vectors[0][0] != 1 || loaded.Vectors[1][1] != 4 {
		t.Fatalf("unexpected vectors: %v", loaded.Vectors)
	}
}

func TestSegmentDimensionMismatch(t *testing.T) {
	store := storage.NewInMemoryStore()
	writer := NewWriter(store, "test")

	docs := []*document.Document{
		{ID: "a", Vector: []float32{1, 2}},
		{ID: "b", Vector: []float32{3}},
	}

	if _, err := writer.WriteSegment(context.Background(), docs); err == nil {
		t.Fatal("expected dimension mismatch error")
	}
}
