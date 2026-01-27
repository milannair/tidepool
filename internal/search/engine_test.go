package search

import (
	"context"
	"testing"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
)

func TestEngineQuery(t *testing.T) {
	store := storage.NewInMemoryStore()
	namespace := "test"

	writer := segment.NewWriter(store, namespace)
	seg, err := writer.WriteSegment(context.Background(), []*document.Document{
		{ID: "a", Vector: []float32{1, 0}, Attributes: map[string]interface{}{"tag": "x"}},
		{ID: "b", Vector: []float32{0, 1}, Attributes: map[string]interface{}{"tag": "y"}},
	})
	if err != nil {
		t.Fatalf("write segment: %v", err)
	}

	m := manifest.NewManifest([]manifest.Segment{*seg})
	manager := manifest.NewManager(store, namespace)
	if err := manager.Save(context.Background(), m); err != nil {
		t.Fatalf("save manifest: %v", err)
	}

	engine := NewEngine(store, namespace, "")
	if err := engine.LoadManifest(context.Background()); err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	resp, err := engine.Query(context.Background(), &document.QueryRequest{
		Vector:  []float32{1, 0},
		TopK:    1,
		Filters: map[string]interface{}{"tag": "x"},
	})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(resp.Results) != 1 || resp.Results[0].ID != "a" {
		t.Fatalf("expected result a, got %+v", resp.Results)
	}
}
