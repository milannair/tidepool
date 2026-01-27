package manifest

import (
	"context"
	"testing"

	"github.com/tidepool/tidepool/internal/storage"
)

func TestManifestSaveLoad(t *testing.T) {
	store := storage.NewInMemoryStore()
	manager := NewManager(store, "test")

	m := NewManifest([]Segment{
		{ID: "s1", SegmentKey: "segments/s1.tpvs", DocCount: 2, Dimensions: 3},
	})

	if err := manager.Save(context.Background(), m); err != nil {
		t.Fatalf("save manifest: %v", err)
	}

	loaded, err := manager.Load(context.Background())
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	if loaded.Version != m.Version {
		t.Fatalf("expected version %s, got %s", m.Version, loaded.Version)
	}
	if loaded.Dimensions != 3 {
		t.Fatalf("expected dimensions 3, got %d", loaded.Dimensions)
	}
	if loaded.TotalDocCount() != 2 {
		t.Fatalf("expected total doc count 2, got %d", loaded.TotalDocCount())
	}
}
