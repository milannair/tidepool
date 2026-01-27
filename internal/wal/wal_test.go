package wal

import (
	"context"
	"testing"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
)

func TestWALWriteRead(t *testing.T) {
	store := storage.NewInMemoryStore()
	writer := NewWriter(store, "test")
	reader := NewReader(store, "test")

	docs := []document.Document{
		{ID: "a", Vector: []float32{1, 2}},
		{ID: "b", Vector: []float32{3, 4}},
	}

	walPath, err := writer.WriteUpsert(context.Background(), docs)
	if err != nil {
		t.Fatalf("write upsert: %v", err)
	}

	entries, err := reader.ReadWALFile(context.Background(), walPath)
	if err != nil {
		t.Fatalf("read wal: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].Op != "upsert" {
		t.Fatalf("expected upsert op, got %s", entries[0].Op)
	}

	if _, err := writer.WriteDelete(context.Background(), []string{"a"}); err != nil {
		t.Fatalf("write delete: %v", err)
	}

	allEntries, walFiles, err := reader.ReadAllWALFiles(context.Background())
	if err != nil {
		t.Fatalf("read all wal: %v", err)
	}
	if len(walFiles) != 2 {
		t.Fatalf("expected 2 wal files, got %d", len(walFiles))
	}
	if len(allEntries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(allEntries))
	}
}
