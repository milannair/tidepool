package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tidepool/tidepool/ingest/internal/compaction"
	"github.com/tidepool/tidepool/ingest/internal/wal"
	"github.com/tidepool/tidepool/internal/config"
	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
)

func setupIngestServer(t *testing.T) (*httptest.Server, *storage.InMemoryStore) {
	t.Helper()

	store := storage.NewInMemoryStore()
	namespace := "default"
	walWriter := wal.NewWriter(store, namespace)
	compactor := compaction.NewCompactor(store, namespace)

	cfg := &config.Config{
		MaxBodyBytes:    1024 * 1024,
		CORSAllowOrigin: "*",
	}
	server := NewServer(walWriter, compactor, namespace, cfg)
	return httptest.NewServer(server), store
}

func TestIngestUpsertAndDelete(t *testing.T) {
	ts, store := setupIngestServer(t)
	defer ts.Close()

	upsertBody, _ := json.Marshal(document.UpsertRequest{
		Vectors: []document.Document{
			{ID: "a", Vector: []float32{1, 0}},
			{ID: "b", Vector: []float32{0, 1}},
		},
	})
	resp, err := http.Post(ts.URL+"/v1/vectors/default", "application/json", bytes.NewReader(upsertBody))
	if err != nil {
		t.Fatalf("post upsert: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	deleteBody, _ := json.Marshal(document.DeleteRequest{IDs: []string{"a"}})
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/v1/vectors/default", bytes.NewReader(deleteBody))
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	reader := wal.NewReader(store, "default")
	entries, _, err := reader.ReadAllWALFiles(context.Background())
	if err != nil {
		t.Fatalf("read wal: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}
}
