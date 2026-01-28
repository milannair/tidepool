package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tidepool/tidepool/internal/config"
	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/segment"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/query/internal/search"
)

func setupQueryServer(t *testing.T) (*httptest.Server, *search.Engine) {
	t.Helper()

	store := storage.NewInMemoryStore()
	namespace := "default"

	writer := segment.NewWriter(store, namespace)
	seg, err := writer.WriteSegment(context.Background(), []*document.Document{
		{ID: "a", Vector: []float32{1, 0}},
		{ID: "b", Vector: []float32{0, 1}},
	})
	if err != nil {
		t.Fatalf("write segment: %v", err)
	}
	m := manifest.NewManifest([]manifest.Segment{*seg})
	manager := manifest.NewManager(store, namespace)
	if err := manager.Save(context.Background(), m); err != nil {
		t.Fatalf("save manifest: %v", err)
	}

	engine := search.NewEngine(store, namespace, "")
	if err := engine.LoadManifest(context.Background()); err != nil {
		t.Fatalf("load manifest: %v", err)
	}

	cfg := &config.Config{
		MaxBodyBytes:    1024 * 1024,
		MaxTopK:         1000,
		CORSAllowOrigin: "*",
	}
	server := NewServer(engine, namespace, cfg)
	return httptest.NewServer(server), engine
}

func TestQueryServerOK(t *testing.T) {
	ts, _ := setupQueryServer(t)
	defer ts.Close()

	body, _ := json.Marshal(document.QueryRequest{
		Vector: []float32{1, 0},
		TopK:   1,
	})
	resp, err := http.Post(ts.URL+"/v1/vectors/default", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
}

func TestQueryServerDimensionMismatch(t *testing.T) {
	ts, _ := setupQueryServer(t)
	defer ts.Close()

	body, _ := json.Marshal(document.QueryRequest{
		Vector: []float32{1, 0, 0},
	})
	resp, err := http.Post(ts.URL+"/v1/vectors/default", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}
