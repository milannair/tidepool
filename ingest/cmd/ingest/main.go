// tidepool-ingest is the background worker service for vector ingestion.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"

	"github.com/tidepool/tidepool/ingest/internal/compaction"
	"github.com/tidepool/tidepool/ingest/internal/wal"
	"github.com/tidepool/tidepool/internal/config"
	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting tidepool-ingest service...")

	// Load configuration
	cfg := config.Load()
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	// Initialize storage client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	storageClient, err := storage.NewClient(ctx, cfg)
	if err != nil {
		log.Fatalf("Failed to initialize storage client: %v", err)
	}

	// Initialize components
	walWriter := wal.NewWriter(storageClient, cfg.Namespace)
	compactor := compaction.NewCompactor(storageClient, cfg.Namespace)

	// Start compaction background worker
	go compactor.RunPeriodically(ctx, cfg.CompactionInterval)

	// Create HTTP server
	server := NewServer(walWriter, compactor, cfg.Namespace, cfg)

	httpServer := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           server,
		ReadTimeout:       cfg.ReadTimeout,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       cfg.IdleTimeout,
	}

	// Graceful shutdown handling
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")

		cancel() // Stop compaction

		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()

		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			log.Printf("HTTP server shutdown error: %v", err)
		}
	}()

	log.Printf("Listening on :%s", cfg.Port)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server error: %v", err)
	}

	log.Println("Server stopped")
}

// Server handles HTTP requests.
type Server struct {
	walWriter       *wal.Writer
	compactor       *compaction.Compactor
	namespace       string
	maxBodyBytes    int64
	corsAllowOrigin string
	mux             *http.ServeMux
}

// NewServer creates a new HTTP server.
func NewServer(walWriter *wal.Writer, compactor *compaction.Compactor, namespace string, cfg *config.Config) *Server {
	s := &Server{
		walWriter:       walWriter,
		compactor:       compactor,
		namespace:       namespace,
		maxBodyBytes:    cfg.MaxBodyBytes,
		corsAllowOrigin: cfg.CORSAllowOrigin,
		mux:             http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	// Turbopuffer-compatible API
	s.mux.HandleFunc("/v1/namespaces/", s.handleNamespace)
	s.mux.HandleFunc("/v1/vectors/", s.handleVectors)

	// Admin endpoints
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/compact", s.handleCompact)
	s.mux.HandleFunc("/status", s.handleStatus)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	w.Header().Set("Access-Control-Allow-Origin", s.corsAllowOrigin)
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	start := time.Now()
	rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	s.mux.ServeHTTP(rec, r)
	log.Printf("%s %s %d %dB %s", r.Method, r.URL.Path, rec.status, rec.bytes, time.Since(start))
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "healthy",
		"service": "tidepool-ingest",
	})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	status, err := s.compactor.GetStatus(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to get status")
		return
	}

	writeJSON(w, http.StatusOK, status)
}

func (s *Server) handleCompact(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	if err := s.compactor.Run(r.Context()); err != nil {
		log.Printf("Manual compaction error: %v", err)
		writeError(w, http.StatusInternalServerError, "compaction failed")
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{
		"status": "compaction completed",
	})
}

// handleNamespace handles /v1/namespaces/{namespace} endpoints
func (s *Server) handleNamespace(w http.ResponseWriter, r *http.Request) {
	// Parse namespace from path
	path := strings.TrimPrefix(r.URL.Path, "/v1/namespaces/")
	parts := strings.Split(path, "/")

	if len(parts) == 0 || parts[0] == "" {
		writeError(w, http.StatusBadRequest, "namespace required")
		return
	}

	namespace := parts[0]
	if namespace != s.namespace {
		writeError(w, http.StatusNotFound, "namespace not found")
		return
	}

	// Handle upsert: POST /v1/namespaces/{namespace}
	if len(parts) == 1 && r.Method == http.MethodPost {
		s.handleUpsert(w, r)
		return
	}

	// Handle delete: DELETE /v1/namespaces/{namespace}
	if len(parts) == 1 && r.Method == http.MethodDelete {
		s.handleDelete(w, r)
		return
	}

	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
}

// handleVectors handles /v1/vectors/{namespace} - Turbopuffer compatibility
func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1/vectors/")
	namespace := strings.TrimSuffix(path, "/")

	if namespace != s.namespace {
		writeError(w, http.StatusNotFound, "namespace not found")
		return
	}

	switch r.Method {
	case http.MethodPost:
		s.handleUpsert(w, r)
	case http.MethodDelete:
		s.handleDelete(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleUpsert(w http.ResponseWriter, r *http.Request) {
	var req document.UpsertRequest
	if err := decodeJSON(w, r, &req, s.maxBodyBytes); err != nil {
		return
	}

	if len(req.Vectors) == 0 {
		writeError(w, http.StatusBadRequest, "no vectors provided")
		return
	}

	// Validate and set IDs
	var dims int
	for i := range req.Vectors {
		if req.Vectors[i].ID == "" {
			req.Vectors[i].ID = uuid.New().String()
		}
		if len(req.Vectors[i].Vector) == 0 {
			writeError(w, http.StatusBadRequest, "vector is required for each document")
			return
		}
		if dims == 0 {
			dims = len(req.Vectors[i].Vector)
		} else if len(req.Vectors[i].Vector) != dims {
			writeError(w, http.StatusBadRequest, "all vectors must have the same dimensions")
			return
		}
	}

	// Write to WAL
	_, err := s.walWriter.WriteUpsert(r.Context(), req.Vectors)
	if err != nil {
		log.Printf("Upsert error: %v", err)
		writeError(w, http.StatusInternalServerError, "upsert failed")
		return
	}

	log.Printf("Upserted %d vectors", len(req.Vectors))

	writeJSON(w, http.StatusOK, document.UpsertResponse{
		Status: "OK",
	})
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	var req document.DeleteRequest
	if err := decodeJSON(w, r, &req, s.maxBodyBytes); err != nil {
		return
	}

	if len(req.IDs) == 0 {
		writeError(w, http.StatusBadRequest, "no ids provided")
		return
	}

	// Write delete to WAL
	_, err := s.walWriter.WriteDelete(r.Context(), req.IDs)
	if err != nil {
		log.Printf("Delete error: %v", err)
		writeError(w, http.StatusInternalServerError, "delete failed")
		return
	}

	log.Printf("Deleted %d vectors", len(req.IDs))

	writeJSON(w, http.StatusOK, document.DeleteResponse{
		Status: "OK",
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(p []byte) (int, error) {
	n, err := r.ResponseWriter.Write(p)
	r.bytes += n
	return n, err
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{
		"error": msg,
	})
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst interface{}, maxBytes int64) error {
	if maxBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var syntaxError *json.SyntaxError
		var unmarshalTypeError *json.UnmarshalTypeError
		switch {
		case errors.As(err, &syntaxError):
			writeError(w, http.StatusBadRequest, "malformed JSON")
		case errors.As(err, &unmarshalTypeError):
			writeError(w, http.StatusBadRequest, "invalid JSON type")
		case errors.Is(err, http.ErrBodyReadAfterClose):
			writeError(w, http.StatusBadRequest, "invalid request body")
		default:
			if strings.Contains(err.Error(), "http: request body too large") {
				writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			} else {
				writeError(w, http.StatusBadRequest, "invalid request body")
			}
		}
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return errors.New("extra data")
	}
	return nil
}
