// tidepool-query is the HTTP API service for Tidepool vector database.
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

	"github.com/tidepool/tidepool/internal/config"
	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/query/internal/search"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting tidepool-query service...")

	// Load configuration
	cfg := config.Load()
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	// Initialize storage client
	ctx := context.Background()
	storageClient, err := storage.NewClient(ctx, cfg)
	if err != nil {
		log.Fatalf("Failed to initialize storage client: %v", err)
	}

	// Initialize search engine
	engine := search.NewEngine(storageClient, cfg.Namespace, cfg.CacheDir)

	// Try to load initial manifest
	if err := engine.LoadManifest(ctx); err != nil {
		log.Printf("Warning: failed to load initial manifest: %v", err)
		log.Println("Will retry on first request")
	}

	// Create HTTP server
	server := NewServer(engine, cfg.Namespace, cfg)

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

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

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
	engine          *search.Engine
	namespace       string
	maxBodyBytes    int64
	maxTopK         int
	corsAllowOrigin string
	mux             *http.ServeMux
}

// NewServer creates a new HTTP server.
func NewServer(engine *search.Engine, namespace string, cfg *config.Config) *Server {
	s := &Server{
		engine:          engine,
		namespace:       namespace,
		maxBodyBytes:    cfg.MaxBodyBytes,
		maxTopK:         cfg.MaxTopK,
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

	// Health and status
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/", s.handleRoot)
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

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"service": "tidepool",
		"version": "0.1.0",
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "healthy",
		"service": "tidepool-query",
	})
}

// handleNamespace handles /v1/namespaces/{namespace} endpoints
func (s *Server) handleNamespace(w http.ResponseWriter, r *http.Request) {
	// Parse namespace from path: /v1/namespaces/{namespace}
	path := strings.TrimPrefix(r.URL.Path, "/v1/namespaces/")
	parts := strings.Split(path, "/")

	if len(parts) == 0 || parts[0] == "" {
		// List namespaces
		s.handleListNamespaces(w, r)
		return
	}

	namespace := parts[0]

	// Check if this is the configured namespace
	if namespace != s.namespace {
		http.Error(w, "Namespace not found", http.StatusNotFound)
		return
	}

	if len(parts) == 1 {
		// GET /v1/namespaces/{namespace} - get namespace info
		if r.Method == http.MethodGet {
			s.handleGetNamespace(w, r)
			return
		}
		// DELETE /v1/namespaces/{namespace} - delete namespace (not supported)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Handle sub-routes
	switch parts[1] {
	case "query":
		s.handleQuery(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (s *Server) handleListNamespaces(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	stats := s.engine.GetStats()
	namespaces := []document.NamespaceInfo{
		{
			Namespace:   s.namespace,
			ApproxCount: stats.TotalVectors,
			Dimensions:  stats.Dimensions,
		},
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"namespaces": namespaces,
	})
}

func (s *Server) handleGetNamespace(w http.ResponseWriter, r *http.Request) {
	stats := s.engine.GetStats()

	writeJSON(w, http.StatusOK, document.NamespaceInfo{
		Namespace:   s.namespace,
		ApproxCount: stats.TotalVectors,
		Dimensions:  stats.Dimensions,
	})
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req document.QueryRequest
	if err := decodeJSON(w, r, &req, s.maxBodyBytes); err != nil {
		return
	}

	if len(req.Vector) == 0 {
		writeError(w, http.StatusBadRequest, "vector is required")
		return
	}

	stats := s.engine.GetStats()
	if stats.Dimensions > 0 && len(req.Vector) != stats.Dimensions {
		writeError(w, http.StatusBadRequest, "vector dimensions do not match namespace")
		return
	}

	// Set defaults
	if req.TopK <= 0 {
		req.TopK = 10
	}
	if s.maxTopK > 0 && req.TopK > s.maxTopK {
		req.TopK = s.maxTopK
	}

	resp, err := s.engine.Query(r.Context(), &req)
	if err != nil {
		log.Printf("Query error: %v", err)
		writeError(w, http.StatusInternalServerError, "query failed")
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

// handleVectors handles /v1/vectors/{namespace} for Turbopuffer compatibility
func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request) {
	// Parse namespace from path: /v1/vectors/{namespace}
	path := strings.TrimPrefix(r.URL.Path, "/v1/vectors/")
	namespace := strings.TrimSuffix(path, "/")

	if namespace != s.namespace {
		writeError(w, http.StatusNotFound, "namespace not found")
		return
	}

	// POST to vectors endpoint is a query
	if r.Method == http.MethodPost {
		s.handleQuery(w, r)
		return
	}

	// GET returns namespace info
	if r.Method == http.MethodGet {
		s.handleGetNamespace(w, r)
		return
	}

	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
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
