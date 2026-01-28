// Package segment handles segment file operations for Tidepool vector database.
package segment

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/google/uuid"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/storage"
	"github.com/tidepool/tidepool/internal/vector"
)

// SegmentData holds the in-memory representation of a segment.
type SegmentData struct {
	IDs        []string
	Vectors    [][]float32
	Attributes []map[string]interface{}
	Dimensions int
}

// Writer handles writing segment files.
type Writer struct {
	storage   storage.Store
	namespace string
}

// NewWriter creates a new segment writer.
func NewWriter(storage storage.Store, namespace string) *Writer {
	return &Writer{
		storage:   storage,
		namespace: namespace,
	}
}

// WriteSegment writes vectors to a new segment file.
// Format: [header][vectors][attributes]
// Header: 4 bytes magic, 4 bytes version, 4 bytes num_vectors, 4 bytes dimensions
// Vectors: num_vectors * dimensions * 4 bytes (float32)
// Attributes: JSON encoded
func (w *Writer) WriteSegment(ctx context.Context, docs []*document.Document) (*manifest.Segment, error) {
	if len(docs) == 0 {
		return nil, nil
	}

	// Determine dimensions from first vector
	var dimensions int
	for _, doc := range docs {
		if len(doc.Vector) > 0 {
			dimensions = len(doc.Vector)
			break
		}
	}

	if dimensions == 0 {
		return nil, fmt.Errorf("no vectors found in documents")
	}

	for i, doc := range docs {
		if len(doc.Vector) == 0 {
			return nil, fmt.Errorf("document %d has empty vector", i)
		}
		if len(doc.Vector) != dimensions {
			return nil, fmt.Errorf("document %d vector dimension mismatch: got %d want %d", i, len(doc.Vector), dimensions)
		}
	}

	segmentID := uuid.New().String()

	// Build segment data
	var buf bytes.Buffer

	// Write header
	buf.Write([]byte("TPVS")) // Magic: TidePool Vector Segment
	if err := binary.Write(&buf, binary.LittleEndian, uint32(1)); err != nil {
		return nil, fmt.Errorf("failed to write version: %w", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(docs))); err != nil {
		return nil, fmt.Errorf("failed to write vector count: %w", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(dimensions)); err != nil {
		return nil, fmt.Errorf("failed to write dimensions: %w", err)
	}

	// Prepare attribute data
	attrData := make([]struct {
		ID         string                 `json:"id"`
		Attributes map[string]interface{} `json:"attributes,omitempty"`
	}, len(docs))

	// Write vectors and collect attributes
	for i, doc := range docs {
		for _, v := range doc.Vector {
			if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
				return nil, fmt.Errorf("failed to write vector %d: %w", i, err)
			}
		}

		attrData[i].ID = doc.ID
		attrData[i].Attributes = doc.Attributes
	}

	// Write attributes as JSON
	attrJSON, err := json.Marshal(attrData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal attributes: %w", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(attrJSON))); err != nil {
		return nil, fmt.Errorf("failed to write attribute length: %w", err)
	}
	buf.Write(attrJSON)

	// Upload segment
	segmentPath := storage.SegmentPath(w.namespace, segmentID)
	if err := w.storage.Put(ctx, segmentPath, buf.Bytes()); err != nil {
		return nil, fmt.Errorf("failed to upload segment: %w", err)
	}

	return &manifest.Segment{
		ID:         segmentID,
		SegmentKey: segmentPath,
		DocCount:   int64(len(docs)),
		Dimensions: dimensions,
	}, nil
}

// Reader handles reading segment files.
type Reader struct {
	storage   storage.Store
	namespace string
	cacheDir  string
}

// NewReader creates a new segment reader.
func NewReader(storage storage.Store, namespace, cacheDir string) *Reader {
	// Ensure cache directory exists
	if cacheDir != "" {
		if err := os.MkdirAll(cacheDir, 0755); err != nil {
			log.Printf("Warning: failed to create cache directory %s: %v", cacheDir, err)
		}
	}
	return &Reader{
		storage:   storage,
		namespace: namespace,
		cacheDir:  cacheDir,
	}
}

// cacheKeyForSegment generates a cache filename for a segment key.
func (r *Reader) cacheKeyForSegment(segmentKey string) string {
	hash := sha256.Sum256([]byte(segmentKey))
	return hex.EncodeToString(hash[:16]) + ".tpvs"
}

// getSegmentData retrieves segment data, using disk cache if available.
func (r *Reader) getSegmentData(ctx context.Context, segmentKey string) ([]byte, error) {
	// Try disk cache first
	if r.cacheDir != "" {
		cachePath := filepath.Join(r.cacheDir, r.cacheKeyForSegment(segmentKey))
		if data, err := os.ReadFile(cachePath); err == nil {
			log.Printf("Cache hit for segment %s", segmentKey)
			return data, nil
		}
	}

	// Download from S3
	log.Printf("Cache miss, downloading segment %s", segmentKey)
	data, err := r.storage.Get(ctx, segmentKey)
	if err != nil {
		return nil, err
	}

	// Save to disk cache
	if r.cacheDir != "" {
		cachePath := filepath.Join(r.cacheDir, r.cacheKeyForSegment(segmentKey))
		if err := os.WriteFile(cachePath, data, 0644); err != nil {
			log.Printf("Warning: failed to cache segment %s: %v", segmentKey, err)
		} else {
			log.Printf("Cached segment %s to %s", segmentKey, cachePath)
		}
	}

	return data, nil
}

// ReadSegment reads a segment file into memory, using disk cache if available.
func (r *Reader) ReadSegment(ctx context.Context, segmentKey string) (*SegmentData, error) {
	data, err := r.getSegmentData(ctx, segmentKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read segment: %w", err)
	}

	reader := bytes.NewReader(data)

	// Read header
	magic := make([]byte, 4)
	if _, err := io.ReadFull(reader, magic); err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if string(magic) != "TPVS" {
		return nil, fmt.Errorf("invalid segment file format")
	}

	var version, numVectors, dimensions uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported segment version: %d", version)
	}
	if err := binary.Read(reader, binary.LittleEndian, &numVectors); err != nil {
		return nil, fmt.Errorf("failed to read vector count: %w", err)
	}
	if err := binary.Read(reader, binary.LittleEndian, &dimensions); err != nil {
		return nil, fmt.Errorf("failed to read dimensions: %w", err)
	}
	if dimensions == 0 {
		return nil, fmt.Errorf("invalid segment dimensions: %d", dimensions)
	}

	// Read vectors
	vectors := make([][]float32, numVectors)
	for i := uint32(0); i < numVectors; i++ {
		vec := make([]float32, dimensions)
		for j := uint32(0); j < dimensions; j++ {
			if err := binary.Read(reader, binary.LittleEndian, &vec[j]); err != nil {
				return nil, fmt.Errorf("failed to read vector %d: %w", i, err)
			}
		}
		vectors[i] = vec
	}

	// Read attributes
	var attrLen uint32
	if err := binary.Read(reader, binary.LittleEndian, &attrLen); err != nil {
		return nil, fmt.Errorf("failed to read attribute length: %w", err)
	}
	if attrLen == 0 {
		return nil, fmt.Errorf("invalid attribute length")
	}

	attrJSON := make([]byte, attrLen)
	if _, err := io.ReadFull(reader, attrJSON); err != nil {
		return nil, fmt.Errorf("failed to read attributes: %w", err)
	}

	var attrData []struct {
		ID         string                 `json:"id"`
		Attributes map[string]interface{} `json:"attributes,omitempty"`
	}
	if err := json.Unmarshal(attrJSON, &attrData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal attributes: %w", err)
	}
	if uint32(len(attrData)) != numVectors {
		return nil, fmt.Errorf("attribute count mismatch: got %d want %d", len(attrData), numVectors)
	}

	// Build segment data
	seg := &SegmentData{
		IDs:        make([]string, numVectors),
		Vectors:    vectors,
		Attributes: make([]map[string]interface{}, numVectors),
		Dimensions: int(dimensions),
	}

	for i := range attrData {
		seg.IDs[i] = attrData[i].ID
		seg.Attributes[i] = attrData[i].Attributes
	}

	return seg, nil
}

// ScoredResult represents a search result with distance.
type ScoredResult struct {
	Index int
	Dist  float32
}

// Search performs brute-force vector search on the segment.
func (s *SegmentData) Search(query []float32, topK int, metric vector.DistanceMetric, filter func(attrs map[string]interface{}) bool) []ScoredResult {
	var results []ScoredResult

	for i, vec := range s.Vectors {
		// Apply filter if provided
		if filter != nil && !filter(s.Attributes[i]) {
			continue
		}

		dist := vector.Distance(query, vec, metric)
		results = append(results, ScoredResult{Index: i, Dist: dist})
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Dist < results[j].Dist
	})

	// Take top K
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	return results
}
