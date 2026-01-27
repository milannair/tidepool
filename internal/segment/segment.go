// Package segment handles segment file operations for Tidepool.
package segment

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/parquet-go/parquet-go"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/manifest"
	"github.com/tidepool/tidepool/internal/storage"
)

// ParquetDocument is a flattened document for parquet storage.
type ParquetDocument struct {
	ID        string   `parquet:"id"`
	Content   string   `parquet:"content"`
	Title     string   `parquet:"title,optional"`
	URL       string   `parquet:"url,optional"`
	Tags      string   `parquet:"tags"` // JSON-encoded array
	Metadata  string   `parquet:"metadata"` // JSON-encoded map
	CreatedAt int64    `parquet:"created_at"`
	UpdatedAt int64    `parquet:"updated_at"`
}

// ToParquet converts a document to parquet format.
func ToParquet(doc *document.Document) ParquetDocument {
	tagsJSON, _ := json.Marshal(doc.Tags)
	metaJSON, _ := json.Marshal(doc.Metadata)

	return ParquetDocument{
		ID:        doc.ID,
		Content:   doc.Content,
		Title:     doc.Title,
		URL:       doc.URL,
		Tags:      string(tagsJSON),
		Metadata:  string(metaJSON),
		CreatedAt: doc.CreatedAt.UnixNano(),
		UpdatedAt: doc.UpdatedAt.UnixNano(),
	}
}

// FromParquet converts a parquet document back to a document.
func FromParquet(pdoc ParquetDocument) *document.Document {
	var tags []string
	json.Unmarshal([]byte(pdoc.Tags), &tags)

	var metadata map[string]interface{}
	json.Unmarshal([]byte(pdoc.Metadata), &metadata)

	return &document.Document{
		ID:        pdoc.ID,
		Content:   pdoc.Content,
		Title:     pdoc.Title,
		URL:       pdoc.URL,
		Tags:      tags,
		Metadata:  metadata,
		CreatedAt: time.Unix(0, pdoc.CreatedAt),
		UpdatedAt: time.Unix(0, pdoc.UpdatedAt),
	}
}

// Writer handles writing segment files.
type Writer struct {
	storage   *storage.Client
	namespace string
}

// NewWriter creates a new segment writer.
func NewWriter(storage *storage.Client, namespace string) *Writer {
	return &Writer{
		storage:   storage,
		namespace: namespace,
	}
}

// WriteSegment writes documents to a new segment and index file.
func (w *Writer) WriteSegment(ctx context.Context, docs []*document.Document) (*manifest.Segment, error) {
	if len(docs) == 0 {
		return nil, nil
	}

	segmentID := uuid.New().String()

	// Convert documents to parquet format
	parquetDocs := make([]ParquetDocument, len(docs))
	for i, doc := range docs {
		parquetDocs[i] = ToParquet(doc)
	}

	// Write parquet segment
	var segmentBuf bytes.Buffer
	if err := parquet.Write(&segmentBuf, parquetDocs); err != nil {
		return nil, fmt.Errorf("failed to write parquet segment: %w", err)
	}

	segmentPath := storage.SegmentPath(w.namespace, segmentID)
	if err := w.storage.Put(ctx, segmentPath, segmentBuf.Bytes()); err != nil {
		return nil, fmt.Errorf("failed to upload segment: %w", err)
	}

	// Build and write index
	index := BuildIndex(docs)
	indexData, err := json.Marshal(index)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize index: %w", err)
	}

	indexPath := storage.IndexPath(w.namespace, segmentID)
	if err := w.storage.Put(ctx, indexPath, indexData); err != nil {
		return nil, fmt.Errorf("failed to upload index: %w", err)
	}

	return &manifest.Segment{
		ID:         segmentID,
		SegmentKey: segmentPath,
		IndexKey:   indexPath,
		DocCount:   int64(len(docs)),
	}, nil
}

// Reader handles reading segment files.
type Reader struct {
	storage   *storage.Client
	namespace string
	cacheDir  string
}

// NewReader creates a new segment reader.
func NewReader(storage *storage.Client, namespace, cacheDir string) *Reader {
	return &Reader{
		storage:   storage,
		namespace: namespace,
		cacheDir:  cacheDir,
	}
}

// ReadSegment reads all documents from a segment file.
func (r *Reader) ReadSegment(ctx context.Context, segmentKey string) ([]*document.Document, error) {
	data, err := r.storage.Get(ctx, segmentKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read segment: %w", err)
	}

	parquetDocs, err := parquet.Read[ParquetDocument](bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("failed to parse parquet segment: %w", err)
	}

	docs := make([]*document.Document, len(parquetDocs))
	for i, pdoc := range parquetDocs {
		docs[i] = FromParquet(pdoc)
	}

	return docs, nil
}

// ReadIndex reads an index file.
func (r *Reader) ReadIndex(ctx context.Context, indexKey string) (*Index, error) {
	data, err := r.storage.Get(ctx, indexKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read index: %w", err)
	}

	var index Index
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("failed to parse index: %w", err)
	}

	return &index, nil
}
