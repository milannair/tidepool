// Package wal handles Write-Ahead Log operations for Tidepool.
package wal

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/tidepool/tidepool/internal/document"
	"github.com/tidepool/tidepool/internal/storage"
)

// Writer handles writing documents to WAL files.
type Writer struct {
	storage   *storage.Client
	namespace string
}

// NewWriter creates a new WAL writer.
func NewWriter(storage *storage.Client, namespace string) *Writer {
	return &Writer{
		storage:   storage,
		namespace: namespace,
	}
}

// Entry represents a single WAL entry.
type Entry struct {
	Timestamp time.Time          `json:"timestamp"`
	Document  *document.Document `json:"document"`
}

// Write appends documents to a new WAL file.
func (w *Writer) Write(ctx context.Context, docs []document.Document) (string, error) {
	if len(docs) == 0 {
		return "", nil
	}

	date := time.Now().UTC().Format("2006-01-02")
	walID := uuid.New().String()
	walPath := storage.WALPath(w.namespace, date, walID)

	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)

	for _, doc := range docs {
		entry := Entry{
			Timestamp: time.Now().UTC(),
			Document:  &doc,
		}
		if err := encoder.Encode(entry); err != nil {
			return "", fmt.Errorf("failed to encode WAL entry: %w", err)
		}
	}

	if err := w.storage.Put(ctx, walPath, buf.Bytes()); err != nil {
		return "", fmt.Errorf("failed to write WAL file: %w", err)
	}

	return walPath, nil
}

// Reader handles reading WAL files.
type Reader struct {
	storage   *storage.Client
	namespace string
}

// NewReader creates a new WAL reader.
func NewReader(storage *storage.Client, namespace string) *Reader {
	return &Reader{
		storage:   storage,
		namespace: namespace,
	}
}

// ListWALFiles returns all WAL files for the namespace.
func (r *Reader) ListWALFiles(ctx context.Context) ([]string, error) {
	prefix := storage.WALPrefix(r.namespace)
	keys, err := r.storage.List(ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to list WAL files: %w", err)
	}

	// Filter to only include .jsonl files
	var walFiles []string
	for _, key := range keys {
		if strings.HasSuffix(key, ".jsonl") {
			walFiles = append(walFiles, key)
		}
	}
	return walFiles, nil
}

// ReadWALFile reads all entries from a WAL file.
func (r *Reader) ReadWALFile(ctx context.Context, walPath string) ([]Entry, error) {
	data, err := r.storage.Get(ctx, walPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read WAL file %s: %w", walPath, err)
	}

	var entries []Entry
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var entry Entry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return nil, fmt.Errorf("failed to parse WAL entry: %w", err)
		}
		entries = append(entries, entry)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning WAL file: %w", err)
	}

	return entries, nil
}

// ReadAllWALFiles reads all WAL entries from all WAL files.
func (r *Reader) ReadAllWALFiles(ctx context.Context) ([]Entry, []string, error) {
	walFiles, err := r.ListWALFiles(ctx)
	if err != nil {
		return nil, nil, err
	}

	var allEntries []Entry
	for _, walFile := range walFiles {
		entries, err := r.ReadWALFile(ctx, walFile)
		if err != nil {
			return nil, nil, err
		}
		allEntries = append(allEntries, entries...)
	}

	return allEntries, walFiles, nil
}

// DeleteWALFile deletes a WAL file.
func (r *Reader) DeleteWALFile(ctx context.Context, walPath string) error {
	return r.storage.Delete(ctx, walPath)
}

// WALInfo represents information about a WAL file.
type WALInfo struct {
	Path      string
	Date      string
	EntryCount int
}

// GetWALInfo parses WAL file path to extract info.
func GetWALInfo(walPath string) WALInfo {
	parts := strings.Split(walPath, "/")
	var date string
	if len(parts) >= 2 {
		date = parts[len(parts)-2]
	}
	return WALInfo{
		Path: walPath,
		Date: date,
	}
}

// ExtractDate extracts the date from a WAL path.
func ExtractDate(walPath string) string {
	// Path format: namespaces/{namespace}/wal/{date}/{uuid}.jsonl
	dir := path.Dir(walPath)
	return path.Base(dir)
}
