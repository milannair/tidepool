// Package storage provides storage implementations for Tidepool.
package storage

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// InMemoryStore is an in-memory implementation of Store for tests.
type InMemoryStore struct {
	mu   sync.RWMutex
	data map[string][]byte
}

// NewInMemoryStore creates a new in-memory store.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		data: make(map[string][]byte),
	}
}

// Get retrieves an object by key.
func (s *InMemoryStore) Get(_ context.Context, key string) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	val, ok := s.data[key]
	if !ok {
		return nil, fmt.Errorf("object not found: %s", key)
	}
	cpy := make([]byte, len(val))
	copy(cpy, val)
	return cpy, nil
}

// Put stores an object by key.
func (s *InMemoryStore) Put(_ context.Context, key string, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cpy := make([]byte, len(data))
	copy(cpy, data)
	s.data[key] = cpy
	return nil
}

// Delete removes an object by key.
func (s *InMemoryStore) Delete(_ context.Context, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.data, key)
	return nil
}

// List returns all keys with the given prefix.
func (s *InMemoryStore) List(_ context.Context, prefix string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0)
	for key := range s.data {
		if strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	return keys, nil
}

// Exists checks if a key exists.
func (s *InMemoryStore) Exists(_ context.Context, key string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.data[key]
	return ok, nil
}
