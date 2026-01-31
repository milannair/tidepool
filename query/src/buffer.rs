//! Optimized hot buffer for real-time vector updates.
//!
//! Features:
//! - HNSW index for O(log n) vector search
//! - Lock-free reads using ArcSwap for high concurrency
//! - Secondary attribute indexes for efficient pre-filtering
//! - Optional Redis Vector Search integration for shared state

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;
use roaring::RoaringBitmap;

use tidepool_common::attributes::AttrValue;
use tidepool_common::document::{Document, VectorResult};
use tidepool_common::index::hnsw::HnswIndex;
use tidepool_common::text::{DefaultTokenizer, Tokenizer, TokenizerConfig};
use tidepool_common::vector::DistanceMetric;

/// HNSW parameters tuned for small, frequently-updated indexes.
const BUFFER_HNSW_M: usize = 12;
const BUFFER_HNSW_EF_CONSTRUCTION: usize = 100;
const BUFFER_HNSW_EF_SEARCH: usize = 50;

// ============================================================================
// IMMUTABLE SNAPSHOT (for lock-free reads)
// ============================================================================

/// Immutable snapshot of buffer state for lock-free reads.
/// Readers get a consistent view without holding locks.
#[derive(Clone)]
struct BufferSnapshot {
    /// HNSW index for fast ANN search
    index: Arc<HnswIndex>,
    /// String ID -> internal index mapping
    id_to_index: Arc<HashMap<String, usize>>,
    /// Internal index -> String ID mapping
    index_to_id: Arc<Vec<String>>,
    /// Attributes for each vector (by internal index)
    attributes: Arc<Vec<Option<AttrValue>>>,
    /// Text for each vector (by internal index)
    texts: Arc<Vec<Option<String>>>,
    /// Bitmap of active (non-deleted) vectors
    active: Arc<RoaringBitmap>,
    /// Dimensions of vectors
    dimensions: Option<usize>,
}

impl BufferSnapshot {
    fn new() -> Self {
        Self {
            index: Arc::new(HnswIndex::new(
                BUFFER_HNSW_M,
                BUFFER_HNSW_EF_CONSTRUCTION,
                BUFFER_HNSW_EF_SEARCH,
                DistanceMetric::Cosine,
            )),
            id_to_index: Arc::new(HashMap::new()),
            index_to_id: Arc::new(Vec::new()),
            attributes: Arc::new(Vec::new()),
            texts: Arc::new(Vec::new()),
            active: Arc::new(RoaringBitmap::new()),
            dimensions: None,
        }
    }
}

// ============================================================================
// ATTRIBUTE INDEX (for pre-filtering)
// ============================================================================

/// Secondary index for attribute-based filtering.
/// Maps attribute key -> value -> set of document IDs.
#[derive(Clone, Default)]
struct AttributeIndex {
    /// key -> value -> bitmap of doc indices
    indexes: HashMap<String, HashMap<String, RoaringBitmap>>,
}

impl AttributeIndex {
    fn new() -> Self {
        Self {
            indexes: HashMap::new(),
        }
    }

    /// Index a document's attributes
    fn index_doc(&mut self, doc_idx: usize, attrs: Option<&AttrValue>) {
        let Some(AttrValue::Object(map)) = attrs else { return };
        
        for (key, value) in map {
            let value_str = value_to_string(value);
            self.indexes
                .entry(key.clone())
                .or_default()
                .entry(value_str)
                .or_default()
                .insert(doc_idx as u32);
        }
    }

    /// Remove a document from all indexes
    fn remove_doc(&mut self, doc_idx: usize, attrs: Option<&AttrValue>) {
        let Some(AttrValue::Object(map)) = attrs else { return };
        
        for (key, value) in map {
            let value_str = value_to_string(value);
            if let Some(key_index) = self.indexes.get_mut(key) {
                if let Some(bitmap) = key_index.get_mut(&value_str) {
                    bitmap.remove(doc_idx as u32);
                }
            }
        }
    }

    /// Get candidate document IDs matching the filter
    /// Returns None if any filter key is unindexed (fall back to post-filter)
    fn get_candidates(&self, filters: &AttrValue) -> Option<RoaringBitmap> {
        let AttrValue::Object(filter_map) = filters else { return None };
        
        let mut result: Option<RoaringBitmap> = None;
        
        for (key, filter_val) in filter_map {
            let Some(key_index) = self.indexes.get(key) else {
                // No index for this key - must fall back to post-filtering
                // to ensure all filter conditions are applied
                return None;
            };
            
            let candidates = match filter_val {
                AttrValue::Array(items) => {
                    // OR of all values in array
                    let mut bitmap = RoaringBitmap::new();
                    for item in items {
                        let value_str = value_to_string(item);
                        if let Some(b) = key_index.get(&value_str) {
                            bitmap |= b;
                        }
                    }
                    bitmap
                }
                _ => {
                    let value_str = value_to_string(filter_val);
                    key_index.get(&value_str).cloned().unwrap_or_default()
                }
            };
            
            // AND with previous results
            result = Some(match result {
                Some(r) => r & &candidates,
                None => candidates,
            });
        }
        
        result
    }
}

fn value_to_string(value: &AttrValue) -> String {
    match value {
        AttrValue::String(s) => s.clone(),
        AttrValue::Number(n) => n.to_string(),
        AttrValue::Bool(b) => b.to_string(),
        AttrValue::Null => "null".to_string(),
        AttrValue::Array(_) | AttrValue::Object(_) => {
            serde_json::to_string(value).unwrap_or_default()
        }
    }
}

// ============================================================================
// TEXT INDEX
// ============================================================================

struct TextBufferIndex {
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_terms: Vec<Vec<(String, u32)>>,
    doc_lengths: Vec<u32>,
    total_length: u64,
}

impl TextBufferIndex {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_terms: Vec::new(),
            doc_lengths: Vec::new(),
            total_length: 0,
        }
    }

    fn upsert(&mut self, doc_id: usize, tokens: &[String]) {
        self.remove_doc(doc_id);

        while self.doc_terms.len() < doc_id + 1 {
            self.doc_terms.push(Vec::new());
            self.doc_lengths.push(0);
        }

        if tokens.is_empty() {
            return;
        }

        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token.clone()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as u32;
        self.doc_lengths[doc_id] = doc_len;
        self.total_length = self.total_length.saturating_add(doc_len as u64);

        let mut terms = Vec::with_capacity(tf_map.len());
        for (term, tf) in tf_map {
            self.postings.entry(term.clone()).or_default().push((doc_id, tf));
            terms.push((term, tf));
        }

        self.doc_terms[doc_id] = terms;
    }

    fn remove_doc(&mut self, doc_id: usize) {
        if doc_id >= self.doc_terms.len() {
            return;
        }
        let terms = std::mem::take(&mut self.doc_terms[doc_id]);
        let old_len = self.doc_lengths.get(doc_id).copied().unwrap_or(0);
        if doc_id < self.doc_lengths.len() {
            self.doc_lengths[doc_id] = 0;
        }
        self.total_length = self.total_length.saturating_sub(old_len as u64);

        for (term, _) in terms {
            let empty = if let Some(list) = self.postings.get_mut(&term) {
                list.retain(|(id, _)| *id != doc_id);
                list.is_empty()
            } else {
                false
            };
            if empty {
                self.postings.remove(&term);
            }
        }
    }

    fn search(
        &self,
        tokens: &[String],
        top_k: usize,
        active: &RoaringBitmap,
        k1: f32,
        b: f32,
    ) -> Vec<(usize, f32)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let doc_count = active.len() as f32;
        if doc_count == 0.0 {
            return Vec::new();
        }

        let avgdl = if self.total_length > 0 {
            self.total_length as f32 / doc_count
        } else {
            1.0
        };

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for token in tokens {
            let Some(postings) = self.postings.get(token) else { continue };
            if postings.is_empty() {
                continue;
            }

            let df = postings.len() as f32;
            let idf = ((doc_count - df + 0.5) / (df + 0.5) + 1.0).ln();

            for &(doc_id, tf) in postings {
                if !active.contains(doc_id as u32) {
                    continue;
                }
                let dl = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                let tf = tf as f32;
                let denom = tf + k1 * (1.0 - b + b * dl / avgdl);
                if denom == 0.0 {
                    continue;
                }
                let score = idf * (tf * (k1 + 1.0)) / denom;
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        if scores.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && results.len() > top_k {
            results.truncate(top_k);
        }
        results
    }
}

// ============================================================================
// HOT BUFFER (Main Structure)
// ============================================================================

/// Thread-safe hot buffer with lock-free reads and pre-filtering support.
pub struct HotBuffer {
    /// Immutable snapshot for lock-free reads
    snapshot: ArcSwap<BufferSnapshot>,
    /// Write lock for mutations (only one writer at a time)
    write_state: RwLock<WriteState>,
    /// Maximum number of vectors to hold
    max_size: usize,
    /// Last manifest version seen (for cleanup)
    last_manifest_version: AtomicU64,
    /// Tokenizer for text
    tokenizer: DefaultTokenizer,
    /// BM25 parameters
    bm25_k1: f32,
    bm25_b: f32,
}

/// Mutable state protected by write lock
struct WriteState {
    /// Mutable HNSW index
    index: HnswIndex,
    /// String ID -> internal index mapping
    id_to_index: HashMap<String, usize>,
    /// Internal index -> String ID mapping
    index_to_id: Vec<String>,
    /// Attributes for each vector
    attributes: Vec<Option<AttrValue>>,
    /// Text for each vector
    texts: Vec<Option<String>>,
    /// In-memory text index
    text_index: TextBufferIndex,
    /// Bitmap of active vectors
    active: RoaringBitmap,
    /// Tombstone set for deleted IDs
    tombstones: HashSet<String>,
    /// Insertion order for FIFO eviction
    insertion_order: Vec<String>,
    /// Dimensions
    dimensions: Option<usize>,
    /// Attribute secondary index
    attr_index: AttributeIndex,
}

impl WriteState {
    fn new() -> Self {
        Self {
            index: HnswIndex::new(
                BUFFER_HNSW_M,
                BUFFER_HNSW_EF_CONSTRUCTION,
                BUFFER_HNSW_EF_SEARCH,
                DistanceMetric::Cosine,
            ),
            id_to_index: HashMap::new(),
            index_to_id: Vec::new(),
            attributes: Vec::new(),
            texts: Vec::new(),
            text_index: TextBufferIndex::new(),
            active: RoaringBitmap::new(),
            tombstones: HashSet::new(),
            insertion_order: Vec::new(),
            dimensions: None,
            attr_index: AttributeIndex::new(),
        }
    }

    /// Create an immutable snapshot for readers
    fn to_snapshot(&self) -> BufferSnapshot {
        BufferSnapshot {
            index: Arc::new(self.index.clone()),
            id_to_index: Arc::new(self.id_to_index.clone()),
            index_to_id: Arc::new(self.index_to_id.clone()),
            attributes: Arc::new(self.attributes.clone()),
            texts: Arc::new(self.texts.clone()),
            active: Arc::new(self.active.clone()),
            dimensions: self.dimensions,
        }
    }
}

impl HotBuffer {
    /// Create a new hot buffer with the given maximum size.
    pub fn new(max_size: usize, tokenizer_config: TokenizerConfig, bm25_k1: f32, bm25_b: f32) -> Self {
        Self {
            snapshot: ArcSwap::from_pointee(BufferSnapshot::new()),
            write_state: RwLock::new(WriteState::new()),
            max_size,
            last_manifest_version: AtomicU64::new(0),
            tokenizer: DefaultTokenizer::new(tokenizer_config),
            bm25_k1,
            bm25_b,
        }
    }

    /// Insert vectors into the buffer.
    /// If the buffer is full, oldest vectors are evicted (FIFO).
    pub async fn insert(&self, docs: Vec<Document>) {
        if docs.is_empty() {
            return;
        }

        let mut state = self.write_state.write();

        // Set dimensions on first insert
        if state.dimensions.is_none() {
            if let Some(doc) = docs.first() {
                if !doc.vector.is_empty() {
                    state.dimensions = Some(doc.vector.len());
                }
            }
        }

        for doc in docs {
            if doc.vector.is_empty() {
                continue;
            }

            let string_id = doc.id.clone();

            // Remove from tombstones if re-inserting
            state.tombstones.remove(&string_id);

            let tokens = doc
                .text
                .as_deref()
                .map(|t| self.tokenizer.tokenize(t))
                .unwrap_or_default();

            if let Some(&existing_idx) = state.id_to_index.get(&string_id) {
                // Update existing vector
                // Remove old attributes from secondary index (extract first to avoid borrow conflict)
                let old_attrs = state.attributes.get(existing_idx).and_then(|a| a.as_ref()).cloned();
                state.attr_index.remove_doc(existing_idx, old_attrs.as_ref());
                
                state.index.insert(existing_idx, doc.vector);
                state.attributes[existing_idx] = doc.attributes.clone();
                if existing_idx < state.texts.len() {
                    state.texts[existing_idx] = doc.text.clone();
                }
                state.text_index.upsert(existing_idx, &tokens);
                state.active.insert(existing_idx as u32);
                
                // Add new attributes to secondary index
                state.attr_index.index_doc(existing_idx, doc.attributes.as_ref());
            } else {
                // New vector
                let new_idx = state.index_to_id.len();
                state.id_to_index.insert(string_id.clone(), new_idx);
                state.index_to_id.push(string_id.clone());
                state.attributes.push(doc.attributes.clone());
                state.texts.push(doc.text.clone());

                state.index.insert(new_idx, doc.vector);
                state.text_index.upsert(new_idx, &tokens);
                state.active.insert(new_idx as u32);
                state.insertion_order.push(string_id);
                
                // Add to secondary index
                state.attr_index.index_doc(new_idx, doc.attributes.as_ref());
            }
        }

        // Evict oldest if over capacity
        while state.active.len() as usize > self.max_size && !state.insertion_order.is_empty() {
            if let Some(oldest_id) = state.insertion_order.first().cloned() {
                state.insertion_order.remove(0);
                if let Some(&idx) = state.id_to_index.get(&oldest_id) {
                    state.active.remove(idx as u32);
                    state.text_index.remove_doc(idx);
                    
                    // Remove from secondary index (extract first to avoid borrow conflict)
                    let old_attrs = state.attributes.get(idx).and_then(|a| a.as_ref()).cloned();
                    state.attr_index.remove_doc(idx, old_attrs.as_ref());
                    
                    if idx < state.texts.len() {
                        state.texts[idx] = None;
                    }
                }
            }
        }

        // Publish new snapshot for readers
        self.snapshot.store(Arc::new(state.to_snapshot()));
    }

    /// Mark IDs as deleted.
    pub async fn delete(&self, ids: Vec<String>) {
        let mut state = self.write_state.write();

        for id in ids {
            state.tombstones.insert(id.clone());

            if let Some(&idx) = state.id_to_index.get(&id) {
                state.active.remove(idx as u32);
                state.text_index.remove_doc(idx);
                
                // Remove from secondary index (extract first to avoid borrow conflict)
                let old_attrs = state.attributes.get(idx).and_then(|a| a.as_ref()).cloned();
                state.attr_index.remove_doc(idx, old_attrs.as_ref());
                
                if idx < state.texts.len() {
                    state.texts[idx] = None;
                }
            }
            state.insertion_order.retain(|x| x != &id);
        }

        // Publish new snapshot
        self.snapshot.store(Arc::new(state.to_snapshot()));
    }

    /// Search the buffer for similar vectors using HNSW index.
    /// Uses lock-free reads and pre-filtering when possible.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _metric: DistanceMetric,
        filters: Option<&AttrValue>,
        include_vectors: bool,
    ) -> Vec<VectorResult> {
        // Lock-free read of snapshot
        let snapshot = self.snapshot.load();

        if snapshot.active.is_empty() {
            return Vec::new();
        }

        // Determine which vectors to search
        let search_candidates = if let Some(filter_val) = filters {
            // Try to use attribute index for pre-filtering
            let state = self.write_state.read();
            if let Some(candidates) = state.attr_index.get_candidates(filter_val) {
                // Intersect with active vectors
                let filtered = &candidates & snapshot.active.as_ref();
                if filtered.is_empty() {
                    return Vec::new();
                }
                Some(filtered)
            } else {
                None // Fall back to post-filtering
            }
        } else {
            None
        };

        let active_filter = search_candidates.as_ref().unwrap_or(&snapshot.active);

        // Use HNSW search with filter
        let ef = (top_k * 2).max(BUFFER_HNSW_EF_SEARCH);
        let results = snapshot.index.search_with_filter(query, top_k, ef, active_filter);

        results
            .into_iter()
            .filter_map(|r| {
                let string_id = snapshot.index_to_id.get(r.id)?;
                
                // Post-filter if we couldn't pre-filter (no index hit)
                if search_candidates.is_none() {
                    if let Some(filter_value) = filters {
                        let attrs = snapshot.attributes.get(r.id).cloned().flatten();
                        if !matches_filters(attrs.as_ref(), filter_value) {
                            return None;
                        }
                    }
                }
                
                Some(VectorResult {
                    id: string_id.clone(),
                    vector: if include_vectors {
                        snapshot.index.nodes.get(r.id).map(|n| n.vector.clone()).unwrap_or_default()
                    } else {
                        Vec::new()
                    },
                    attributes: snapshot.attributes.get(r.id).cloned().flatten(),
                    score: distance_to_score(r.dist),
                })
            })
            .collect()
    }

    /// Search the buffer by text using BM25.
    pub async fn text_search(
        &self,
        tokens: &[String],
        top_k: usize,
        filters: Option<&AttrValue>,
        include_vectors: bool,
    ) -> Vec<VectorResult> {
        let snapshot = self.snapshot.load();
        let state = self.write_state.read();

        if snapshot.active.is_empty() {
            return Vec::new();
        }

        // Determine search candidates using attribute index
        let search_candidates = if let Some(filter_val) = filters {
            if let Some(candidates) = state.attr_index.get_candidates(filter_val) {
                let filtered = &candidates & snapshot.active.as_ref();
                if filtered.is_empty() {
                    return Vec::new();
                }
                Some(filtered)
            } else {
                None
            }
        } else {
            None
        };

        let active_filter = search_candidates.as_ref().unwrap_or(&snapshot.active);

        let results = state.text_index.search(tokens, top_k, active_filter, self.bm25_k1, self.bm25_b);
        
        results
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let string_id = snapshot.index_to_id.get(doc_id)?;
                
                // Post-filter if needed
                if search_candidates.is_none() {
                    if let Some(filter_value) = filters {
                        let attrs = snapshot.attributes.get(doc_id).cloned().flatten();
                        if !matches_filters(attrs.as_ref(), filter_value) {
                            return None;
                        }
                    }
                }
                
                Some(VectorResult {
                    id: string_id.clone(),
                    vector: if include_vectors {
                        snapshot.index.nodes.get(doc_id).map(|n| n.vector.clone()).unwrap_or_default()
                    } else {
                        Vec::new()
                    },
                    attributes: snapshot.attributes.get(doc_id).cloned().flatten(),
                    score,
                })
            })
            .collect()
    }

    /// Check if an ID is deleted.
    pub async fn is_deleted(&self, id: &str) -> bool {
        let snapshot = self.snapshot.load();
        match snapshot.id_to_index.get(id) {
            Some(&idx) => !snapshot.active.contains(idx as u32),
            None => false,
        }
    }

    /// Get all deleted IDs.
    pub async fn get_deleted_ids(&self) -> HashSet<String> {
        let snapshot = self.snapshot.load();
        let state = self.write_state.read();

        let mut deleted = state.tombstones.clone();
        for (id, &idx) in snapshot.id_to_index.iter() {
            if !snapshot.active.contains(idx as u32) {
                deleted.insert(id.clone());
            }
        }
        deleted
    }

    /// Get buffer statistics.
    pub async fn stats(&self) -> BufferStats {
        let snapshot = self.snapshot.load();
        let state = self.write_state.read();

        let buffer_deleted = snapshot.id_to_index.len() - snapshot.active.len() as usize;
        let total_deleted = buffer_deleted + state.tombstones.len();

        BufferStats {
            vector_count: snapshot.active.len() as usize,
            deleted_count: total_deleted,
            max_size: self.max_size,
            dimensions: snapshot.dimensions,
        }
    }

    /// Clear all vectors from the buffer after compaction.
    pub async fn clear_compacted(&self, manifest_version: u64) {
        let old_version = self.last_manifest_version.swap(manifest_version, Ordering::SeqCst);
        if manifest_version > old_version && old_version > 0 {
            let mut state = self.write_state.write();

            let old_count = state.active.len();
            let old_tombstones = state.tombstones.len();

            // Reset all state
            *state = WriteState::new();

            // Publish empty snapshot
            self.snapshot.store(Arc::new(BufferSnapshot::new()));

            tracing::info!(
                "Buffer cleared after compaction: {} vectors, {} tombstones removed (version {} -> {})",
                old_count,
                old_tombstones,
                old_version,
                manifest_version
            );
        }
    }

    /// Prune tombstones that are already in persistent storage.
    pub async fn prune_tombstones(&self, persistent_tombstones: &HashSet<String>) -> usize {
        let mut state = self.write_state.write();
        let before = state.tombstones.len();
        state.tombstones.retain(|id| !persistent_tombstones.contains(id));
        before - state.tombstones.len()
    }

    /// Get the number of active vectors.
    pub async fn len(&self) -> usize {
        self.snapshot.load().active.len() as usize
    }

    /// Check if the buffer is empty.
    pub async fn is_empty(&self) -> bool {
        self.snapshot.load().active.is_empty()
    }

    /// Get dimensions of vectors.
    pub async fn dimensions(&self) -> Option<usize> {
        self.snapshot.load().dimensions
    }
}

/// Convert cosine distance to score.
fn distance_to_score(dist: f32) -> f32 {
    if dist.is_finite() {
        (1.0 - dist).max(0.0)
    } else {
        0.0
    }
}

fn matches_filters(attrs: Option<&AttrValue>, filters: &AttrValue) -> bool {
    let AttrValue::Object(filter_map) = filters else { return false };
    if filter_map.is_empty() {
        return false;
    }
    let Some(AttrValue::Object(attr_map)) = attrs else { return false };

    for (key, filter_val) in filter_map {
        let Some(attr_val) = attr_map.get(key) else { return false };
        if !value_matches(attr_val, filter_val) {
            return false;
        }
    }
    true
}

fn value_matches(attr: &AttrValue, filter: &AttrValue) -> bool {
    match filter {
        AttrValue::Array(items) => items.iter().any(|item| value_matches(attr, item)),
        _ => attr == filter,
    }
}

/// Buffer statistics.
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub vector_count: usize,
    pub deleted_count: usize,
    pub max_size: usize,
    pub dimensions: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, vector: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            vector,
            text: None,
            attributes: None,
        }
    }

    fn make_doc_with_attrs(id: &str, vector: Vec<f32>, attrs: AttrValue) -> Document {
        Document {
            id: id.to_string(),
            vector,
            text: None,
            attributes: Some(attrs),
        }
    }

    #[tokio::test]
    async fn test_insert_and_search() {
        let buffer = HotBuffer::new(100, TokenizerConfig::default(), 1.2, 0.75);

        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
            make_doc("c", vec![0.0, 0.0, 1.0]),
        ]).await;

        let results = buffer.search(&[1.0, 0.0, 0.0], 2, DistanceMetric::Cosine, None, false).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[tokio::test]
    async fn test_attribute_filtering() {
        let buffer = HotBuffer::new(100, TokenizerConfig::default(), 1.2, 0.75);

        let mut attrs1 = std::collections::BTreeMap::new();
        attrs1.insert("category".to_string(), AttrValue::String("tech".to_string()));
        
        let mut attrs2 = std::collections::BTreeMap::new();
        attrs2.insert("category".to_string(), AttrValue::String("sports".to_string()));

        buffer.insert(vec![
            make_doc_with_attrs("a", vec![1.0, 0.0, 0.0], AttrValue::Object(attrs1)),
            make_doc_with_attrs("b", vec![0.9, 0.1, 0.0], AttrValue::Object(attrs2)),
        ]).await;

        // Filter for tech category
        let mut filter = std::collections::BTreeMap::new();
        filter.insert("category".to_string(), AttrValue::String("tech".to_string()));
        let filter_val = AttrValue::Object(filter);

        let results = buffer.search(&[1.0, 0.0, 0.0], 10, DistanceMetric::Cosine, Some(&filter_val), false).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[tokio::test]
    async fn test_eviction() {
        let buffer = HotBuffer::new(2, TokenizerConfig::default(), 1.2, 0.75);

        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
        ]).await;
        assert_eq!(buffer.len().await, 2);

        buffer.insert(vec![make_doc("c", vec![0.0, 0.0, 1.0])]).await;
        assert_eq!(buffer.len().await, 2);

        // 'a' should be evicted (FIFO)
        let results = buffer.search(&[1.0, 0.0, 0.0], 10, DistanceMetric::Cosine, None, false).await;
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[tokio::test]
    async fn test_delete() {
        let buffer = HotBuffer::new(100, TokenizerConfig::default(), 1.2, 0.75);

        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
        ]).await;

        buffer.delete(vec!["a".to_string()]).await;

        let results = buffer.search(&[1.0, 0.0, 0.0], 10, DistanceMetric::Cosine, None, false).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }
}
