//! Hot buffer for real-time vector updates.
//!
//! Maintains an in-memory buffer of recently upserted vectors that haven't
//! been compacted into segments yet. Enables sub-second write-to-query latency.
//!
//! Uses an HNSW index for O(log n) search instead of brute-force O(n).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use roaring::RoaringBitmap;
use tokio::sync::RwLock;

use tidepool_common::attributes::AttrValue;
use tidepool_common::document::{Document, VectorResult};
use tidepool_common::index::hnsw::HnswIndex;
use tidepool_common::text::{DefaultTokenizer, Tokenizer, TokenizerConfig};
use tidepool_common::vector::DistanceMetric;

/// HNSW parameters tuned for small, frequently-updated indexes.
const BUFFER_HNSW_M: usize = 12;
const BUFFER_HNSW_EF_CONSTRUCTION: usize = 100;
const BUFFER_HNSW_EF_SEARCH: usize = 50;

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

/// Thread-safe hot buffer for real-time vector updates with HNSW indexing.
pub struct HotBuffer {
    /// HNSW index for fast ANN search
    index: RwLock<HnswIndex>,
    /// String ID -> internal index mapping
    id_to_index: RwLock<HashMap<String, usize>>,
    /// Internal index -> String ID mapping
    index_to_id: RwLock<Vec<String>>,
    /// Attributes for each vector (by internal index)
    attributes: RwLock<Vec<Option<AttrValue>>>,
    /// Text for each vector (by internal index)
    texts: RwLock<Vec<Option<String>>>,
    /// In-memory text index
    text_index: RwLock<TextBufferIndex>,
    /// Bitmap of active (non-deleted) vectors
    active: RwLock<RoaringBitmap>,
    /// Insertion order for FIFO eviction (stores string IDs)
    insertion_order: RwLock<Vec<String>>,
    /// Maximum number of vectors to hold
    max_size: usize,
    /// Last manifest version seen (for cleanup)
    last_manifest_version: AtomicU64,
    /// Dimensions (set on first insert)
    dimensions: RwLock<Option<usize>>,
    /// Tokenizer for text
    tokenizer: DefaultTokenizer,
    /// BM25 parameters
    bm25_k1: f32,
    bm25_b: f32,
}

impl HotBuffer {
    /// Create a new hot buffer with the given maximum size.
    pub fn new(max_size: usize, tokenizer_config: TokenizerConfig, bm25_k1: f32, bm25_b: f32) -> Self {
        Self {
            index: RwLock::new(HnswIndex::new(
                BUFFER_HNSW_M,
                BUFFER_HNSW_EF_CONSTRUCTION,
                BUFFER_HNSW_EF_SEARCH,
                DistanceMetric::Cosine, // Default, will be used per-query
            )),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
            attributes: RwLock::new(Vec::new()),
            texts: RwLock::new(Vec::new()),
            text_index: RwLock::new(TextBufferIndex::new()),
            active: RwLock::new(RoaringBitmap::new()),
            insertion_order: RwLock::new(Vec::new()),
            max_size,
            last_manifest_version: AtomicU64::new(0),
            dimensions: RwLock::new(None),
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

        let mut index = self.index.write().await;
        let mut id_to_index = self.id_to_index.write().await;
        let mut index_to_id = self.index_to_id.write().await;
        let mut attributes = self.attributes.write().await;
        let mut texts = self.texts.write().await;
        let mut text_index = self.text_index.write().await;
        let mut active = self.active.write().await;
        let mut order = self.insertion_order.write().await;
        let mut dims = self.dimensions.write().await;

        // Set dimensions on first insert
        if dims.is_none() {
            if let Some(doc) = docs.first() {
                if !doc.vector.is_empty() {
                    *dims = Some(doc.vector.len());
                }
            }
        }

        for doc in docs {
            if doc.vector.is_empty() {
                continue;
            }

            let string_id = doc.id.clone();

            let tokens = doc
                .text
                .as_deref()
                .map(|t| self.tokenizer.tokenize(t))
                .unwrap_or_default();

            if let Some(&existing_idx) = id_to_index.get(&string_id) {
                // Update existing vector - reinsert into HNSW at same index
                index.insert(existing_idx, doc.vector);
                attributes[existing_idx] = doc.attributes;
                if existing_idx < texts.len() {
                    texts[existing_idx] = doc.text.clone();
                }
                text_index.upsert(existing_idx, &tokens);
                active.insert(existing_idx as u32);
                // Don't change insertion order for updates
            } else {
                // New vector - allocate new index
                let new_idx = index_to_id.len();
                id_to_index.insert(string_id.clone(), new_idx);
                index_to_id.push(string_id.clone());
                attributes.push(doc.attributes);
                texts.push(doc.text.clone());
                
                // Insert into HNSW index
                index.insert(new_idx, doc.vector);
                text_index.upsert(new_idx, &tokens);
                active.insert(new_idx as u32);
                order.push(string_id);
            }
        }

        // Evict oldest if over capacity
        while active.len() as usize > self.max_size && !order.is_empty() {
            if let Some(oldest_id) = order.first().cloned() {
                order.remove(0);
                if let Some(&idx) = id_to_index.get(&oldest_id) {
                    active.remove(idx as u32);
                    text_index.remove_doc(idx);
                    if idx < texts.len() {
                        texts[idx] = None;
                    }
                    // Note: We don't remove from HNSW (expensive), just mark inactive
                }
            }
        }
    }

    /// Mark IDs as deleted.
    pub async fn delete(&self, ids: Vec<String>) {
        let id_to_index = self.id_to_index.read().await;
        let mut active = self.active.write().await;
        let mut order = self.insertion_order.write().await;
        let mut text_index = self.text_index.write().await;
        let mut texts = self.texts.write().await;

        for id in ids {
            if let Some(&idx) = id_to_index.get(&id) {
                active.remove(idx as u32);
                text_index.remove_doc(idx);
                if idx < texts.len() {
                    texts[idx] = None;
                }
            }
            order.retain(|x| x != &id);
        }
    }

    /// Search the buffer for similar vectors using HNSW index.
    /// Returns results sorted by score (descending).
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _metric: DistanceMetric, // HNSW uses its configured metric
        filters: Option<&AttrValue>,
        include_vectors: bool,
    ) -> Vec<VectorResult> {
        let index = self.index.read().await;
        let index_to_id = self.index_to_id.read().await;
        let attributes = self.attributes.read().await;
        let active = self.active.read().await;

        if active.is_empty() {
            return Vec::new();
        }

        // Use HNSW search with filter for active vectors
        let ef = (top_k * 2).max(BUFFER_HNSW_EF_SEARCH);
        let results = index.search_with_filter(query, top_k, ef, &active);

        results
            .into_iter()
            .filter_map(|r| {
                let string_id = index_to_id.get(r.id)?;
                if let Some(filter_value) = filters {
                    let attrs = attributes.get(r.id).cloned().flatten();
                    if !matches_filters(attrs.as_ref(), filter_value) {
                        return None;
                    }
                }
                Some(VectorResult {
                    id: string_id.clone(),
                    vector: if include_vectors {
                        index.nodes.get(r.id).map(|n| n.vector.clone()).unwrap_or_default()
                    } else {
                        Vec::new()
                    },
                    attributes: attributes.get(r.id).cloned().flatten(),
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
        let index = self.index.read().await;
        let index_to_id = self.index_to_id.read().await;
        let attributes = self.attributes.read().await;
        let active = self.active.read().await;
        let text_index = self.text_index.read().await;

        if active.is_empty() {
            return Vec::new();
        }

        let results = text_index.search(tokens, top_k, &active, self.bm25_k1, self.bm25_b);
        results
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let string_id = index_to_id.get(doc_id)?;
                if let Some(filter_value) = filters {
                    let attrs = attributes.get(doc_id).cloned().flatten();
                    if !matches_filters(attrs.as_ref(), filter_value) {
                        return None;
                    }
                }
                Some(VectorResult {
                    id: string_id.clone(),
                    vector: if include_vectors {
                        index.nodes.get(doc_id).map(|n| n.vector.clone()).unwrap_or_default()
                    } else {
                        Vec::new()
                    },
                    attributes: attributes.get(doc_id).cloned().flatten(),
                    score,
                })
            })
            .collect()
    }

    /// Check if an ID is deleted (not active).
    pub async fn is_deleted(&self, id: &str) -> bool {
        let id_to_index = self.id_to_index.read().await;
        let active = self.active.read().await;
        
        match id_to_index.get(id) {
            Some(&idx) => !active.contains(idx as u32),
            None => false, // Not in buffer at all
        }
    }

    /// Get all deleted IDs (IDs that were in buffer but marked deleted).
    pub async fn get_deleted_ids(&self) -> std::collections::HashSet<String> {
        let id_to_index = self.id_to_index.read().await;
        let active = self.active.read().await;
        
        id_to_index
            .iter()
            .filter(|(_, &idx)| !active.contains(idx as u32))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get buffer statistics.
    pub async fn stats(&self) -> BufferStats {
        let id_to_index = self.id_to_index.read().await;
        let active = self.active.read().await;
        let dims = self.dimensions.read().await;

        BufferStats {
            vector_count: active.len() as usize,
            deleted_count: id_to_index.len() - active.len() as usize,
            max_size: self.max_size,
            dimensions: *dims,
        }
    }

    /// Clear all vectors from the buffer after compaction.
    /// Called when manifest changes - all buffered data is now in segments.
    pub async fn clear_compacted(&self, manifest_version: u64) {
        let old_version = self.last_manifest_version.swap(manifest_version, Ordering::SeqCst);
        if manifest_version > old_version && old_version > 0 {
            // Clear all buffer state - compacted data is now in segments
            let mut index = self.index.write().await;
            let mut id_to_index = self.id_to_index.write().await;
            let mut index_to_id = self.index_to_id.write().await;
            let mut attributes = self.attributes.write().await;
            let mut texts = self.texts.write().await;
            let mut text_index = self.text_index.write().await;
            let mut active = self.active.write().await;
            let mut order = self.insertion_order.write().await;
            let mut dims = self.dimensions.write().await;

            let old_count = active.len();
            
            // Reset all state
            *index = HnswIndex::new(
                BUFFER_HNSW_M,
                BUFFER_HNSW_EF_CONSTRUCTION,
                BUFFER_HNSW_EF_SEARCH,
                DistanceMetric::Cosine,
            );
            id_to_index.clear();
            index_to_id.clear();
            attributes.clear();
            texts.clear();
            *text_index = TextBufferIndex::new();
            active.clear();
            order.clear();
            *dims = None;

            tracing::info!(
                "Buffer cleared after compaction: {} vectors removed (version {} -> {})",
                old_count,
                old_version,
                manifest_version
            );
        }
    }

    /// Get the number of active vectors in the buffer.
    pub async fn len(&self) -> usize {
        self.active.read().await.len() as usize
    }

    /// Check if the buffer is empty.
    pub async fn is_empty(&self) -> bool {
        self.active.read().await.is_empty()
    }

    /// Get dimensions of vectors in the buffer.
    pub async fn dimensions(&self) -> Option<usize> {
        *self.dimensions.read().await
    }
}

/// Convert cosine distance to score.
/// Cosine distance [0, 2]: 0=identical (score 1), 1=orthogonal (score 0), 2=opposite (score 0)
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
    async fn test_eviction() {
        let buffer = HotBuffer::new(2, TokenizerConfig::default(), 1.2, 0.75);

        // Use 3D vectors for HNSW to work properly
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

    #[tokio::test]
    async fn test_upsert_existing() {
        let buffer = HotBuffer::new(100, TokenizerConfig::default(), 1.2, 0.75);

        buffer.insert(vec![make_doc("a", vec![1.0, 0.0, 0.0])]).await;
        buffer.insert(vec![make_doc("a", vec![0.0, 1.0, 0.0])]).await;

        assert_eq!(buffer.len().await, 1);

        // After update, searching for the new vector should find it with score near 1
        let results = buffer.search(&[0.0, 1.0, 0.0], 1, DistanceMetric::Cosine, None, false).await;
        assert_eq!(results[0].id, "a");
        assert!(results[0].score > 0.99); // Should be very close to 1
    }
}
