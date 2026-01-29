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
use tidepool_common::vector::DistanceMetric;

/// HNSW parameters tuned for small, frequently-updated indexes.
const BUFFER_HNSW_M: usize = 12;
const BUFFER_HNSW_EF_CONSTRUCTION: usize = 100;
const BUFFER_HNSW_EF_SEARCH: usize = 50;

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
}

impl HotBuffer {
    /// Create a new hot buffer with the given maximum size.
    pub fn new(max_size: usize) -> Self {
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
            active: RwLock::new(RoaringBitmap::new()),
            insertion_order: RwLock::new(Vec::new()),
            max_size,
            last_manifest_version: AtomicU64::new(0),
            dimensions: RwLock::new(None),
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

            if let Some(&existing_idx) = id_to_index.get(&string_id) {
                // Update existing vector - reinsert into HNSW at same index
                index.insert(existing_idx, doc.vector);
                attributes[existing_idx] = doc.attributes;
                active.insert(existing_idx as u32);
                // Don't change insertion order for updates
            } else {
                // New vector - allocate new index
                let new_idx = index_to_id.len();
                id_to_index.insert(string_id.clone(), new_idx);
                index_to_id.push(string_id.clone());
                attributes.push(doc.attributes);
                
                // Insert into HNSW index
                index.insert(new_idx, doc.vector);
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

        for id in ids {
            if let Some(&idx) = id_to_index.get(&id) {
                active.remove(idx as u32);
            }
            order.retain(|x| x != &id);
        }
    }

    /// Search the buffer for similar vectors using HNSW index.
    /// Returns results sorted by distance (ascending).
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _metric: DistanceMetric, // HNSW uses its configured metric
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
                Some(VectorResult {
                    id: string_id.clone(),
                    vector: Vec::new(), // Don't include vectors by default
                    attributes: attributes.get(r.id).cloned().flatten(),
                    dist: r.dist,
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

    /// Clear vectors that are now in compacted segments.
    /// Called after compaction with the new manifest version.
    pub async fn clear_compacted(&self, manifest_version: u64) {
        let old_version = self.last_manifest_version.swap(manifest_version, Ordering::SeqCst);
        if manifest_version > old_version {
            // For now, we keep the buffer as-is.
            // Vectors will naturally age out via FIFO eviction.
            // A more sophisticated approach would track which vectors
            // were written before the compaction cutoff.
            tracing::debug!(
                "Buffer notified of compaction: version {} -> {}",
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
            attributes: None,
        }
    }

    #[tokio::test]
    async fn test_insert_and_search() {
        let buffer = HotBuffer::new(100);

        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
            make_doc("c", vec![0.0, 0.0, 1.0]),
        ]).await;

        let results = buffer.search(&[1.0, 0.0, 0.0], 2, DistanceMetric::Cosine).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[tokio::test]
    async fn test_eviction() {
        let buffer = HotBuffer::new(2);

        // Use 3D vectors for HNSW to work properly
        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
        ]).await;
        assert_eq!(buffer.len().await, 2);

        buffer.insert(vec![make_doc("c", vec![0.0, 0.0, 1.0])]).await;
        assert_eq!(buffer.len().await, 2);

        // 'a' should be evicted (FIFO)
        let results = buffer.search(&[1.0, 0.0, 0.0], 10, DistanceMetric::Cosine).await;
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[tokio::test]
    async fn test_delete() {
        let buffer = HotBuffer::new(100);

        buffer.insert(vec![
            make_doc("a", vec![1.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0]),
        ]).await;

        buffer.delete(vec!["a".to_string()]).await;

        let results = buffer.search(&[1.0, 0.0, 0.0], 10, DistanceMetric::Cosine).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[tokio::test]
    async fn test_upsert_existing() {
        let buffer = HotBuffer::new(100);

        buffer.insert(vec![make_doc("a", vec![1.0, 0.0, 0.0])]).await;
        buffer.insert(vec![make_doc("a", vec![0.0, 1.0, 0.0])]).await;

        assert_eq!(buffer.len().await, 1);

        // After update, searching for the new vector should find it with distance 0
        let results = buffer.search(&[0.0, 1.0, 0.0], 1, DistanceMetric::Cosine).await;
        assert_eq!(results[0].id, "a");
        assert!(results[0].dist < 0.01); // Should be very close to 0
    }
}
