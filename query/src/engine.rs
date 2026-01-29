use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::task::JoinSet;
use tracing::{info, warn};

use tidepool_common::document::{QueryRequest, QueryResponse, VectorResult};
use tidepool_common::manifest::{Manifest, Manager};
use tidepool_common::segment::{Reader, ReaderOptions, SegmentData};
use tidepool_common::storage::{segment_index_path, segment_ivf_path, segment_quant_path, Store};
use tidepool_common::tombstone::Manager as TombstoneManager;
use tidepool_common::vector::DistanceMetric;
use tidepool_common::wal::Reader as WalReader;

use crate::buffer::HotBuffer;

pub struct Engine<S: Store + Clone> {
    #[allow(dead_code)]
    storage: S,
    namespace: String,
    manifest_manager: Manager<S>,
    segment_reader: Reader<S>,
    tombstone_manager: TombstoneManager<S>,
    wal_reader: WalReader<S>,
    current_manifest: RwLock<Option<Manifest>>,
    loaded_segments: RwLock<HashMap<String, Arc<SegmentData>>>,
    tombstones: RwLock<HashSet<String>>,
    id_versions: RwLock<HashMap<String, usize>>,
    hot_buffer: Option<Arc<HotBuffer>>,
    /// Track the latest WAL entry timestamp we've processed
    last_wal_ts: RwLock<i64>,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub hnsw_ef_search: usize,
    pub quantization_rerank_factor: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            hnsw_ef_search: 0,
            quantization_rerank_factor: 4,
        }
    }
}

impl<S: Store + Clone + 'static> Engine<S> {
    pub fn new(storage: S, namespace: impl Into<String>, cache_dir: Option<String>) -> Self {
        Self::new_with_options(storage, namespace, cache_dir, EngineOptions::default())
    }

    pub fn new_with_options(
        storage: S,
        namespace: impl Into<String>,
        cache_dir: Option<String>,
        opts: EngineOptions,
    ) -> Self {
        Self::new_with_buffer(storage, namespace, cache_dir, opts, None)
    }

    pub fn new_with_buffer(
        storage: S,
        namespace: impl Into<String>,
        cache_dir: Option<String>,
        opts: EngineOptions,
        hot_buffer: Option<Arc<HotBuffer>>,
    ) -> Self {
        let namespace = namespace.into();
        let manifest_manager = Manager::new(storage.clone(), &namespace);
        let segment_reader = Reader::new_with_options(
            storage.clone(),
            &namespace,
            cache_dir.clone(),
            ReaderOptions {
                hnsw_ef_search: opts.hnsw_ef_search,
                quantization_rerank_factor: opts.quantization_rerank_factor,
            },
        );
        let wal_reader = WalReader::new(storage.clone(), &namespace);

        Self {
            storage: storage.clone(),
            tombstone_manager: TombstoneManager::new(storage, &namespace),
            namespace,
            manifest_manager,
            segment_reader,
            wal_reader,
            current_manifest: RwLock::new(None),
            loaded_segments: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashSet::new()),
            id_versions: RwLock::new(HashMap::new()),
            hot_buffer,
            last_wal_ts: RwLock::new(0),
        }
    }

    /// Get a reference to the hot buffer if available.
    pub fn hot_buffer(&self) -> Option<&Arc<HotBuffer>> {
        self.hot_buffer.as_ref()
    }

    pub async fn load_manifest(&self) -> Result<bool, String> {
        let manifest = self
            .manifest_manager
            .load()
            .await
            .map_err(|err| format!("failed to load manifest: {}", err))?;
        let mut guard = self.current_manifest.write().await;
        let changed = guard
            .as_ref()
            .map(|current| current.version != manifest.version)
            .unwrap_or(true);
        if changed {
            *guard = Some(manifest.clone());
            info!(
                "Loaded manifest version {} with {} segments",
                manifest.version,
                manifest.segments.len()
            );
        }
        Ok(changed)
    }

    pub async fn ensure_segments_loaded(&self) -> Result<(), String> {
        let manifest = { self.current_manifest.read().await.clone() };
        let Some(manifest) = manifest else {
            return Err("no manifest loaded".to_string());
        };

        self.prune_segments(&manifest).await;

        let mut to_load = Vec::new();
        {
            let guard = self.loaded_segments.read().await;
            for seg in &manifest.segments {
                if !guard.contains_key(&seg.id) {
                    to_load.push((seg.id.clone(), seg.segment_key.clone()));
                }
            }
        }

        if to_load.is_empty() {
            return Ok(());
        }

        let mut join_set = JoinSet::new();
        for (seg_id, segment_key) in to_load {
            let reader = self.segment_reader.clone();
            join_set.spawn(async move {
                let data = reader
                    .read_segment(&segment_key)
                    .await
                    .map_err(|err| format!("failed to load segment {}: {}", seg_id, err))?;
                Ok::<(String, SegmentData), String>((seg_id, data))
            });
        }

        let mut guard = self.loaded_segments.write().await;
        while let Some(res) = join_set.join_next().await {
            let (seg_id, data) = res
                .map_err(|err| format!("failed to join segment load: {}", err))??;
            guard.insert(seg_id, Arc::new(data));
        }

        Ok(())
    }

    async fn prune_segments(&self, manifest: &Manifest) {
        let keep: HashSet<String> = manifest.segments.iter().map(|s| s.id.clone()).collect();
        let mut guard = self.loaded_segments.write().await;
        guard.retain(|seg_id, _| keep.contains(seg_id));
    }

    async fn refresh_tombstones(&self) -> Result<(), String> {
        let tombstones = self
            .tombstone_manager
            .load()
            .await
            .map_err(|err| format!("failed to load tombstones: {}", err))?;
        let mut guard = self.tombstones.write().await;
        *guard = tombstones;
        Ok(())
    }

    async fn rebuild_id_versions(&self, manifest: &Manifest) {
        let segments = self.loaded_segments.read().await;
        let mut id_versions = HashMap::new();
        for (seg_index, seg_meta) in manifest.segments.iter().enumerate() {
            if let Some(seg_data) = segments.get(&seg_meta.id) {
                for id in &seg_data.ids {
                    id_versions.insert(id.clone(), seg_index);
                }
            }
        }
        let mut guard = self.id_versions.write().await;
        *guard = id_versions;
    }

    /// Scan WAL files from S3 and populate the hot buffer.
    /// Uses timestamp-based tracking to ensure new entries are always picked up.
    async fn scan_wal(&self) {
        let Some(buffer) = &self.hot_buffer else {
            return;
        };

        // Get manifest watermark (entries before this are already in segments)
        let manifest_created_at = {
            let manifest = self.current_manifest.read().await;
            manifest.as_ref().map(|m| m.created_at).unwrap_or(0)
        };

        // Get the last processed timestamp
        let last_ts = *self.last_wal_ts.read().await;

        // List all WAL files (always re-list to catch new files)
        let wal_files = match self.wal_reader.list_wal_files().await {
            Ok(files) => files,
            Err(err) => {
                warn!("Failed to list WAL files: {}", err);
                return;
            }
        };

        if wal_files.is_empty() {
            return;
        }

        // Read all WAL files and extract entries newer than last_ts
        let mut docs_to_insert = Vec::new();
        let mut ids_to_delete = Vec::new();
        let mut max_ts = last_ts;

        for wal_file in wal_files {
            match self.wal_reader.read_wal_file(&wal_file).await {
                Ok(entries) => {
                    for entry in entries {
                        // Skip entries already in segments (compacted)
                        if entry.ts <= manifest_created_at {
                            continue;
                        }

                        // Skip entries we've already processed
                        if entry.ts <= last_ts {
                            continue;
                        }

                        // Track max timestamp
                        if entry.ts > max_ts {
                            max_ts = entry.ts;
                        }

                        match entry.op.as_str() {
                            "upsert" => {
                                if let Some(doc) = entry.doc {
                                    docs_to_insert.push(doc);
                                }
                            }
                            "delete" => {
                                ids_to_delete.extend(entry.delete_ids);
                            }
                            _ => {}
                        }
                    }
                }
                Err(err) => {
                    warn!("Failed to read WAL file: {}", err);
                }
            }
        }

        // Update buffer with new entries
        if !docs_to_insert.is_empty() {
            let count = docs_to_insert.len();
            buffer.insert(docs_to_insert).await;
            info!("WAL scan: loaded {} vectors into hot buffer", count);
        }

        if !ids_to_delete.is_empty() {
            let count = ids_to_delete.len();
            buffer.delete(ids_to_delete).await;
            info!("WAL scan: applied {} deletes to hot buffer", count);
        }

        // Update last processed timestamp
        if max_ts > last_ts {
            let mut ts = self.last_wal_ts.write().await;
            *ts = max_ts;
        }
    }

    /// Reset WAL tracking and clear buffer when manifest changes.
    /// Compacted data is now in segments, so buffer should be cleared.
    async fn reset_wal_state(&self) {
        // Reset WAL timestamp tracking (will rescan from manifest watermark)
        {
            let mut ts = self.last_wal_ts.write().await;
            *ts = 0;
        }

        // Clear buffer when manifest changes (compacted data now in segments)
        if let Some(buffer) = &self.hot_buffer {
            let manifest = self.current_manifest.read().await;
            if let Some(m) = manifest.as_ref() {
                if let Ok(version) = m.version.parse::<u64>() {
                    buffer.clear_compacted(version).await;
                }
            }
        }
    }

    async fn refresh_state(&self) {
        let changed = self.load_manifest().await.unwrap_or(false);
        if changed {
            // Clean up stale cache files when manifest changes
            if let Some(manifest) = self.current_manifest.read().await.clone() {
                let mut valid_keys: Vec<String> = Vec::new();
                for seg in &manifest.segments {
                    valid_keys.push(seg.segment_key.clone());
                    valid_keys.push(segment_index_path(&self.namespace, &seg.id));
                    valid_keys.push(segment_ivf_path(&self.namespace, &seg.id));
                    valid_keys.push(segment_quant_path(&self.namespace, &seg.id));
                }
                let removed = self.segment_reader.cleanup_cache(&valid_keys).await;
                if removed > 0 {
                    info!("Cache cleanup: removed {} stale files", removed);
                }
            }
            // Reset WAL state when manifest changes (data now in segments)
            self.reset_wal_state().await;
        }
        let _ = self.ensure_segments_loaded().await;
        if changed || self.id_versions.read().await.is_empty() {
            let _ = self.refresh_tombstones().await;
            if let Some(manifest) = self.current_manifest.read().await.clone() {
                self.rebuild_id_versions(&manifest).await;
            }
        }

        // Scan WAL for recent writes (provides real-time visibility)
        self.scan_wal().await;
    }

    pub async fn query(&self, req: &QueryRequest) -> Result<QueryResponse, String> {
        self.refresh_state().await;

        let top_k = if req.top_k == 0 { 10 } else { req.top_k };
        let metric = DistanceMetric::parse(req.distance_metric.as_deref());

        // Search hot buffer first (if available)
        let buffer_results = if let Some(buffer) = &self.hot_buffer {
            buffer.search(&req.vector, top_k * 2, metric).await
        } else {
            Vec::new()
        };

        // Get buffer tombstones
        let buffer_deleted = if let Some(buffer) = &self.hot_buffer {
            buffer.get_deleted_ids().await
        } else {
            HashSet::new()
        };

        // Get IDs in buffer (for deduplication with segments)
        let buffer_ids: HashSet<String> = buffer_results.iter().map(|r| r.id.clone()).collect();

        let manifest = { self.current_manifest.read().await.clone() };
        
        // If no manifest or segments, return buffer results only
        let manifest = match manifest {
            Some(m) if !m.segments.is_empty() => m,
            _ => {
                let mut results = buffer_results;
                results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(top_k);
                return Ok(QueryResponse {
                    results,
                    namespace: self.namespace.clone(),
                });
            }
        };

        let segments = { self.loaded_segments.read().await.clone() };
        let per_segment_k = top_k.saturating_mul(2).max(top_k + 10);
        let query_vector = req.vector.clone();
        let filters = req.filters.clone();
        let include_vectors = req.include_vectors;
        let ef_search = req.ef_search;
        let nprobe = req.nprobe;
        let mut handles = Vec::new();
        for (seg_index, seg_meta) in manifest.segments.iter().enumerate() {
            let Some(seg_data) = segments.get(&seg_meta.id) else { continue };
            let seg_data = Arc::clone(seg_data);
            let query_vector = query_vector.clone();
            let filters = filters.clone();
            let nprobe = nprobe;
            let handle = tokio::task::spawn_blocking(move || {
                let max_k = seg_data.len().min(per_segment_k);
                let results = seg_data.search(&query_vector, max_k, metric, filters.as_ref(), ef_search, nprobe);
                let mut out = Vec::with_capacity(results.len());
                for r in results {
                    let mut result = VectorResult {
                        id: seg_data.ids[r.index].clone(),
                        vector: Vec::new(),
                        attributes: seg_data.attributes[r.index].clone(),
                        dist: r.dist,
                    };
                    if include_vectors {
                        result.vector = seg_data.vector_owned(r.index).unwrap_or_default();
                    }
                    out.push(result);
                }
                Ok::<(usize, Vec<VectorResult>), String>((seg_index, out))
            });
            handles.push(handle);
        }

        let mut segment_results = Vec::with_capacity(handles.len());
        for handle in handles {
            let (seg_index, results) = handle
                .await
                .map_err(|err| format!("failed to join query task: {}", err))??;
            segment_results.push((seg_index, results));
        }

        let tombstones = self.tombstones.read().await;
        let id_versions = self.id_versions.read().await;
        let mut all_results = Vec::new();
        
        // Add segment results, filtering out:
        // - Tombstoned IDs (from S3 tombstones)
        // - IDs deleted in buffer
        // - IDs that exist in buffer (buffer version is fresher)
        // - Duplicate IDs from older segments
        for (seg_index, mut results) in segment_results {
            results.retain(|r| {
                // Skip if in S3 tombstones
                if tombstones.contains(&r.id) {
                    return false;
                }
                // Skip if deleted in buffer
                if buffer_deleted.contains(&r.id) {
                    return false;
                }
                // Skip if exists in buffer (buffer has fresher version)
                if buffer_ids.contains(&r.id) {
                    return false;
                }
                // Skip if not latest segment version
                match id_versions.get(&r.id) {
                    Some(&latest) => latest == seg_index,
                    None => true,
                }
            });
            all_results.extend(results);
        }

        // Add buffer results (already filtered for buffer tombstones)
        all_results.extend(buffer_results);

        // Sort and truncate
        all_results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        if all_results.len() > top_k {
            all_results.truncate(top_k);
        }

        Ok(QueryResponse {
            results: all_results,
            namespace: self.namespace.clone(),
        })
    }

    pub async fn get_manifest(&self) -> Option<Manifest> {
        self.current_manifest.read().await.clone()
    }

    pub async fn get_stats(&self) -> Stats {
        self.refresh_state().await;
        let manifest = self.current_manifest.read().await;
        let segments = self.loaded_segments.read().await;
        let tombstones = self.tombstones.read().await;
        let id_versions = self.id_versions.read().await;
        
        // Get buffer stats
        let (buffer_vectors, buffer_deleted) = if let Some(buffer) = &self.hot_buffer {
            let stats = buffer.stats().await;
            (stats.vector_count, stats.deleted_count)
        } else {
            (0, 0)
        };

        let mut stats = Stats {
            namespace: self.namespace.clone(),
            manifest_version: None,
            segment_count: 0,
            total_vectors: 0,
            dimensions: 0,
            loaded_segments: segments.len(),
            buffer_vectors,
            buffer_deleted,
        };
        if let Some(manifest) = manifest.as_ref() {
            stats.manifest_version = Some(manifest.version.clone());
            stats.segment_count = manifest.segments.len();
            if !id_versions.is_empty() {
                let active = id_versions.len().saturating_sub(tombstones.len());
                stats.total_vectors = active as i64;
            } else {
                stats.total_vectors = manifest.total_doc_count().saturating_sub(tombstones.len() as i64);
            }
            // Add buffer vectors to total (approximate, may have overlap)
            stats.total_vectors += buffer_vectors as i64;
            stats.dimensions = manifest.dimensions;
        } else if buffer_vectors > 0 {
            // No manifest yet but buffer has vectors
            stats.total_vectors = buffer_vectors as i64;
            if let Some(buffer) = &self.hot_buffer {
                if let Some(dims) = buffer.dimensions().await {
                    stats.dimensions = dims;
                }
            }
        }
        stats
    }

    pub async fn invalidate_cache(&self) {
        let mut segments = self.loaded_segments.write().await;
        segments.clear();
        let mut tombstones = self.tombstones.write().await;
        tombstones.clear();
        let mut versions = self.id_versions.write().await;
        versions.clear();
        let mut manifest = self.current_manifest.write().await;
        *manifest = None;
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub namespace: String,
    pub manifest_version: Option<String>,
    pub segment_count: usize,
    pub total_vectors: i64,
    pub dimensions: usize,
    pub loaded_segments: usize,
    pub buffer_vectors: usize,
    pub buffer_deleted: usize,
}
