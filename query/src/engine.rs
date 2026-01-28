use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::task::JoinSet;
use tracing::info;

use tidepool_common::document::{QueryRequest, QueryResponse, VectorResult};
use tidepool_common::manifest::{Manifest, Manager};
use tidepool_common::segment::{Reader, ReaderOptions, SegmentData};
use tidepool_common::storage::Store;
use tidepool_common::tombstone::Manager as TombstoneManager;
use tidepool_common::vector::DistanceMetric;

pub struct Engine<S: Store + Clone> {
    namespace: String,
    manifest_manager: Manager<S>,
    segment_reader: Reader<S>,
    tombstone_manager: TombstoneManager<S>,
    current_manifest: RwLock<Option<Manifest>>,
    loaded_segments: RwLock<HashMap<String, Arc<SegmentData>>>,
    tombstones: RwLock<HashSet<String>>,
    id_versions: RwLock<HashMap<String, usize>>,
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub hnsw_ef_search: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self { hnsw_ef_search: 0 }
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
        let namespace = namespace.into();
        let manifest_manager = Manager::new(storage.clone(), &namespace);
        let segment_reader = Reader::new_with_options(
            storage.clone(),
            &namespace,
            cache_dir.clone(),
            ReaderOptions {
                hnsw_ef_search: opts.hnsw_ef_search,
            },
        );

        Self {
            tombstone_manager: TombstoneManager::new(storage, &namespace),
            namespace,
            manifest_manager,
            segment_reader,
            current_manifest: RwLock::new(None),
            loaded_segments: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashSet::new()),
            id_versions: RwLock::new(HashMap::new()),
        }
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

    async fn refresh_state(&self) {
        let changed = self.load_manifest().await.unwrap_or(false);
        if changed {
            // Clean up stale cache files when manifest changes
            if let Some(manifest) = self.current_manifest.read().await.clone() {
                let valid_keys: Vec<String> = manifest
                    .segments
                    .iter()
                    .map(|s| s.segment_key.clone())
                    .collect();
                let removed = self.segment_reader.cleanup_cache(&valid_keys).await;
                if removed > 0 {
                    info!("Cache cleanup: removed {} stale files", removed);
                }
            }
        }
        let _ = self.ensure_segments_loaded().await;
        if changed || self.id_versions.read().await.is_empty() {
            let _ = self.refresh_tombstones().await;
            if let Some(manifest) = self.current_manifest.read().await.clone() {
                self.rebuild_id_versions(&manifest).await;
            }
        }
    }

    pub async fn query(&self, req: &QueryRequest) -> Result<QueryResponse, String> {
        self.refresh_state().await;

        let manifest = { self.current_manifest.read().await.clone() };
        if manifest.is_none() {
            return Ok(QueryResponse {
                results: Vec::new(),
                namespace: self.namespace.clone(),
            });
        }
        let manifest = manifest.unwrap();

        if manifest.segments.is_empty() {
            return Ok(QueryResponse {
                results: Vec::new(),
                namespace: self.namespace.clone(),
            });
        }

        let top_k = if req.top_k == 0 { 10 } else { req.top_k };
        let metric = DistanceMetric::parse(req.distance_metric.as_deref());

        let segments = { self.loaded_segments.read().await.clone() };
        let per_segment_k = top_k.saturating_mul(2).max(top_k + 10);
        let query_vector = req.vector.clone();
        let filters = req.filters.clone();
        let include_vectors = req.include_vectors;
        let ef_search = req.ef_search;
        let mut handles = Vec::new();
        for (seg_index, seg_meta) in manifest.segments.iter().enumerate() {
            let Some(seg_data) = segments.get(&seg_meta.id) else { continue };
            let seg_data = Arc::clone(seg_data);
            let query_vector = query_vector.clone();
            let filters = filters.clone();
            let handle = tokio::task::spawn_blocking(move || {
                let max_k = seg_data.vectors.len().min(per_segment_k);
                let results = seg_data.search(&query_vector, max_k, metric, filters.as_ref(), ef_search);
                let mut out = Vec::with_capacity(results.len());
                for r in results {
                    let mut result = VectorResult {
                        id: seg_data.ids[r.index].clone(),
                        vector: Vec::new(),
                        attributes: seg_data.attributes[r.index].clone(),
                        dist: r.dist,
                    };
                    if include_vectors {
                        result.vector = seg_data.vectors[r.index].clone();
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
        for (seg_index, mut results) in segment_results {
            results.retain(|r| {
                if tombstones.contains(&r.id) {
                    return false;
                }
                match id_versions.get(&r.id) {
                    Some(&latest) => latest == seg_index,
                    None => true,
                }
            });
            all_results.extend(results);
        }

        all_results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
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
        let mut stats = Stats {
            namespace: self.namespace.clone(),
            manifest_version: None,
            segment_count: 0,
            total_vectors: 0,
            dimensions: 0,
            loaded_segments: segments.len(),
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
            stats.dimensions = manifest.dimensions;
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
}
