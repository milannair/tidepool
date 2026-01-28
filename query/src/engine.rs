use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::info;

use tidepool_common::document::{QueryRequest, QueryResponse, VectorResult};
use tidepool_common::manifest::{Manifest, Manager};
use tidepool_common::segment::{Reader, ReaderOptions, SegmentData};
use tidepool_common::storage::Store;
use tidepool_common::vector::DistanceMetric;

pub struct Engine<S: Store + Clone> {
    namespace: String,
    manifest_manager: Manager<S>,
    segment_reader: Reader<S>,
    current_manifest: RwLock<Option<Manifest>>,
    loaded_segments: RwLock<HashMap<String, Arc<SegmentData>>>,
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

impl<S: Store + Clone> Engine<S> {
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
            namespace,
            manifest_manager,
            segment_reader,
            current_manifest: RwLock::new(None),
            loaded_segments: RwLock::new(HashMap::new()),
        }
    }

    pub async fn load_manifest(&self) -> Result<(), String> {
        let manifest = self
            .manifest_manager
            .load()
            .await
            .map_err(|err| format!("failed to load manifest: {}", err))?;
        {
            let mut guard = self.current_manifest.write().await;
            *guard = Some(manifest.clone());
        }
        info!(
            "Loaded manifest version {} with {} segments",
            manifest.version,
            manifest.segments.len()
        );
        Ok(())
    }

    pub async fn ensure_segments_loaded(&self) -> Result<(), String> {
        let manifest = { self.current_manifest.read().await.clone() };
        let Some(manifest) = manifest else {
            return Err("no manifest loaded".to_string());
        };

        for seg in &manifest.segments {
            self.load_segment_if_needed(seg.id.clone(), seg.segment_key.clone())
                .await?;
        }

        Ok(())
    }

    async fn load_segment_if_needed(&self, seg_id: String, segment_key: String) -> Result<(), String> {
        {
            let guard = self.loaded_segments.read().await;
            if guard.contains_key(&seg_id) {
                return Ok(());
            }
        }

        info!("Loading segment {}", seg_id);
        let seg_data = self
            .segment_reader
            .read_segment(&segment_key)
            .await
            .map_err(|err| format!("failed to load segment {}: {}", seg_id, err))?;

        let mut guard = self.loaded_segments.write().await;
        guard.insert(seg_id, Arc::new(seg_data));
        Ok(())
    }

    pub async fn query(&self, req: &QueryRequest) -> Result<QueryResponse, String> {
        let _ = self.load_manifest().await;
        let _ = self.ensure_segments_loaded().await;

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

        let mut all_results = Vec::new();

        let segments = { self.loaded_segments.read().await.clone() };
        for seg in &manifest.segments {
            let Some(seg_data) = segments.get(&seg.id) else { continue };
            let results = seg_data.search(
                &req.vector,
                top_k,
                metric,
                req.filters.as_ref(),
                req.ef_search,
            );
            for r in results {
                let mut result = VectorResult {
                    id: seg_data.ids[r.index].clone(),
                    vector: Vec::new(),
                    attributes: seg_data.attributes[r.index].clone(),
                    dist: r.dist,
                };
                if req.include_vectors {
                    result.vector = seg_data.vectors[r.index].clone();
                }
                all_results.push(result);
            }
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
        let manifest = self.current_manifest.read().await;
        let segments = self.loaded_segments.read().await;
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
            stats.total_vectors = manifest.total_doc_count();
            stats.dimensions = manifest.dimensions;
        }
        stats
    }

    pub async fn invalidate_cache(&self) {
        let mut segments = self.loaded_segments.write().await;
        segments.clear();
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
