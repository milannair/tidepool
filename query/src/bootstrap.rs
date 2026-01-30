//! Two-phase bootstrap for cold-start-safe startup.
//!
//! Phase 1 (Control Plane): Load manifests, Bloom filters, and metadata only.
//!                          Service can start accepting queries immediately.
//! Phase 2 (Data Plane):    Background rehydration of segment data based on priority.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{info, warn};

use tidepool_common::bloom::BloomFilter;
use tidepool_common::manifest::{parse_manifest_bytes, Manifest, Segment};
use tidepool_common::storage::{
    latest_manifest_path, segment_index_path, segment_ivf_path,
    segment_quant_path, segment_text_index_path, tombstone_path, LocalStore, S3Store,
    StorageError, Store,
};

/// State after Phase 1 bootstrap (control plane ready).
#[derive(Debug)]
pub struct BootstrapState {
    /// Manifests by namespace.
    pub manifests: HashMap<String, Manifest>,
    /// Bloom filters by segment ID.
    pub blooms: HashMap<String, BloomFilter>,
    /// Segments that are locally present (segment_id -> content_hash).
    pub local_segments: HashSet<String>,
    /// Segments to download in Phase 2 (priority ordered).
    pub to_download: Vec<SegmentDownloadInfo>,
    /// Statistics.
    pub stats: Phase1Stats,
}

#[derive(Debug, Clone)]
pub struct SegmentDownloadInfo {
    pub namespace: String,
    pub segment: Segment,
    /// Priority score (higher = download first). Based on recency, size, etc.
    pub priority: i64,
}

#[derive(Debug, Clone, Default)]
pub struct Phase1Stats {
    pub duration: Duration,
    pub namespaces: usize,
    pub segments: usize,
    pub blooms_loaded: usize,
    pub bytes_downloaded: u64,
}

/// Local Merkle state: tracks which segments are locally present.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LocalMerkleState {
    /// Map of segment_id -> content_hash for locally present segments.
    pub segments: HashMap<String, String>,
}

impl LocalMerkleState {
    pub fn load(data_dir: &str) -> Self {
        let path = format!("{}/local_merkle.json", data_dir);
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, data_dir: &str) -> Result<(), String> {
        let path = format!("{}/local_merkle.json", data_dir);
        let contents = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, contents).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn has_segment(&self, segment_id: &str, content_hash: &Option<String>) -> bool {
        match (self.segments.get(segment_id), content_hash) {
            (Some(local_hash), Some(remote_hash)) => local_hash == remote_hash,
            (Some(_), None) => true, // No remote hash to compare, assume valid
            (None, _) => false,
        }
    }

    pub fn add_segment(&mut self, segment_id: &str, content_hash: &str) {
        self.segments.insert(segment_id.to_string(), content_hash.to_string());
    }

    pub fn remove_segment(&mut self, segment_id: &str) {
        self.segments.remove(segment_id);
    }
}

/// Bootstrap manager for two-phase startup.
pub struct Bootstrap {
    remote: S3Store,
    local_dir: PathBuf,
    #[allow(dead_code)]
    max_local_disk: usize,
    #[allow(dead_code)]
    target_local_disk: usize,
    #[allow(dead_code)]
    eager_sync_all: bool,
}

impl Bootstrap {
    pub fn new(
        remote: S3Store,
        local_dir: PathBuf,
        max_local_disk: usize,
        target_local_disk: usize,
        eager_sync_all: bool,
    ) -> Self {
        Self {
            remote,
            local_dir,
            max_local_disk,
            target_local_disk,
            eager_sync_all,
        }
    }

    /// Phase 1: Load manifests, Bloom filters, and build segment registry.
    /// This is fast and allows the service to start accepting queries immediately.
    pub async fn phase1(&self) -> Result<BootstrapState, String> {
        let start = Instant::now();
        let local = LocalStore::new(&self.local_dir);

        let mut manifests = HashMap::new();
        let mut blooms = HashMap::new();
        let mut all_segments = Vec::new();
        let mut bytes_downloaded: u64 = 0;

        // List namespaces from S3
        let namespaces = self.list_namespaces().await?;

        for ns in &namespaces {
            // Download manifest
            let manifest_key = latest_manifest_path(ns);
            let manifest_bytes = match self.remote.get(&manifest_key).await {
                Ok(data) => data,
                Err(StorageError::NotFound(_)) => continue,
                Err(e) => {
                    warn!("Failed to get manifest for {}: {}", ns, e);
                    continue;
                }
            };

            // Save manifest locally
            if let Err(e) = local.put(&manifest_key, manifest_bytes.clone()).await {
                warn!("Failed to save manifest locally for {}: {}", ns, e);
            }
            bytes_downloaded += manifest_bytes.len() as u64;

            let manifest = match parse_manifest_bytes(&manifest_bytes) {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to parse manifest for {}: {}", ns, e);
                    continue;
                }
            };

            // Download Bloom filters for each segment (small files)
            for seg in &manifest.segments {
                if let Some(bloom_key) = &seg.bloom_key {
                    match self.remote.get(bloom_key).await {
                        Ok(bloom_data) => {
                            bytes_downloaded += bloom_data.len() as u64;
                            // Save Bloom locally
                            let _ = local.put(bloom_key, bloom_data.clone()).await;
                            if let Some(bloom) = BloomFilter::from_bytes(&bloom_data) {
                                blooms.insert(seg.id.clone(), bloom);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to get Bloom filter for segment {}: {}", seg.id, e);
                        }
                    }
                }

                all_segments.push((ns.clone(), seg.clone()));
            }

            // Download tombstones (small file)
            let tomb_key = tombstone_path(ns);
            if let Ok(data) = self.remote.get(&tomb_key).await {
                bytes_downloaded += data.len() as u64;
                let _ = local.put(&tomb_key, data).await;
            }

            manifests.insert(ns.clone(), manifest);
        }

        // Load local Merkle state to determine which segments are already present
        let merkle_state = LocalMerkleState::load(self.local_dir.to_str().unwrap_or("/data"));
        let local_segments: HashSet<String> = merkle_state.segments.keys().cloned().collect();

        // Compute segments to download (not locally present)
        let mut to_download: Vec<SegmentDownloadInfo> = all_segments
            .into_iter()
            .filter(|(_, seg)| !merkle_state.has_segment(&seg.id, &seg.content_hash))
            .map(|(ns, seg)| {
                // Priority: higher generation (more recent) = higher priority
                // Also factor in size (smaller segments first for quick wins)
                let manifest = manifests.get(&ns);
                let generation = manifest.map(|m| m.generation).unwrap_or(0);
                let size_factor = if seg.size_bytes > 0 {
                    // Invert size: smaller segments get higher priority
                    (1_000_000_000 / seg.size_bytes.max(1)) as i64
                } else {
                    0
                };
                let priority = (generation as i64 * 1000) + size_factor;

                SegmentDownloadInfo {
                    namespace: ns,
                    segment: seg,
                    priority,
                }
            })
            .collect();

        // Sort by priority (highest first)
        to_download.sort_by(|a, b| b.priority.cmp(&a.priority));

        let stats = Phase1Stats {
            duration: start.elapsed(),
            namespaces: manifests.len(),
            segments: to_download.len() + local_segments.len(),
            blooms_loaded: blooms.len(),
            bytes_downloaded,
        };

        info!(
            "Phase 1 complete in {:?}: {} namespaces, {} segments ({} local, {} to download), {} Bloom filters, {} bytes",
            stats.duration,
            stats.namespaces,
            stats.segments,
            local_segments.len(),
            to_download.len(),
            stats.blooms_loaded,
            stats.bytes_downloaded
        );

        Ok(BootstrapState {
            manifests,
            blooms,
            local_segments,
            to_download,
            stats,
        })
    }

    async fn list_namespaces(&self) -> Result<Vec<String>, String> {
        let keys = self
            .remote
            .list("namespaces/")
            .await
            .map_err(|e| format!("list namespaces: {}", e))?;
        let mut namespaces = HashSet::new();
        for key in keys {
            if let Some(ns) = parse_namespace_from_key(&key) {
                namespaces.insert(ns.to_string());
            }
        }
        let mut list: Vec<String> = namespaces.into_iter().collect();
        list.sort();
        Ok(list)
    }
}

/// Background rehydration task for Phase 2.
pub struct Rehydrator {
    remote: S3Store,
    local_dir: PathBuf,
    max_local_disk: usize,
    /// Segments waiting to be downloaded.
    queue: Arc<RwLock<Vec<SegmentDownloadInfo>>>,
    /// Local Merkle state.
    merkle_state: Arc<RwLock<LocalMerkleState>>,
    /// Callback when a segment is downloaded.
    on_segment_ready: Option<Box<dyn Fn(&str, &str) + Send + Sync>>,
}

impl Rehydrator {
    pub fn new(
        remote: S3Store,
        local_dir: PathBuf,
        max_local_disk: usize,
        initial_queue: Vec<SegmentDownloadInfo>,
    ) -> Self {
        let merkle_state = LocalMerkleState::load(local_dir.to_str().unwrap_or("/data"));
        Self {
            remote,
            local_dir,
            max_local_disk,
            queue: Arc::new(RwLock::new(initial_queue)),
            merkle_state: Arc::new(RwLock::new(merkle_state)),
            on_segment_ready: None,
        }
    }

    /// Get current local disk usage.
    async fn local_disk_usage(&self) -> u64 {
        let state = self.merkle_state.read().await;
        // Sum up sizes from segments we have
        // For now, just count number of segments * estimated size
        // TODO: Track actual sizes in merkle state
        (state.segments.len() * 10_000_000) as u64 // Rough estimate
    }

    /// Run the rehydration loop. Never returns.
    pub async fn run(&self) {
        loop {
            let segment_info = {
                let mut queue = self.queue.write().await;
                queue.pop()
            };

            let Some(info) = segment_info else {
                // Queue empty, sleep and check again
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            };

            // Check disk budget
            let usage = self.local_disk_usage().await;
            if usage + info.segment.size_bytes > self.max_local_disk as u64 {
                info!(
                    "Disk budget exceeded ({} + {} > {}), skipping segment {}",
                    usage, info.segment.size_bytes, self.max_local_disk, info.segment.id
                );
                // TODO: Could trigger eviction here
                continue;
            }

            // Download segment
            if let Err(e) = self.download_segment(&info).await {
                warn!("Failed to download segment {}: {}", info.segment.id, e);
                // Re-add to queue with lower priority
                let mut queue = self.queue.write().await;
                let mut retry_info = info;
                retry_info.priority -= 1000; // Lower priority for retry
                queue.push(retry_info);
                queue.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
        }
    }

    async fn download_segment(&self, info: &SegmentDownloadInfo) -> Result<(), String> {
        let local = LocalStore::new(&self.local_dir);
        let seg = &info.segment;
        let ns = &info.namespace;

        info!("Downloading segment {} ({} bytes)", seg.id, seg.size_bytes);
        let start = Instant::now();

        // Download segment data
        let segment_data = self.remote.get(&seg.segment_key).await
            .map_err(|e| format!("get segment: {}", e))?;
        local.put(&seg.segment_key, segment_data).await
            .map_err(|e| format!("put segment: {}", e))?;

        // Download index files
        for key in [
            segment_index_path(ns, &seg.id),
            segment_ivf_path(ns, &seg.id),
            segment_quant_path(ns, &seg.id),
            segment_text_index_path(ns, &seg.id),
        ] {
            if let Ok(data) = self.remote.get(&key).await {
                let _ = local.put(&key, data).await;
            }
        }

        // Update local Merkle state
        if let Some(hash) = &seg.content_hash {
            let mut state = self.merkle_state.write().await;
            state.add_segment(&seg.id, hash);
            let _ = state.save(self.local_dir.to_str().unwrap_or("/data"));
        }

        info!(
            "Downloaded segment {} in {:?}",
            seg.id,
            start.elapsed()
        );

        // Notify callback
        if let Some(callback) = &self.on_segment_ready {
            callback(ns, &seg.id);
        }

        Ok(())
    }

    /// Add segments to the download queue.
    pub async fn enqueue(&self, segments: Vec<SegmentDownloadInfo>) {
        let mut queue = self.queue.write().await;
        queue.extend(segments);
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Check if a segment is locally present.
    pub async fn is_local(&self, segment_id: &str) -> bool {
        let state = self.merkle_state.read().await;
        state.segments.contains_key(segment_id)
    }

    /// Get the local Merkle state.
    pub async fn merkle_state(&self) -> LocalMerkleState {
        self.merkle_state.read().await.clone()
    }
}

fn parse_namespace_from_key(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("namespaces/")?;
    let ns = rest.split('/').next()?;
    if ns.is_empty() {
        None
    } else {
        Some(ns)
    }
}
