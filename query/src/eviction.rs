//! Disk budget enforcement and background segment eviction.
//!
//! Monitors local disk usage and evicts least-recently-accessed segments
//! when usage exceeds MAX_LOCAL_DISK, targeting TARGET_LOCAL_DISK.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::{info, warn};

use tidepool_common::manifest::parse_manifest_bytes;
use tidepool_common::storage::{
    latest_manifest_path, segment_index_path, segment_ivf_path, segment_quant_path,
    segment_text_index_path, LocalStore, Store,
};

use crate::bootstrap::LocalMerkleState;

/// Tracks segment access patterns for LRU eviction.
#[derive(Debug, Clone, Default)]
pub struct AccessTracker {
    /// Last access time by segment ID.
    access_times: HashMap<String, u64>,
}

impl AccessTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record access to a segment.
    pub fn record_access(&mut self, segment_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.access_times.insert(segment_id.to_string(), now);
    }

    /// Get last access time for a segment (0 if never accessed).
    pub fn last_access(&self, segment_id: &str) -> u64 {
        self.access_times.get(segment_id).copied().unwrap_or(0)
    }

    /// Get all segment IDs sorted by last access time (oldest first).
    pub fn oldest_first(&self, segment_ids: &[String]) -> Vec<String> {
        let mut ids: Vec<_> = segment_ids
            .iter()
            .map(|id| (id.clone(), self.last_access(id)))
            .collect();
        ids.sort_by_key(|(_, access)| *access);
        ids.into_iter().map(|(id, _)| id).collect()
    }
}

/// Segment info for eviction decisions.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub id: String,
    pub namespace: String,
    pub segment_key: String,
    pub size_bytes: u64,
}

/// Background eviction manager.
pub struct Evictor {
    local_dir: PathBuf,
    max_local_disk: usize,
    target_local_disk: usize,
    access_tracker: Arc<RwLock<AccessTracker>>,
}

impl Evictor {
    pub fn new(
        local_dir: PathBuf,
        max_local_disk: usize,
        target_local_disk: usize,
    ) -> Self {
        Self {
            local_dir,
            max_local_disk,
            target_local_disk,
            access_tracker: Arc::new(RwLock::new(AccessTracker::new())),
        }
    }

    /// Get the access tracker for recording segment accesses.
    pub fn access_tracker(&self) -> Arc<RwLock<AccessTracker>> {
        Arc::clone(&self.access_tracker)
    }

    /// Calculate current local disk usage from tracked segment sizes.
    pub async fn disk_usage(&self) -> u64 {
        // Use LocalMerkleState for accurate segment size tracking
        let merkle_state = LocalMerkleState::load(self.local_dir.to_str().unwrap_or("/data"));
        merkle_state.total_size_bytes()
    }

    /// Check if eviction is needed.
    pub async fn needs_eviction(&self) -> bool {
        let usage = self.disk_usage().await;
        usage > self.max_local_disk as u64
    }

    /// Run eviction if needed. Returns number of segments evicted.
    pub async fn evict_if_needed(&self) -> Result<usize, String> {
        let usage = self.disk_usage().await;
        if usage <= self.max_local_disk as u64 {
            return Ok(0);
        }

        info!(
            "Disk usage {} exceeds max {}, starting eviction (target: {})",
            usage, self.max_local_disk, self.target_local_disk
        );

        let to_free = usage.saturating_sub(self.target_local_disk as u64);
        self.evict_bytes(to_free).await
    }

    /// Evict segments until at least `bytes_to_free` are freed.
    async fn evict_bytes(&self, bytes_to_free: u64) -> Result<usize, String> {
        let local = LocalStore::new(&self.local_dir);

        // Load Merkle state to get list of local segments
        let mut merkle_state = LocalMerkleState::load(self.local_dir.to_str().unwrap_or("/data"));
        let segment_ids: Vec<String> = merkle_state.segments.keys().cloned().collect();

        if segment_ids.is_empty() {
            return Ok(0);
        }

        // Get segment info from manifests
        let mut segments: Vec<SegmentInfo> = Vec::new();
        let namespaces = self.list_namespaces(&local).await;

        for ns in namespaces {
            let manifest_key = latest_manifest_path(&ns);
            let manifest_bytes = match local.get(&manifest_key).await {
                Ok(data) => data,
                Err(_) => continue,
            };
            let manifest = match parse_manifest_bytes(&manifest_bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            for seg in manifest.segments {
                if segment_ids.contains(&seg.id) {
                    segments.push(SegmentInfo {
                        id: seg.id,
                        namespace: ns.clone(),
                        segment_key: seg.segment_key,
                        size_bytes: seg.size_bytes,
                    });
                }
            }
        }

        // Sort by LRU (oldest access first)
        let tracker = self.access_tracker.read().await;
        let ids: Vec<String> = segments.iter().map(|s| s.id.clone()).collect();
        let sorted_ids = tracker.oldest_first(&ids);
        drop(tracker);

        // Create a map for quick lookup
        let segment_map: HashMap<String, SegmentInfo> = segments
            .into_iter()
            .map(|s| (s.id.clone(), s))
            .collect();

        let mut freed: u64 = 0;
        let mut evicted: usize = 0;

        for seg_id in sorted_ids {
            if freed >= bytes_to_free {
                break;
            }

            let Some(seg) = segment_map.get(&seg_id) else {
                continue;
            };

            info!("Evicting segment {} ({} bytes)", seg.id, seg.size_bytes);

            // Delete segment files
            let _ = local.delete(&seg.segment_key).await;
            let _ = local.delete(&segment_index_path(&seg.namespace, &seg.id)).await;
            let _ = local.delete(&segment_ivf_path(&seg.namespace, &seg.id)).await;
            let _ = local.delete(&segment_quant_path(&seg.namespace, &seg.id)).await;
            let _ = local.delete(&segment_text_index_path(&seg.namespace, &seg.id)).await;

            // Update Merkle state
            merkle_state.remove_segment(&seg.id);

            freed += seg.size_bytes;
            evicted += 1;
        }

        // Save updated Merkle state
        if let Err(e) = merkle_state.save(self.local_dir.to_str().unwrap_or("/data")) {
            warn!("Failed to save Merkle state after eviction: {}", e);
        }

        info!(
            "Eviction complete: freed {} bytes from {} segments",
            freed, evicted
        );

        Ok(evicted)
    }

    async fn list_namespaces(&self, local: &LocalStore) -> Vec<String> {
        let keys = match local.list("namespaces/").await {
            Ok(keys) => keys,
            Err(_) => return Vec::new(),
        };

        let mut namespaces = std::collections::HashSet::new();
        for key in keys {
            if let Some(rest) = key.strip_prefix("namespaces/") {
                if let Some(ns) = rest.split('/').next() {
                    if !ns.is_empty() {
                        namespaces.insert(ns.to_string());
                    }
                }
            }
        }
        namespaces.into_iter().collect()
    }

    /// Run the eviction loop. Never returns; call with tokio::spawn.
    pub async fn run(self, check_interval: Duration) {
        loop {
            tokio::time::sleep(check_interval).await;
            if let Err(e) = self.evict_if_needed().await {
                warn!("Eviction error: {}", e);
            }
        }
    }
}
