//! Background sync: poll S3 for updates and reload local state.
//!
//! Uses Merkle-based incremental sync to only download segments that have
//! changed or are missing locally.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use tidepool_common::manifest::parse_manifest_bytes;
use tidepool_common::storage::{
    latest_manifest_path, segment_index_path, segment_ivf_path, segment_quant_path,
    segment_text_index_path, tombstone_path, LocalStore, S3Store, StorageError, Store,
};

use crate::bootstrap::LocalMerkleState;
use crate::loader::DataLoader;
use crate::namespace_manager::NamespaceManager;

/// Polls S3 periodically, syncs new data to local disk, and reloads all engines.
pub struct BackgroundSync {
    remote: S3Store,
    local_dir: PathBuf,
    loader: DataLoader,
    namespaces: std::sync::Arc<NamespaceManager<LocalStore>>,
    interval: Duration,
    /// If true, use eager full sync (legacy mode).
    eager_sync_all: bool,
}

impl BackgroundSync {
    pub fn new(
        remote: S3Store,
        local_dir: PathBuf,
        namespaces: std::sync::Arc<NamespaceManager<LocalStore>>,
        interval: Duration,
    ) -> Self {
        Self {
            remote: remote.clone(),
            local_dir: local_dir.clone(),
            loader: DataLoader::new(remote, local_dir.clone()),
            namespaces,
            interval,
            eager_sync_all: false,
        }
    }

    pub fn with_eager_sync(mut self, eager: bool) -> Self {
        self.eager_sync_all = eager;
        self
    }

    /// Run the sync loop. Never returns; call with tokio::spawn.
    pub async fn run(self) {
        loop {
            tokio::time::sleep(self.interval).await;

            let result = if self.eager_sync_all {
                // Legacy mode: sync everything
                self.loader.sync_all().await.map(|_| ())
            } else {
                // Merkle-based incremental sync
                self.incremental_sync().await
            };

            match result {
                Ok(()) => {
                    self.namespaces.reload_all_engines().await;
                }
                Err(e) => {
                    warn!("Background sync failed: {}", e);
                }
            }
        }
    }

    /// Merkle-based incremental sync: only download changed/missing segments.
    async fn incremental_sync(&self) -> Result<(), String> {
        let start = Instant::now();
        let local = LocalStore::new(&self.local_dir);

        // Load local Merkle state
        let mut merkle_state = LocalMerkleState::load(self.local_dir.to_str().unwrap_or("/data"));

        let namespaces = self.list_namespaces().await?;
        let mut segments_synced = 0;
        let mut bytes_synced: u64 = 0;

        for ns in &namespaces {
            let (synced, bytes) = match self.sync_namespace_incremental(&local, &mut merkle_state, ns).await {
                Ok(stats) => stats,
                Err(e) => {
                    warn!("Failed to sync namespace {}: {}", ns, e);
                    continue;
                }
            };
            segments_synced += synced;
            bytes_synced += bytes;
        }

        // Save updated Merkle state
        if let Err(e) = merkle_state.save(self.local_dir.to_str().unwrap_or("/data")) {
            warn!("Failed to save Merkle state: {}", e);
        }

        if segments_synced > 0 {
            info!(
                "Incremental sync complete: {} segments, {} bytes in {:?}",
                segments_synced,
                bytes_synced,
                start.elapsed()
            );
        }

        Ok(())
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

    async fn sync_namespace_incremental(
        &self,
        local: &LocalStore,
        merkle_state: &mut LocalMerkleState,
        namespace: &str,
    ) -> Result<(usize, u64), String> {
        let mut bytes: u64 = 0;
        let mut segments_synced: usize = 0;

        // Get remote manifest
        let manifest_key = latest_manifest_path(namespace);
        let manifest_bytes = match self.remote.get(&manifest_key).await {
            Ok(data) => data,
            Err(StorageError::NotFound(_)) => return Ok((0, 0)),
            Err(e) => return Err(format!("get manifest {}: {}", namespace, e)),
        };

        // Save manifest locally
        local
            .put(&manifest_key, manifest_bytes.clone())
            .await
            .map_err(|e| e.to_string())?;
        bytes += manifest_bytes.len() as u64;

        let manifest = parse_manifest_bytes(&manifest_bytes)
            .map_err(|e| format!("parse manifest {}: {}", namespace, e))?;

        // Check each segment against local Merkle state
        for seg in &manifest.segments {
            // Skip if we already have this segment with matching hash
            if merkle_state.has_segment(&seg.id, &seg.content_hash) {
                continue;
            }

            // Download segment (only if missing or hash mismatch)
            info!(
                "Syncing segment {} (hash: {:?})",
                seg.id,
                seg.content_hash.as_ref().map(|h| &h[..8.min(h.len())])
            );

            let segment_data = match self.remote.get(&seg.segment_key).await {
                Ok(data) => data,
                Err(e) => {
                    warn!("Failed to get segment {}: {}", seg.id, e);
                    continue;
                }
            };
            local
                .put(&seg.segment_key, segment_data.clone())
                .await
                .map_err(|e| e.to_string())?;
            bytes += segment_data.len() as u64;

            // Download index files
            for key in [
                segment_index_path(namespace, &seg.id),
                segment_ivf_path(namespace, &seg.id),
                segment_quant_path(namespace, &seg.id),
                segment_text_index_path(namespace, &seg.id),
            ] {
                if let Ok(data) = self.remote.get(&key).await {
                    let _ = local.put(&key, data.clone()).await;
                    bytes += data.len() as u64;
                }
            }

            // Download Bloom filter
            if let Some(bloom_key) = &seg.bloom_key {
                if let Ok(data) = self.remote.get(bloom_key).await {
                    let _ = local.put(bloom_key, data.clone()).await;
                    bytes += data.len() as u64;
                }
            }

            // Update Merkle state
            if let Some(hash) = &seg.content_hash {
                merkle_state.add_segment(&seg.id, hash, seg.size_bytes);
            }

            segments_synced += 1;
        }

        // Sync tombstones
        let tomb_key = tombstone_path(namespace);
        if let Ok(data) = self.remote.get(&tomb_key).await {
            local
                .put(&tomb_key, data.clone())
                .await
                .map_err(|e| e.to_string())?;
            bytes += data.len() as u64;
        }

        Ok((segments_synced, bytes))
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
