//! Startup data sync: copy all namespace data from S3 to local disk.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use tidepool_common::manifest::parse_manifest_bytes;
use tidepool_common::storage::{
    latest_manifest_path, segment_index_path, segment_ivf_path, segment_quant_path,
    segment_text_index_path, tombstone_path, LocalStore, StorageError, Store,
};

/// Syncs all data from a remote store (S3) to local disk at startup.
pub struct DataLoader {
    /// Remote store (S3) to read from.
    remote: tidepool_common::storage::S3Store,
    /// Local directory to write to (used to create LocalStore).
    local_dir: PathBuf,
}

/// Statistics from a full sync.
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub bytes_synced: u64,
    pub duration: Duration,
    pub namespaces: usize,
    pub segments: usize,
}

impl DataLoader {
    pub fn new(remote: tidepool_common::storage::S3Store, local_dir: impl Into<PathBuf>) -> Self {
        Self {
            remote,
            local_dir: local_dir.into(),
        }
    }

    /// Sync all namespaces from remote to local. Blocks until complete.
    pub async fn sync_all(&self) -> Result<SyncStats, String> {
        let start = Instant::now();
        let local = LocalStore::new(&self.local_dir);

        let namespaces = self.list_namespaces().await?;
        let mut total_bytes: u64 = 0;
        let mut total_segments: usize = 0;

        for ns in &namespaces {
            match self.sync_namespace(&self.remote, &local, ns).await {
                Ok((bytes, segments)) => {
                    total_bytes += bytes;
                    total_segments += segments;
                }
                Err(e) => {
                    warn!("Failed to sync namespace {}: {}", ns, e);
                }
            }
        }

        let duration = start.elapsed();
        info!(
            "Synced {} bytes in {:?} ({} namespaces, {} segments)",
            total_bytes, duration, namespaces.len(), total_segments
        );

        Ok(SyncStats {
            bytes_synced: total_bytes,
            duration,
            namespaces: namespaces.len(),
            segments: total_segments,
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

    async fn sync_namespace(
        &self,
        remote: &tidepool_common::storage::S3Store,
        local: &LocalStore,
        namespace: &str,
    ) -> Result<(u64, usize), String> {
        let mut bytes: u64 = 0;

        // Manifest (latest)
        let manifest_key = latest_manifest_path(namespace);
        let manifest_bytes = match remote.get(&manifest_key).await {
            Ok(data) => data,
            Err(StorageError::NotFound(_)) => return Ok((0, 0)),
            Err(e) => return Err(format!("get manifest {}: {}", namespace, e)),
        };
        local
            .put(&manifest_key, manifest_bytes.clone())
            .await
            .map_err(|e| e.to_string())?;
        bytes += manifest_bytes.len() as u64;

        let manifest = parse_manifest_bytes(&manifest_bytes)
            .map_err(|e| format!("parse manifest {}: {}", namespace, e))?;

        let mut segment_count = 0;
        for seg in &manifest.segments {
            bytes += self
                .copy_key_if_exists(remote, local, &seg.segment_key)
                .await?;
            bytes += self
                .copy_key_if_exists(remote, local, &segment_index_path(namespace, &seg.id))
                .await?;
            bytes += self
                .copy_key_if_exists(remote, local, &segment_ivf_path(namespace, &seg.id))
                .await?;
            bytes += self
                .copy_key_if_exists(remote, local, &segment_quant_path(namespace, &seg.id))
                .await?;
            bytes += self
                .copy_key_if_exists(remote, local, &segment_text_index_path(namespace, &seg.id))
                .await?;
            segment_count += 1;
        }

        // Tombstones (optional)
        let tomb_key = tombstone_path(namespace);
        if let Ok(data) = remote.get(&tomb_key).await {
            local
                .put(&tomb_key, data.clone())
                .await
                .map_err(|e| e.to_string())?;
            bytes += data.len() as u64;
        }

        Ok((bytes, segment_count))
    }

    async fn copy_key_if_exists(
        &self,
        remote: &tidepool_common::storage::S3Store,
        local: &LocalStore,
        key: &str,
    ) -> Result<u64, String> {
        match remote.get(key).await {
            Ok(data) => {
                local
                    .put(key, data.clone())
                    .await
                    .map_err(|e| format!("put {}: {}", key, e.to_string()))?;
                Ok(data.len() as u64)
            }
            // Treat any error as "not found" for optional index files
            Err(_) => Ok(0),
        }
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
