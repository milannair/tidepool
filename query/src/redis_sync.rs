//! Redis hot buffer synchronization for real-time vector updates.
//!
//! This module provides real-time synchronization from Redis Streams to the
//! in-memory hot buffer, enabling sub-100ms write-to-query latency.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use tidepool_common::document::Document;
use tidepool_common::redis::RedisStore;
use tidepool_common::storage::LocalStore;

use crate::namespace_manager::NamespaceManager;

/// Background task that syncs hot buffer from Redis Streams.
pub struct RedisHotBufferSync {
    redis: Arc<RedisStore>,
    namespaces: Arc<NamespaceManager<LocalStore>>,
    poll_interval: Duration,
    /// Track last seen stream ID per namespace
    last_ids: RwLock<HashMap<String, String>>,
}

impl RedisHotBufferSync {
    pub fn new(
        redis: Arc<RedisStore>,
        namespaces: Arc<NamespaceManager<LocalStore>>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            redis,
            namespaces,
            poll_interval,
            last_ids: RwLock::new(HashMap::new()),
        }
    }

    /// Run the sync loop indefinitely
    pub async fn run(self) {
        info!(
            "Redis hot buffer sync started (poll interval: {:?})",
            self.poll_interval
        );

        let mut ticker = tokio::time::interval(self.poll_interval);

        loop {
            ticker.tick().await;

            if let Err(e) = self.sync_all_namespaces().await {
                warn!("Redis hot buffer sync error: {}", e);
            }
        }
    }

    /// Sync hot buffer for all known namespaces
    async fn sync_all_namespaces(&self) -> Result<(), String> {
        // Get list of namespaces from Redis
        let namespaces = self
            .redis
            .list_namespaces()
            .await
            .map_err(|e| format!("list namespaces: {}", e))?;

        for namespace in namespaces {
            if let Err(e) = self.sync_namespace(&namespace).await {
                warn!("Failed to sync namespace {}: {}", namespace, e);
            }
        }

        Ok(())
    }

    /// Sync hot buffer for a single namespace
    async fn sync_namespace(&self, namespace: &str) -> Result<(), String> {
        // Get last seen ID for this namespace
        let last_id = {
            let ids = self.last_ids.read().await;
            ids.get(namespace).cloned()
        };

        // Read new entries from Redis
        let entries = if let Some(ref id) = last_id {
            self.redis
                .read_wal_after(namespace, id)
                .await
                .map_err(|e| format!("read wal after {}: {}", id, e))?
        } else {
            self.redis
                .read_wal(namespace, None, Some(10000)) // Initial load: last 10k entries
                .await
                .map_err(|e| format!("read wal: {}", e))?
        };

        if entries.is_empty() {
            return Ok(());
        }

        debug!(
            "Syncing {} entries for namespace {} (from {:?})",
            entries.len(),
            namespace,
            last_id
        );

        // Get or create engine for this namespace
        let engine = self
            .namespaces
            .get_engine(namespace)
            .await
            .map_err(|e| format!("get engine: {}", e))?;

        // Apply entries to hot buffer
        let mut upserts: Vec<Document> = Vec::new();
        let mut deletes: Vec<String> = Vec::new();
        let mut latest_id = last_id.clone();

        for stream_entry in &entries {
            latest_id = Some(stream_entry.id.clone());

            match stream_entry.entry.op.as_str() {
                "upsert" => {
                    if let Some(doc) = stream_entry.entry.to_document() {
                        upserts.push(doc);
                    }
                }
                "delete" => {
                    deletes.extend(stream_entry.entry.delete_ids.clone());
                }
                _ => {}
            }
        }

        // Apply to hot buffer (batch for efficiency)
        if !upserts.is_empty() {
            engine.hot_buffer_insert(upserts).await;
        }
        if !deletes.is_empty() {
            engine.hot_buffer_delete(deletes).await;
        }

        // Update last seen ID
        if let Some(id) = latest_id {
            let mut ids = self.last_ids.write().await;
            ids.insert(namespace.to_string(), id);
        }

        debug!(
            "Synced {} upserts, {} deletes for namespace {}",
            entries.iter().filter(|e| e.entry.op == "upsert").count(),
            entries.iter().filter(|e| e.entry.op == "delete").count(),
            namespace
        );

        Ok(())
    }
}
