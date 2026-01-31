use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use tidepool_common::document::Document;
use tidepool_common::manifest::{Manager, Manifest};
use tidepool_common::redis::RedisStore;
use tidepool_common::segment::{Writer, WriterOptions};
use tidepool_common::storage::{Store, StorageError};
use tidepool_common::tombstone::Manager as TombstoneManager;
use tidepool_common::wal::{DeserializedEntry, Reader as WalReader};

#[derive(Clone)]
pub struct Compactor<S: Store + Clone> {
    namespace: String,
    wal_reader: WalReader<S>,
    segment_writer: Writer<S>,
    manifest_manager: Manager<S>,
    tombstone_manager: TombstoneManager<S>,
    redis: Option<Arc<RedisStore>>,
    last_run: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// Enable distributed locking (default: true if Redis is available)
    use_distributed_lock: bool,
    /// Lock TTL in seconds
    lock_ttl_secs: u64,
    /// Enable pub/sub invalidation notifications
    use_pubsub_invalidation: bool,
}

/// Options for compactor behavior
#[derive(Clone)]
pub struct CompactorOptions {
    pub use_distributed_lock: bool,
    pub lock_ttl_secs: u64,
    pub use_pubsub_invalidation: bool,
}

impl Default for CompactorOptions {
    fn default() -> Self {
        Self {
            use_distributed_lock: true,
            lock_ttl_secs: 300, // 5 minutes
            use_pubsub_invalidation: true,
        }
    }
}

impl<S: Store + Clone> Compactor<S> {
    #[allow(dead_code)]
    pub fn new_with_options(storage: S, namespace: impl Into<String>, opts: WriterOptions) -> Self {
        Self::new_with_redis(storage, namespace, opts, None, CompactorOptions::default())
    }

    pub fn new_with_redis(
        storage: S,
        namespace: impl Into<String>,
        opts: WriterOptions,
        redis: Option<Arc<RedisStore>>,
        compactor_opts: CompactorOptions,
    ) -> Self {
        let namespace = namespace.into();
        Self {
            namespace: namespace.clone(),
            wal_reader: WalReader::new(storage.clone(), &namespace),
            segment_writer: Writer::new_with_options(storage.clone(), &namespace, opts),
            manifest_manager: Manager::new(storage.clone(), &namespace),
            tombstone_manager: TombstoneManager::new(storage, &namespace),
            redis,
            last_run: Arc::new(RwLock::new(None)),
            use_distributed_lock: compactor_opts.use_distributed_lock,
            lock_ttl_secs: compactor_opts.lock_ttl_secs,
            use_pubsub_invalidation: compactor_opts.use_pubsub_invalidation,
        }
    }

    pub async fn run(&self) -> Result<(), String> {
        info!("Starting compaction cycle...");

        // Acquire distributed lock if enabled and Redis is available
        let _lock = if self.use_distributed_lock {
            if let Some(redis) = &self.redis {
                let lock_name = format!("compaction:{}", self.namespace);
                match redis.try_lock(&lock_name, self.lock_ttl_secs).await {
                    Ok(Some(lock)) => {
                        info!("Acquired compaction lock for namespace {}", self.namespace);
                        Some(lock)
                    }
                    Ok(None) => {
                        info!(
                            "Compaction skipped - lock held by another instance for namespace {}",
                            self.namespace
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        warn!("Failed to acquire compaction lock: {} - proceeding without lock", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let (mut entries, wal_files) = self
            .wal_reader
            .read_all_wal_files()
            .await
            .map_err(|err| format!("failed to read WAL files: {}", err))?;

        if entries.is_empty() {
            info!("No WAL entries to compact");
            // Lock will be released when _lock is dropped
            return Ok(());
        }

        entries.sort_by(|a, b| a.ts.cmp(&b.ts));

        let mut deleted_ids: HashSet<String> = HashSet::new();
        let mut doc_map: HashMap<String, Document> = HashMap::new();

        for entry in entries {
            apply_entry(&mut doc_map, &mut deleted_ids, &entry);
        }

        let mut existing_segments = Vec::new();
        let mut existing_dimensions = 0usize;
        if let Ok(manifest) = self.manifest_manager.load().await {
            existing_dimensions = manifest.dimensions;
            existing_segments = manifest.segments;
        }

        // Filter to documents with vectors (required for segment storage)
        // Note: text-only documents are rejected at ingest time, but we filter here
        // as a safety measure to avoid silent failures if any slip through
        let total_docs = doc_map.len();
        let mut docs: Vec<Document> = doc_map
            .into_values()
            .filter(|doc| !doc.vector.is_empty())
            .collect();
        
        let skipped = total_docs - docs.len();
        if skipped > 0 {
            warn!(
                "Skipped {} documents without vectors during compaction (text-only documents not yet supported)",
                skipped
            );
        }

        docs.sort_by(|a, b| a.id.cmp(&b.id));

        // Validate vector dimensions
        let mut dims = existing_dimensions;
        for (i, doc) in docs.iter().enumerate() {
            if dims == 0 {
                dims = doc.vector.len();
            } else if doc.vector.len() != dims {
                return Err(format!(
                    "dimension mismatch during compaction at doc {}: got {} want {}",
                    i,
                    doc.vector.len(),
                    dims
                ));
            }
        }

        let mut segments = existing_segments.clone();
        if !docs.is_empty() {
            info!("Compacting {} vectors", docs.len());

            let seg = self
                .segment_writer
                .write_segment(&docs)
                .await
                .map_err(|err| format!("failed to write segment: {}", err))?
                .ok_or_else(|| "failed to create segment".to_string())?;

            info!(
                "Created segment {} with {} vectors ({} dimensions)",
                seg.id, seg.doc_count, seg.dimensions
            );

            // ManifestSegment guarantees these are populated by write_segment()
            if seg.content_hash.is_empty() || seg.bloom_key.is_empty() {
                return Err(format!("segment {} missing content_hash or bloom_key", seg.id));
            }

            segments.push(tidepool_common::manifest::Segment {
                id: seg.id.clone(),
                segment_key: seg.segment_key.clone(),
                doc_count: seg.doc_count,
                dimensions: seg.dimensions,
                size_bytes: seg.size_bytes,
                content_hash: Some(seg.content_hash.clone()),
                bloom_key: Some(seg.bloom_key.clone()),
            });
        } else {
            info!("No vectors to compact, applying tombstones only");
        }

        let mut tombstones = self
            .tombstone_manager
            .load()
            .await
            .unwrap_or_else(|_| HashSet::new());

        for id in &deleted_ids {
            tombstones.insert(id.clone());
        }
        for doc in &docs {
            tombstones.remove(&doc.id);
        }

        // Increment generation from previous manifest (or start at 1)
        let prev_generation = if let Ok(manifest) = self.manifest_manager.load().await {
            manifest.generation
        } else {
            0
        };
        let mut new_manifest = Manifest::new_with_generation(segments, prev_generation + 1);
        if new_manifest.dimensions == 0 {
            new_manifest.dimensions = dims;
        }

        self.manifest_manager
            .save(&new_manifest)
            .await
            .map_err(|err| format!("failed to save manifest: {}", err))?;

        self.tombstone_manager
            .save(&tombstones)
            .await
            .map_err(|err| format!("failed to save tombstones: {}", err))?;

        self.delete_wal_files(&wal_files).await;

        // Trim Redis WAL if enabled
        if let Some(redis) = &self.redis {
            self.trim_redis_wal(redis).await;
            
            // Invalidate query cache for this namespace
            if let Err(e) = redis.invalidate_query_cache(&self.namespace).await {
                warn!("Failed to invalidate query cache: {}", e);
            }
            
            // Clear hot vectors (now compacted into segments)
            if let Err(e) = redis.clear_hot_vectors(&self.namespace).await {
                warn!("Failed to clear hot vectors: {}", e);
            }
            
            // Publish invalidation event via pub/sub
            if self.use_pubsub_invalidation {
                let payload = serde_json::json!({
                    "manifest_version": new_manifest.version,
                    "generation": new_manifest.generation,
                    "total_vectors": new_manifest.total_doc_count(),
                    "segment_count": new_manifest.segments.len(),
                });
                if let Err(e) = redis
                    .publish_invalidation(&self.namespace, "compaction_complete", &payload.to_string())
                    .await
                {
                    warn!("Failed to publish invalidation event: {}", e);
                }
            }
        }

        *self.last_run.write().await = Some(Utc::now());

        info!("Compaction complete: {} vectors", new_manifest.total_doc_count());
        // Lock will be automatically released when _lock is dropped
        Ok(())
    }

    /// Trim Redis WAL entries that have been compacted
    async fn trim_redis_wal(&self, redis: &RedisStore) {
        // Get current stream length first
        match redis.wal_len(&self.namespace).await {
            Ok(len) if len == 0 => {
                debug!("Redis WAL is empty, nothing to trim");
                return;
            }
            Ok(len) => {
                debug!("Redis WAL has {} entries before trim", len);
            }
            Err(e) => {
                warn!("Failed to get Redis WAL length: {}", e);
                return;
            }
        }

        // Read all entries to find the last one
        match redis.read_wal(&self.namespace, None, None).await {
            Ok(entries) if entries.is_empty() => {
                debug!("No Redis WAL entries to trim");
            }
            Ok(entries) => {
                if let Some(last) = entries.last() {
                    // Trim all entries up to and including the last one
                    match redis.trim_wal(&self.namespace, &last.id).await {
                        Ok(trimmed) => {
                            info!(
                                "Trimmed {} entries from Redis WAL for namespace {}",
                                trimmed, self.namespace
                            );
                        }
                        Err(e) => {
                            warn!("Failed to trim Redis WAL: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to read Redis WAL for trimming: {}", e);
            }
        }
    }

    pub async fn run_periodically(&self, interval: std::time::Duration) {
        let mut ticker = tokio::time::interval(interval);
        if let Err(err) = self.run().await {
            warn!("Compaction error: {}", err);
        }
        loop {
            ticker.tick().await;
            if let Err(err) = self.run().await {
                warn!("Compaction error: {}", err);
            }
        }
    }

    pub async fn get_status(&self) -> Result<Status, StorageError> {
        let wal_files = self.wal_reader.list_wal_files().await?;
        let mut wal_entries = 0usize;
        for wal_file in &wal_files {
            if let Ok(entries) = self.wal_reader.read_wal_file(wal_file).await {
                wal_entries += entries.len();
            }
        }
        let manifest = self.manifest_manager.load().await.ok();
        let (segments, total_vecs, dimensions) = if let Some(manifest) = manifest {
            (
                manifest.segments.len(),
                manifest.total_doc_count(),
                manifest.dimensions,
            )
        } else {
            (0, 0, 0)
        };

        let last_run = self.last_run.read().await.clone();
        Ok(Status {
            last_run,
            wal_files: wal_files.len(),
            wal_entries,
            segments,
            total_vecs,
            dimensions,
        })
    }

    async fn delete_wal_files(&self, wal_files: &[String]) {
        for wal_file in wal_files {
            if let Err(err) = self.wal_reader.delete_wal_file(wal_file).await {
                warn!("Warning: failed to delete WAL file {}: {}", wal_file, err);
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Status {
    pub last_run: Option<DateTime<Utc>>,
    pub wal_files: usize,
    pub wal_entries: usize,
    pub segments: usize,
    pub total_vecs: i64,
    pub dimensions: usize,
}

fn apply_entry(doc_map: &mut HashMap<String, Document>, deleted: &mut HashSet<String>, entry: &DeserializedEntry) {
    match entry.op.as_str() {
        "upsert" => {
            if let Some(doc) = &entry.doc {
                doc_map.insert(doc.id.clone(), doc.clone());
                deleted.remove(&doc.id);
            }
        }
        "delete" => {
            for id in &entry.delete_ids {
                deleted.insert(id.clone());
                doc_map.remove(id);
            }
        }
        _ => {}
    }
}
