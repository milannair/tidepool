use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::{info, warn};

use tidepool_common::document::Document;
use tidepool_common::manifest::{Manager, Manifest};
use tidepool_common::segment::{Reader, Writer, WriterOptions};
use tidepool_common::storage::{segment_index_path, Store, StorageError};
use tidepool_common::wal::{DeserializedEntry, Reader as WalReader};

#[derive(Clone)]
pub struct Compactor<S: Store + Clone> {
    storage: S,
    namespace: String,
    wal_reader: WalReader<S>,
    segment_writer: Writer<S>,
    manifest_manager: Manager<S>,
    last_run: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl<S: Store + Clone> Compactor<S> {
    pub fn new_with_options(storage: S, namespace: impl Into<String>, opts: WriterOptions) -> Self {
        let namespace = namespace.into();
        Self {
            storage: storage.clone(),
            namespace: namespace.clone(),
            wal_reader: WalReader::new(storage.clone(), &namespace),
            segment_writer: Writer::new_with_options(storage.clone(), &namespace, opts),
            manifest_manager: Manager::new(storage, &namespace),
            last_run: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn run(&self) -> Result<(), String> {
        info!("Starting compaction cycle...");

        let (mut entries, wal_files) = self
            .wal_reader
            .read_all_wal_files()
            .await
            .map_err(|err| format!("failed to read WAL files: {}", err))?;

        if entries.is_empty() {
            info!("No WAL entries to compact");
            return Ok(());
        }

        entries.sort_by(|a, b| a.ts.cmp(&b.ts));

        let mut deleted_ids: HashSet<String> = HashSet::new();
        let mut doc_map: HashMap<String, Document> = HashMap::new();

        for entry in entries {
            apply_entry(&mut doc_map, &mut deleted_ids, &entry);
        }

        let mut existing_segments = Vec::new();
        if let Ok(manifest) = self.manifest_manager.load().await {
            existing_segments = manifest.segments;
        }

        let seg_reader = Reader::new(self.storage.clone(), &self.namespace, None);
        for seg in &existing_segments {
            let seg_data = match seg_reader.read_segment(&seg.segment_key).await {
                Ok(data) => data,
                Err(err) => {
                    warn!("Warning: failed to read existing segment {}: {}", seg.id, err);
                    continue;
                }
            };

            for i in 0..seg_data.ids.len() {
                let id = seg_data.ids[i].clone();
                if deleted_ids.contains(&id) {
                    continue;
                }
                if !doc_map.contains_key(&id) {
                    doc_map.insert(
                        id.clone(),
                        Document {
                            id,
                            vector: seg_data.vectors[i].clone(),
                            attributes: seg_data.attributes[i].clone(),
                        },
                    );
                }
            }
        }

        let mut docs: Vec<Document> = doc_map
            .into_values()
            .filter(|doc| !doc.vector.is_empty())
            .collect();

        if docs.is_empty() {
            info!("No vectors to compact");
            let empty = Manifest::new(Vec::new());
            self.manifest_manager
                .save(&empty)
                .await
                .map_err(|err| format!("failed to save empty manifest: {}", err))?;
            self.delete_wal_files(&wal_files).await;
            return Ok(());
        }

        docs.sort_by(|a, b| a.id.cmp(&b.id));

        let mut dims = 0usize;
        for (i, doc) in docs.iter().enumerate() {
            if doc.vector.is_empty() {
                return Err(format!("document {} has empty vector during compaction", i));
            }
            if dims == 0 {
                dims = doc.vector.len();
            } else if doc.vector.len() != dims {
                return Err(format!(
                    "dimension mismatch during compaction: got {} want {}",
                    doc.vector.len(),
                    dims
                ));
            }
        }

        info!("Compacting {} vectors", docs.len());

        let docs_ref: Vec<Document> = docs;
        let seg = self
            .segment_writer
            .write_segment(&docs_ref)
            .await
            .map_err(|err| format!("failed to write segment: {}", err))?
            .ok_or_else(|| "failed to create segment".to_string())?;

        info!(
            "Created segment {} with {} vectors ({} dimensions)",
            seg.id, seg.doc_count, seg.dimensions
        );

        let new_manifest = Manifest::new(vec![tidepool_common::manifest::Segment {
            id: seg.id.clone(),
            segment_key: seg.segment_key.clone(),
            doc_count: seg.doc_count,
            dimensions: seg.dimensions,
        }]);

        self.manifest_manager
            .save(&new_manifest)
            .await
            .map_err(|err| format!("failed to save manifest: {}", err))?;

        for seg in &existing_segments {
            if let Err(err) = self.storage.delete(&seg.segment_key).await {
                warn!("Warning: failed to delete old segment {}: {}", seg.id, err);
            }
            let index_key = segment_index_path(&self.namespace, &seg.id);
            if let Err(err) = self.storage.delete(&index_key).await {
                warn!("Warning: failed to delete old segment index {}: {}", seg.id, err);
            }
        }

        self.delete_wal_files(&wal_files).await;

        *self.last_run.write().await = Some(Utc::now());

        info!("Compaction complete: {} vectors", new_manifest.total_doc_count());
        Ok(())
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
