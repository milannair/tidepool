use chrono::Utc;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::storage::{latest_manifest_path, manifest_path, Store, StorageError};

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct Manifest {
    pub version: String,
    pub created_at: i64,
    pub segments: Vec<Segment>,
    #[serde(default)]
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct Segment {
    pub id: String,
    pub segment_key: String,
    pub doc_count: i64,
    #[serde(default)]
    pub dimensions: usize,
}

#[derive(Clone)]
pub struct Manager<S: Store> {
    storage: S,
    namespace: String,
}

impl<S: Store> Manager<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
        }
    }

    pub async fn load(&self) -> Result<Manifest, StorageError> {
        let data = self.storage.get(&latest_manifest_path(&self.namespace)).await?;
        // SAFETY: We trust our own serialized data format
        let archived = unsafe { rkyv::archived_root::<Manifest>(&data) };
        let manifest: Manifest = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|err| StorageError::Other(format!("parse manifest: {}", err)))?;
        Ok(manifest)
    }

    pub async fn load_version(&self, version: &str) -> Result<Manifest, StorageError> {
        let data = self
            .storage
            .get(&manifest_path(&self.namespace, version))
            .await?;
        // SAFETY: We trust our own serialized data format
        let archived = unsafe { rkyv::archived_root::<Manifest>(&data) };
        let manifest: Manifest = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|err| StorageError::Other(format!("parse manifest: {}", err)))?;
        Ok(manifest)
    }

    pub async fn save(&self, manifest: &Manifest) -> Result<(), StorageError> {
        let data = rkyv::to_bytes::<_, 256>(manifest)
            .map_err(|err| StorageError::Other(format!("serialize manifest: {}", err)))?
            .as_ref()
            .to_vec();
        let version_path = manifest_path(&self.namespace, &manifest.version);
        self.storage.put(&version_path, data.clone()).await?;
        let latest_path = latest_manifest_path(&self.namespace);
        self.storage.put(&latest_path, data).await?;
        Ok(())
    }
}

impl Manifest {
    pub fn new(segments: Vec<Segment>) -> Self {
        let dimensions = segments.first().map(|s| s.dimensions).unwrap_or(0);
        Self {
            version: format!(
                "{}",
                Utc::now()
                    .timestamp_nanos_opt()
                    .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000)
            ),
            created_at: Utc::now()
                .timestamp_nanos_opt()
                .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000),
            segments,
            dimensions,
        }
    }

    pub fn total_doc_count(&self) -> i64 {
        self.segments.iter().map(|s| s.doc_count).sum()
    }
}
