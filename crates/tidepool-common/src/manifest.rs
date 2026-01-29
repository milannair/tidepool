use std::path::PathBuf;

use chrono::Utc;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;

use crate::storage::{latest_manifest_path, manifest_path, Store, StorageError};

fn manifest_cache_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let hash = hasher.finalize();
    format!("{:x}", hash)[..16].to_string()
}

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
    cache_dir: Option<PathBuf>,
}

impl<S: Store> Manager<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
            cache_dir: None,
        }
    }

    pub fn new_with_cache(
        storage: S,
        namespace: impl Into<String>,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
            cache_dir,
        }
    }

    pub async fn load(&self) -> Result<Manifest, StorageError> {
        let data = self.storage.get(&latest_manifest_path(&self.namespace)).await?;
        parse_manifest_bytes(&data)
    }

    /// Load manifest only if it has changed (HEAD + cache). Returns None if unchanged.
    pub async fn load_if_changed(&self) -> Result<Option<Manifest>, StorageError> {
        let key = latest_manifest_path(&self.namespace);

        let meta = match self.storage.head(&key).await {
            Ok(m) => m,
            Err(StorageError::NotFound(_)) => {
                // No manifest yet; if we have cache, leave it; otherwise caller will retry
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        let Some(ref cache_dir) = self.cache_dir else {
            let data = self.storage.get(&key).await?;
            return Ok(Some(parse_manifest_bytes(&data)?));
        };

        let cache_key = manifest_cache_key(&key);
        let cache_rkyv = cache_dir.join("manifest").join(format!("{}.rkyv", cache_key));
        let cache_etag = cache_dir.join("manifest").join(format!("{}.etag", cache_key));

        if let (Some(ref etag), true) = (&meta.etag, cache_etag.is_file()) {
            if let Ok(cached_etag) = fs::read_to_string(&cache_etag).await {
                let cached_etag = cached_etag.trim();
                if cached_etag == etag {
                    if let Ok(data) = fs::read(&cache_rkyv).await {
                        if parse_manifest_bytes(&data).is_ok() {
                            return Ok(None);
                        }
                    }
                }
            }
        }

        let data = self.storage.get(&key).await?;
        let manifest = parse_manifest_bytes(&data)?;

        let manifest_dir = cache_dir.join("manifest");
        let _ = std::fs::create_dir_all(&manifest_dir);
        let _ = fs::write(&cache_rkyv, &data).await;
        if let Some(ref etag) = meta.etag {
            let _ = fs::write(&cache_etag, etag).await;
        }

        Ok(Some(manifest))
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

fn parse_manifest_bytes(data: &[u8]) -> Result<Manifest, StorageError> {
    let archived = unsafe { rkyv::archived_root::<Manifest>(data) };
    let manifest: Manifest = archived
        .deserialize(&mut rkyv::Infallible)
        .map_err(|err| StorageError::Other(format!("parse manifest: {}", err)))?;
    Ok(manifest)
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
