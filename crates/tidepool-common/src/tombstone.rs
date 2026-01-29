use std::collections::HashSet;
use std::path::PathBuf;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use sha2::{Digest, Sha256};
use tokio::fs;

use crate::storage::{tombstone_path, Store, StorageError};

fn tombstone_cache_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let hash = hasher.finalize();
    format!("{:x}", hash)[..16].to_string()
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
struct TombstoneSet {
    ids: Vec<String>,
}

fn parse_tombstone_bytes(data: &[u8]) -> Result<HashSet<String>, StorageError> {
    let archived = unsafe { rkyv::archived_root::<TombstoneSet>(data) };
    let set: TombstoneSet = archived
        .deserialize(&mut rkyv::Infallible)
        .map_err(|err| StorageError::Other(format!("parse tombstones: {}", err)))?;
    Ok(set.ids.into_iter().collect())
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

    pub async fn load(&self) -> Result<HashSet<String>, StorageError> {
        let data = match self.storage.get(&tombstone_path(&self.namespace)).await {
            Ok(data) => data,
            Err(StorageError::NotFound(_)) => return Ok(HashSet::new()),
            Err(err) => return Err(err),
        };
        parse_tombstone_bytes(&data)
    }

    /// Load tombstones only if changed (HEAD + cache). Returns None if unchanged.
    pub async fn load_if_changed(&self) -> Result<Option<HashSet<String>>, StorageError> {
        let key = tombstone_path(&self.namespace);

        let meta = match self.storage.head(&key).await {
            Ok(m) => m,
            Err(StorageError::NotFound(_)) => {
                if self.cache_dir.is_some() {
                    let cache_key = tombstone_cache_key(&key);
                    let cache_dir = self.cache_dir.as_ref().unwrap();
                    let cache_rkyv = cache_dir.join("tombstones").join(format!("{}.rkyv", cache_key));
                    let cache_etag = cache_dir.join("tombstones").join(format!("{}.etag", cache_key));
                    if cache_etag.as_path().exists() {
                        std::fs::remove_file(&cache_etag).ok();
                        std::fs::remove_file(&cache_rkyv).ok();
                    }
                }
                return Ok(Some(HashSet::new()));
            }
            Err(e) => return Err(e),
        };

        let Some(ref cache_dir) = self.cache_dir else {
            let data = match self.storage.get(&key).await {
                Ok(d) => d,
                Err(StorageError::NotFound(_)) => return Ok(Some(HashSet::new())),
                Err(e) => return Err(e),
            };
            return Ok(Some(parse_tombstone_bytes(&data)?));
        };

        let cache_key = tombstone_cache_key(&key);
        let cache_rkyv = cache_dir.join("tombstones").join(format!("{}.rkyv", cache_key));
        let cache_etag = cache_dir.join("tombstones").join(format!("{}.etag", cache_key));

        if let (Some(ref etag), true) = (&meta.etag, cache_etag.as_path().exists()) {
            if let Ok(cached_etag) = fs::read_to_string(&cache_etag).await {
                let cached_etag = cached_etag.trim();
                if cached_etag == etag {
                    if let Ok(data) = fs::read(&cache_rkyv).await {
                        if parse_tombstone_bytes(&data).is_ok() {
                            return Ok(None);
                        }
                    }
                }
            }
        }

        let data = match self.storage.get(&key).await {
            Ok(d) => d,
            Err(StorageError::NotFound(_)) => return Ok(Some(HashSet::new())),
            Err(e) => return Err(e),
        };
        let tombstones = parse_tombstone_bytes(&data)?;

        let tomb_dir = cache_dir.join("tombstones");
        let _ = std::fs::create_dir_all(&tomb_dir);
        let _ = fs::write(&cache_rkyv, &data).await;
        if let Some(ref etag) = meta.etag {
            let _ = fs::write(&cache_etag, etag).await;
        }

        Ok(Some(tombstones))
    }

    pub async fn save(&self, tombstones: &HashSet<String>) -> Result<(), StorageError> {
        let mut ids: Vec<String> = tombstones.iter().cloned().collect();
        ids.sort();
        let data = rkyv::to_bytes::<_, 256>(&TombstoneSet { ids })
            .map_err(|err| StorageError::Other(format!("serialize tombstones: {}", err)))?
            .as_ref()
            .to_vec();
        self.storage.put(&tombstone_path(&self.namespace), data).await
    }
}
