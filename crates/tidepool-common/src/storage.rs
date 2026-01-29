use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use aws_config::Region;
use aws_sdk_s3::Client as S3Client;
use tokio::sync::RwLock;

use crate::config::Config;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("object not found: {0}")]
    NotFound(String),
    #[error("storage error: {0}")]
    Other(String),
}

/// Object metadata from HEAD request (for cache invalidation).
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub etag: Option<String>,
    pub last_modified: Option<i64>,
    pub size: u64,
}

#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: &str) -> Result<Vec<u8>, StorageError>;
    async fn put(&self, key: &str, data: Vec<u8>) -> Result<(), StorageError>;
    async fn delete(&self, key: &str) -> Result<(), StorageError>;
    async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError>;
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;
    /// Return object metadata without downloading body. NotFound if key does not exist.
    async fn head(&self, key: &str) -> Result<ObjectMeta, StorageError>;
}

#[derive(Clone)]
pub struct InMemoryStore {
    data: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
}

#[async_trait]
impl Store for InMemoryStore {
    async fn get(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let data = self.data.read().await;
        match data.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(StorageError::NotFound(key.to_string())),
        }
    }

    async fn put(&self, key: &str, data: Vec<u8>) -> Result<(), StorageError> {
        let mut map = self.data.write().await;
        map.insert(key.to_string(), data);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let mut map = self.data.write().await;
        map.remove(key);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let map = self.data.read().await;
        let keys = map
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }

    async fn exists(&self, key: &str) -> Result<bool, StorageError> {
        let map = self.data.read().await;
        Ok(map.contains_key(key))
    }

    async fn head(&self, key: &str) -> Result<ObjectMeta, StorageError> {
        let map = self.data.read().await;
        match map.get(key) {
            Some(data) => Ok(ObjectMeta {
                etag: Some(format!("{:x}", simple_hash(data))),
                last_modified: None,
                size: data.len() as u64,
            }),
            None => Err(StorageError::NotFound(key.to_string())),
        }
    }
}

fn simple_hash(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    hasher.write(data);
    hasher.finish()
}

#[derive(Clone)]
pub struct S3Store {
    client: S3Client,
    bucket: String,
}

impl S3Store {
    pub async fn new(cfg: &Config) -> Result<Self, StorageError> {
        let base_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(Region::new(cfg.aws_region.clone()))
            .load()
            .await;

        let s3_config = aws_sdk_s3::config::Builder::from(&base_config)
            .endpoint_url(cfg.aws_endpoint_url.clone())
            .force_path_style(true)
            .build();

        let client = S3Client::from_conf(s3_config);

        Ok(Self {
            client,
            bucket: cfg.bucket_name.clone(),
        })
    }
}

#[async_trait]
impl Store for S3Store {
    async fn get(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|err| StorageError::Other(format!("get object {}: {}", key, err)))?;

        let data = resp
            .body
            .collect()
            .await
            .map_err(|err| StorageError::Other(format!("read object {}: {}", key, err)))?
            .into_bytes()
            .to_vec();

        Ok(data)
    }

    async fn put(&self, key: &str, data: Vec<u8>) -> Result<(), StorageError> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(data.into())
            .send()
            .await
            .map_err(|err| StorageError::Other(format!("put object {}: {}", key, err)))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|err| StorageError::Other(format!("delete object {}: {}", key, err)))?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let mut keys = Vec::new();
        let mut continuation = None;
        loop {
            let mut req = self.client.list_objects_v2().bucket(&self.bucket).prefix(prefix);
            if let Some(token) = continuation.clone() {
                req = req.continuation_token(token);
            }

            let resp = req
                .send()
                .await
                .map_err(|err| StorageError::Other(format!("list objects {}: {}", prefix, err)))?;

            if let Some(contents) = resp.contents {
                for obj in contents {
                    if let Some(key) = obj.key {
                        keys.push(key);
                    }
                }
            }

            if resp.is_truncated.unwrap_or(false) {
                continuation = resp.next_continuation_token;
            } else {
                break;
            }
        }
        Ok(keys)
    }

    async fn exists(&self, key: &str) -> Result<bool, StorageError> {
        match self.head(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn head(&self, key: &str) -> Result<ObjectMeta, StorageError> {
        let resp = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|err| {
                let msg = err.to_string();
                if msg.contains("NotFound") || msg.contains("NoSuchKey") {
                    StorageError::NotFound(key.to_string())
                } else {
                    StorageError::Other(format!("head object {}: {}", key, err))
                }
            })?;

        let etag = resp.e_tag().map(|s| s.trim_matches('"').to_string());
        let last_modified = resp
            .last_modified()
            .and_then(|dt| dt.secs().try_into().ok());
        let size = resp.content_length().unwrap_or(0) as u64;

        Ok(ObjectMeta {
            etag,
            last_modified,
            size,
        })
    }
}

pub fn namespace_path(namespace: &str, subpath: &str) -> String {
    format!("namespaces/{}/{}", namespace, subpath)
}

pub fn manifest_path(namespace: &str, version: &str) -> String {
    namespace_path(namespace, &format!("manifests/{}.rkyv", version))
}

pub fn latest_manifest_path(namespace: &str) -> String {
    manifest_path(namespace, "latest")
}

pub fn wal_path(namespace: &str, date: &str, uuid: &str) -> String {
    namespace_path(namespace, &format!("wal/{}/{}.wal", date, uuid))
}

pub fn wal_prefix(namespace: &str) -> String {
    namespace_path(namespace, "wal/")
}

pub fn segment_path(namespace: &str, segment_id: &str) -> String {
    namespace_path(namespace, &format!("segments/{}.tpvs", segment_id))
}

pub fn segment_index_path(namespace: &str, segment_id: &str) -> String {
    namespace_path(namespace, &format!("segments/{}.hnsw", segment_id))
}

pub fn segment_ivf_path(namespace: &str, segment_id: &str) -> String {
    namespace_path(namespace, &format!("segments/{}.ivf", segment_id))
}

pub fn segment_quant_path(namespace: &str, segment_id: &str) -> String {
    namespace_path(namespace, &format!("segments/{}.tpq", segment_id))
}

pub fn segment_text_index_path(namespace: &str, segment_id: &str) -> String {
    namespace_path(namespace, &format!("segments/{}.tpti", segment_id))
}

pub fn tombstone_path(namespace: &str) -> String {
    namespace_path(namespace, "tombstones/latest.rkyv")
}
