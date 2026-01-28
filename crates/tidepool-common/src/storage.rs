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

#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: &str) -> Result<Vec<u8>, StorageError>;
    async fn put(&self, key: &str, data: Vec<u8>) -> Result<(), StorageError>;
    async fn delete(&self, key: &str) -> Result<(), StorageError>;
    async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError>;
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;
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
        let resp = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await;

        match resp {
            Ok(_) => Ok(true),
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("NotFound") || msg.contains("NoSuchKey") {
                    Ok(false)
                } else {
                    Err(StorageError::Other(format!("head object {}: {}", key, err)))
                }
            }
        }
    }
}

pub fn namespace_path(namespace: &str, subpath: &str) -> String {
    format!("namespaces/{}/{}", namespace, subpath)
}

pub fn manifest_path(namespace: &str, version: &str) -> String {
    namespace_path(namespace, &format!("manifests/{}.json", version))
}

pub fn latest_manifest_path(namespace: &str) -> String {
    manifest_path(namespace, "latest")
}

pub fn wal_path(namespace: &str, date: &str, uuid: &str) -> String {
    namespace_path(namespace, &format!("wal/{}/{}.jsonl", date, uuid))
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
