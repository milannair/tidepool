use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::document::Document;
use crate::storage::{wal_path, wal_prefix, Store, StorageError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub ts: DateTime<Utc>,
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<Document>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delete_ids: Vec<String>,
}

#[derive(Clone)]
pub struct Writer<S: Store> {
    storage: S,
    namespace: String,
}

impl<S: Store> Writer<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
        }
    }

    pub async fn write_upsert(&self, docs: Vec<Document>) -> Result<Option<String>, StorageError> {
        if docs.is_empty() {
            return Ok(None);
        }

        let date = Utc::now().format("%Y-%m-%d").to_string();
        let wal_id = Uuid::new_v4().to_string();
        let wal_path = wal_path(&self.namespace, &date, &wal_id);

        let mut buf = Vec::new();
        for doc in docs {
            let entry = Entry {
                ts: Utc::now(),
                op: "upsert".to_string(),
                doc: Some(doc),
                delete_ids: Vec::new(),
            };
            let mut line = serde_json::to_vec(&entry)
                .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
            line.push(b'\n');
            buf.extend_from_slice(&line);
        }

        self.storage.put(&wal_path, buf).await?;
        Ok(Some(wal_path))
    }

    pub async fn write_delete(&self, ids: Vec<String>) -> Result<Option<String>, StorageError> {
        if ids.is_empty() {
            return Ok(None);
        }

        let date = Utc::now().format("%Y-%m-%d").to_string();
        let wal_id = Uuid::new_v4().to_string();
        let wal_path = wal_path(&self.namespace, &date, &wal_id);

        let entry = Entry {
            ts: Utc::now(),
            op: "delete".to_string(),
            doc: None,
            delete_ids: ids,
        };
        let mut buf = serde_json::to_vec(&entry)
            .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
        buf.push(b'\n');

        self.storage.put(&wal_path, buf).await?;
        Ok(Some(wal_path))
    }
}

#[derive(Clone)]
pub struct Reader<S: Store> {
    storage: S,
    namespace: String,
}

impl<S: Store> Reader<S> {
    pub fn new(storage: S, namespace: impl Into<String>) -> Self {
        Self {
            storage,
            namespace: namespace.into(),
        }
    }

    pub async fn list_wal_files(&self) -> Result<Vec<String>, StorageError> {
        let prefix = wal_prefix(&self.namespace);
        let mut keys = self.storage.list(&prefix).await?;
        keys.retain(|k| k.ends_with(".jsonl"));
        keys.sort();
        Ok(keys)
    }

    pub async fn read_wal_file(&self, wal_path: &str) -> Result<Vec<Entry>, StorageError> {
        let data = self.storage.get(wal_path).await?;
        let mut entries = Vec::new();
        for line in data.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let entry: Entry = serde_json::from_slice(line)
                .map_err(|err| StorageError::Other(format!("parse WAL entry: {}", err)))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    pub async fn read_all_wal_files(&self) -> Result<(Vec<Entry>, Vec<String>), StorageError> {
        let wal_files = self.list_wal_files().await?;
        let mut all_entries = Vec::new();
        for wal_file in &wal_files {
            let entries = self.read_wal_file(wal_file).await?;
            all_entries.extend(entries);
        }
        Ok((all_entries, wal_files))
    }

    pub async fn delete_wal_file(&self, wal_path: &str) -> Result<(), StorageError> {
        self.storage.delete(wal_path).await
    }
}

pub fn extract_date(wal_path: &str) -> Option<String> {
    let parts: Vec<&str> = wal_path.split('/').collect();
    if parts.len() < 5 {
        return None;
    }
    let date = parts.get(parts.len().saturating_sub(2))?;
    Some((*date).to_string())
}
