use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use chrono::Utc;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use uuid::Uuid;

use crate::document::{Document, RkyvDocument};
use crate::storage::{wal_path, wal_prefix, Store, StorageError};

/// WAL Entry stored with rkyv.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct Entry {
    pub ts: i64,
    pub op: String,
    pub doc: Option<RkyvDocument>,
    pub delete_ids: Vec<String>,
}

/// Deserialized WAL entry for API use.
#[derive(Debug, Clone)]
pub struct DeserializedEntry {
    pub ts: i64,
    pub op: String,
    pub doc: Option<Document>,
    pub delete_ids: Vec<String>,
}

impl From<Entry> for DeserializedEntry {
    fn from(entry: Entry) -> Self {
        Self {
            ts: entry.ts,
            op: entry.op,
            doc: entry.doc.map(|d| d.into()),
            delete_ids: entry.delete_ids,
        }
    }
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
                ts: now_ts_nanos(),
                op: "upsert".to_string(),
                doc: Some(RkyvDocument::from(&doc)),
                delete_ids: Vec::new(),
            };
            let entry_bytes = rkyv::to_bytes::<_, 256>(&entry)
                .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
            let entry_bytes = entry_bytes.as_ref();
            buf.write_u32::<LittleEndian>(entry_bytes.len() as u32)
                .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
            buf.extend_from_slice(entry_bytes);
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
            ts: now_ts_nanos(),
            op: "delete".to_string(),
            doc: None,
            delete_ids: ids,
        };
        let entry_bytes = rkyv::to_bytes::<_, 256>(&entry)
            .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
        let entry_bytes = entry_bytes.as_ref();
        let mut buf = Vec::with_capacity(entry_bytes.len() + 4);
        buf.write_u32::<LittleEndian>(entry_bytes.len() as u32)
            .map_err(|err| StorageError::Other(format!("encode WAL entry: {}", err)))?;
        buf.extend_from_slice(entry_bytes);

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
        keys.retain(|k| k.ends_with(".wal"));
        keys.sort();
        Ok(keys)
    }

    pub async fn read_wal_file(&self, wal_path: &str) -> Result<Vec<DeserializedEntry>, StorageError> {
        let data = self.storage.get(wal_path).await?;
        let mut entries = Vec::new();
        let mut cursor = Cursor::new(&data);
        while (cursor.position() as usize) < data.len() {
            let len = cursor
                .read_u32::<LittleEndian>()
                .map_err(|err| StorageError::Other(format!("parse WAL entry: {}", err)))? as usize;
            if len == 0 {
                continue;
            }
            let mut buf = vec![0u8; len];
            cursor
                .read_exact(&mut buf)
                .map_err(|err| StorageError::Other(format!("parse WAL entry: {}", err)))?;
            // SAFETY: We trust our own serialized data format
            let archived = unsafe { rkyv::archived_root::<Entry>(&buf) };
            let entry: Entry = archived
                .deserialize(&mut rkyv::Infallible)
                .map_err(|err| StorageError::Other(format!("parse WAL entry: {}", err)))?;
            entries.push(entry.into());
        }
        Ok(entries)
    }

    pub async fn read_all_wal_files(&self) -> Result<(Vec<DeserializedEntry>, Vec<String>), StorageError> {
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

fn now_ts_nanos() -> i64 {
    Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000)
}
