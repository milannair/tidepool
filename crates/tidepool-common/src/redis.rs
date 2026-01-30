//! Redis-based shared hot buffer for real-time vector updates.
//!
//! Uses Redis Streams for WAL entries and simple keys for manifest/Bloom filters.
//! This enables all query nodes to see writes instantly without polling S3.

use std::collections::HashMap;
use std::sync::Arc;

use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client, RedisError};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::document::{Document, RkyvDocument};
use crate::wal::Entry;

/// Redis key helpers
pub struct RedisKeys {
    prefix: String,
}

impl RedisKeys {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    /// WAL stream key: {prefix}:wal:{namespace}
    pub fn wal_stream(&self, namespace: &str) -> String {
        format!("{}:wal:{}", self.prefix, namespace)
    }

    /// Manifest key: {prefix}:manifest:{namespace}
    pub fn manifest(&self, namespace: &str) -> String {
        format!("{}:manifest:{}", self.prefix, namespace)
    }

    /// Bloom filter key: {prefix}:bloom:{namespace}:{segment_id}
    pub fn bloom(&self, namespace: &str, segment_id: &str) -> String {
        format!("{}:bloom:{}:{}", self.prefix, namespace, segment_id)
    }

    /// Last compacted ID key: {prefix}:compacted:{namespace}
    pub fn last_compacted(&self, namespace: &str) -> String {
        format!("{}:compacted:{}", self.prefix, namespace)
    }

    /// Namespace list key: {prefix}:namespaces
    pub fn namespaces(&self) -> String {
        format!("{}:namespaces", self.prefix)
    }
}

/// WAL entry stored in Redis Stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisWalEntry {
    pub ts: i64,
    pub op: String,
    /// Serialized document (JSON for simplicity, could use msgpack for efficiency)
    pub doc: Option<String>,
    pub delete_ids: Vec<String>,
}

impl RedisWalEntry {
    pub fn from_entry(entry: &Entry) -> Self {
        Self {
            ts: entry.ts,
            op: entry.op.clone(),
            doc: entry.doc.as_ref().map(|d| {
                let doc: Document = d.clone().into();
                serde_json::to_string(&doc).unwrap_or_default()
            }),
            delete_ids: entry.delete_ids.clone(),
        }
    }

    pub fn to_entry(&self) -> Entry {
        Entry {
            ts: self.ts,
            op: self.op.clone(),
            doc: self.doc.as_ref().and_then(|s| {
                serde_json::from_str::<Document>(s)
                    .ok()
                    .map(|d| RkyvDocument::from(&d))
            }),
            delete_ids: self.delete_ids.clone(),
        }
    }

    pub fn to_document(&self) -> Option<Document> {
        self.doc
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
    }
}

/// Parsed stream entry with ID
#[derive(Debug, Clone)]
pub struct StreamEntry {
    pub id: String,
    pub entry: RedisWalEntry,
}

/// Redis client wrapper for Tidepool operations
#[derive(Clone)]
pub struct RedisStore {
    conn: ConnectionManager,
    keys: Arc<RedisKeys>,
    #[allow(dead_code)]
    wal_ttl_secs: u64,
}

impl RedisStore {
    /// Connect to Redis
    pub async fn new(url: &str, prefix: &str, wal_ttl_secs: u64) -> Result<Self, RedisError> {
        info!("Connecting to Redis at {}", url);
        let client = Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;
        info!("Redis connection established");

        Ok(Self {
            conn,
            keys: Arc::new(RedisKeys::new(prefix)),
            wal_ttl_secs,
        })
    }

    /// Check if Redis is connected
    pub async fn ping(&self) -> Result<(), RedisError> {
        let mut conn = self.conn.clone();
        redis::cmd("PING").query_async(&mut conn).await
    }

    // ========== WAL Stream Operations ==========

    /// Append entries to the WAL stream for a namespace
    pub async fn append_wal(
        &self,
        namespace: &str,
        entries: &[Entry],
    ) -> Result<Vec<String>, RedisError> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let key = self.keys.wal_stream(namespace);
        let mut conn = self.conn.clone();
        let mut ids = Vec::with_capacity(entries.len());

        // Register namespace
        let ns_key = self.keys.namespaces();
        let _: () = conn.sadd(&ns_key, namespace).await?;

        for entry in entries {
            let redis_entry = RedisWalEntry::from_entry(entry);
            let json = serde_json::to_string(&redis_entry).unwrap_or_default();

            // XADD with auto-generated ID
            let id: String = redis::cmd("XADD")
                .arg(&key)
                .arg("*")
                .arg("data")
                .arg(&json)
                .query_async(&mut conn)
                .await?;

            ids.push(id);
        }

        debug!(
            "Appended {} entries to Redis WAL for namespace {}",
            entries.len(),
            namespace
        );

        Ok(ids)
    }

    /// Read WAL entries from a namespace, optionally starting from a specific ID
    /// Returns entries and their stream IDs
    pub async fn read_wal(
        &self,
        namespace: &str,
        from_id: Option<&str>,
        count: Option<usize>,
    ) -> Result<Vec<StreamEntry>, RedisError> {
        let key = self.keys.wal_stream(namespace);
        let mut conn = self.conn.clone();

        let start = from_id.unwrap_or("0");
        let end = "+";

        // XRANGE key start end [COUNT count]
        let mut cmd = redis::cmd("XRANGE");
        cmd.arg(&key).arg(start).arg(end);
        if let Some(c) = count {
            cmd.arg("COUNT").arg(c);
        }

        let result: Vec<(String, HashMap<String, String>)> = cmd.query_async(&mut conn).await?;

        let mut entries = Vec::with_capacity(result.len());
        for (id, fields) in result {
            if let Some(data) = fields.get("data") {
                if let Ok(entry) = serde_json::from_str::<RedisWalEntry>(data) {
                    entries.push(StreamEntry { id, entry });
                }
            }
        }

        Ok(entries)
    }

    /// Read WAL entries newer than a given ID (exclusive)
    pub async fn read_wal_after(
        &self,
        namespace: &str,
        after_id: &str,
    ) -> Result<Vec<StreamEntry>, RedisError> {
        let key = self.keys.wal_stream(namespace);
        let mut conn = self.conn.clone();

        // Use exclusive range by adding a small increment
        let start = format!("({}", after_id);

        let result: Vec<(String, HashMap<String, String>)> = redis::cmd("XRANGE")
            .arg(&key)
            .arg(&start)
            .arg("+")
            .query_async(&mut conn)
            .await?;

        let mut entries = Vec::with_capacity(result.len());
        for (id, fields) in result {
            if let Some(data) = fields.get("data") {
                if let Ok(entry) = serde_json::from_str::<RedisWalEntry>(data) {
                    entries.push(StreamEntry { id, entry });
                }
            }
        }

        Ok(entries)
    }

    /// Trim WAL entries up to (and including) the given ID
    /// Called after compaction to remove processed entries
    pub async fn trim_wal(&self, namespace: &str, up_to_id: &str) -> Result<u64, RedisError> {
        let key = self.keys.wal_stream(namespace);
        let mut conn = self.conn.clone();

        // Parse the stream ID to create one that's just after it
        // Stream IDs are formatted as "timestamp-sequence"
        let next_id = if let Some((ts, seq)) = up_to_id.split_once('-') {
            if let (Ok(ts_num), Ok(seq_num)) = (ts.parse::<u64>(), seq.parse::<u64>()) {
                format!("{}-{}", ts_num, seq_num + 1)
            } else {
                up_to_id.to_string()
            }
        } else {
            up_to_id.to_string()
        };

        // XTRIM key MINID ~ id (keeps entries with ID >= id)
        // Using MINID with next_id to remove all entries <= up_to_id
        let trimmed: u64 = redis::cmd("XTRIM")
            .arg(&key)
            .arg("MINID")
            .arg(&next_id)
            .query_async(&mut conn)
            .await?;

        // Store the last compacted ID
        let compacted_key = self.keys.last_compacted(namespace);
        let _: () = conn.set(&compacted_key, up_to_id).await?;

        info!(
            "Trimmed {} entries from Redis WAL for namespace {} (up to {})",
            trimmed, namespace, up_to_id
        );

        Ok(trimmed)
    }

    /// Get the last compacted stream ID for a namespace
    pub async fn get_last_compacted_id(&self, namespace: &str) -> Result<Option<String>, RedisError> {
        let key = self.keys.last_compacted(namespace);
        let mut conn = self.conn.clone();
        conn.get(&key).await
    }

    /// Get the length of the WAL stream
    pub async fn wal_len(&self, namespace: &str) -> Result<u64, RedisError> {
        let key = self.keys.wal_stream(namespace);
        let mut conn = self.conn.clone();
        let len: u64 = redis::cmd("XLEN")
            .arg(&key)
            .query_async(&mut conn)
            .await?;
        Ok(len)
    }

    // ========== Manifest Operations ==========

    /// Store the current manifest JSON
    pub async fn set_manifest(&self, namespace: &str, manifest_json: &str) -> Result<(), RedisError> {
        let key = self.keys.manifest(namespace);
        let mut conn = self.conn.clone();
        conn.set(&key, manifest_json).await
    }

    /// Get the current manifest JSON
    pub async fn get_manifest(&self, namespace: &str) -> Result<Option<String>, RedisError> {
        let key = self.keys.manifest(namespace);
        let mut conn = self.conn.clone();
        conn.get(&key).await
    }

    // ========== Bloom Filter Operations ==========

    /// Store a Bloom filter
    pub async fn set_bloom(
        &self,
        namespace: &str,
        segment_id: &str,
        data: &[u8],
    ) -> Result<(), RedisError> {
        let key = self.keys.bloom(namespace, segment_id);
        let mut conn = self.conn.clone();
        conn.set(&key, data).await
    }

    /// Get a Bloom filter
    pub async fn get_bloom(
        &self,
        namespace: &str,
        segment_id: &str,
    ) -> Result<Option<Vec<u8>>, RedisError> {
        let key = self.keys.bloom(namespace, segment_id);
        let mut conn = self.conn.clone();
        conn.get(&key).await
    }

    /// Delete a Bloom filter
    pub async fn delete_bloom(&self, namespace: &str, segment_id: &str) -> Result<(), RedisError> {
        let key = self.keys.bloom(namespace, segment_id);
        let mut conn = self.conn.clone();
        conn.del(&key).await
    }

    // ========== Namespace Operations ==========

    /// List all namespaces with WAL data
    pub async fn list_namespaces(&self) -> Result<Vec<String>, RedisError> {
        let key = self.keys.namespaces();
        let mut conn = self.conn.clone();
        let members: Vec<String> = conn.smembers(&key).await?;
        Ok(members)
    }
}

/// Thread-safe wrapper for optional Redis store
#[derive(Clone)]
pub struct OptionalRedis {
    store: Option<RedisStore>,
}

impl OptionalRedis {
    pub fn new(store: Option<RedisStore>) -> Self {
        Self { store }
    }

    pub fn none() -> Self {
        Self { store: None }
    }

    pub fn is_enabled(&self) -> bool {
        self.store.is_some()
    }

    pub fn store(&self) -> Option<&RedisStore> {
        self.store.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests require a running Redis instance
    // Run with: REDIS_TEST_URL=redis://localhost:6379 cargo test redis --features redis-tests

    #[tokio::test]
    #[ignore = "requires Redis"]
    async fn test_redis_wal_operations() {
        let url = std::env::var("REDIS_TEST_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let store = RedisStore::new(&url, "test", 3600).await.unwrap();

        // Clean up
        let mut conn = store.conn.clone();
        let _: () = redis::cmd("DEL")
            .arg("test:wal:test_ns")
            .query_async(&mut conn)
            .await
            .unwrap();

        // Append entries
        let entries = vec![
            Entry {
                ts: 1000,
                op: "upsert".to_string(),
                doc: None,
                delete_ids: vec![],
            },
            Entry {
                ts: 2000,
                op: "delete".to_string(),
                doc: None,
                delete_ids: vec!["id1".to_string()],
            },
        ];

        let ids = store.append_wal("test_ns", &entries).await.unwrap();
        assert_eq!(ids.len(), 2);

        // Read all entries
        let read = store.read_wal("test_ns", None, None).await.unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].entry.ts, 1000);
        assert_eq!(read[1].entry.ts, 2000);

        // Read after first entry
        let read_after = store.read_wal_after("test_ns", &ids[0]).await.unwrap();
        assert_eq!(read_after.len(), 1);
        assert_eq!(read_after[0].entry.ts, 2000);

        // Check length
        let len = store.wal_len("test_ns").await.unwrap();
        assert_eq!(len, 2);
    }
}
