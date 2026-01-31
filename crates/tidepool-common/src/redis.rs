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
    url: String,
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
            url: url.to_string(),
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

// ============================================================================
// DISTRIBUTED LOCKING
// ============================================================================

/// Distributed lock for coordinating operations across instances
pub struct DistributedLock {
    conn: ConnectionManager,
    key: String,
    token: String,
    ttl_secs: u64,
}

impl DistributedLock {
    /// Release the lock
    pub async fn release(self) -> Result<bool, RedisError> {
        let mut conn = self.conn;
        
        // Use Lua script for atomic check-and-delete
        let script = r#"
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        "#;
        
        let result: i32 = redis::Script::new(script)
            .key(&self.key)
            .arg(&self.token)
            .invoke_async(&mut conn)
            .await?;
        
        Ok(result == 1)
    }
    
    /// Extend the lock TTL
    pub async fn extend(&mut self, extra_secs: u64) -> Result<bool, RedisError> {
        let script = r#"
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
        "#;
        
        let new_ttl = self.ttl_secs + extra_secs;
        let result: i32 = redis::Script::new(script)
            .key(&self.key)
            .arg(&self.token)
            .arg(new_ttl)
            .invoke_async(&mut self.conn)
            .await?;
        
        if result == 1 {
            self.ttl_secs = new_ttl;
        }
        
        Ok(result == 1)
    }
}

// ============================================================================
// QUERY CACHE ENTRY
// ============================================================================

/// Cached query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedQueryResult {
    pub results_json: String,
    pub created_at: i64,
    pub namespace: String,
}

impl RedisStore {
    // ========== Distributed Locking ==========

    /// Try to acquire a distributed lock
    /// Returns Some(lock) if acquired, None if lock is held by another instance
    pub async fn try_lock(
        &self,
        name: &str,
        ttl_secs: u64,
    ) -> Result<Option<DistributedLock>, RedisError> {
        let key = format!("{}:lock:{}", self.keys.prefix, name);
        let token = uuid::Uuid::new_v4().to_string();
        let mut conn = self.conn.clone();
        
        // SET key value NX EX ttl
        let result: Option<String> = redis::cmd("SET")
            .arg(&key)
            .arg(&token)
            .arg("NX")
            .arg("EX")
            .arg(ttl_secs)
            .query_async(&mut conn)
            .await?;
        
        if result.is_some() {
            info!("Acquired lock: {} (token: {})", name, &token[..8]);
            Ok(Some(DistributedLock {
                conn: self.conn.clone(),
                key,
                token,
                ttl_secs,
            }))
        } else {
            debug!("Failed to acquire lock: {} (already held)", name);
            Ok(None)
        }
    }
    
    /// Acquire a lock with retry
    pub async fn lock_with_retry(
        &self,
        name: &str,
        ttl_secs: u64,
        max_retries: usize,
        retry_delay_ms: u64,
    ) -> Result<Option<DistributedLock>, RedisError> {
        for attempt in 0..max_retries {
            if let Some(lock) = self.try_lock(name, ttl_secs).await? {
                return Ok(Some(lock));
            }
            if attempt < max_retries - 1 {
                tokio::time::sleep(std::time::Duration::from_millis(retry_delay_ms)).await;
            }
        }
        Ok(None)
    }

    // ========== Query Result Caching ==========
    
    /// Cache a query result
    pub async fn cache_query(
        &self,
        namespace: &str,
        query_hash: &str,
        results_json: &str,
        ttl_secs: u64,
    ) -> Result<(), RedisError> {
        let key = format!("{}:cache:{}:{}", self.keys.prefix, namespace, query_hash);
        let mut conn = self.conn.clone();
        
        let entry = CachedQueryResult {
            results_json: results_json.to_string(),
            created_at: chrono::Utc::now().timestamp(),
            namespace: namespace.to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap_or_default();
        
        conn.set_ex::<_, _, ()>(&key, &json, ttl_secs).await?;
        debug!("Cached query result: {} (TTL: {}s)", query_hash, ttl_secs);
        
        Ok(())
    }
    
    /// Get a cached query result
    pub async fn get_cached_query(
        &self,
        namespace: &str,
        query_hash: &str,
    ) -> Result<Option<CachedQueryResult>, RedisError> {
        let key = format!("{}:cache:{}:{}", self.keys.prefix, namespace, query_hash);
        let mut conn = self.conn.clone();
        
        let result: Option<String> = conn.get(&key).await?;
        
        match result {
            Some(json) => {
                match serde_json::from_str(&json) {
                    Ok(entry) => {
                        debug!("Cache hit: {}", query_hash);
                        Ok(Some(entry))
                    }
                    Err(_) => Ok(None),
                }
            }
            None => Ok(None),
        }
    }
    
    /// Invalidate all cached queries for a namespace
    pub async fn invalidate_query_cache(&self, namespace: &str) -> Result<u64, RedisError> {
        let pattern = format!("{}:cache:{}:*", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        // Use SCAN instead of KEYS to avoid blocking
        let mut cursor = 0u64;
        let mut total_deleted = 0u64;
        
        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut conn)
                .await?;
            
            if !keys.is_empty() {
                for key in keys {
                    let _: () = conn.del(&key).await?;
                    total_deleted += 1;
                }
            }
            
            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }
        
        info!("Invalidated {} cached queries for namespace {}", total_deleted, namespace);
        Ok(total_deleted)
    }

    // ========== Pub/Sub for Cache Invalidation ==========
    
    /// Publish a cache invalidation event
    pub async fn publish_invalidation(
        &self,
        namespace: &str,
        event_type: &str,
        payload: &str,
    ) -> Result<u64, RedisError> {
        let channel = format!("{}:invalidate:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let message = serde_json::json!({
            "type": event_type,
            "namespace": namespace,
            "payload": payload,
            "timestamp": chrono::Utc::now().timestamp_millis()
        });
        
        let subscribers: u64 = conn.publish(&channel, message.to_string()).await?;
        debug!(
            "Published invalidation event {} to {} ({} subscribers)",
            event_type, channel, subscribers
        );
        
        Ok(subscribers)
    }
    
    /// Subscribe to invalidation events for a namespace
    /// Returns a receiver that yields invalidation messages
    pub async fn subscribe_invalidations(
        &self,
        namespace: &str,
    ) -> Result<redis::aio::PubSub, RedisError> {
        let channel = format!("{}:invalidate:{}", self.keys.prefix, namespace);
        
        // Create a new connection for pub/sub (can't reuse ConnectionManager)
        let client = redis::Client::open(&*self.url)?;
        let mut pubsub = client.get_async_pubsub().await?;
        
        pubsub.subscribe(&channel).await?;
        info!("Subscribed to invalidation channel: {}", channel);
        
        Ok(pubsub)
    }
    
    /// Subscribe to invalidation events for all namespaces (pattern subscribe)
    pub async fn subscribe_all_invalidations(&self) -> Result<redis::aio::PubSub, RedisError> {
        let pattern = format!("{}:invalidate:*", self.keys.prefix);
        
        let client = redis::Client::open(&*self.url)?;
        let mut pubsub = client.get_async_pubsub().await?;
        
        pubsub.psubscribe(&pattern).await?;
        info!("Subscribed to invalidation pattern: {}", pattern);
        
        Ok(pubsub)
    }

    // ========== Shared Hot Buffer (Vector Storage) ==========
    
    /// Store vectors in Redis for shared hot buffer
    pub async fn store_hot_vectors(
        &self,
        namespace: &str,
        vectors: &[(String, Vec<f32>, Option<String>)], // (id, vector, attributes_json)
    ) -> Result<(), RedisError> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let key = format!("{}:hotvec:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        // Use HSET to store vectors as hash fields
        for (id, vector, attrs) in vectors {
            let vector_json = serde_json::to_string(vector).unwrap_or_default();
            let value = serde_json::json!({
                "vector": vector_json,
                "attributes": attrs,
            });
            let _: () = conn.hset(&key, id, value.to_string()).await?;
        }
        
        debug!("Stored {} hot vectors for namespace {}", vectors.len(), namespace);
        Ok(())
    }
    
    /// Get a hot vector by ID
    pub async fn get_hot_vector(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<(Vec<f32>, Option<String>)>, RedisError> {
        let key = format!("{}:hotvec:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let result: Option<String> = conn.hget(&key, id).await?;
        
        match result {
            Some(json) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json) {
                    let vector: Vec<f32> = value.get("vector")
                        .and_then(|v| v.as_str())
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or_default();
                    let attrs = value.get("attributes")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    Ok(Some((vector, attrs)))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }
    
    /// Delete hot vectors by ID
    pub async fn delete_hot_vectors(
        &self,
        namespace: &str,
        ids: &[String],
    ) -> Result<u64, RedisError> {
        if ids.is_empty() {
            return Ok(0);
        }
        
        let key = format!("{}:hotvec:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let deleted: u64 = conn.hdel(&key, ids).await?;
        debug!("Deleted {} hot vectors from namespace {}", deleted, namespace);
        
        Ok(deleted)
    }
    
    /// Get count of hot vectors
    pub async fn hot_vector_count(&self, namespace: &str) -> Result<u64, RedisError> {
        let key = format!("{}:hotvec:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let count: u64 = conn.hlen(&key).await?;
        Ok(count)
    }
    
    /// Clear all hot vectors for a namespace (after compaction)
    pub async fn clear_hot_vectors(&self, namespace: &str) -> Result<u64, RedisError> {
        let key = format!("{}:hotvec:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let count: u64 = conn.hlen(&key).await?;
        let _: () = conn.del(&key).await?;
        
        info!("Cleared {} hot vectors for namespace {}", count, namespace);
        Ok(count)
    }

    // ========== Redis Vector Search (FT module) ==========
    
    /// Create a vector search index for a namespace.
    /// Requires Redis Stack with RediSearch module.
    pub async fn create_vector_index(
        &self,
        namespace: &str,
        dimensions: usize,
    ) -> Result<(), RedisError> {
        let index_name = format!("{}:vecidx:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        // FT.CREATE idx ON HASH PREFIX 1 prefix: SCHEMA
        //   vector VECTOR HNSW 6 TYPE FLOAT32 DIM dims DISTANCE_METRIC COSINE
        //   id TEXT
        //   attrs TEXT
        let result: Result<String, _> = redis::cmd("FT.CREATE")
            .arg(&index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(format!("{}:vec:{}:", self.keys.prefix, namespace))
            .arg("SCHEMA")
            .arg("vector")
            .arg("VECTOR")
            .arg("HNSW")
            .arg("6")
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dimensions)
            .arg("DISTANCE_METRIC")
            .arg("COSINE")
            .arg("id")
            .arg("TEXT")
            .arg("attrs")
            .arg("TEXT")
            .query_async(&mut conn)
            .await;
        
        match result {
            Ok(_) => {
                info!("Created vector index {} with {} dimensions", index_name, dimensions);
                Ok(())
            }
            Err(e) => {
                // Index might already exist
                if e.to_string().contains("Index already exists") {
                    debug!("Vector index {} already exists", index_name);
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }
    
    /// Drop a vector search index.
    pub async fn drop_vector_index(&self, namespace: &str) -> Result<(), RedisError> {
        let index_name = format!("{}:vecidx:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        let _: Result<String, _> = redis::cmd("FT.DROPINDEX")
            .arg(&index_name)
            .arg("DD") // Delete documents too
            .query_async(&mut conn)
            .await;
        
        info!("Dropped vector index {}", index_name);
        Ok(())
    }
    
    /// Add vectors to the search index.
    pub async fn index_vectors(
        &self,
        namespace: &str,
        vectors: &[(String, Vec<f32>, Option<String>)], // (id, vector, attributes_json)
    ) -> Result<(), RedisError> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let mut conn = self.conn.clone();
        let prefix = format!("{}:vec:{}:", self.keys.prefix, namespace);
        
        for (id, vector, attrs) in vectors {
            let key = format!("{}{}", prefix, id);
            
            // Convert vector to bytes (little-endian f32)
            let vector_bytes: Vec<u8> = vector
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            
            // HSET key id value vector bytes attrs json
            let _: () = redis::cmd("HSET")
                .arg(&key)
                .arg("id")
                .arg(id)
                .arg("vector")
                .arg(&vector_bytes)
                .arg("attrs")
                .arg(attrs.as_deref().unwrap_or("{}"))
                .query_async(&mut conn)
                .await?;
        }
        
        debug!("Indexed {} vectors for namespace {}", vectors.len(), namespace);
        Ok(())
    }
    
    /// Remove vectors from the search index.
    pub async fn unindex_vectors(
        &self,
        namespace: &str,
        ids: &[String],
    ) -> Result<u64, RedisError> {
        if ids.is_empty() {
            return Ok(0);
        }
        
        let mut conn = self.conn.clone();
        let prefix = format!("{}:vec:{}:", self.keys.prefix, namespace);
        let mut deleted = 0u64;
        
        for id in ids {
            let key = format!("{}{}", prefix, id);
            let result: u64 = conn.del(&key).await?;
            deleted += result;
        }
        
        debug!("Unindexed {} vectors from namespace {}", deleted, namespace);
        Ok(deleted)
    }
    
    /// Search vectors using Redis Vector Search.
    /// Returns (id, score, attributes_json) tuples.
    pub async fn search_vectors(
        &self,
        namespace: &str,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&str>, // Optional FT.SEARCH filter expression
    ) -> Result<Vec<(String, f32, Option<String>)>, RedisError> {
        let index_name = format!("{}:vecidx:{}", self.keys.prefix, namespace);
        let mut conn = self.conn.clone();
        
        // Convert query vector to bytes
        let vector_bytes: Vec<u8> = query_vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        // Build query string
        let query = if let Some(f) = filter {
            format!("({})=>[KNN {} @vector $BLOB AS score]", f, top_k)
        } else {
            format!("*=>[KNN {} @vector $BLOB AS score]", top_k)
        };
        
        // FT.SEARCH idx query PARAMS 2 BLOB vector_bytes RETURN 3 id score attrs SORTBY score LIMIT 0 top_k DIALECT 2
        let result: Vec<redis::Value> = redis::cmd("FT.SEARCH")
            .arg(&index_name)
            .arg(&query)
            .arg("PARAMS")
            .arg("2")
            .arg("BLOB")
            .arg(&vector_bytes)
            .arg("RETURN")
            .arg("3")
            .arg("id")
            .arg("score")
            .arg("attrs")
            .arg("SORTBY")
            .arg("score")
            .arg("LIMIT")
            .arg("0")
            .arg(top_k)
            .arg("DIALECT")
            .arg("2")
            .query_async(&mut conn)
            .await?;
        
        // Parse results
        // Format: [total, key1, [field, value, field, value, ...], key2, [...], ...]
        let mut results = Vec::new();
        
        if result.len() < 2 {
            return Ok(results);
        }
        
        let mut i = 1; // Skip total count
        while i + 1 < result.len() {
            // Skip key name
            i += 1;
            
            if i >= result.len() {
                break;
            }
            
            // Parse field-value array
            if let redis::Value::Array(fields) = &result[i] {
                let mut id = String::new();
                let mut score = 0.0f32;
                let mut attrs = None;
                
                let mut j = 0;
                while j + 1 < fields.len() {
                    let field_name = match &fields[j] {
                        redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                        redis::Value::SimpleString(s) => s.clone(),
                        _ => String::new(),
                    };
                    
                    let field_value = &fields[j + 1];
                    
                    match field_name.as_str() {
                        "id" => {
                            id = match field_value {
                                redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                                redis::Value::SimpleString(s) => s.clone(),
                                _ => String::new(),
                            };
                        }
                        "score" => {
                            score = match field_value {
                                redis::Value::BulkString(b) => {
                                    String::from_utf8_lossy(b).parse().unwrap_or(0.0)
                                }
                                redis::Value::SimpleString(s) => s.parse().unwrap_or(0.0),
                                _ => 0.0,
                            };
                        }
                        "attrs" => {
                            attrs = match field_value {
                                redis::Value::BulkString(b) => {
                                    Some(String::from_utf8_lossy(b).to_string())
                                }
                                redis::Value::SimpleString(s) => Some(s.clone()),
                                _ => None,
                            };
                        }
                        _ => {}
                    }
                    
                    j += 2;
                }
                
                if !id.is_empty() {
                    // Convert distance to similarity score (cosine: score = 1 - distance)
                    let similarity = 1.0 - score;
                    results.push((id, similarity, attrs));
                }
            }
            
            i += 1;
        }
        
        debug!("Vector search returned {} results for namespace {}", results.len(), namespace);
        Ok(results)
    }
    
    /// Check if Redis Stack vector search is available.
    pub async fn has_vector_search(&self) -> bool {
        let mut conn = self.conn.clone();
        
        // Try to get module list
        let result: Result<Vec<redis::Value>, _> = redis::cmd("MODULE")
            .arg("LIST")
            .query_async(&mut conn)
            .await;
        
        match result {
            Ok(modules) => {
                // Check if 'search' or 'ReJSON' module is loaded
                for module in modules {
                    if let redis::Value::Array(arr) = module {
                        for item in arr {
                            if let redis::Value::BulkString(b) = item {
                                let name = String::from_utf8_lossy(&b).to_lowercase();
                                if name.contains("search") {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            }
            Err(_) => false,
        }
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
