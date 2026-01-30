use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use tidepool_common::document::Document;
use tidepool_common::document::RkyvDocument;
use tidepool_common::redis::RedisStore;
use tidepool_common::storage::{Store, StorageError};
use tidepool_common::wal::{Entry, Writer as WalWriter};

struct WalRequest {
    entries: Vec<Entry>,
    respond_to: oneshot::Sender<Result<(), StorageError>>,
}

#[derive(Clone)]
pub struct BufferedWalWriter<S: Store + Clone + Send + Sync + 'static> {
    sender: mpsc::Sender<WalRequest>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Store + Clone + Send + Sync + 'static> BufferedWalWriter<S> {
    pub fn new(writer: WalWriter<S>, max_entries: usize, flush_interval: Duration) -> Self {
        Self::new_with_redis(writer, max_entries, flush_interval, None, "".to_string())
    }

    /// Create a new buffered WAL writer with optional Redis dual-write
    pub fn new_with_redis(
        writer: WalWriter<S>,
        max_entries: usize,
        flush_interval: Duration,
        redis: Option<Arc<RedisStore>>,
        namespace: String,
    ) -> Self {
        let max_entries = max_entries.max(1);
        let (sender, mut receiver) = mpsc::channel::<WalRequest>(max_entries * 4);
        let flush_every = if flush_interval.is_zero() {
            Duration::from_secs(3600)
        } else {
            flush_interval
        };

        tokio::spawn(async move {
            let mut buffer: Vec<WalRequest> = Vec::new();
            let mut entry_count = 0usize;
            let mut ticker = tokio::time::interval(flush_every);

            loop {
                tokio::select! {
                    maybe_req = receiver.recv() => {
                        match maybe_req {
                            Some(req) => {
                                entry_count += req.entries.len();
                                buffer.push(req);
                                if max_entries > 0 && entry_count >= max_entries {
                                    flush_batch_with_redis(&writer, &mut buffer, &mut entry_count, redis.as_deref(), &namespace).await;
                                }
                            }
                            None => {
                                if !buffer.is_empty() {
                                    flush_batch_with_redis(&writer, &mut buffer, &mut entry_count, redis.as_deref(), &namespace).await;
                                }
                                break;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if !buffer.is_empty() {
                            flush_batch_with_redis(&writer, &mut buffer, &mut entry_count, redis.as_deref(), &namespace).await;
                        }
                    }
                }
            }
        });

        Self {
            sender,
            _phantom: std::marker::PhantomData,
        }
    }

    pub async fn write_upsert(&self, docs: Vec<Document>) -> Result<Option<String>, StorageError> {
        if docs.is_empty() {
            return Ok(None);
        }

        let entries: Vec<Entry> = docs
            .into_iter()
            .map(|doc| Entry {
                ts: now_ts_nanos(),
                op: "upsert".to_string(),
                doc: Some(RkyvDocument::from(&doc)),
                delete_ids: Vec::new(),
            })
            .collect();

        self.send_entries(entries).await?;
        Ok(None)
    }

    pub async fn write_delete(&self, ids: Vec<String>) -> Result<Option<String>, StorageError> {
        if ids.is_empty() {
            return Ok(None);
        }

        let entry = Entry {
            ts: now_ts_nanos(),
            op: "delete".to_string(),
            doc: None,
            delete_ids: ids,
        };

        self.send_entries(vec![entry]).await?;
        Ok(None)
    }

    async fn send_entries(&self, entries: Vec<Entry>) -> Result<(), StorageError> {
        let (respond_to, response) = oneshot::channel();
        let req = WalRequest { entries, respond_to };
        self.sender
            .send(req)
            .await
            .map_err(|_| StorageError::Other("wal batch channel closed".to_string()))?;
        response
            .await
            .map_err(|_| StorageError::Other("wal batch response dropped".to_string()))?
    }
}

async fn flush_batch_with_redis<S: Store>(
    writer: &WalWriter<S>,
    buffer: &mut Vec<WalRequest>,
    entry_count: &mut usize,
    redis: Option<&RedisStore>,
    namespace: &str,
) {
    if buffer.is_empty() {
        return;
    }

    let mut entries = Vec::with_capacity(*entry_count);
    let mut responders = Vec::with_capacity(buffer.len());
    for req in buffer.drain(..) {
        *entry_count = entry_count.saturating_sub(req.entries.len());
        entries.extend(req.entries);
        responders.push(req.respond_to);
    }

    // Dual-write: Redis first (for real-time visibility), then S3 (for durability)
    if let Some(redis_store) = redis {
        match redis_store.append_wal(namespace, &entries).await {
            Ok(ids) => {
                debug!("Wrote {} entries to Redis WAL (last id: {:?})", entries.len(), ids.last());
            }
            Err(e) => {
                // Log but don't fail - S3 is the durable store
                warn!("Failed to write to Redis WAL: {} (S3 write will continue)", e);
            }
        }
    }

    // Always write to S3 for durability
    let result = writer.write_entries(&entries).await.map(|_| ());
    for responder in responders {
        let _ = responder.send(result.as_ref().map(|_| ()).map_err(clone_error));
    }
}

fn clone_error(err: &StorageError) -> StorageError {
    match err {
        StorageError::NotFound(value) => StorageError::NotFound(value.clone()),
        StorageError::Other(value) => StorageError::Other(value.clone()),
    }
}

fn now_ts_nanos() -> i64 {
    Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000)
}
